from __future__ import annotations

import os
import json
import random
import argparse
import faiss
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from spider_official import build_kmaps_from_tables_json, spider_official_exec_match

# ================= 1. 配置参数 =================
_ROOT = Path(__file__).resolve().parent
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TOP_K = 3
MAX_NEW_TOKENS = 150

# 本地 Spider 数据集路径 (请根据实际情况修改)
TRAIN_DATA_PATH = "spider_data/train_spider.json"
DEV_DATA_PATH = "spider_data/dev.json"
TEST_DATA_PATH = "spider_data/test.json"
TABLES_DATA_PATH = "spider_data/tables.json"
TEST_TABLES_DATA_PATH = "spider_data/test_tables.json"
DATABASE_DIR = "spider_data/database"
TEST_DATABASE_DIR = "spider_data/test_database"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. 加载模型 =================
print("正在加载 Embedding 模型...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

print("正在加载 LLM 模型...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)


# ================= 3. 构建本地 RAG 向量库 =================
def build_vector_db():
    print(f"正在读取本地训练集构建向量库: {TRAIN_DATA_PATH}")
    train_file = _ROOT / TRAIN_DATA_PATH
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    questions = [item["question"] for item in train_data]
    sqls = [item["query"] for item in train_data]

    question_embeddings = embedding_model.encode(
        questions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(question_embeddings.shape[1])
    index.add(question_embeddings)
    return index, questions, sqls


# ================= 4. Schema 获取与 RAG 检索 =================
_schema_cache = None


def _schema_text_from_db_info(db_info):
    schema_text = ""
    table_names = db_info["table_names_original"]
    column_names = db_info["column_names_original"]

    tables_dict = {t_name: [] for t_name in table_names}
    for col in column_names:
        if col[0] >= 0:
            tables_dict[table_names[col[0]]].append(col[1])

    for t_name, cols in tables_dict.items():
        schema_text += f"Table {t_name}, columns: {', '.join(cols)}\n"
    return schema_text.strip()


def _ensure_schema_cache():
    global _schema_cache
    if _schema_cache is not None:
        return
    _schema_cache = {}
    for rel in (TABLES_DATA_PATH, TEST_TABLES_DATA_PATH):
        path = _ROOT / rel
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for db_info in json.load(f):
                _schema_cache[db_info["db_id"]] = _schema_text_from_db_info(db_info)


def get_schema(db_id):
    _ensure_schema_cache()
    return _schema_cache.get(db_id, "")


def retrieve_few_shots(query, index, questions_list, sqls_list):
    query_vector = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    distances, indices = index.search(query_vector, TOP_K)
    few_shot_prompt = ""
    for idx in indices[0]:
        few_shot_prompt += f"Question: {questions_list[idx]}\nSQL: {sqls_list[idx]}\n\n"
    return few_shot_prompt.strip()


def generate_sql(user_question, db_id, faiss_index, train_qs, train_sqls):
    schema_info = get_schema(db_id)
    few_shot_examples = retrieve_few_shots(user_question, faiss_index, train_qs, train_sqls)

    prompt = f"""You are a powerful Text-to-SQL assistant. 
Please generate ONLY a valid SQL query based on the following database schema and the user's question. 
Do not output any explanation.

[Database Schema]
{schema_info}

[Similar Examples]
{few_shot_examples}

[User Question]
{user_question}

[SQL Query]
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only outputs valid SQL code."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = llm_model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 清理大模型可能输出的 Markdown 符号
    return response.strip().replace("```sql", "").replace("```", "").strip()


# ================= 5. 评测（EX = Spider 官方 eval_exec_match，见 spider_official）=================
_spider_kmaps: dict | None = None


def _get_spider_kmaps() -> dict:
    global _spider_kmaps
    if _spider_kmaps is None:
        _spider_kmaps = build_kmaps_from_tables_json(
            [_ROOT / TABLES_DATA_PATH, _ROOT / TEST_TABLES_DATA_PATH]
        )
    return _spider_kmaps


def evaluate_execution_accuracy(faiss_index, train_qs, train_sqls, num_samples=100, *, eval_json: str, eval_db_dir: str):
    """批量评测执行准确率"""
    data_file = _ROOT / eval_json
    if not data_file.is_file():
        raise FileNotFoundError(f"评测数据不存在: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    test_data = rows[:num_samples] if num_samples else rows
    db_root = _ROOT / eval_db_dir

    correct_count = 0
    total_count = len(test_data)

    kmaps = _get_spider_kmaps()
    print(f"\n开始评测，共测试 {total_count} 条数据...")
    print("EX 判定：Spider 官方 eval_exec_match（taoyds/spider，与 Text2SQL.py 一致）")

    for item in tqdm(test_data, desc="Evaluating"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        # 1. 大模型生成预测 SQL
        pred_sql = generate_sql(question, db_id, faiss_index, train_qs, train_sqls)

        # 2. 定位对应的 SQLite 数据库文件
        db_path = str(db_root / db_id / f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"[警告] 找不到数据库文件: {db_path}，跳过此题")
            total_count -= 1
            continue

        if spider_official_exec_match(pred_sql, gold_sql, db_path, db_id, kmaps):
            correct_count += 1

    # 计算并打印最终准确率
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n" + "=" * 40)
    print("评测完成！")
    print(f"总测试数: {total_count}")
    print(f"正确执行数: {correct_count}")
    print(f"Execution Accuracy (EX): {accuracy:.2f}%")
    print("=" * 40)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================= 6. 主程序运行 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2SQL2 baseline (simple RAG + Qwen)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（与主实验对齐；贪心解码下 EX 几乎不随 seed 变）",
    )
    parser.add_argument("--eval-samples", type=int, default=500, help="评测条数，0 表示全量")
    parser.add_argument("--split", type=str, choices=["test", "dev"], default="test", help="test 或 dev")
    args = parser.parse_args()

    _set_seed(args.seed)

    if args.split == "dev":
        eval_json, eval_db_dir = DEV_DATA_PATH, DATABASE_DIR
    else:
        eval_json, eval_db_dir = TEST_DATA_PATH, TEST_DATABASE_DIR

    n_eval = args.eval_samples if args.eval_samples else 0

    print(f"seed={args.seed} split={args.split} eval_samples={args.eval_samples}")
    print(f"eval_json={eval_json} eval_db_dir={eval_db_dir}")

    index, train_questions, train_queries = build_vector_db()
    evaluate_execution_accuracy(
        index,
        train_questions,
        train_queries,
        num_samples=n_eval if n_eval > 0 else 0,
        eval_json=eval_json,
        eval_db_dir=eval_db_dir,
    )
