from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

from spider_official import build_kmaps_from_tables_json, spider_official_exec_match

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
CHAT_MODEL_NAME = os.environ.get("CHAT_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
CHAT_LORA_PATH = os.environ.get("CHAT_LORA_PATH", "").strip()
TOP_K = 3
MAX_COMPLETION_TOKENS = 300
EMBED_BATCH_SIZE = 256
EMBED_CACHE_PATH = "cache/train_question_embeddings.npz"
REPAIR_ON_ERROR = True
WIDE_RECALL_TOP_N = 30
LEXICAL_WEIGHT = 0.3
EVAL_SAMPLES = 500
SEED = int(os.environ.get("TEXT2SQL_SEED", "42"))
ENABLE_TRACE_JSON = True
TRACE_JSON_PATH = "cache/eval_trace.jsonl"
PLAN_MAX_TOKENS = 300

TRAIN_DATA_PATH = "spider_data/train_spider.json"
DEV_DATA_PATH = "spider_data/dev.json"
TEST_DATA_PATH = "spider_data/test.json"
TABLES_DATA_PATH = "spider_data/tables.json"
TEST_TABLES_DATA_PATH = "spider_data/test_tables.json"
DATABASE_DIR = "spider_data/database"
TEST_DATABASE_DIR = "spider_data/test_database"
SPIDER_EVAL_SPLIT = os.environ.get("SPIDER_EVAL_SPLIT", "test").strip().lower()

_schema_cache = None
_embedding_model = None
_tokenizer = None
_llm_model = None
_torch = None
_device = None

_ROOT = Path(__file__).resolve().parent


def _resolve_data_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else _ROOT / p


def _resolve_lora_dir() -> Path | None:
    raw = (CHAT_LORA_PATH or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return p if p.is_dir() else None


def _lora_weights_look_valid(lora_dir: Path) -> bool:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = lora_dir / name
        if f.is_file() and f.stat().st_size > 10_000:
            return True
    return False


def _load_local_model_deps():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Install torch, sentence-transformers, and transformers for local Qwen inference."
        ) from e
    return torch, SentenceTransformer, AutoTokenizer, AutoModelForCausalLM


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _, _ = _load_local_model_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> str:
    global _torch, _device
    if _device is None:
        _torch, _, _, _ = _load_local_model_deps()
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    return _device


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _, SentenceTransformer, _, _ = _load_local_model_deps()
        device = _get_device()
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _embedding_model


def _get_llm():
    global _tokenizer, _llm_model, _torch
    if _tokenizer is None or _llm_model is None:
        _torch, _, AutoTokenizer, AutoModelForCausalLM = _load_local_model_deps()
        device = _get_device()
        print("Loading LLM...")
        _tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME, trust_remote_code=True)
        load_kwargs: dict = {"trust_remote_code": True}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["dtype"] = _torch.float16
        else:
            load_kwargs["torch_dtype"] = _torch.float32
        _llm_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME, **load_kwargs)
        lora_dir = _resolve_lora_dir()
        if lora_dir is not None:
            if not _lora_weights_look_valid(lora_dir):
                print(f"[warn] LoRA adapter missing or too small; using base only: {lora_dir}")
            else:
                try:
                    from peft import PeftModel
                except ImportError as e:
                    raise RuntimeError("pip install peft for LoRA") from e
                print(f"Loading LoRA: {lora_dir}")
                _llm_model = PeftModel.from_pretrained(_llm_model, str(lora_dir))
        if device != "cuda":
            _llm_model = _llm_model.to(device)
        _llm_model.eval()
    return _tokenizer, _llm_model


def embed_texts(texts: list[str], desc: str = "Embedding") -> np.ndarray:
    embedding_model = _get_embedding_model()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc=desc):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        arr = embedding_model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        chunks.append(arr)
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def _load_embedding_cache():
    if not os.path.exists(EMBED_CACHE_PATH):
        return None, None, None
    try:
        data = np.load(EMBED_CACHE_PATH, allow_pickle=True)
        vectors = data["vectors"].astype("float32")
        signatures = data["signatures"].tolist()
        model_name = data["model_name"].tolist() if "model_name" in data.files else None
        return vectors, signatures, model_name
    except Exception:
        return None, None, None


def _save_embedding_cache(vectors: np.ndarray, signatures: list[str]) -> None:
    os.makedirs(os.path.dirname(EMBED_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        EMBED_CACHE_PATH,
        vectors=vectors.astype("float32"),
        signatures=np.array(signatures, dtype=object),
        model_name=np.array(EMBED_MODEL_NAME, dtype=object),
    )


def build_vector_db():
    print(f"Loading train split for index: {TRAIN_DATA_PATH}")
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    questions = [item["question"] for item in train_data]
    sqls = [item["query"] for item in train_data]
    db_ids = [item["db_id"] for item in train_data]
    signatures = [f"{db}|||{q}" for db, q in zip(db_ids, questions)]

    cached_vectors, cached_signatures, cached_model_name = _load_embedding_cache()
    if (
        cached_vectors is not None
        and cached_signatures is not None
        and cached_model_name == EMBED_MODEL_NAME
        and len(cached_signatures) == len(signatures)
        and cached_signatures == signatures
    ):
        print(f"Using embedding cache: {EMBED_CACHE_PATH}")
        question_embeddings = cached_vectors
    else:
        if cached_vectors is not None and cached_model_name != EMBED_MODEL_NAME:
            print(f"Embedding model changed ({cached_model_name} -> {EMBED_MODEL_NAME}); rebuilding cache.")
        question_embeddings = embed_texts(questions, desc="Embedding train questions")
        _save_embedding_cache(question_embeddings, signatures)
        print(f"Wrote embedding cache: {EMBED_CACHE_PATH}")

    index = faiss.IndexFlatIP(question_embeddings.shape[1])
    index.add(question_embeddings)
    return index, questions, sqls, db_ids


def _schema_cache_from_tables_json(tables_data: list) -> dict:
    cache = {}
    for db_info in tables_data:
        table_names = db_info["table_names_original"]
        column_names = db_info["column_names_original"]
        column_types = db_info.get("column_types", [])
        primary_keys = set(db_info.get("primary_keys", []))
        foreign_keys = db_info.get("foreign_keys", [])

        tables_dict = {t_name: [] for t_name in table_names}
        col_idx_to_fullname = {}
        for idx, col in enumerate(column_names):
            t_idx, c_name = col[0], col[1]
            if t_idx < 0:
                continue
            col_type = column_types[idx] if idx < len(column_types) else "unknown"
            pk_flag = " [PK]" if idx in primary_keys else ""
            t_name = table_names[t_idx]
            tables_dict[t_name].append(f"{t_name}.{c_name} ({col_type}){pk_flag}")
            col_idx_to_fullname[idx] = f"{table_names[t_idx]}.{c_name}"

        lines = []
        for t_name, cols in tables_dict.items():
            lines.append(f"Table {t_name}, columns: {', '.join(cols)}")

        if foreign_keys:
            lines.append("Foreign Keys:")
            for src_idx, tgt_idx in foreign_keys:
                left = col_idx_to_fullname.get(src_idx, f"col_{src_idx}")
                right = col_idx_to_fullname.get(tgt_idx, f"col_{tgt_idx}")
                lines.append(f"- {left} -> {right}")

        cache[db_info["db_id"]] = "\n".join(lines).strip()
    return cache


def _load_schema_cache():
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    cache = {}
    for rel in (TABLES_DATA_PATH, TEST_TABLES_DATA_PATH):
        path = _resolve_data_path(rel)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
        cache.update(_schema_cache_from_tables_json(tables_data))

    _schema_cache = cache
    return _schema_cache


def get_schema(db_id: str) -> str:
    return _load_schema_cache().get(db_id, "")


def _tokenize_for_lexical(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def retrieve_few_shots_two_stage(
    query: str,
    db_id: str,
    index,
    questions_list: list[str],
    sqls_list: list[str],
    db_ids_list: list[str],
) -> str:
    qv = embed_texts([query], desc="Embedding query")[0:1]
    search_k = min(max(WIDE_RECALL_TOP_N, TOP_K), len(questions_list))
    distances, indices = index.search(qv, search_k)

    query_tokens = _tokenize_for_lexical(query)
    db_ids_arr = np.array(db_ids_list)
    candidates: list[tuple[float, int]] = []

    for rank, idx in enumerate(indices[0]):
        idx = int(idx)
        emb_score = float(distances[0][rank])
        cand_tokens = _tokenize_for_lexical(questions_list[idx])
        if query_tokens or cand_tokens:
            lex_score = len(query_tokens & cand_tokens) / max(len(query_tokens | cand_tokens), 1)
        else:
            lex_score = 0.0
        same_db_bonus = 0.1 if db_ids_arr[idx] == db_id else 0.0
        final_score = (1.0 - LEXICAL_WEIGHT) * emb_score + LEXICAL_WEIGHT * lex_score + same_db_bonus
        candidates.append((final_score, idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [idx for _, idx in candidates[:TOP_K]]

    parts = []
    for idx in selected:
        parts.append(f"Question: {questions_list[idx]}\nSQL: {sqls_list[idx]}\n")
    return "\n".join(parts).strip()


def _llm_completion(messages: list[dict], max_new_tokens: int) -> str:
    tokenizer, llm_model = _get_llm()
    device = _get_device()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if device == "cuda":
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with _torch.no_grad():
        outputs = llm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _strip_code_fences(text: str) -> str:
    return text.replace("```sql", "").replace("```", "").strip()


def _finalize_sql_text(text: str) -> str:
    m = re.search(r"<SQL>\s*(.*?)\s*</SQL>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return _strip_code_fences(m.group(1))
    return _strip_code_fences(text)


def _chat_generate(messages: list[dict], max_new_tokens: int) -> str:
    return _strip_code_fences(_llm_completion(messages, max_new_tokens))


def _chat_sql(messages: list[dict]) -> str:
    return _finalize_sql_text(_llm_completion(messages, MAX_COMPLETION_TOKENS))


def _extract_json_object(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _schema_foreign_keys(schema_info: str) -> set[tuple[str, str]]:
    fks = set()
    if not schema_info:
        return fks
    in_fk = False
    for line in schema_info.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("foreign keys"):
            in_fk = True
            continue
        if in_fk and line.startswith("- "):
            parts = line[2:].split("->")
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                fks.add((left, right))
                fks.add((right, left))
    return fks


def _schema_table_names(schema_info: str) -> set[str]:
    tables = set()
    if not schema_info:
        return tables
    for line in schema_info.splitlines():
        line = line.strip()
        if line.startswith("Table ") and ", columns:" in line:
            tables.add(line[len("Table ") : line.index(", columns:")].strip())
    return tables


def _tables_referenced_in_plan(plan: dict, schema_info: str) -> set[str]:
    tables = set()
    if not isinstance(plan, dict):
        return tables
    schema_tables = _schema_table_names(schema_info)

    def add_from_val(v):
        if isinstance(v, str):
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", v):
                table_name = match.group(1).strip()
                if table_name in schema_tables:
                    tables.add(table_name)
        elif isinstance(v, list):
            for x in v:
                add_from_val(x)
        elif isinstance(v, dict):
            for x in v.values():
                add_from_val(x)

    for key in ("select", "group_by", "order_by", "filters"):
        add_from_val(plan.get(key))
    add_from_val(plan.get("joins"))
    return tables


def validate_plan_json(plan: dict, schema_info: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return False, ["plan is not an object"]

    fk_pairs = _schema_foreign_keys(schema_info)
    joins = plan.get("joins") or []
    if not isinstance(joins, list):
        errors.append("joins must be a list")
        joins = []
    if fk_pairs:
        for j in joins:
            if not isinstance(j, dict):
                errors.append("join item must be an object")
                continue
            left, right = j.get("left"), j.get("right")
            if not (isinstance(left, str) and isinstance(right, str)):
                errors.append("join.left and join.right must be strings")
                continue
            if (left.strip(), right.strip()) not in fk_pairs:
                errors.append(f"join keys not in foreign keys: {left} -> {right}")

    plan_tables = plan.get("tables") or []
    if not isinstance(plan_tables, list) or not all(isinstance(t, str) for t in plan_tables):
        errors.append("tables must be a list of strings")
        plan_tables = []
    plan_tables_set = {t.strip() for t in plan_tables if t and t.strip()}

    referenced_tables = _tables_referenced_in_plan(plan, schema_info)
    missing = sorted(referenced_tables - plan_tables_set)
    if missing:
        errors.append(f"tables missing referenced tables: {missing}")

    unused = sorted(plan_tables_set - referenced_tables)
    if unused:
        errors.append(f"tables include unused tables: {unused}")

    expected = plan.get("expected_row_behavior")
    limit = plan.get("limit")
    if expected is not None and expected not in ("single", "multi", "one_per_group"):
        errors.append("expected_row_behavior must be single|multi|one_per_group|null")
    if limit is not None and not (isinstance(limit, int) and limit >= 1):
        errors.append("limit must be a positive integer or null")
    if expected in ("multi", "one_per_group") and limit == 1:
        errors.append("limit=1 conflicts with expected_row_behavior != single")

    distinct = plan.get("distinct")
    if distinct is not None and not isinstance(distinct, bool):
        errors.append("distinct must be boolean or null")

    return len(errors) == 0, errors


def plan_query_json(user_question: str, schema_info: str) -> tuple[dict | None, bool, str | None]:
    fk_hint = "Foreign keys listed under [Database Schema] MUST be used for any JOIN keys.\n"

    plan_prompt = f"""You will create a JOIN plan for a Text-to-SQL task.
Return ONLY valid JSON, no markdown.

Schema fidelity (apply to select/filters/group_by/order_by):
- Treat [Database Schema] as the only source of tables/columns; do NOT use prior knowledge or assume standard normalization.
- If a word in the question matches or closely matches a physical column name, plan to use that column rather than substituting an aggregate on another interpretation.
- When listing or grouping by entities (e.g. "each stadium", "which singer"), prefer human-readable name/title columns over id columns unless the question explicitly asks for id.

{fk_hint}
JSON schema:
{{
  "tables": ["table1", "table2"],
  "joins": [{{"left": "tableA.col", "right": "tableB.col", "type": "inner|left"}}],
  "select": ["table.col", "..."],
  "filters": ["..."],
  "group_by": ["table.col", "..."],
  "order_by": ["table.col ASC|DESC", "..."],
  "distinct": false,
  "expected_row_behavior": "single|multi|one_per_group",
  "limit": null,
  "table_usage": {{"table1": "why needed", "table2": "why needed"}}
}}

Rules:
- Every column must be fully-qualified as table.column.
- If a JOIN is needed, each join key pair MUST match one of the foreign keys in the schema (either direction).
- Include ONLY the minimal set of tables needed. Do NOT include unused tables.
- Ensure plan.tables contains exactly all tables referenced in joins/select/filters/group_by/order_by.
- If the question asks for min/max/youngest/oldest/most/least, do not assume LIMIT 1; plan for ties.

[Database Schema]
{schema_info}

[User Question]
{user_question}
"""
    raw = _chat_generate(
        [
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": plan_prompt},
        ],
        int(PLAN_MAX_TOKENS),
    )
    plan = _extract_json_object(raw)
    if not isinstance(plan, dict):
        return None, False, raw

    ok, errors = validate_plan_json(plan, schema_info)
    if not ok:
        plan["_validation_errors"] = errors
    return plan, ok, raw


def repair_plan_json(
    user_question: str,
    schema_info: str,
    bad_plan: dict | None,
    raw_text: str | None,
    validation_errors: list[str],
) -> tuple[dict | None, bool, str | None]:
    bad_plan_text = json.dumps(bad_plan, ensure_ascii=False) if isinstance(bad_plan, dict) else "null"
    raw_text = raw_text or ""
    error_text = "\n".join(f"- {e}" for e in validation_errors) if validation_errors else "- invalid JSON structure"
    repair_prompt = f"""You will repair a JOIN plan for a Text-to-SQL task.
Return ONLY valid JSON, no markdown.

Schema fidelity (apply to select/filters/group_by/order_by):
- Treat [Database Schema] as the only source of tables/columns; do NOT use prior knowledge or assume standard normalization.
- If a word in the question matches or closely matches a physical column name, plan to use that column rather than substituting an aggregate on another interpretation.
- When listing or grouping by entities (e.g. "each stadium", "which singer"), prefer human-readable name/title columns over id columns unless the question explicitly asks for id.

The repaired JSON must follow this schema:
{{
  "tables": ["table1", "table2"],
  "joins": [{{"left": "tableA.col", "right": "tableB.col", "type": "inner|left"}}],
  "select": ["table.col", "..."],
  "filters": ["..."],
  "group_by": ["table.col", "..."],
  "order_by": ["table.col ASC|DESC", "..."],
  "distinct": false,
  "expected_row_behavior": "single|multi|one_per_group",
  "limit": null,
  "table_usage": {{"table1": "why needed", "table2": "why needed"}}
}}

Hard rules:
- Every column must be fully-qualified as table.column.
- If a JOIN is needed, each join key pair MUST match one of the foreign keys in the schema (either direction).
- Include ONLY the minimal set of tables needed. Do NOT include unused tables.
- Ensure plan.tables contains exactly all tables referenced in joins/select/filters/group_by/order_by.
- If the question asks for min/max/youngest/oldest/most/least, do not assume LIMIT 1; plan for ties.

[Database Schema]
{schema_info}

[User Question]
{user_question}

[Previous Raw Output]
{raw_text}

[Previous Parsed Plan]
{bad_plan_text}

[Validation Errors]
{error_text}
"""
    raw = _chat_generate(
        [
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": repair_prompt},
        ],
        int(PLAN_MAX_TOKENS),
    )
    plan = _extract_json_object(raw)
    if not isinstance(plan, dict):
        return None, False, raw

    ok, errors = validate_plan_json(plan, schema_info)
    if not ok:
        plan["_validation_errors"] = errors
    return plan, ok, raw


def generate_sql_from_plan(
    user_question: str,
    schema_info: str,
    few_shot_examples: str,
    plan: dict,
) -> str:
    plan_text = json.dumps(plan, ensure_ascii=False)
    distinct_rule = ""
    if isinstance(plan, dict) and plan.get("distinct") is True:
        distinct_rule = "- Use DISTINCT in the SELECT clause.\n"
    expected = plan.get("expected_row_behavior")
    behavior_rule = ""
    if expected == "single":
        behavior_rule = "- The query should return a single row (unless the question implies multiple).\n"
    elif expected == "one_per_group":
        behavior_rule = "- The query should return one row per group as implied by GROUP BY.\n"
    elif expected == "multi":
        behavior_rule = "- The query may return multiple rows.\n"
    prompt = f"""You are an expert Text-to-SQL assistant.

CRITICAL INSTRUCTIONS:
1. STRICTLY rely on the provided [Database Schema]. Do NOT use prior knowledge or assume standard database normalization; the schema is the ground truth even if it looks counter-intuitive.
2. COLUMN vs FUNCTION: If a word in the question matches or closely matches an existing physical column name, use that column—do not replace it with an SQL aggregate (e.g. COUNT/SUM) unless the question clearly asks for aggregation.
3. ID vs NAME: When listing, identifying, or grouping by an entity (e.g. "for each stadium", "which singer"), select human-readable name/title columns rather than id columns unless the question explicitly requests an id.
4. Use ONLY tables/columns that appear in the schema.
5. SQL surface form (helps standard execution evaluators align predicted vs reference SQL):
   - If FROM uses a single table (no JOIN), write bare column names in SELECT, GROUP BY, ORDER BY, and HAVING (e.g. SELECT Manufacturer, COUNT(*) FROM club GROUP BY Manufacturer). Do not add table. prefixes unless SQLite needs them.
   - If FROM joins multiple tables, qualify every selected column and ambiguous reference as table.column.
   - For counts per group or "how many rows/records", prefer COUNT(*) over COUNT(some_id) unless the question explicitly counts non-NULL values of one column.
   - Put SELECT list order to mirror the question when natural: grouping or entity columns first, then aggregates or metrics.
6. Follow the [Join Plan JSON] exactly for tables and join keys; do not invent join keys. When emitting SQL, apply the qualification rules in (5): single-table → bare columns; multi-table → qualified columns.
{distinct_rule}{behavior_rule}
7. Think step-by-step about which tables and columns are needed (brief reasoning is OK).
8. Output ONLY one valid SQLite query, wrapped exactly in <SQL> and </SQL> tags. No markdown code fences.

[Database Schema]
{schema_info}

[Join Plan JSON]
{plan_text}

[Similar Examples]
{few_shot_examples}

[User Question]
{user_question}
"""
    return _chat_sql(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert Text-to-SQL assistant. After brief reasoning, output only one SQLite query inside <SQL></SQL>. "
                    "Prefer Spider-style SQL: single-table queries use bare column names; multi-table queries use table.column; prefer COUNT(*) for per-group row counts."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )


def generate_sql(
    user_question: str,
    db_id: str,
    faiss_index,
    train_qs: list[str],
    train_sqls: list[str],
    train_db_ids: list[str],
):
    schema_info = get_schema(db_id)
    few_shot_examples = retrieve_few_shots_two_stage(
        user_question, db_id, faiss_index, train_qs, train_sqls, train_db_ids
    )

    plan, plan_fk_ok, plan_raw = plan_query_json(user_question, schema_info)
    if not (isinstance(plan, dict) and plan_fk_ok):
        validation_errors = []
        if isinstance(plan, dict):
            validation_errors = plan.get("_validation_errors") or []
        plan, plan_fk_ok, plan_raw = repair_plan_json(
            user_question,
            schema_info,
            plan if isinstance(plan, dict) else None,
            plan_raw,
            validation_errors,
        )

    if not isinstance(plan, dict):
        return "SELECT 1 WHERE 1 = 0", None, False, plan_raw

    sql = generate_sql_from_plan(user_question, schema_info, few_shot_examples, plan)
    return sql, plan, bool(plan_fk_ok), plan_raw


def repair_sql(pred_sql: str, error_message: str, user_question: str, db_id: str) -> str:
    schema_info = get_schema(db_id)
    repair_prompt = f"""Fix the SQL based on the execution error.

CRITICAL:
- Obey [Database Schema] only; do not assume normalization from prior knowledge.
- Prefer matching physical column names over inappropriate aggregates; prefer name/title columns over ids when the question asks for entities unless id is explicit.
- SQL style: single-table FROM → bare column names (no table. prefix) in SELECT/GROUP BY/ORDER BY; multi-table JOIN → use table.column everywhere needed. Prefer COUNT(*) for counting rows per group when appropriate.

Return ONLY one corrected SQLite query inside <SQL> and </SQL> tags.

[Database Schema]
{schema_info}

[User Question]
{user_question}

[Previous SQL]
{pred_sql}

[Execution Error]
{error_message}
"""

    return _chat_sql(
        [
            {
                "role": "system",
                "content": (
                    "You fix broken SQLite and return only the corrected query inside <SQL></SQL>. "
                    "Keep Spider-style qualification: bare columns if one table; table.column if joined; prefer COUNT(*) for group counts."
                ),
            },
            {"role": "user", "content": repair_prompt},
        ]
    )


def execute_sql(sql: str, db_path: str) -> str | None:
    """Return None on success; otherwise 'Error: ...'. Used only to detect runtime SQL failures."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 3000")
        conn.text_factory = bytes
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchall()
        return None
    except Exception as e:
        return f"Error: {e}"
    finally:
        if conn is not None:
            conn.close()

_spider_kmaps: dict | None = None


def _get_spider_kmaps() -> dict:
    global _spider_kmaps
    if _spider_kmaps is None:
        _spider_kmaps = build_kmaps_from_tables_json(
            [_resolve_data_path(TABLES_DATA_PATH), _resolve_data_path(TEST_TABLES_DATA_PATH)]
        )
    return _spider_kmaps


def evaluate_execution_accuracy(
    faiss_index,
    train_qs: list[str],
    train_sqls: list[str],
    train_db_ids: list[str],
    num_samples: int = 100,
    *,
    eval_data_path: str,
    eval_database_dir: str,
    trace_json_path: str | None = None,
) -> float:
    data_file = _resolve_data_path(eval_data_path)
    if not data_file.is_file():
        raise FileNotFoundError(f"Eval data not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        eval_rows = json.load(f)

    test_data = eval_rows[:num_samples] if num_samples else eval_rows
    db_root = _resolve_data_path(eval_database_dir)

    correct_count = 0
    repaired_count = 0
    total_count = len(test_data)
    trace_rows: list[dict] = []

    kmaps = _get_spider_kmaps()
    print(f"\nEvaluating {total_count} examples (Spider official eval_exec_match)...")

    for item in tqdm(test_data, desc="Evaluating"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        repaired_sql = None

        pred_sql, plan, plan_fk_ok, plan_raw = generate_sql(
            question, db_id, faiss_index, train_qs, train_sqls, train_db_ids
        )

        db_path = str(db_root / db_id / f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"[warn] Missing database: {db_path}, skipping")
            total_count -= 1
            continue

        pred_result = execute_sql(pred_sql, db_path)
        pred_error = pred_result if isinstance(pred_result, str) and pred_result.startswith("Error:") else None

        if REPAIR_ON_ERROR and pred_error:
            repaired_sql = repair_sql(pred_sql, pred_result, question, db_id)
            repaired_result = execute_sql(repaired_sql, db_path)
            if repaired_result is None:
                pred_result = None
                repaired_count += 1

        final_pred_sql = repaired_sql if repaired_sql is not None else pred_sql
        is_correct = spider_official_exec_match(final_pred_sql, gold_sql, db_path, db_id, kmaps)
        if is_correct:
            correct_count += 1

        if ENABLE_TRACE_JSON:
            trace_rows.append(
                {
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "plan": plan,
                    "plan_fk_ok": plan_fk_ok,
                    "plan_raw": plan_raw,
                    "pred_error": pred_error,
                    "repaired_sql": repaired_sql,
                    "final_status": "ok" if pred_result is None else "error",
                    "final_error": pred_result if isinstance(pred_result, str) else None,
                    "is_correct": is_correct,
                }
            )

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
    print("\n" + "=" * 40)
    print("Done.")
    print(f"Total: {total_count}")
    print(f"Repaired to runnable SQL: {repaired_count}")
    print(f"Correct (EX): {correct_count}")
    print(f"Execution Accuracy (EX): {accuracy:.2f}%")
    print("=" * 40)

    trace_out = trace_json_path or TRACE_JSON_PATH
    if ENABLE_TRACE_JSON:
        parent = os.path.dirname(trace_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(trace_out, "w", encoding="utf-8") as wf:
            for row in trace_rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Trace: {trace_out}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spider Text-to-SQL: multi-stage pipeline + local Qwen eval")
    parser.add_argument(
        "--lora",
        type=str,
        default="",
        metavar="DIR",
        help="PEFT adapter directory (overrides CHAT_LORA_PATH; relative paths are under nlp/)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help=f"Number of eval examples (default {EVAL_SAMPLES}; 0 = all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "dev"],
        default=None,
        help="test=test.json+test_database (default); dev=dev.json+database. Override with SPIDER_EVAL_SPLIT.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default TEXT2SQL_SEED or {SEED})",
    )
    args, _unknown = parser.parse_known_args()
    if args.lora:
        CHAT_LORA_PATH = args.lora.strip()

    n_eval = EVAL_SAMPLES if args.eval_samples is None else args.eval_samples
    eval_seed = SEED if args.seed is None else args.seed
    split = (args.split or SPIDER_EVAL_SPLIT or "test").strip().lower()
    if split not in ("test", "dev"):
        split = "test"
    if split == "dev":
        eval_data_path, eval_db_dir = DEV_DATA_PATH, DATABASE_DIR
        trace_out = "cache/eval_trace.jsonl"
    else:
        eval_data_path, eval_db_dir = TEST_DATA_PATH, TEST_DATABASE_DIR
        trace_out = "cache/eval_trace_test.jsonl"

    _set_eval_seed(eval_seed)
    print(f"CHAT_MODEL_NAME={CHAT_MODEL_NAME}")
    print(f"CHAT_LORA_PATH={CHAT_LORA_PATH or '(none)'}")
    print(f"SPIDER_EVAL_SPLIT={split}")
    print(f"eval_data={eval_data_path}")
    print(f"eval_database_dir={eval_db_dir}")
    print(f"eval_samples={n_eval}")
    print(f"seed={eval_seed}")

    _get_embedding_model()
    _get_llm()
    index, train_questions, train_queries, train_db_ids = build_vector_db()
    evaluate_execution_accuracy(
        index,
        train_questions,
        train_queries,
        train_db_ids,
        num_samples=n_eval,
        eval_data_path=eval_data_path,
        eval_database_dir=eval_db_dir,
        trace_json_path=trace_out,
    )
