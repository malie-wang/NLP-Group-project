#!/usr/bin/env python3
"""
Spider CoT 数据整理 → 微调格式（JSONL）

不加载本地 Qwen，只复用 Text2SQL 里与 schema 相关的路径与 get_schema。

流程：
  1) export-prompts  从 train_spider.json 导出「给强模型用的」提示词 JSONL（可拷到 Gemini 网页或走 API）
  2) generate       用 OpenAI 兼容 API（.env 里 OPENAI_API_KEY 等）批量写 cot 字段
  3) to-sft         把带 cot 的行转成 sharegpt/messages 风格，供 LLaMA-Factory、Axolotl、Unsloth 等 SFT

示例：
  cd /path/to/nlp
  python build_cot_ft_dataset.py export-prompts --limit 200 --out data/cot_prompts.jsonl
  python build_cot_ft_dataset.py generate --in data/cot_prompts.jsonl --out data/cot_filled.jsonl
  python build_cot_ft_dataset.py to-sft --in data/cot_filled.jsonl --out data/sft_messages.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 与 Text2SQL 同目录，保证相对路径一致
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_ROOT / ".env")
except ImportError:
    pass

import Text2SQL as t2s


def _teacher_prompt(schema: str, question: str, gold_sql: str) -> str:
    return f"""You are an expert Text-to-SQL teacher. The following Gold SQL is AUTHORITATIVE and correct for this database.

Write a SHORT chain-of-thought (3–8 short bullet lines) explaining how this SQL answers the question:
- which tables are needed and why
- how joins follow foreign keys (if any)
- filters / GROUP BY / ORDER BY / LIMIT intent

Rules:
- Do NOT contradict or rewrite the Gold SQL; your explanation must justify THAT query only.
- Do not output SQL in the CoT; reasoning text only.

[Database Schema]
{schema}

[Question]
{question}

[Gold SQL]
{gold_sql}

Chain-of-thought:"""


def _sft_user_message(schema: str, question: str) -> str:
    return f"""You are an expert Text-to-SQL assistant.

Think step-by-step, then output exactly one valid SQLite query wrapped in <SQL> and </SQL> tags.

[Database Schema]
{schema}

[User Question]
{question}"""


def _sft_assistant_message(cot: str, gold_sql: str) -> str:
    cot = (cot or "").strip()
    sql = (gold_sql or "").strip()
    return f"{cot}\n\n<SQL>\n{sql}\n</SQL>"


def _load_train(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gold_runs(db_id: str, sql: str) -> tuple[bool, str]:
    db_path = _ROOT / t2s.DATABASE_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        return False, f"missing db file: {db_path}"
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout = 3000")
        conn.execute(sql)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def cmd_export_prompts(args: argparse.Namespace) -> None:
    train_path = _ROOT / args.train
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_train(train_path)
    if args.limit:
        rows = rows[: int(args.limit)]

    n_skip = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for i, item in enumerate(rows):
            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            schema = t2s.get_schema(db_id)
            if not schema.strip():
                n_skip += 1
                continue
            if args.check_sql:
                ok, err = _gold_runs(db_id, gold_sql)
                if not ok:
                    n_skip += 1
                    continue
            rec = {
                "id": f"{db_id}_{i}",
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "schema_info": schema,
                "teacher_user": _teacher_prompt(schema, question, gold_sql),
                "cot": "",
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} (skipped {n_skip} rows without schema or failed gold check)")


def _openai_chat(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def _write_cot_jsonl(out_path: Path, records: list[dict]) -> None:
    """完整重写输出文件，便于断点续跑时随时落盘。"""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as wf:
        for rec in records:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def cmd_generate(args: argparse.Namespace) -> None:
    in_path = _ROOT / args.input
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        print(
            f"ERROR: input file not found: {in_path}\n"
            "Run this first (from nlp/):\n"
            "  python build_cot_ft_dataset.py export-prompts --check-sql --out data/cot_prompts.jsonl",
            file=sys.stderr,
        )
        sys.exit(2)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    model = os.environ.get("COT_TEACHER_MODEL", "gpt-4o-mini")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY in environment or .env", file=sys.stderr)
        sys.exit(2)

    merged: dict[str, dict] = {}
    with open(in_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            merged[rec.get("id", json.dumps(rec, sort_keys=True)[:120])] = rec

    if args.resume and out_path.is_file():
        with open(out_path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                oid = o.get("id", "")
                if oid and o.get("cot", "").strip():
                    if oid in merged:
                        merged[oid]["cot"] = o["cot"]
                        merged[oid].pop("cot_error", None)

    records = list(merged.values())
    records.sort(key=lambda r: r.get("id", ""))

    already = sum(1 for r in records if (r.get("cot") or "").strip())
    need = len(records) - already
    print(
        f"Loaded {len(records)} rows ({already} already have cot, {need} API calls left).",
        flush=True,
    )
    print(f"Using model={model!r} base={base_url!r}", flush=True)
    if need == 0:
        _write_cot_jsonl(out_path, records)
        print(f"Nothing to do → wrote {out_path}", flush=True)
        return

    done_calls = 0
    for rec in records:
        if rec.get("cot", "").strip():
            continue
        user = rec.get("teacher_user") or _teacher_prompt(
            rec["schema_info"], rec["question"], rec["gold_sql"]
        )
        try:
            cot = _openai_chat(
                [
                    {
                        "role": "system",
                        "content": "You write concise, accurate reasoning for SQL. No SQL in the answer.",
                    },
                    {"role": "user", "content": user},
                ],
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=args.timeout,
            )
            rec["cot"] = cot
            rec.pop("cot_error", None)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            rec["cot"] = ""
            rec["cot_error"] = str(e)
            if done_calls == 0:
                print(f"First API error (check key / URL / proxy): {e}", flush=True)

        done_calls += 1
        if done_calls == 1 or done_calls % int(args.progress_every) == 0:
            n_cot = sum(1 for r in records if (r.get("cot") or "").strip())
            print(
                f"[cot] API {done_calls}/{need} | total with cot: {n_cot}/{len(records)} | last_id={rec.get('id','')}",
                flush=True,
            )
        if done_calls % int(args.checkpoint_every) == 0:
            _write_cot_jsonl(out_path, records)
            print(f"  (checkpoint saved → {out_path})", flush=True)

        time.sleep(args.sleep)

    _write_cot_jsonl(out_path, records)
    print(
        f"Done → {out_path} ({sum(1 for r in records if r.get('cot','').strip())} with cot)",
        flush=True,
    )


def cmd_to_sft(args: argparse.Namespace) -> None:
    in_path = _ROOT / args.input
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        print(
            f"ERROR: input file not found: {in_path}\n"
            "Generate CoT first:\n"
            "  python build_cot_ft_dataset.py generate -i data/cot_prompts.jsonl -o data/cot_filled.jsonl",
            file=sys.stderr,
        )
        sys.exit(2)

    n = 0
    n_skip = 0
    with open(in_path, "r", encoding="utf-8") as rf, open(
        out_path, "w", encoding="utf-8"
    ) as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cot = (rec.get("cot") or "").strip()
            if not cot:
                n_skip += 1
                continue
            schema = rec["schema_info"]
            question = rec["question"]
            gold = rec["gold_sql"]
            user = _sft_user_message(schema, question)
            assistant = _sft_assistant_message(cot, gold)
            out = {
                "id": rec.get("id", ""),
                "db_id": rec.get("db_id", ""),
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} SFT rows → {out_path} (skipped {n_skip} without cot)")


def cmd_to_alpaca(args: argparse.Namespace) -> None:
    """messages JSONL → 单个 JSON 数组，供 AI6130_Assignment2/finetune.py（instruction/input/output）使用。"""
    in_path = _ROOT / args.input
    out_path = _ROOT / args.out
    if not in_path.is_file():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(2)
    rows: list[dict] = []
    with open(in_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            msgs = o.get("messages") or []
            user = ""
            asst = ""
            for m in msgs:
                if m.get("role") == "user":
                    user = m.get("content") or ""
                elif m.get("role") == "assistant":
                    asst = m.get("content") or ""
            if not user.strip() or not asst.strip():
                continue
            rows.append({"instruction": user, "input": "", "output": asst})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(rows, wf, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} Alpaca-style records → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Spider CoT export / generate / SFT JSONL")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-prompts", help="Export teacher prompts JSONL from train split")
    e.add_argument("--train", default=t2s.TRAIN_DATA_PATH, help="train json path")
    e.add_argument("--out", default="data/cot_prompts.jsonl")
    e.add_argument("--limit", type=int, default=0, help="max rows (0=all)")
    e.add_argument(
        "--check-sql",
        action="store_true",
        help="skip rows where gold SQL fails on sqlite (recommended)",
    )
    e.set_defaults(func=cmd_export_prompts)

    g = sub.add_parser("generate", help="Fill cot via OpenAI-compatible API")
    g.add_argument("--input", "-i", default="data/cot_prompts.jsonl")
    g.add_argument("--out", "-o", default="data/cot_filled.jsonl")
    g.add_argument("--resume", action="store_true", help="merge non-empty cot from --out then fill the rest")
    g.add_argument("--sleep", type=float, default=0.2, help="seconds between API calls")
    g.add_argument("--timeout", type=int, default=120)
    g.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="print progress every N new API calls",
    )
    g.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="rewrite output JSONL every N new API calls (crash-safe)",
    )
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("to-sft", help="Convert filled JSONL to messages JSONL for SFT")
    t.add_argument("--input", "-i", default="data/cot_filled.jsonl")
    t.add_argument("--out", "-o", default="data/sft_messages.jsonl")
    t.set_defaults(func=cmd_to_sft)

    a = sub.add_parser(
        "to-alpaca",
        help="Convert messages JSONL to one .json array for AI6130 finetune.py",
    )
    a.add_argument("--input", "-i", default="data/sft_messages_524.jsonl")
    a.add_argument("--out", "-o", default="data/spider_alpaca_524.json")
    a.set_defaults(func=cmd_to_alpaca)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
