"""
Spider 官方 EX 判定（Yale Spider / taoyds/spider 的 eval_exec_match 路径）。
与仓库内 vendored 的 process_sql.py、spider_evaluation.py 一致，不另起子进程。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .process_sql import Schema, get_schema, get_sql
from .spider_evaluation import (
    build_foreign_key_map,
    build_valid_col_units,
    eval_exec_match,
    rebuild_sql_col,
    rebuild_sql_val,
)

# evaluation.py 中解析失败时的空 SQL 结构
_EMPTY_PRED_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


def build_kmaps_from_tables_json(paths: list[Path | str]) -> dict[str, dict]:
    """合并多份 tables.json（如 tables.json + test_tables.json），构建 db_id -> foreign_key_map。"""
    kmaps: dict[str, dict] = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            kmaps[entry["db_id"]] = build_foreign_key_map(entry)
    return kmaps


def spider_official_exec_match(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    db_id: str,
    kmaps: dict[str, dict],
) -> bool:
    """
    与 Spider 官方 evaluation.py 中 evaluate(..., etype exec) 的 exec 分支一致：
    get_sql -> rebuild_sql_val -> rebuild_sql_col -> eval_exec_match。
    """
    schema = Schema(get_schema(db_path))
    try:
        g_sql = get_sql(schema, gold_sql)
    except Exception:
        return False

    try:
        p_sql = get_sql(schema, pred_sql)
    except Exception:
        p_sql = copy.deepcopy(_EMPTY_PRED_SQL)

    kmap = kmaps.get(db_id, {})

    g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)

    p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

    return bool(eval_exec_match(db_path, pred_sql, gold_sql, p_sql, g_sql))
