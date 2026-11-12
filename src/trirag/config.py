"""Đọc cấu hình thí nghiệm (YAML) và hợp nhất với giá trị mặc định."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import PROJECT_ROOT

DEFAULTS: dict[str, Any] = {
    "experiment": "default",
    "seed": 42,
    "paths": {
        "data_dir": "data",
        "index_dir": "indexes",
        "results_dir": "results",
        "cache_dir": ".cache",
    },
    # LLM sinh câu trả lời (dùng CHUNG cho mọi phương pháp)
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash", "rpm": 10, "thinking_budget": 0},
    # LLM dùng khi lập chỉ mục (RAPTOR tóm tắt, LightRAG trích thực thể, SG-RAG tách câu hỏi)
    "index_llm": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "rpm": 15, "thinking_budget": 0},
    # LLM chấm điểm — nên khác model sinh
    "judge": None,
    "embedding": {
        "model": "BAAI/bge-m3",
        "device": "auto",
        "batch_size": 16,
        "query_prefix": "",
        "doc_prefix": "",
        "max_seq_length": 512,
    },
    "retrieval": {"top_k": 5, "context_token_budget": 2000},
    "generation": {"prompt": "qa_short", "max_tokens": 64, "temperature": 0.0},
    "methods": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(DEFAULTS, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    cfg["_config_path"] = str(path)
    return cfg
