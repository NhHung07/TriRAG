"""Lưu / nạp bộ dữ liệu đã tiền xử lý và chọn loader theo cấu hình."""

from __future__ import annotations

from pathlib import Path

from ..utils import append_jsonl, read_json, read_jsonl, write_json
from .schema import Dataset, Document, QAExample


def processed_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "processed" / cfg["experiment"]


def save_dataset(ds: Dataset, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "corpora").mkdir(parents=True, exist_ok=True)
    ex_path = out_dir / "examples.jsonl"
    ex_path.unlink(missing_ok=True)
    append_jsonl(ex_path, (e.to_dict() for e in ds.examples))
    for cid, docs in ds.corpora.items():
        p = out_dir / "corpora" / f"{_safe(cid)}.jsonl"
        p.unlink(missing_ok=True)
        append_jsonl(p, (d.to_dict() for d in docs))
    write_json(out_dir / "meta.json", {**ds.stats(), "corpus_files": {cid: f"{_safe(cid)}.jsonl" for cid in ds.corpora}})


def load_processed(out_dir: str | Path) -> Dataset:
    out_dir = Path(out_dir)
    meta = read_json(out_dir / "meta.json")
    examples = [QAExample.from_dict(d) for d in read_jsonl(out_dir / "examples.jsonl")]
    corpora = {
        cid: [Document.from_dict(d) for d in read_jsonl(out_dir / "corpora" / fname)]
        for cid, fname in meta["corpus_files"].items()
    }
    return Dataset(meta["name"], examples, corpora)


def _safe(cid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in cid)


def build_dataset(cfg: dict, llm=None) -> Dataset:
    """Tạo Dataset từ dữ liệu thô theo ``cfg['dataset']``."""
    from ..utils import PROJECT_ROOT

    d = dict(cfg["dataset"])
    name = d.pop("name")

    def path(key: str) -> Path:
        p = Path(d.pop(key))
        return p if p.is_absolute() else PROJECT_ROOT / p

    if name == "hotpotqa":
        from .hotpotqa import load_hotpotqa

        return load_hotpotqa(path("path"), d.get("n_questions", 300), cfg.get("seed", 42), d.get("stratify", "balanced"))
    if name == "qasper":
        from .qasper import load_qasper

        return load_qasper(path("path"), d.get("n_papers", 15), cfg.get("seed", 42), d.get("max_questions", 100))
    if name == "ultradomain":
        from .ultradomain import load_ultradomain

        return load_ultradomain(
            path("path"),
            d.get("max_tokens", 150_000),
            d.get("chunk_tokens", 1200),
            llm=llm,
            **d.get("questions", {}),
        )
    if name == "vi_docs":
        from .vi_docs import load_vi

        qp = path("questions_path") if d.get("questions_path") else None
        return load_vi(path("input_dir"), qp, d.get("max_tokens", 400), d.get("overlap_tokens", 50))
    if name == "erqa":
        from .erqa import load_erqa

        return load_erqa(path("path"), **d)
    raise ValueError(f"Bộ dữ liệu không hỗ trợ: {name}")
