"""Bộ chạy thí nghiệm thống nhất: index → retrieve → pack context → generate → đo.

Kết quả mỗi phương pháp ghi vào ``results/<experiment>/<method>.jsonl`` (mỗi dòng một câu hỏi),
hỗ trợ chạy tiếp (resume) khi bị ngắt giữa chừng (VD hết quota).
"""

from __future__ import annotations

import logging
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from ..data.io import _safe, load_processed, processed_dir
from ..data.schema import Dataset, Document
from ..generation import Generator, pack_context
from ..retrievers import Components, Retriever, make_retriever
from ..utils import PROJECT_ROOT, append_jsonl, read_jsonl, write_json
from .metrics import answer_metrics, retrieval_metrics

log = logging.getLogger(__name__)


def run_meta(cfg: dict) -> dict:
    def _git():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return None

    meta = {"git": _git(), "python": platform.python_version(), "platform": platform.platform(), "cpu_count": None}
    try:
        import os

        meta["cpu_count"] = os.cpu_count()
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    meta["config"] = {k: v for k, v in cfg.items() if k not in ("paths",)}
    return meta


class ExperimentRunner:
    def __init__(self, cfg: dict, components: Components | None = None, dataset: Dataset | None = None):
        self.cfg = cfg
        self.c = components or Components(cfg)
        self.dataset = dataset or load_processed(processed_dir(cfg))
        self.results_dir = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- chỉ mục ----------
    def index_dir(self, method: str, corpus_id: str) -> Path:
        return Path(self.cfg["paths"]["index_dir"]) / self.cfg["experiment"] / method / _safe(corpus_id)

    def method_params(self, method: str) -> dict:
        methods = self.cfg.get("methods", {})
        if method not in methods:
            raise KeyError(f"Phương pháp '{method}' không có trong config. Có: {list(methods)}")
        return methods[method] or {}

    def get_retriever(self, method: str, corpus_id: str, docs: list[Document], rebuild: bool = False) -> Retriever:
        r = make_retriever(method, self.method_params(method), self.c)
        idx = self.index_dir(method, corpus_id)
        r.experiment_index_root = Path(self.cfg["paths"]["index_dir"]) / self.cfg["experiment"]  # cho sgrag_lite
        r.corpus_id = corpus_id
        if rebuild or not Retriever.is_built(idx):
            log.info("[%s] lập chỉ mục corpus=%s (%d tài liệu)", method, corpus_id, len(docs))
            stats = r.build_index(docs, idx)
            log.info("[%s] xong: %s", method, {k: stats[k] for k in ("index_seconds", "llm")})
        else:
            r.load(idx)
        return r

    def build(self, method: str, rebuild: bool = False) -> None:
        for cid, docs in self.dataset.corpora.items():
            self.get_retriever(method, cid, docs, rebuild=rebuild)

    # ---------- chạy đánh giá ----------
    def run(self, method: str, limit: int | None = None, retrieval_only: bool = False) -> Path:
        out_path = self.results_dir / f"{method}.jsonl"
        existing = list(read_jsonl(out_path))
        if not retrieval_only and any("prediction" not in r for r in existing):
            kept = [r for r in existing if "prediction" in r]
            out_path.unlink(missing_ok=True)
            if kept:
                append_jsonl(out_path, kept)
            existing = kept
        done = {r["qid"] for r in existing}
        examples = self.dataset.examples[:limit] if limit else self.dataset.examples
        todo = [e for e in examples if e.qid not in done]
        log.info("[%s] %d câu hỏi (%d đã có)", method, len(todo), len(examples) - len(todo))
        write_json(self.results_dir / "run_meta.json", run_meta(self.cfg))
        if not todo:
            return out_path

        rcfg, gcfg = self.cfg["retrieval"], self.cfg["generation"]
        top_k, budget = rcfg["top_k"], rcfg["context_token_budget"]
        retrieve_k = max(top_k, rcfg.get("retrieve_k", 10))
        generator = None
        if not retrieval_only:
            generator = Generator(self.c.answer_llm, gcfg["prompt"], gcfg["max_tokens"], gcfg["temperature"])

        by_corpus: dict[str, list] = defaultdict(list)
        for e in todo:
            by_corpus[e.corpus_id].append(e)

        for cid, exs in by_corpus.items():
            retriever = self.get_retriever(method, cid, self.dataset.corpora[cid])
            for ex in tqdm(exs, desc=f"{method}:{cid}"[:40]):
                res = retriever.retrieve(ex.question, retrieve_k)
                ctx_items = res.items[:top_k] if getattr(retriever, "chunk_ranked", True) else res.items
                ctx = pack_context(ctx_items, budget)
                ranked = res.ranked_doc_ids()
                rec = {
                    "qid": ex.qid,
                    "method": method,
                    "question": ex.question,
                    "answers": ex.answers,
                    "gold_doc_ids": ex.gold_doc_ids,
                    "meta": ex.meta,
                    "ranked_doc_ids": ranked[:20],
                    "context_doc_ids": ctx.doc_ids,
                    "context_tokens": ctx.n_tokens,
                    "item_kinds": dict(Counter(it.kind for it in ctx.items)),
                    "context": ctx.text,
                    "retrieval_s": round(res.latency_s, 4),
                    "retrieval_extra": res.extra,
                }
                if ex.gold_doc_ids:
                    rec["retrieval_metrics"] = retrieval_metrics(ranked, ctx.doc_ids, ex.gold_doc_ids)
                if generator is not None:
                    ans = generator.answer(ex.question, ctx)
                    rec.update(
                        prediction=ans.text,
                        generation_s=round(ans.latency_s, 4),
                        gen_cached=ans.cached,
                        gen_input_tokens=ans.input_tokens,
                        gen_output_tokens=ans.output_tokens,
                    )
                    if ex.answers:
                        rec["answer_metrics"] = answer_metrics(ans.text, ex.answers)
                append_jsonl(out_path, [rec])
        return out_path
