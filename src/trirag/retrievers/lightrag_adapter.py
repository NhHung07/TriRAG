"""Adapter cho LightRAG (Findings of EMNLP 2025) — gói ``lightrag-hku``.

* Dùng chung ``Embedder`` (``BAAI/bge-m3``) và ``CachedLLM`` (Gemini Flash-Lite khi index).
* Hỗ trợ 4 chế độ trong bài báo: ``hybrid`` (mặc định), ``local``, ``global``, ``naive`` (+ ``mix``).
* Các chế độ ablation có thể khai báo ``share_index_from: lightrag`` để dùng chung chỉ mục đã dựng
  (không tốn thêm thời gian hay lời gọi LLM).
* Dùng ``query_data`` để lấy cấu trúc ``chunks`` + ``relationships`` + ``entities`` kèm ``file_path``
  (chính là ``doc_id`` nguồn) mà không gọi LLM sinh câu trả lời của LightRAG.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import numpy as np

from ..data.schema import Document
from ..utils import count_tokens, read_json, truncate_tokens, write_json
from .base import RetrievalResult, RetrievedItem, Retriever

log = logging.getLogger(__name__)


class _OfflineTokenizerImpl:
    """Tokenizer dự phòng không mất dữ liệu (thread-safe, deepcopy-safe) khi chạy offline chưa có cache tiktoken."""

    def encode(self, content: str, **_kwargs) -> list[int]:
        return list(content.encode("utf-8"))

    def decode(self, tokens: list[int], **_kwargs) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def _make_tokenizer():
    from lightrag.utils import TiktokenTokenizer, Tokenizer

    try:
        return TiktokenTokenizer("gpt-4o-mini")
    except Exception:
        return Tokenizer("offline", _OfflineTokenizerImpl())


def _split_sources(val: str | None, valid_ids: set[str]) -> list[str]:
    if not val:
        return []
    parts = [p.strip() for p in re.split(r"<SEP>|[,;\n]+", str(val)) if p.strip()]
    return [p for p in dict.fromkeys(parts) if not valid_ids or p in valid_ids]


class LightRAGRetriever(Retriever):
    type_name = "lightrag"
    chunk_ranked = False  # giữ cả chunk + quan hệ + thực thể khi đóng gói ngữ cảnh

    def _make_rag(self, working_dir: Path, llm_tag: str):
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

        embedder = self.c.embedder
        llm = self.c.index_llm
        self._active_tag = llm_tag

        async def _llm_func(prompt: str, system_prompt: str | None = None, history_messages: list | None = None,
                            keyword_extraction: bool = False, **kwargs) -> str:
            full_prompt = prompt
            if history_messages:
                hist = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history_messages)
                full_prompt = f"{hist}\n\n{prompt}"
            max_tok = int(kwargs.get("max_tokens", 512 if keyword_extraction else 1536))
            return await asyncio.to_thread(
                llm.generate,
                full_prompt,
                system=system_prompt,
                temperature=0.0,
                max_tokens=max_tok,
                tag=self._active_tag,
            )

        async def _embed_func(texts: list[str], **kwargs) -> np.ndarray:
            is_q = kwargs.get("context") == "query"
            return await asyncio.to_thread(embedder.encode, list(texts), is_query=is_q)

        addon = {"language": self.params.get("language", "English")}
        if "entity_types" in self.params:
            addon["entity_types"] = self.params["entity_types"]

        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=_llm_func,
            llm_model_name=llm.model,
            llm_model_max_async=int(self.params.get("max_async", 2)),
            entity_extract_max_gleaning=int(self.params.get("max_gleaning", 0)),
            chunk_token_size=int(self.params.get("chunk_token_size", 1200)),
            chunk_overlap_token_size=int(self.params.get("chunk_overlap_token_size", 100)),
            tokenizer=_make_tokenizer(),
            embedding_func=EmbeddingFunc(
                embedding_dim=embedder.dim,
                max_token_size=8192,
                func=_embed_func,
            ),
            embedding_batch_num=int(self.params.get("embedding_batch_num", 16)),
            embedding_func_max_async=int(self.params.get("embedding_max_async", 2)),
            cosine_better_than_threshold=float(self.params.get("cosine_threshold", 0.1)),
            addon_params=addon,
        )
        loop = always_get_an_event_loop()
        loop.run_until_complete(rag.initialize_storages())
        return rag

    def _shared_dir(self) -> Path | None:
        share = self.params.get("share_index_from")
        root = getattr(self, "experiment_index_root", None)
        cid = getattr(self, "corpus_id", "all")
        if share and root:
            from ..data.io import _safe

            target = Path(root) / share / _safe(cid)
            if Retriever.is_built(target):
                return target
        return None

    def _build(self, docs: list[Document], index_dir: Path):
        shared = self._shared_dir()
        if shared is not None:
            log.info("[%s] dùng chung chỉ mục LightRAG từ %s", self.name, shared)
            write_json(index_dir / "shared_from.json", {"path": str(shared)})
            self._load_from(shared)
            return {"shared_from": str(shared)}

        self.rag = self._make_rag(index_dir, self.llm_tag)
        texts = [d.full_text for d in docs]
        ids = [d.doc_id for d in docs]
        sources = [d.meta.get("source_doc_id", d.doc_id) for d in docs]
        self.valid_doc_ids = set(sources)

        batch = int(self.params.get("insert_batch_size", 32))
        for i in range(0, len(docs), batch):
            self.rag.insert(texts[i : i + batch], ids=ids[i : i + batch], file_paths=sources[i : i + batch])

        from lightrag.utils import always_get_an_event_loop

        always_get_an_event_loop().run_until_complete(self.rag.finalize_storages())
        # Khởi tạo lại để sẵn sàng truy vấn ngay trong cùng tiến trình
        self.rag = self._make_rag(index_dir, f"retrieve:{self.name}")
        self._load_chunk_map(index_dir)
        write_json(index_dir / "valid_doc_ids.json", sorted(self.valid_doc_ids))
        return {"mode": self.params.get("mode", "hybrid")}

    def _load_chunk_map(self, work_dir: Path) -> None:
        self.chunk_to_doc: dict[str, str] = {}
        p = work_dir / "kv_store_text_chunks.json"
        if p.exists():
            data = read_json(p)
            for cid, entry in data.items():
                fp = entry.get("file_path") or entry.get("full_doc_id")
                if fp:
                    self.chunk_to_doc[cid] = str(fp)

    def _load_from(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        vpath = work_dir / "valid_doc_ids.json"
        self.valid_doc_ids = set(read_json(vpath)) if vpath.exists() else set()
        self._load_chunk_map(work_dir)
        self.rag = self._make_rag(work_dir, f"retrieve:{self.name}")

    def _load(self, index_dir: Path):
        if (index_dir / "shared_from.json").exists():
            target = Path(read_json(index_dir / "shared_from.json")["path"])
            self._load_from(target)
        else:
            self._load_from(index_dir)

    def _resolve_sources(self, obj: dict) -> list[str]:
        srcs = _split_sources(obj.get("file_path"), self.valid_doc_ids)
        if srcs:
            return srcs
        for cid in _split_sources(obj.get("source_id"), set()):
            if cid in self.chunk_to_doc:
                srcs.append(self.chunk_to_doc[cid])
        return list(dict.fromkeys(srcs))

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        from lightrag import QueryParam

        mode = self.params.get("mode", "hybrid")
        self._active_tag = f"retrieve:{self.name}"
        param = QueryParam(
            mode=mode,
            top_k=int(self.params.get("kg_top_k", max(10, top_k * 2))),
            chunk_top_k=top_k,
            enable_rerank=False,
        )
        raw = self.rag.query_data(query, param)
        data = (raw or {}).get("data") or {}
        if not any(data.get(k) for k in ("chunks", "relationships", "entities")) and mode != "naive":
            # Dự phòng: nếu truy vấn KG không khớp ngưỡng, lấy thêm vector chunks (naive)
            raw = self.rag.query_data(query, QueryParam(mode="naive", top_k=top_k, chunk_top_k=top_k))
            data = (raw or {}).get("data") or {}
        chunks = data.get("chunks") or []
        rels = data.get("relationships") or []
        ents = data.get("entities") or []

        max_chunk_tokens = int(self.params.get("max_chunk_tokens", 280))
        max_rels = int(self.params.get("max_relations", 10))
        max_ents = int(self.params.get("max_entities", 10))

        items: list[RetrievedItem] = []
        # 1. Các đoạn văn bản (chunks) lên trước để ưu tiên xếp hạng nguồn và bằng chứng trực tiếp
        for i, ch in enumerate(chunks[:top_k]):
            text = (ch.get("content") or "").strip()
            if not text:
                continue
            if count_tokens(text) > max_chunk_tokens:
                text = truncate_tokens(text, max_chunk_tokens)
            srcs = self._resolve_sources(ch)
            items.append(
                RetrievedItem(
                    text=text,
                    source_doc_ids=srcs,
                    score=1.0 / (i + 1),
                    kind="chunk",
                    title=srcs[0] if srcs else "",
                )
            )
        # 2. Quan hệ và thực thể từ đồ thị tri thức
        for j, r in enumerate(rels[:max_rels]):
            desc = (r.get("description") or "").strip()
            kw = (r.get("keywords") or "").strip()
            text = f"{r.get('src_id', '')} -> {r.get('tgt_id', '')} ({kw}): {desc}".strip()
            srcs = self._resolve_sources(r)
            items.append(
                RetrievedItem(
                    text=text,
                    source_doc_ids=srcs,
                    score=0.5 / (j + 1),
                    kind="relation",
                    title=f"{r.get('src_id', '')} — {r.get('tgt_id', '')}",
                )
            )
        for k, e in enumerate(ents[:max_ents]):
            name = e.get("entity_name", "")
            etype = e.get("entity_type", "")
            desc = (e.get("description") or "").strip()
            text = f"{name} [{etype}]: {desc}".strip()
            srcs = self._resolve_sources(e)
            items.append(
                RetrievedItem(
                    text=text,
                    source_doc_ids=srcs,
                    score=0.25 / (k + 1),
                    kind="entity",
                    title=name,
                )
            )

        kw_meta = ((raw or {}).get("metadata") or {}).get("keywords") or {}
        return RetrievalResult(
            items,
            extra={
                "mode": mode,
                "n_chunks": len(chunks),
                "n_relations": len(rels),
                "n_entities": len(ents),
                "keywords": kw_meta,
            },
        )
