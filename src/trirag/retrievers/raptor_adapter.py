"""Adapter cho RAPTOR (ICLR 2024) — kế thừa mã nguồn ``third_party/raptor``.

* Dùng chung ``Embedder`` (``BAAI/bge-m3``) và ``CachedLLM`` (Gemini Flash-Lite) của dự án.
* Nút lá (tầng 0) = các tài liệu/chunk đầu vào → giữ nguyên ``doc_id`` nguồn.
* Nút tóm tắt (tầng ≥ 1) = tóm tắt của một cụm nút tầng dưới → ``source_doc_ids`` là hợp
  đệ quy các ``doc_id`` của toàn bộ nút lá con cháu.
* Hỗ trợ cả 2 chế độ truy hồi trong bài báo: ``collapsed_tree`` (mặc định, tốt nhất trong paper)
  và ``tree_traversal``.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from ..data.chunking import split_document
from ..data.schema import Document
from ..utils import PROJECT_ROOT, count_tokens, truncate_tokens
from .base import RetrievalResult, RetrievedItem, Retriever

log = logging.getLogger(__name__)

RAPTOR_DIR = PROJECT_ROOT / "third_party" / "raptor"


class _FallbackTokenizer:
    """Dùng khi chạy offline chưa có cache BPE của tiktoken."""

    def encode(self, text: str, **_kwargs) -> list[int]:
        return list(range(max(1, len(text) // 4)))

    def decode(self, tokens: list[int], **_kwargs) -> str:
        return " " * len(tokens)


def _import_raptor():
    import types

    if str(RAPTOR_DIR) not in sys.path:
        sys.path.insert(0, str(RAPTOR_DIR))
    # third_party/raptor/__init__.py import FaissRetriever dù ClusterTreeBuilder không dùng faiss
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
    import tiktoken

    try:
        tiktoken.get_encoding("cl100k_base")
    except Exception:
        tiktoken.get_encoding = lambda *_a, **_kw: _FallbackTokenizer()  # type: ignore[assignment]

    from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
    from raptor.cluster_utils import RAPTOR_Clustering
    from raptor.EmbeddingModels import BaseEmbeddingModel
    from raptor.SummarizationModels import BaseSummarizationModel
    from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
    from raptor.tree_structures import Node, Tree

    return (
        BaseEmbeddingModel,
        BaseSummarizationModel,
        ClusterTreeBuilder,
        ClusterTreeConfig,
        RAPTOR_Clustering,
        Node,
        Tree,
        TreeRetriever,
        TreeRetrieverConfig,
    )


def _leaf_doc_ids(node_idx: int, all_nodes: dict, leaf_sources: dict[int, str], memo: dict[int, list[str]]) -> list[str]:
    if node_idx in memo:
        return memo[node_idx]
    if node_idx in leaf_sources:
        memo[node_idx] = [leaf_sources[node_idx]]
        return memo[node_idx]
    node = all_nodes.get(node_idx)
    children = getattr(node, "children", None) or []
    seen: dict[str, None] = {}
    for ch in children:
        for did in _leaf_doc_ids(int(ch), all_nodes, leaf_sources, memo):
            seen.setdefault(did, None)
    memo[node_idx] = list(seen)
    return memo[node_idx]


class RaptorRetriever(Retriever):
    type_name = "raptor"

    def _build(self, docs: list[Document], index_dir: Path):
        (
            BaseEmbeddingModel,
            BaseSummarizationModel,
            ClusterTreeBuilder,
            ClusterTreeConfig,
            RAPTOR_Clustering,
            Node,
            Tree,
            _,
            _,
        ) = _import_raptor()

        max_leaf_tokens = int(self.params.get("leaf_tokens", 200))
        leaf_docs: list[Document] = []
        for d in docs:
            leaf_docs.extend(split_document(d, max_leaf_tokens))

        embedder = self.c.embedder
        llm = self.c.index_llm
        tag = self.llm_tag
        max_ctx_tokens = int(self.params.get("summary_input_tokens", 3000))

        class _EmbedModel(BaseEmbeddingModel):
            def create_embedding(self, text: str):
                return embedder.encode([text])[0].tolist()

        class _SumModel(BaseSummarizationModel):
            def summarize(self, context: str, max_tokens: int = 150):
                ctx = truncate_tokens(context, max_ctx_tokens)
                prompt = (
                    "Write a concise summary of the following passages, preserving key entities, "
                    f"numbers, dates, and factual relations:\n\n{ctx}\n\nSummary:"
                )
                return llm.generate(prompt, max_tokens=max_tokens, temperature=0.0, tag=tag)

        class _SafeClustering(RAPTOR_Clustering):
            @staticmethod
            def perform_clustering(nodes, embedding_model_name, max_length_in_cluster=3500,
                                   tokenizer=None, reduction_dimension=10, threshold=0.1, verbose=False):
                n = len(nodes)
                if n <= 2:
                    return [list(nodes)]
                # UMAP cần n_neighbors >= 2 và n_components < n - 2
                if n <= max(6, reduction_dimension + 3):
                    mid = max(1, n // 2)
                    return [list(nodes[:mid]), list(nodes[mid:])]
                try:
                    clusters = RAPTOR_Clustering.perform_clustering(
                        nodes,
                        embedding_model_name,
                        max_length_in_cluster=max_length_in_cluster,
                        reduction_dimension=min(reduction_dimension, n - 3),
                        threshold=threshold,
                        verbose=verbose,
                    )
                    return clusters or [list(nodes)]
                except Exception as e:
                    log.warning("RAPTOR UMAP/GMM fallback (%d nodes): %s", n, e)
                    step = max(2, n // 3)
                    return [list(nodes[i : i + step]) for i in range(0, n, step)]

        emb_key = "default"
        cfg = ClusterTreeConfig(
            tokenizer=_FallbackTokenizer(),
            max_tokens=max_leaf_tokens,
            num_layers=int(self.params.get("num_layers", 3)),
            threshold=float(self.params.get("threshold", 0.1)),
            top_k=int(self.params.get("top_k", 5)),
            selection_mode="top_k",
            summarization_length=int(self.params.get("summarization_length", 150)),
            summarization_model=_SumModel(),
            embedding_models={emb_key: _EmbedModel()},
            cluster_embedding_model=emb_key,
            reduction_dimension=int(self.params.get("reduction_dimension", 10)),
            clustering_algorithm=_SafeClustering,
        )
        builder = ClusterTreeBuilder(cfg)

        # Mã hoá toàn bộ nút lá theo lô để tận dụng GPU + cache
        leaf_vecs = embedder.encode([d.full_text for d in leaf_docs])
        leaf_nodes: dict[int, object] = {}
        self.leaf_sources: dict[int, str] = {}
        self.node_titles: dict[int, str] = {}
        for i, (d, v) in enumerate(zip(leaf_docs, leaf_vecs)):
            leaf_nodes[i] = Node(d.full_text, i, set(), {emb_key: v.tolist()})
            self.leaf_sources[i] = d.meta.get("source_doc_id", d.doc_id)
            self.node_titles[i] = d.title

        all_nodes = dict(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}
        root_nodes = builder.construct_tree(dict(leaf_nodes), all_nodes, layer_to_nodes, use_multithreading=False)
        num_layers = max(layer_to_nodes.keys())
        self.tree = Tree(all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes)
        self._prepare_Arrays(emb_key)

        with (index_dir / "raptor_tree.pkl").open("wb") as f:
            pickle.dump(
                {
                    "tree": self.tree,
                    "leaf_sources": self.leaf_sources,
                    "node_titles": self.node_titles,
                    "emb_key": emb_key,
                },
                f,
            )
        return {
            "n_leaf_nodes": len(leaf_nodes),
            "n_total_nodes": len(all_nodes),
            "num_layers": num_layers,
        }

    def _prepare_Arrays(self, emb_key: str) -> None:
        self.node_layer: dict[int, int] = {}
        for layer, nodes in self.tree.layer_to_nodes.items():
            for n in nodes:
                self.node_layer[n.index] = layer
        self.node_list = [self.tree.all_nodes[k] for k in sorted(self.tree.all_nodes)]
        mat = np.asarray([n.embeddings[emb_key] for n in self.node_list], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.node_emb = mat / norms
        self._memo_sources: dict[int, list[str]] = {}

    def _load(self, index_dir: Path):
        _import_raptor()
        with (index_dir / "raptor_tree.pkl").open("rb") as f:
            obj = pickle.load(f)
        self.tree = obj["tree"]
        self.leaf_sources = obj["leaf_sources"]
        self.node_titles = obj.get("node_titles", {})
        self._prepare_Arrays(obj.get("emb_key", "default"))

    def _node_to_item(self, node, score: float) -> RetrievedItem:
        layer = self.node_layer.get(node.index, 0)
        srcs = _leaf_doc_ids(node.index, self.tree.all_nodes, self.leaf_sources, self._memo_sources)
        kind = "chunk" if layer == 0 else "summary"
        title = self.node_titles.get(node.index) or (f"RAPTOR summary (layer {layer}, {len(srcs)} docs)")
        return RetrievedItem(
            text=node.text,
            source_doc_ids=srcs,
            score=score,
            kind=kind,
            title=title,
            meta={"node_index": node.index, "layer": layer},
        )

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        mode = self.params.get("mode", "collapsed")
        q = self.c.embedder.encode([query], is_query=True)[0]

        if mode == "traversal" and self.tree.num_layers > 0:
            # Duyệt từ tầng cao nhất xuống tầng 0, mỗi tầng chọn branch_k nút gần câu hỏi nhất
            branch_k = int(self.params.get("branch_k", max(2, top_k // 2)))
            idx_to_pos = {n.index: pos for pos, n in enumerate(self.node_list)}
            current = list(self.tree.layer_to_nodes[self.tree.num_layers])
            selected: list[tuple[object, float]] = []
            for _layer in range(self.tree.num_layers, -1, -1):
                if not current:
                    break
                sims = [float(self.node_emb[idx_to_pos[n.index]] @ q) for n in current]
                order = np.argsort(-np.asarray(sims))[:branch_k]
                chosen = [current[i] for i in order]
                selected.extend((current[i], sims[i]) for i in order)
                nxt_ids = {ch for n in chosen for ch in (n.children or [])}
                current = [self.tree.all_nodes[i] for i in sorted(nxt_ids) if i in self.tree.all_nodes]
            selected.sort(key=lambda x: -x[1])
            return RetrievalResult([self._node_to_item(n, s) for n, s in selected[:top_k]], extra={"mode": mode})

        # Mặc định: collapsed tree — xếp hạng toàn bộ nút (lá + tóm tắt) theo cosine
        sims = self.node_emb @ q
        order = np.argsort(-sims)[:top_k]
        items = [self._node_to_item(self.node_list[i], float(sims[i])) for i in order]
        return RetrievalResult(items, extra={"mode": "collapsed"})
