"""SG-RAG (Structure Guided RAG — ACL 2026): khớp đồ thị con dạng sao-cầu (bridge-star).

Hỗ trợ 2 nguồn đồ thị tri thức (KG):
1. **Đồ thị có sẵn trong dữ liệu** (``doc.meta["triples"]``, VD bộ ERQA ``.graphml``) hoặc
   tái sử dụng đồ thị ``graph_chunk_entity_relation.graphml`` do LightRAG đã dựng.
2. **Tự trích xuất từ văn bản** (HotpotQA, QASPER, tài liệu tiếng Việt): trích thực thể từ tiêu đề
   tài liệu, cụm danh từ viết hoa, năm/số đo và tạo cạnh quan hệ theo câu, không bắt buộc gọi LLM.

Khi truy hồi cho câu hỏi nhiều điều kiện $Q$:
1. **Phân rã điều kiện (constraint decomposition)**: tách các thực thể mỏ neo $C = \\{c_1, \\dots, c_m\\}$
   (kể cả mở rộng cấu trúc phối hợp như ``"1988, 2012, and 1984 Summer Olympics"`` → 3 ràng buộc riêng)
   và gợi ý loại thực thể đích (sport / event / award / person / location...).
2. **Khớp nút mỏ neo (entity normalization)** bằng đối sánh chuỗi + độ tương đồng vector trên
   biểu diễn nút trội (dominant neighborhood embedding).
3. **Khớp đồ thị con sao-cầu (bridge-star subgraph matching)**: chấm điểm mọi nút ứng viên $u$
   lân cận 1–2 bước nhảy theo số lượng điều kiện $c_i$ được thoả đồng thời + độ khớp quan hệ.
4. **Tạo ngữ cảnh có hướng dẫn cấu trúc**: trả về các cấu trúc sao-cầu khớp tốt nhất (``kind="path"``)
   kèm các đoạn văn bản nguồn (``kind="chunk"``).
"""

from __future__ import annotations

import logging
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from ..data.schema import Document
from .base import RetrievalResult, RetrievedItem, Retriever

log = logging.getLogger(__name__)

_COORD_YEARS_RE = re.compile(
    r"\b((?:(?:18|19|20)\d{2}\s*(?:,\s*|\s+and\s+|\s+or\s+))+(?:18|19|20)\d{2})\s+"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"
)
_QUOTED_RE = re.compile(r'["“”\']([^"“”\']{2,80})["“”\']')
_CAP_SPAN_RE = re.compile(
    r"\b(?:[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9'\-]*(?:\s+(?:of|the|for|in|on|and|de|van|von|[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9'\-]*)){0,5})"
)
_YEAR_EVENT_RE = re.compile(r"\b((?:18|19|20)\d{2}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})")
_STOP_SPANS = {
    "Which",
    "What",
    "Where",
    "When",
    "Who",
    "Whom",
    "Whose",
    "How",
    "Why",
    "The",
    "In",
    "On",
    "At",
    "From",
    "With",
    "Both",
    "Are",
    "Is",
    "Was",
    "Were",
}

_TARGET_HINTS = {
    "sport": ("sport", "athletes", "olympic_sport", "discipline"),
    "event": ("olympic_games", "olympics", "event", "games", "tournament", "championship"),
    "award": ("award", "nominee", "winner", "honor", "prize"),
    "film": ("film", "movie", "actor", "director", "screenplay"),
    "institution": ("institution", "university", "college", "school", "education", "alumni"),
    "university": ("university", "college", "education", "graduates", "institution"),
    "country": ("country", "nation", "republic", "location", "state"),
    "city": ("city", "capital", "location", "town"),
    "person": ("person", "people", "profession", "actor", "author", "politician", "athlete"),
    "organization": ("organization", "company", "member", "agency", "institution"),
}


def _norm_ent(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().strip(".,;:!?\"'“”()[]"))
    return s.lower()


def extract_entities_from_text(text: str) -> list[str]:
    out: list[str] = []
    for m in _QUOTED_RE.finditer(text):
        val = m.group(1).strip()
        if len(val) >= 2:
            out.append(val)
    for m in _YEAR_EVENT_RE.finditer(text):
        out.append(m.group(1).strip())
    for m in _CAP_SPAN_RE.finditer(text):
        span = m.group(0).strip()
        words = span.split()
        while words and words[0] in _STOP_SPANS:
            words.pop(0)
        span = " ".join(words)
        if len(span) >= 3 and span not in _STOP_SPANS:
            out.append(span)
    seen: dict[str, str] = {}
    for e in out:
        k = _norm_ent(e)
        if k and len(k) >= 2:
            seen.setdefault(k, e)
    return list(seen.values())


def extract_query_constraints(query: str) -> tuple[list[str], list[str]]:
    """Tách danh sách ràng buộc thực thể trong câu hỏi và từ khoá loại thực thể đích."""
    constraints: list[str] = []
    # 1. Mở rộng cấu trúc phối hợp năm: "1988, 2012, and 1984 Summer Olympics"
    for m in _COORD_YEARS_RE.finditer(query):
        years = re.findall(r"(?:18|19|20)\d{2}", m.group(1))
        suffix = m.group(2).strip()
        for y in years:
            constraints.append(f"{y} {suffix}")

    # 2. Các cụm trong ngoặc kép & cụm viết hoa
    for e in extract_entities_from_text(query):
        # Bỏ các cụm mở đầu câu hỏi như "Which Olympic"
        e_clean = re.sub(r"^(Which|What)\s+(Olympic\s+)?(sport|event|award|institution|country|film)\b\s*", "", e, flags=re.I).strip()
        if len(e_clean) >= 2 and _norm_ent(e_clean) not in {"olympic", "olympics", "summer", "winter"}:
            constraints.append(e_clean)

    # Khử trùng, ưu tiên cụm dài hơn nhưng giữ các năm-sự-kiện riêng biệt
    dedup: dict[str, str] = {}
    for c in constraints:
        k = _norm_ent(c)
        if k and k not in dedup:
            dedup[k] = c

    # Gợi ý loại thực thể đích từ đầu câu hỏi ("Which sport...", "Which event...", ...)
    q_low = query.lower()[:80]
    target_keywords: list[str] = []
    for key, hints in _TARGET_HINTS.items():
        if re.search(rf"\b{key}s?\b", q_low):
            target_keywords.extend(hints)
    return list(dedup.values()), target_keywords


class SGRAGLiteRetriever(Retriever):
    type_name = "sgrag_lite"
    chunk_ranked = False  # giữ cả cấu trúc đường đi (path) + đoạn văn bản (chunk)

    def _try_load_lightrag_graph(self, g: nx.MultiGraph) -> int:
        root = getattr(self, "experiment_index_root", None)
        cid = getattr(self, "corpus_id", "all")
        if not root:
            return 0
        from ..data.io import _safe

        gml = Path(root) / "lightrag" / _safe(cid) / "graph_chunk_entity_relation.graphml"
        if not gml.exists():
            return 0
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        tree = ET.parse(gml).getroot()
        key_map = {
            k.attrib["id"]: k.attrib.get("attr.name", k.attrib["id"]) for k in tree.findall("g:key", ns)
        }
        added = 0
        for edge in tree.findall(".//g:edge", ns):
            u = edge.attrib.get("source", "").strip()
            v = edge.attrib.get("target", "").strip()
            if not u or not v:
                continue
            attrs = {key_map.get(d.attrib["key"], d.attrib["key"]): (d.text or "") for d in edge.findall("g:data", ns)}
            srcs = [s.strip() for s in re.split(r"<SEP>|[,;]", attrs.get("file_path", "")) if s.strip()]
            self._add_triple(g, u, attrs.get("keywords", "related_to"), attrs.get("description", ""), v, srcs)
            added += 1
        return added

    @staticmethod
    def _add_triple(
        g: nx.MultiGraph, head: str, rel: str, desc: str, tail: str, source_doc_ids: list[str]
    ) -> None:
        u, v = _norm_ent(head), _norm_ent(tail)
        if not u or not v or u == v:
            return
        for nid, label in ((u, head.strip()), (v, tail.strip())):
            if nid not in g:
                g.add_node(nid, label=label, sources=set(), rel_tokens=set())
            g.nodes[nid]["sources"].update(source_doc_ids)
            for tok in re.findall(r"[a-z]{3,}", f"{rel} {desc}".lower()):
                g.nodes[nid]["rel_tokens"].add(tok)
        g.add_edge(u, v, relation=rel, description=desc, sources=set(source_doc_ids))

    def _build_text_triples(self, docs: list[Document], g: nx.MultiGraph) -> None:
        sent_re = re.compile(r"(?<=[.!?])\s+|\n+")
        for d in docs:
            src_id = d.meta.get("source_doc_id", d.doc_id)
            title_ent = d.title.split("—")[0].strip() if d.title else ""
            for sent in sent_re.split(d.text):
                ents = extract_entities_from_text(sent)
                if title_ent and _norm_ent(title_ent) not in {_norm_ent(e) for e in ents}:
                    ents = [title_ent] + ents
                if len(ents) < 2:
                    continue
                rel_snippet = " ".join(sent.split()[:18])
                anchor = ents[0]
                for other in ents[1:8]:
                    self._add_triple(g, anchor, "mentioned_with", rel_snippet, other, [src_id])
                for i in range(1, min(len(ents) - 1, 5)):
                    self._add_triple(g, ents[i], "co_occurs", rel_snippet, ents[i + 1], [src_id])

    def _build(self, docs: list[Document], index_dir: Path):
        self.docs = docs
        self.doc_by_id = {d.meta.get("source_doc_id", d.doc_id): d for d in docs}
        g = nx.MultiGraph()

        n_erqa = 0
        for d in docs:
            src_id = d.meta.get("source_doc_id", d.doc_id)
            for h, rel, desc, t in d.meta.get("triples") or []:
                self._add_triple(g, h, rel, desc, t, [src_id])
                n_erqa += 1

        n_lightrag = 0
        if n_erqa == 0:
            n_lightrag = self._try_load_lightrag_graph(g)
            self._build_text_triples(docs, g)

        self.graph = g
        self.node_ids = list(g.nodes)
        # Biểu diễn nút trội (dominant neighborhood representation): nhãn nút + các từ khoá quan hệ kề
        node_texts = [
            f"{g.nodes[n]['label']} ({' '.join(sorted(g.nodes[n]['rel_tokens'])[:15])})" for n in self.node_ids
        ]
        self.node_emb = (
            self.c.embedder.encode(node_texts)
            if self.node_ids
            else np.zeros((0, self.c.embedder.dim), dtype=np.float32)
        )
        self.doc_emb = self.c.embedder.encode([d.full_text for d in docs])

        with (index_dir / "sgrag.pkl").open("wb") as f:
            pickle.dump(
                {
                    "docs": [d.to_dict() for d in docs],
                    "graph": g,
                    "node_ids": self.node_ids,
                    "node_emb": self.node_emb,
                    "doc_emb": self.doc_emb,
                },
                f,
            )
        return {
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
            "erqa_triples": n_erqa,
            "lightrag_edges": n_lightrag,
        }

    def _load(self, index_dir: Path):
        with (index_dir / "sgrag.pkl").open("rb") as f:
            obj = pickle.load(f)
        self.docs = [Document.from_dict(d) for d in obj["docs"]]
        self.doc_by_id = {d.meta.get("source_doc_id", d.doc_id): d for d in self.docs}
        self.graph = obj["graph"]
        self.node_ids = obj["node_ids"]
        self.node_emb = obj["node_emb"]
        self.doc_emb = obj["doc_emb"]

    def _match_anchors(self, query: str, constraints: list[str], top_per_constraint: int = 3) -> list[dict[str, float]]:
        """Với mỗi điều kiện c_i, tìm các nút mỏ neo trong KG cùng điểm khớp."""
        if not self.node_ids:
            return []
        q_low = query.lower()
        groups: list[dict[str, float]] = []
        used_exact: set[str] = set()

        # 1. Khớp theo từng điều kiện tách được
        if constraints:
            c_vecs = self.c.embedder.encode(constraints, is_query=True)
            sims = c_vecs @ self.node_emb.T  # (m, |V|)
            for i, c in enumerate(constraints):
                cn = _norm_ent(c)
                cand: dict[str, float] = {}
                if cn in self.graph:
                    cand[cn] = 1.0
                    used_exact.add(cn)
                else:
                    for j, nid in enumerate(self.node_ids):
                        if cn in nid or nid in cn:
                            cand[nid] = max(cand.get(nid, 0.0), 0.92)
                for j in np.argsort(-sims[i])[:top_per_constraint]:
                    nid = self.node_ids[int(j)]
                    score = float(sims[i, j])
                    if score >= 0.45:
                        cand[nid] = max(cand.get(nid, 0.0), score * 0.85)
                if cand:
                    groups.append(cand)

        # 2. Bổ sung mọi nút trong KG có nhãn xuất hiện nguyên vẹn trong câu hỏi nhưng chưa có trong groups
        covered = {nid for g in groups for nid in g if g[nid] >= 0.9}
        for nid in self.node_ids:
            if len(nid) >= 4 and nid not in covered and nid not in _STOP_SPANS:
                if re.search(rf"\b{re.escape(nid)}\b", q_low):
                    groups.append({nid: 0.95})
        return groups

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        q_vec = self.c.embedder.encode([query], is_query=True)[0]
        doc_sims = self.doc_emb @ q_vec

        constraints, target_hints = extract_query_constraints(query)
        anchor_groups = self._match_anchors(query, constraints)
        exact_anchors = {nid for grp in anchor_groups for nid, sc in grp.items() if sc >= 0.9}

        # Chấm điểm từng nút ứng viên u theo cấu trúc sao-cầu (bridge-star)
        star_scores: dict[str, float] = {}
        star_matched_groups: dict[str, int] = {}
        star_paths: dict[str, list[tuple[str, str, str, list[str], float]]] = defaultdict(list)
        doc_graph_votes: dict[str, float] = defaultdict(float)

        for gi, grp in enumerate(anchor_groups):
            best_for_candidate: dict[str, tuple[float, tuple[str, str, str, list[str], float]]] = {}
            for anchor, a_score in grp.items():
                if anchor not in self.graph:
                    continue
                for nbr in self.graph.neighbors(anchor):
                    if nbr in exact_anchors and len(anchor_groups) >= 2:
                        continue
                    edge_dict = self.graph.get_edge_data(nbr, anchor) or {}
                    for ed in edge_dict.values():
                        rel = ed.get("relation", "")
                        desc = ed.get("description", "")
                        srcs = sorted(ed.get("sources") or [])
                        sc = a_score
                        if sc > best_for_candidate.get(nbr, (0.0, None))[0]:  # type: ignore[arg-type]
                            best_for_candidate[nbr] = (sc, (nbr, f"{rel} ({desc})".strip(), anchor, srcs, sc))

            for cand, (sc, path_info) in best_for_candidate.items():
                star_scores[cand] = star_scores.get(cand, 0.0) + sc
                star_matched_groups[cand] = star_matched_groups.get(cand, 0) + 1
                star_paths[cand].append(path_info)

        # Cộng điểm khớp loại thực thể đích (sport / event / award / ...)
        if target_hints:
            hint_set = set(target_hints)
            for cand in list(star_scores):
                rtoks = self.graph.nodes[cand].get("rel_tokens", set())
                overlap = sum(1 for h in hint_set if any(h in tok for tok in rtoks))
                if overlap:
                    star_scores[cand] += 0.35 * overlap

        # Xếp hạng các tâm sao u* theo (số ràng buộc thoả, tổng điểm cấu trúc)
        ranked_stars = sorted(
            star_scores,
            key=lambda u: (star_matched_groups[u], star_scores[u]),
            reverse=True,
        )

        items: list[RetrievedItem] = []
        n_constraints = max(1, len(anchor_groups))
        max_paths = int(self.params.get("max_star_subgraphs", 5))
        for rank_idx, u in enumerate(ranked_stars[:max_paths]):
            label = self.graph.nodes[u]["label"]
            paths = star_paths[u]
            srcs: list[str] = []
            edge_strs: list[str] = []
            # Tìm doc nguồn chứa đồng thời nhiều cạnh của sao u nhất
            doc_edge_counts: dict[str, int] = defaultdict(int)
            for _, rel_desc, anchor, e_srcs, _ in paths:
                a_label = self.graph.nodes[anchor]["label"]
                edge_strs.append(f"{label} --[{rel_desc}]--> {a_label}")
                for s in e_srcs:
                    doc_edge_counts[s] += 1
            srcs = [d for d, _ in sorted(doc_edge_counts.items(), key=lambda x: -x[1])]
            weight = (star_matched_groups[u] / n_constraints) * star_scores[u]
            for s, cnt in doc_edge_counts.items():
                doc_graph_votes[s] += weight * (cnt / max(1, len(paths))) / (rank_idx + 1)

            text = (
                f"Candidate Answer Entity: {label} "
                f"(satisfies {star_matched_groups[u]}/{n_constraints} query constraints)\n"
                + "\n".join(f"  - {es}" for es in edge_strs)
            )
            items.append(
                RetrievedItem(
                    text=text,
                    source_doc_ids=srcs,
                    score=float(star_scores[u]),
                    kind="path",
                    title=f"SG-RAG Subgraph: {label}",
                    meta={"entity": label, "matched_constraints": star_matched_groups[u]},
                )
            )

        # Kết hợp điểm cấu trúc đồ thị + điểm tương đồng dense để chọn các chunk nguồn tốt nhất
        combined_doc_scores: list[tuple[int, float]] = []
        for i, d in enumerate(self.docs):
            did = d.meta.get("source_doc_id", d.doc_id)
            score = float(doc_sims[i]) + 1.5 * doc_graph_votes.get(did, 0.0)
            combined_doc_scores.append((i, score))
        combined_doc_scores.sort(key=lambda x: -x[1])

        for i, sc in combined_doc_scores[:top_k]:
            d = self.docs[i]
            did = d.meta.get("source_doc_id", d.doc_id)
            items.append(
                RetrievedItem(
                    text=d.text,
                    source_doc_ids=[did],
                    score=sc,
                    kind="chunk",
                    title=d.title,
                )
            )

        return RetrievalResult(
            items,
            extra={
                "constraints": constraints,
                "n_anchor_groups": len(anchor_groups),
                "top_candidates": [self.graph.nodes[u]["label"] for u in ranked_stars[:3]],
            },
        )
