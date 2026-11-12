"""Smoke test cho toàn bộ các phương pháp truy hồi trên kho 10 tài liệu nhỏ (không tốn quota API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trirag.config import DEFAULTS
from trirag.data.schema import Document
from trirag.embeddings import HashEmbedder
from trirag.llm.base import CachedLLM, SQLiteCache
from trirag.llm.fake import FakeBackend
from trirag.retrievers import Components, Retriever, make_retriever


def _toy_docs() -> list[Document]:
    items = [
        ("doc-01", "Paris", "Paris is the capital and most populous city of France, home to the Eiffel Tower."),
        ("doc-02", "Eiffel Tower", "The Eiffel Tower was constructed by Gustave Eiffel for the 1889 World's Fair in Paris."),
        ("doc-03", "Gustave Eiffel", "Alexandre Gustave Eiffel was a French civil engineer born in Dijon in 1832."),
        ("doc-04", "Berlin", "Berlin is the capital and largest city of Germany, known for the Brandenburg Gate."),
        ("doc-05", "Tokyo", "Tokyo is the capital of Japan and hosted the 1964 and 2020 Summer Olympics."),
        ("doc-06", "Dressage", "Dressage is an equestrian sport featured in the 1984 Summer Olympics and 2012 Summer Olympics with athletes from Ukraine."),
        ("doc-07", "Rowing", "Rowing is a water sport featured in the 2008 Summer Olympics with athletes from Finland and Myanmar."),
        ("doc-08", "Fencing", "Fencing is a combat sport associated with Portugal in the 2000 and 2012 Summer Olympics."),
        ("doc-09", "Python", "Python is a high-level programming language created by Guido van Rossum in 1991."),
        ("doc-10", "RAPTOR", "RAPTOR builds a recursive abstractive tree using UMAP dimensionality reduction and Gaussian Mixture Models."),
    ]
    return [Document(doc_id=i, title=t, text=txt) for i, t, txt in items]


def _make_components(tmp_path: Path) -> Components:
    cfg = dict(DEFAULTS)
    cfg["paths"] = {k: str(tmp_path / v) for k, v in DEFAULTS["paths"].items()}
    fake_llm = CachedLLM(FakeBackend(), SQLiteCache(tmp_path / "cache.sqlite"))
    embedder = HashEmbedder(dim=128)
    return Components(cfg, answer_llm=fake_llm, index_llm=fake_llm, judge_llm=fake_llm, embedder=embedder)


@pytest.mark.parametrize("method,params", [
    ("bm25", {"type": "bm25"}),
    ("dense", {"type": "dense"}),
    ("hybrid", {"type": "hybrid", "rerank": False}),
    ("raptor", {"type": "raptor", "num_layers": 2, "reduction_dimension": 3}),
    ("lightrag", {"type": "lightrag", "mode": "hybrid", "max_async": 1, "embedding_max_async": 1}),
    ("sgrag", {"type": "sgrag"}),
])
def test_retriever_build_load_and_retrieve(tmp_path: Path, method: str, params: dict):
    docs = _toy_docs()
    valid_ids = {d.doc_id for d in docs}
    comp = _make_components(tmp_path)
    idx_dir = tmp_path / "indexes" / method

    r = make_retriever(method, params, comp)
    stats = r.build_index(docs, idx_dir)
    assert Retriever.is_built(idx_dir)
    assert stats["n_docs"] == len(docs)

    # Nạp lại từ đĩa và truy hồi
    r2 = make_retriever(method, params, comp)
    r2.load(idx_dir)
    res = r2.retrieve("Which city is the capital of France and has the Eiffel Tower?", top_k=3)
    assert len(res.items) > 0
    returned_doc_ids = res.ranked_doc_ids()
    assert len(returned_doc_ids) > 0
    assert set(returned_doc_ids) <= valid_ids


def test_sgrag_exact_star_matching_on_erqa_triples(tmp_path: Path):
    """Kiểm tra SG-RAG tìm đúng thực thể tâm sao thoả mãn đồng thời nhiều điều kiện trên bộ ERQA."""
    docs = [
        Document(
            doc_id="g1.graphml",
            title="g1.graphml",
            text="Dressage connected to 1984, 1988, 2012 Summer Olympics and Ukraine.",
            meta={
                "triples": [
                    ("Dressage", "/olympics/olympic_games/sports", "Sports", "1984 Summer Olympics"),
                    ("Dressage", "/olympics/olympic_sport/athletes/olympics", "Athletes", "1988 Summer Olympics"),
                    ("Dressage", "/olympics/olympic_sport/athletes/olympics", "Athletes", "2012 Summer Olympics"),
                    ("Dressage", "/olympics/olympic_sport/athletes/country", "Country", "Ukraine"),
                    ("Dominican Republic", "/olympics/medals", "Medals", "1984 Summer Olympics"),
                    ("Dominican Republic", "/olympics/medals", "Medals", "2012 Summer Olympics"),
                ]
            },
        ),
        Document(
            doc_id="g2.graphml",
            title="g2.graphml",
            text="Judo connected to 2004 and 2008 Summer Olympics and Barbados.",
            meta={
                "triples": [
                    ("Judo", "/olympics/olympic_sport/athletes/olympics", "Athletes", "2004 Summer Olympics"),
                    ("Judo", "/olympics/olympic_sport/athletes/olympics", "Athletes", "2008 Summer Olympics"),
                    ("Judo", "/olympics/olympic_sport/athletes/country", "Country", "Barbados"),
                ]
            },
        ),
    ]
    comp = _make_components(tmp_path)
    r = make_retriever("sgrag", {"type": "sgrag"}, comp)
    r.build_index(docs, tmp_path / "sgrag_erqa")
    q = "Which sport is associated with athletes who competed in the 1988, 2012, and 1984 Summer Olympics and has a connection to Ukraine?"
    res = r.retrieve(q, top_k=2)
    assert res.extra["top_candidates"][0] == "Dressage"
    assert res.ranked_doc_ids()[0] == "g1.graphml"
