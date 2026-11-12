"""ERQA (Exact Retrieval Question Answering — đi kèm SG-RAG, ACL 2026).

Đọc trực tiếp dữ liệu có sẵn trong ``third_party/SG-RAG/SG-RAG-EXAMPLES``:
* ``unique_answer_rows_all_combined.csv``: các cặp (file GraphML, câu hỏi đa ràng buộc, đáp án chuẩn)
* ``subgraph/*.graphml``: 2.003 đồ thị con dạng sao-cầu (bridge-star subgraph) trích từ FB15K-237.

Mỗi file ``.graphml`` trong tập mẫu được chuyển thành một :class:`Document`:
* ``doc_id`` = tên file ``.graphml`` (khớp với cột ``file`` trong CSV → ``gold_doc_ids = [file]``),
* ``text`` = các câu mô tả quan hệ bằng tiếng Anh (để mọi phương pháp BM25 / Dense / RAPTOR /
  LightRAG cũng chạy được trên ERQA),
* ``meta["triples"]`` = danh sách bộ 4 ``(head, relation, description, tail)`` để SG-RAG dựng
  đồ thị tri thức chính xác mà không tốn lời gọi LLM.
"""

from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from ..utils import PROJECT_ROOT
from .schema import Dataset, Document, QAExample

DEFAULT_ERQA_DIR = PROJECT_ROOT / "third_party" / "SG-RAG" / "SG-RAG-EXAMPLES"
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def _clean_rel(rel: str) -> str:
    parts = [p for p in rel.replace(".", "/").split("/") if p and p not in ("user", "jg", "default_domain")]
    if len(parts) >= 2:
        parts = parts[-2:]
    return " / ".join(p.replace("_", " ") for p in parts) or rel


def parse_graphml_file(path: Path) -> tuple[list[tuple[str, str, str, str]], str]:
    """Trả về (danh sách (head, relation, desc, tail), văn bản tiếng Anh mô tả các cạnh)."""
    root = ET.parse(path).getroot()
    labels: dict[str, str] = {}
    for node in root.findall(".//g:node", NS):
        nid = node.attrib.get("id", "")
        d0 = node.find("g:data[@key='d0']", NS)
        labels[nid] = (d0.text or nid).strip() if d0 is not None else nid

    triples: list[tuple[str, str, str, str]] = []
    lines: list[str] = []
    for edge in root.findall(".//g:edge", NS):
        src = labels.get(edge.attrib.get("source", ""), edge.attrib.get("source", ""))
        tgt = labels.get(edge.attrib.get("target", ""), edge.attrib.get("target", ""))
        d1 = edge.find("g:data[@key='d1']", NS)
        d2 = edge.find("g:data[@key='d2']", NS)
        rel = (d1.text or "").strip() if d1 is not None else ""
        desc = (d2.text or "").strip() if d2 is not None else ""
        rel_short = _clean_rel(rel)
        triples.append((src, rel, desc, tgt))
        lines.append(f"{src} --[{rel_short}: {desc}]--> {tgt}.")
    return triples, "\n".join(lines)


def load_erqa(
    path: str | Path = DEFAULT_ERQA_DIR,
    n_questions: int = 300,
    n_distractor_graphs: int = 100,
    seed: int = 42,
    **_kwargs,
) -> Dataset:
    base = Path(path)
    csv_path = base / "unique_answer_rows_all_combined.csv" if base.is_dir() else base
    subgraph_dir = csv_path.parent / "subgraph"

    available = {p.name: p for p in sorted(subgraph_dir.glob("*.graphml"))}
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r.get("file") in available and (r.get("query") or "").strip() and (r.get("answer") or "").strip()
        ]

    rng = random.Random(seed)
    selected = rng.sample(rows, min(n_questions, len(rows)))

    used_files = list(dict.fromkeys(r["file"] for r in selected))
    unused_files = [fn for fn in sorted(available) if fn not in set(used_files)]
    if n_distractor_graphs > 0 and unused_files:
        used_files.extend(rng.sample(unused_files, min(n_distractor_graphs, len(unused_files))))

    docs: list[Document] = []
    for fn in used_files:
        triples, text = parse_graphml_file(available[fn])
        docs.append(Document(doc_id=fn, title=fn, text=text, meta={"triples": triples}))

    examples = [
        QAExample(
            qid=f"erqa-{i:04d}",
            question=r["query"].strip(),
            answers=[r["answer"].strip()],
            gold_doc_ids=[r["file"]],
            corpus_id="all",
            meta={"subgraph_file": r["file"]},
        )
        for i, r in enumerate(selected)
    ]
    return Dataset("erqa", examples, {"all": docs})
