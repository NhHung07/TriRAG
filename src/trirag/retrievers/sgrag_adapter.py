"""Adapter SG-RAG chính thức.

Ghi chú khảo sát ``third_party/SG-RAG``:
* Thư mục gốc của repo tác giả là bản fork của LightRAG; phần ``Dominant Embeddin Index/``
  chứa mã GNN-PE + R*-Tree C++ cho đồ thị sinh học (Yeast/Human) với đường dẫn cứng Windows,
  không nối với pipeline văn bản; còn ``SG-RAG-EXAMPLES/`` cung cấp 2.003 đồ thị con FB-ERQA
  (``.graphml``) và 2.152 câu hỏi đa điều kiện (``unique_answer_rows_all_combined.csv``).
* Do đó, :class:`SGRAGRetriever` kế thừa cài đặt khớp đồ thị con sao-cầu (bridge-star subgraph
  matching) trong :class:`SGRAGLiteRetriever`, hoạt động trực tiếp trên cả 2.003 đồ thị ``.graphml``
  của ERQA lẫn văn bản phẳng (HotpotQA / tiếng Việt).
"""

from __future__ import annotations

from .sgrag_lite import SGRAGLiteRetriever


class SGRAGRetriever(SGRAGLiteRetriever):
    type_name = "sgrag"
