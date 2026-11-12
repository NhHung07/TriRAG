"""Đăng ký phương pháp truy hồi: tên trong config → lớp cài đặt.

Trong config, mỗi mục ``methods.<tên>`` có thể khai báo ``type`` để tạo nhiều biến thể
của cùng phương pháp (VD ``lightrag_local: {type: lightrag, mode: local}``).
"""

from __future__ import annotations

from .base import Components, Retriever


def _registry() -> dict[str, type[Retriever]]:
    from .bm25 import BM25Retriever
    from .dense import DenseRetriever
    from .hybrid import HybridRetriever

    reg: dict[str, type[Retriever]] = {
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
    }
    # Adapter nặng: import khi cần
    try:
        from .raptor_adapter import RaptorRetriever

        reg["raptor"] = RaptorRetriever
    except ImportError:
        pass
    try:
        from .lightrag_adapter import LightRAGRetriever

        reg["lightrag"] = LightRAGRetriever
    except ImportError:
        pass
    from .sgrag_lite import SGRAGLiteRetriever

    reg["sgrag_lite"] = SGRAGLiteRetriever
    try:
        from .sgrag_adapter import SGRAGRetriever

        reg["sgrag"] = SGRAGRetriever
    except ImportError:
        pass
    return reg


def make_retriever(name: str, params: dict | None, components: Components) -> Retriever:
    params = dict(params or {})
    type_name = params.pop("type", name)
    reg = _registry()
    if type_name not in reg:
        raise ValueError(f"Phương pháp '{type_name}' không có hoặc thiếu thư viện. Có: {sorted(reg)}")
    return reg[type_name](name, params, components)
