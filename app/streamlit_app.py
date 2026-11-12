"""Giao diện Demo Chatbot Hỏi Đáp TriRAG (Streamlit).

Chạy ứng dụng:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from trirag.config import load_config
from trirag.data.io import build_dataset, load_processed, processed_dir, save_dataset
from trirag.eval.runner import ExperimentRunner
from trirag.eval.summary import MAIN_COLS, summary_tables, to_markdown
from trirag.generation import Generator, pack_context
from trirag.utils import PROJECT_ROOT, load_env

load_env()

st.set_page_config(page_title="TriRAG — Hỏi Đáp Tài Liệu & So Sánh Truy Hồi", layout="wide")
st.title("TriRAG — Hệ thống Hỏi Đáp Tài Liệu & So Sánh Truy Hồi Tri Thức")
st.caption("Nhóm 6 NLP · BM25 · Dense · Hybrid+Rerank · RAPTOR · LightRAG · SG-RAG")

CONFIG_FILES = {
    "Demo Đề cương môn học (Tiếng Việt)": PROJECT_ROOT / "configs" / "demo_vi.yaml",
    "So sánh chung — HotpotQA": PROJECT_ROOT / "configs" / "hotpot_common.yaml",
    "Tái lập SG-RAG — FB-ERQA": PROJECT_ROOT / "configs" / "repro_sgrag_erqa.yaml",
    "Tái lập RAPTOR — QASPER": PROJECT_ROOT / "configs" / "repro_raptor_qasper.yaml",
}


@st.cache_resource(show_spinner="Đang nạp cấu hình & kho tài liệu...")
def get_runner(config_path: str) -> ExperimentRunner:
    cfg = load_config(config_path)
    pdir = processed_dir(cfg)
    if not (pdir / "meta.json").exists():
        ds = build_dataset(cfg)
        save_dataset(ds, pdir)
    else:
        ds = load_processed(pdir)
    return ExperimentRunner(cfg, dataset=ds)


with st.sidebar:
    st.header("Cấu hình")
    cfg_label = st.selectbox("Bộ dữ liệu / Cấu hình", list(CONFIG_FILES.keys()))
    cfg_path = str(CONFIG_FILES[cfg_label])
    runner = get_runner(cfg_path)
    methods = list(runner.cfg["methods"].keys())
    top_k = st.slider("Số mục truy hồi (top-k)", min_value=1, max_value=10, value=int(runner.cfg["retrieval"]["top_k"]))
    budget = st.slider(
        "Ngân sách ngữ cảnh (tokens)", min_value=500, max_value=4000, value=int(runner.cfg["retrieval"]["context_token_budget"]), step=250
    )
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    retrieval_only = st.checkbox(
        "Chỉ truy hồi (không gọi LLM sinh câu trả lời)", value=not has_api_key,
        help="Bật khi chưa cấu hình GEMINI_API_KEY hoặc muốn kiểm tra nhanh các đoạn truy hồi."
    )

    if runner.cfg["dataset"]["name"] == "vi_docs":
        st.divider()
        st.subheader("Tải thêm PDF đề cương tiếng Việt")
        uploaded = st.file_uploader("Chọn file .pdf / .md / .txt", type=["pdf", "md", "txt"], accept_multiple_files=True)
        if uploaded and st.button("Lưu & lập lại chỉ mục"):
            vi_dir = PROJECT_ROOT / runner.cfg["dataset"]["input_dir"]
            vi_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded:
                (vi_dir / uf.name).write_bytes(uf.getvalue())
            ds = build_dataset(runner.cfg)
            save_dataset(ds, processed_dir(runner.cfg))
            st.cache_resource.clear()
            st.success(f"Đã cập nhật {len(uploaded)} tài liệu ({ds.stats()['n_docs']} chunks).")
            st.rerun()


def run_single_query(method: str, question: str, corpus_id: str = "all"):
    docs = runner.dataset.corpora[corpus_id]
    retriever = runner.get_retriever(method, corpus_id, docs)
    res = retriever.retrieve(question, max(top_k, 10))
    ctx_items = res.items[:top_k] if getattr(retriever, "chunk_ranked", True) else res.items
    packed = pack_context(ctx_items, budget)
    answer = None
    if not retrieval_only:
        gcfg = runner.cfg["generation"]
        gen = Generator(runner.c.answer_llm, gcfg["prompt"], gcfg["max_tokens"], gcfg["temperature"])
        answer = gen.answer(question, packed)
    return res, packed, answer


def render_result(method: str, res, packed, answer):
    st.markdown(f"#### Phương pháp: `{method}`")
    cols = st.columns(3)
    cols[0].metric("Độ trễ truy hồi", f"{res.latency_s * 1000:.1f} ms")
    cols[1].metric("Ngữ cảnh đóng gói", f"{packed.n_tokens} tokens ({len(packed.items)} mục)")
    if answer is not None:
        cols[2].metric("Độ trễ sinh", f"{answer.latency_s * 1000:.1f} ms" + (" (cache)" if answer.cached else ""))
        st.markdown("**Câu trả lời:**")
        st.info(answer.text)
    else:
        cols[2].metric("Chế độ", "Chỉ truy hồi")

    st.markdown("**Nguồn tham chiếu đã truy hồi:**")
    for idx, it in enumerate(packed.items, 1):
        pages = (it.meta or {}).get("pages")
        page_str = f" · Trang {pages[0]}–{pages[-1]}" if pages else ""
        src_str = ", ".join(it.source_doc_ids) if it.source_doc_ids else "N/A"
        header = f"[{idx}] {it.title or src_str} ({it.kind}, score={it.score:.3f}{page_str})"
        with st.expander(header, expanded=(idx <= 2)):
            st.caption(f"ID nguồn: `{src_str}`")
            st.write(it.text)


tab_qa, tab_cmp, tab_bench = st.tabs([
    "💬 Hỏi đáp & Trích dẫn nguồn",
    "⚖️ So sánh trực tiếp 2 phương pháp",
    "📊 Bảng kết quả thực nghiệm",
])

sample_questions = [ex.question for ex in runner.dataset.examples[:15]]
corpus_ids = list(runner.dataset.corpora.keys())

with tab_qa:
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_sample = st.selectbox("Câu hỏi mẫu (tuỳ chọn):", ["(Tự nhập câu hỏi bên dưới)"] + sample_questions)
    with c2:
        method_single = st.selectbox("Phương pháp truy hồi:", methods, index=min(2, len(methods) - 1))
        cid_single = st.selectbox("Kho tài liệu:", corpus_ids) if len(corpus_ids) > 1 else corpus_ids[0]

    default_q = "" if selected_sample.startswith("(") else selected_sample
    question_single = st.text_input("Nhập câu hỏi:", value=default_q, key="q_single")
    if st.button("Truy hồi & Trả lời", type="primary", disabled=not question_single.strip()):
        with st.spinner(f"Đang chạy `{method_single}`..."):
            res, packed, ans = run_single_query(method_single, question_single.strip(), cid_single)
        render_result(method_single, res, packed, ans)

with tab_cmp:
    col_a, col_b = st.columns(2)
    with col_a:
        m_a = st.selectbox("Phương pháp A:", methods, index=min(2, len(methods) - 1), key="cmp_a")
    with col_b:
        m_b = st.selectbox("Phương pháp B:", methods, index=min(len(methods) - 1, 4), key="cmp_b")
    question_cmp = st.text_input("Câu hỏi so sánh:", value=sample_questions[0] if sample_questions else "", key="q_cmp")
    if st.button("Chạy so sánh song song", type="primary", disabled=not question_cmp.strip()):
        col_left, col_right = st.columns(2)
        with col_left:
            with st.spinner(f"Đang chạy `{m_a}`..."):
                ra, pa, aa = run_single_query(m_a, question_cmp.strip(), corpus_ids[0])
            render_result(m_a, ra, pa, aa)
        with col_right:
            with st.spinner(f"Đang chạy `{m_b}`..."):
                rb, pb, ab = run_single_query(m_b, question_cmp.strip(), corpus_ids[0])
            render_result(m_b, rb, pb, ab)

with tab_bench:
    tables = summary_tables(runner.cfg, group_by="type")
    if not tables["main"]:
        st.info("Chưa có file kết quả trong `results/`. Hãy chạy `scripts/run_eval.py` trước.")
    else:
        st.subheader("1. Chất lượng truy hồi & câu trả lời")
        st.markdown(to_markdown(tables["main"], ["method", "n"] + MAIN_COLS + ["context_tokens", "retrieval_ms_p50", "generation_ms_p50"]))
        st.subheader("2. Chi phí lập chỉ mục")
        st.markdown(to_markdown(tables["cost"]))
        if tables["grouped"]:
            st.subheader("3. Phân tích theo loại câu hỏi")
            st.markdown(to_markdown(tables["grouped"], ["method", "type", "n", "em", "f1", "evidence_recall", "evidence_all"]))
