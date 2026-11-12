# TriRAG — So sánh các phương pháp truy hồi tri thức trong hệ thống hỏi đáp dựa trên RAG

> **Đề tài Nhóm 6 — Xử lý Ngôn ngữ Tự nhiên (NLP)**
> Nghiên cứu, tái lập và so sánh thực nghiệm 3 hướng tiếp cận truy hồi tri thức hiện đại trong RAG cùng các baseline truyền thống và cải tiến kết hợp:
> 1. **Cấu trúc phân cấp tài liệu**: **RAPTOR** (*ICLR 2024*)
> 2. **Kết hợp đồ thị tri thức & biểu diễn vector**: **LightRAG** (*Findings of EMNLP 2025*)
> 3. **Khớp cấu trúc đồ thị cho câu hỏi đa điều kiện**: **SG-RAG** (*ACL 2026*)
> 4. **Baselines & Cải tiến của nhóm**: **BM25**, **Dense RAG** (`BAAI/bge-m3`), và **Hybrid RRF (BM25 + Dense) + Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`).

---

## 1. Kiến trúc hệ thống & Nguyên tắc so sánh công bằng

Mọi phương pháp truy hồi đều cài đặt chung giao diện [`Retriever`](src/trirag/retrievers/base.py) và chỉ trả về **ngữ cảnh đã truy hồi** (`list[RetrievedItem]` kèm `source_doc_ids` truy vết về tài liệu gốc). Việc đóng gói ngữ cảnh theo ngân sách cố định (`context_token_budget = 2000` tokens) và gọi LLM sinh câu trả lời do [`Generator`](src/trirag/generation/generator.py) dùng chung đảm nhận:

- **Cùng mô hình embedding đa ngữ chạy local**: `BAAI/bge-m3` (có cache vector SQLite tại `.cache/`).
- **Cùng mô hình sinh & prompt**: Gemini (`gemini-2.5-flash` cho sinh câu trả lời, `gemini-2.5-flash-lite` cho lập chỉ mục) đi qua [`CachedLLM`](src/trirag/llm/base.py) có cache SQLite, giới hạn tốc độ RPM (token bucket), retry/backoff tự động và bộ đếm token/lời gọi.
- **Cùng bộ độ đo**:
  - *Truy hồi (quy về ID tài liệu nguồn)*: `Evidence Recall`, `All-Evidence Recall` (đủ toàn bộ tài liệu vàng cho câu hỏi multi-hop), `Recall@k`, `MRR`, `nDCG@k`.
  - *Câu trả lời*: `Exact Match (EM)`, `Token-level F1` (theo chuẩn chính thức của HotpotQA), và `Win-rate` chấm cặp bằng `LLM-as-a-Judge` (đảo thứ tự A/B 2 lượt để khử thiên lệch vị trí).
  - *Chi phí tính toán*: Thời gian lập chỉ mục (`index_s`), số lời gọi và số token LLM khi index, dung lượng chỉ mục (`index_mb`), độ trễ truy hồi/sinh trung vị (`p50 ms`).

---

## 2. Cấu trúc thư mục & Minh bạch phần kế thừa / tự thực hiện

```text
TriRAG/
├── configs/                        # Cấu hình YAML cho từng thực nghiệm
│   ├── hotpot_common.yaml          # (A) So sánh chung trên HotpotQA (300 câu)
│   ├── repro_raptor_qasper.yaml    # (B1) Tái lập thu nhỏ RAPTOR trên QASPER
│   ├── repro_lightrag_ud.yaml      # (B2) Tái lập thu nhỏ LightRAG trên UltraDomain
│   ├── repro_sgrag_erqa.yaml       # (B3) Tái lập thu nhỏ SG-RAG trên FB-ERQA
│   └── demo_vi.yaml                # (C) Demo hỏi đáp đề cương môn học tiếng Việt
├── src/trirag/                     # [NHÓM TỰ XÂY DỰNG]
│   ├── llm/                        # Client Gemini + OpenAI-compat, cache SQLite, rate limiter, usage tracker
│   ├── embeddings.py               # SentenceTransformer (bge-m3) có cache vector + HashEmbedder cho test
│   ├── data/                       # Schema, chunking, loaders: HotpotQA, QASPER, UltraDomain, ERQA, tài liệu tiếng Việt
│   ├── retrievers/                 # Interface Retriever + BM25, Dense, Hybrid(+Rerank), adapters cho RAPTOR/LightRAG/SG-RAG
│   ├── generation/                 # Đóng gói ngữ cảnh theo ngân sách token + prompt chuẩn + Generator
│   └── eval/                       # Độ đo EM/F1/Recall/MRR/nDCG, LLM-judge đảo vị trí, ExperimentRunner, bảng tổng hợp
├── scripts/                        # [NHÓM TỰ XÂY DỰNG]
│   ├── prepare_data.py             # Tải & tiền xử lý dữ liệu theo config
│   ├── build_index.py              # Xây dựng chỉ mục + ghi nhận thống kê chi phí
│   ├── run_eval.py                 # Chạy thực nghiệm (hỗ trợ resume) + chấm cặp LLM-judge
│   ├── make_tables.py              # Xuất bảng kết quả Markdown & LaTeX
│   └── error_sample.py             # Phân loại tự động 4 nhóm lỗi & lấy mẫu phân tích định tính
├── app/
│   └── streamlit_app.py            # [NHÓM TỰ XÂY DỰNG] Giao diện Chatbot hỏi đáp kèm trích dẫn & so sánh song song
├── tests/                          # [NHÓM TỰ XÂY DỰNG] Unit test & smoke test cho toàn bộ pipeline
└── third_party/                    # [KẾ THỪA MÃ NGUỒN TÁC GIẢ — git submodule]
    ├── raptor/                     # parthsarthi03/raptor (commit 7da1d48)
    └── SG-RAG/                     # CAU-X-AI-Lab/SG-RAG (commit 49bad56, kèm bộ dữ liệu FB-ERQA trong SG-RAG-EXAMPLES/)
```

| Thành phần | Nguồn kế thừa | Phần nhóm tự xây dựng / điều chỉnh |
|---|---|---|
| **RAPTOR** | `third_party/raptor` (`7da1d48`) | [`raptor_adapter.py`](src/trirag/retrievers/raptor_adapter.py): nối với `CachedLLM` & `bge-m3`, xử lý an toàn UMAP/GMM khi cụm nhỏ, truy vết đệ quy `source_doc_ids` từ nút tóm tắt về các nút lá. |
| **LightRAG** | `lightrag-hku` | [`lightrag_adapter.py`](src/trirag/retrievers/lightrag_adapter.py): bọc async `CachedLLM` & `bge-m3`, dùng `query_data` để tách ngữ cảnh (`chunks` + `relationships` + `entities`) kèm `file_path` nguồn, chia sẻ chỉ mục giữa 4 chế độ `hybrid`/`local`/`global`/`naive`. |
| **SG-RAG & ERQA** | `third_party/SG-RAG` (`49bad56`, `SG-RAG-EXAMPLES/`) | [`erqa.py`](src/trirag/data/erqa.py) & [`sgrag_lite.py`](src/trirag/retrievers/sgrag_lite.py): đọc trực tiếp 2.003 đồ thị `.graphml` và câu hỏi đa ràng buộc của FB-ERQA, cài đặt thuật toán phân rã ràng buộc câu hỏi + khớp đồ thị con dạng sao-cầu (*bridge-star subgraph matching*) hoạt động trên cả ERQA và văn bản phẳng. |
| **Hybrid + Rerank, Đánh giá, Demo** | — | Toàn bộ `bm25.py`, `dense.py`, `hybrid.py` (RRF + `bge-reranker-v2-m3`), `eval/`, `scripts/`, `app/streamlit_app.py`, `tests/`. |

---

## 3. Cài đặt môi trường

```bash
# 1. Cài đặt dependencies vào môi trường Python 3.11 (.venv)
~/.local/bin/uv sync --extra all --group dev

# 2. Cấu hình khoá API (mỗi thành viên dùng API key của chính mình)
cp .env.example .env
# Điền GEMINI_API_KEY=... (và JUDGE_API_KEY / JUDGE_BASE_URL nếu dùng model khác để chấm điểm)

# 3. Chạy kiểm thử tự động (không tốn quota API)
.venv/bin/pytest tests/ -v
```

---

## 4. Hướng dẫn chạy các thực nghiệm

### Thực nghiệm (A) — So sánh chung trên HotpotQA (300 câu hỏi)

```bash
# Bước 1: Tự động tải HotpotQA dev (distractor) và lấy mẫu 300 câu cố định theo seed
.venv/bin/python scripts/prepare_data.py --config configs/hotpot_common.yaml

# Bước 2: Chạy cột mốc M1 (BM25 + Dense trên 20 câu đầu; thêm --retrieval-only nếu chưa điền GEMINI_API_KEY)
.venv/bin/python scripts/run_eval.py --config configs/hotpot_common.yaml --methods bm25,dense --limit 20

# Bước 3: Chạy đầy đủ tất cả các phương pháp (hỗ trợ tạm dừng và chạy tiếp tự động)
.venv/bin/python scripts/run_eval.py --config configs/hotpot_common.yaml --methods bm25,dense,hybrid,sgrag,raptor,lightrag,lightrag_local,lightrag_global,lightrag_naive

# Bước 4: Xuất bảng Markdown + LaTeX (tách theo bridge vs comparison) và phân tích lỗi
.venv/bin/python scripts/make_tables.py --config configs/hotpot_common.yaml --group-by type
.venv/bin/python scripts/error_sample.py --config configs/hotpot_common.yaml
```

### Thực nghiệm (B) — Tái lập thu nhỏ trên dữ liệu gốc của từng bài báo

```bash
# (B3) SG-RAG trên FB-ERQA (dữ liệu có sẵn trong third_party/SG-RAG/SG-RAG-EXAMPLES, không cần tải thêm)
.venv/bin/python scripts/prepare_data.py --config configs/repro_sgrag_erqa.yaml
.venv/bin/python scripts/run_eval.py --config configs/repro_sgrag_erqa.yaml --methods bm25,dense,hybrid,sgrag

# (B1) RAPTOR trên QASPER (15 bài báo khoa học, tối đa 100 câu hỏi)
.venv/bin/python scripts/prepare_data.py --config configs/repro_raptor_qasper.yaml
.venv/bin/python scripts/run_eval.py --config configs/repro_raptor_qasper.yaml --methods bm25,dense,raptor

# (B2) LightRAG trên UltraDomain + chấm cặp Win-rate bằng LLM-as-a-Judge
.venv/bin/python scripts/prepare_data.py --config configs/repro_lightrag_ud.yaml
.venv/bin/python scripts/run_eval.py --config configs/repro_lightrag_ud.yaml --methods dense,lightrag --judge-pair lightrag,dense
```

### Thực nghiệm (C) & Giao diện Demo Chatbot tiếng Việt

1. Đặt file PDF đề cương môn học (50–70 trang) vào thư mục `data/raw/vi_docs/` (đã có sẵn file mẫu [`de_cuong_hoc_phan_nlp_cntt.md`](data/raw/vi_docs/de_cuong_hoc_phan_nlp_cntt.md) và bộ câu hỏi [`vi_questions.jsonl`](data/raw/vi_questions.jsonl)).
2. Tiền xử lý và mở giao diện Streamlit:

```bash
.venv/bin/python scripts/prepare_data.py --config configs/demo_vi.yaml
.venv/bin/streamlit run app/streamlit_app.py
```
