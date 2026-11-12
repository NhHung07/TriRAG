# Kết quả thực nghiệm: `repro_sgrag_erqa`

## 1. Chất lượng truy hồi & sinh câu trả lời

| method | n | evidence_recall | evidence_all | recall@5 | all_recall@5 | mrr | ndcg@10 | context_tokens | retrieval_ms_p50 | generation_ms_p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 300 | 86.7 | 86.7 | 87.7 | 87.7 | 79.3 | 83.1 | 1,915.8 | 0.9 | – |

## 2. Chi phí lập chỉ mục

| method | index_s | index_llm_calls | index_llm_in_tok | index_llm_out_tok | index_mb |
|---|---|---|---|---|---|
| bm25 | 0.0 | 0 | 0 | 0 | 1.6 |

## 3. Phân tích theo `type`

| method | type | n | evidence_recall | evidence_all |
|---|---|---|---|---|
| bm25 | None | 300 | 86.7 | 86.7 |
