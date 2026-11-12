# Kết quả thực nghiệm: `hotpot_common`

## 1. Chất lượng truy hồi & sinh câu trả lời

| method | n | evidence_recall | evidence_all | recall@5 | all_recall@5 | mrr | ndcg@10 | context_tokens | retrieval_ms_p50 | generation_ms_p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 300 | 75.5 | 55.3 | 75.5 | 55.3 | 84.3 | 77.1 | 621.2 | 5.5 | – |

## 2. Chi phí lập chỉ mục

| method | index_s | index_llm_calls | index_llm_in_tok | index_llm_out_tok | index_mb |
|---|---|---|---|---|---|
| bm25 | 0.1 | 0 | 0 | 0 | 4.0 |

## 3. Phân tích theo `type`

| method | type | n | evidence_recall | evidence_all |
|---|---|---|---|---|
| bm25 | bridge | 150 | 70.3 | 46.0 |
| bm25 | comparison | 150 | 80.7 | 64.7 |
