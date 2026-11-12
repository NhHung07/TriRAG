# ĐỀ CƯƠNG CHI TIẾT HỌC PHẦN: XỬ LÝ NGÔN NGỮ TỰ NHIÊN & TRÍ TUỆ NHÂN TẠO

## Chương 1. Thông tin chung về học phần

1.1. Tên học phần: Xử lý ngôn ngữ tự nhiên (Natural Language Processing — mã học phần: IT4868).
Số tín chỉ: 3 tín chỉ (2 tín chỉ lý thuyết, 1 tín chỉ thực hành/bài tập lớn).
Thời lượng: 30 tiết lý thuyết, 15 tiết bài tập và thảo luận nhóm, 90 giờ tự học.
Học phần tiên quyết: Nhập môn Học máy (IT4850) và Xác suất thống kê (MI2020).

1.2. Mục tiêu học phần
Học phần trang bị cho sinh viên kiến thức nền tảng và hiện đại về xử lý ngôn ngữ tự nhiên, từ biểu diễn từ vựng, mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models), đến các hệ thống Hỏi Đáp dựa trên Truy hồi tăng cường sinh (Retrieval-Augmented Generation — RAG) và Đồ thị tri thức (Knowledge Graph).

## Chương 2. Quy định đánh giá và trọng số điểm

2.1. Thành phần điểm học phần
Điểm tổng kết học phần được tính theo thang điểm 10, gồm 3 thành phần:
- Điểm chuyên cần và bài tập tuần: 10% tổng điểm (yêu cầu tham dự tối thiểu 80% số buổi học trên lớp).
- Điểm Bài tập lớn (Project nhóm 3–4 sinh viên): 40% tổng điểm, đánh giá qua báo cáo kỹ thuật, mã nguồn thực nghiệm và buổi vấn đáp demo.
- Điểm Thi cuối kỳ: 50% tổng điểm, hình thức thi viết tự luận kết hợp trắc nghiệm trong 90 phút.

2.2. Điều kiện dự thi và điểm liệt
Sinh viên vắng quá 20% số tiết học hoặc không nộp Bài tập lớn sẽ bị điểm 0 quá trình và không đủ điều kiện dự thi cuối kỳ. Điểm liệt của bài thi cuối kỳ là dưới 3,0 điểm trên thang 10.

## Chương 3. Nội dung chi tiết theo tuần học

Tuần 1. Giới thiệu tổng quan về NLP và tiền xử lý văn bản tiếng Việt (tách từ, chuẩn hoá Unicode NFC, loại bỏ từ dừng).
Tuần 2. Biểu diễn văn bản truyền thống: Bag-of-Words, TF-IDF và mô hình xác suất BM25 (Okapi BM25 với hai siêu tham số k1 và b).
Tuần 3. Biểu diễn từ phân tán: Word2Vec (CBOW, Skip-gram), GloVe và FastText cho ngôn ngữ đơn âm tiết.
Tuần 4. Kiến trúc Transformer, cơ chế Self-Attention và các mô hình Encoder (BERT, PhoBERT, BGE-M3 đa ngữ).
Tuần 5. Mô hình ngôn ngữ lớn (Large Language Models — LLM), kỹ thuật Prompting và vấn đề ảo giác (Hallucination) trong hỏi đáp thực tế.
Tuần 6. Hệ thống Retrieval-Augmented Generation (RAG) cơ bản: cắt đoạn văn bản (chunking), tìm kiếm vector dày (Dense Retrieval) và kết hợp lai (Hybrid BM25 + Dense với Reciprocal Rank Fusion).
Tuần 7. Truy hồi phân cấp trên tài liệu dài: phương pháp RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) sử dụng giảm chiều UMAP, phân cụm GMM và tóm tắt đệ quy nhiều tầng.
Tuần 8. Truy hồi kết hợp Đồ thị tri thức và Vector: phương pháp LightRAG trích xuất thực thể, quan hệ và truy hồi hai mức (Low-level & High-level keywords).
Tuần 9. Truy hồi hướng dẫn bởi cấu trúc đồ thị cho câu hỏi đa điều kiện: phương pháp SG-RAG (Structure Guided RAG) và bài toán Exact Retrieval Problem (ERP) trên bộ dữ liệu ERQA.
Tuần 10. Phương pháp đánh giá hệ thống Hỏi Đáp RAG: độ đo truy hồi (Recall@k, MRR, nDCG@k), độ đo câu trả lời (Exact Match, Token-level F1) và LLM-as-a-Judge.

## Chương 4. Quy định về Bài tập lớn (Project)

4.1. Yêu cầu triển khai
Mỗi nhóm từ 3 đến 4 sinh viên lựa chọn một đề tài thực nghiệm so sánh tối thiểu 3 phương pháp hiện đại từ các hội nghị ICLR, ACL, EMNLP cùng với các baseline BM25 và Dense RAG. Nhóm bắt buộc phải:
1. Xây dựng quy trình đánh giá thống nhất (cùng mô hình sinh, cùng prompt và ngân sách ngữ cảnh).
2. Thực hiện cả thí nghiệm so sánh trên thiết lập chung và thí nghiệm tái lập thu nhỏ theo bài báo gốc.
3. Đề xuất tối thiểu một cải tiến kỹ thuật (ví dụ: kết hợp Hybrid BM25 + Dense kèm Cross-Encoder Reranker) và xây dựng giao diện demo có trích dẫn nguồn tham chiếu.

4.2. Mốc thời gian nộp bài
- Tuần 1: Chốt đề tài, khảo sát mã nguồn tác giả và hoàn thiện hạ tầng đánh giá cơ sở.
- Tuần 2: Chạy thực nghiệm trên tập dữ liệu benchmark (HotpotQA, QASPER, UltraDomain, ERQA) và thu thập bảng kết quả.
- Tuần 3: Phân tích lỗi định lượng/định tính, hoàn thiện giao diện demo trên tài liệu tiếng Việt và nộp báo cáo cuối cùng.
