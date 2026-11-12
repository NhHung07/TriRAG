## **ĐỀ TÀI: SO SÁNH CÁC PHƯƠNG PHÁP TRUY HỒI TRI THỨC TRONG HỆ THỐNG HỎI ĐÁP DỰA TRÊN RAG (RETRIEVAL-AUGMENTED GENERATION)**

### **1\. Giới thiệu đề tài**

Retrieval-Augmented Generation (RAG) là phương pháp kết hợp mô hình ngôn ngữ lớn (Large Language Model – LLM) với hệ thống truy hồi thông tin nhằm cung cấp tri thức bên ngoài cho quá trình sinh câu trả lời. Thay vì chỉ dựa vào kiến thức đã học trong quá trình huấn luyện, hệ thống RAG truy hồi các tài liệu liên quan đến câu hỏi của người dùng, sau đó sử dụng những tài liệu này làm ngữ cảnh để tạo câu trả lời.

Tuy nhiên, chất lượng của chatbot RAG phụ thuộc đáng kể vào khả năng tìm kiếm và tổ chức tri thức. Các phương pháp truy hồi truyền thống thường chia tài liệu thành những đoạn văn bản độc lập và tìm kiếm dựa trên độ tương đồng giữa câu hỏi với các đoạn đó. Cách tiếp cận này có thể gặp hạn chế khi câu hỏi yêu cầu tổng hợp thông tin từ nhiều phần của tài liệu, khai thác mối quan hệ giữa các thực thể hoặc đáp ứng đồng thời nhiều điều kiện.

Trong đề tài này, nhóm đề xuất nghiên cứu và so sánh các phương pháp truy hồi tri thức hiện đại trong hệ thống hỏi đáp dựa trên RAG. Nhóm tập trung vào ba hướng tiếp cận: truy hồi theo cấu trúc phân cấp của tài liệu, truy hồi kết hợp đồ thị tri thức với biểu diễn vector, và truy hồi có xét đến cấu trúc quan hệ của câu hỏi.

Thông qua việc tái lập các phương pháp từ những công trình nghiên cứu đã công bố, nhóm hướng tới đánh giá ưu điểm, hạn chế và chi phí tính toán của từng phương pháp, từ đó lựa chọn phương pháp phù hợp để xây dựng một chatbot hỏi đáp có khả năng trả lời dựa trên tài liệu và cung cấp nguồn tham chiếu.

### **2\. Lý do lựa chọn đề tài**

* **Tính ứng dụng:** Chatbot hỏi đáp dựa trên tri thức có thể được ứng dụng trong tra cứu tài liệu học tập, hỗ trợ sinh viên, hỏi đáp trên tài liệu chuyên ngành và hệ thống hỗ trợ khách hàng.  
* **Tính nghiên cứu:** RAG là một hướng nghiên cứu quan trọng trong NLP hiện đại. Các phương pháp tổ chức và truy hồi tri thức khác nhau tạo ra cơ sở để so sánh thực nghiệm và phân tích những yếu tố ảnh hưởng đến chất lượng câu trả lời.  
* **Khả năng đánh giá:** Có thể sử dụng các bộ dữ liệu hỏi đáp công khai, các độ đo truy hồi thông tin và các độ đo đánh giá câu trả lời để thực hiện thí nghiệm có kiểm soát.  
* **Khả năng tái lập:** Các công trình được lựa chọn có bài báo công khai và mã nguồn tham khảo, cho phép nhóm triển khai lại trên tập dữ liệu gốc hoặc tập con phù hợp với tài nguyên tính toán.  
* **Khả năng mở rộng:** Ngoài việc tái lập và so sánh các phương pháp, nhóm có thể thử nghiệm cải tiến nhỏ bằng cách kết hợp các phương pháp truy hồi hoặc điều chỉnh cách lựa chọn tài liệu đầu vào cho mô hình sinh.

### **3\. Các hướng nghiên cứu và paper tiêu biểu**

#### **3.1. Hướng 1: Truy hồi dựa trên cấu trúc phân cấp của tài liệu**

**Paper 1: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024).**

* **Ý tưởng:** RAPTOR xây dựng cấu trúc cây phân cấp bằng cách mã hóa, phân cụm và tóm tắt các đoạn văn bản theo nhiều cấp độ. Khi trả lời câu hỏi, hệ thống có thể truy hồi thông tin ở cấp đoạn văn bản hoặc cấp tóm tắt, qua đó khai thác được cả thông tin chi tiết lẫn nội dung tổng quát của tài liệu.  
* **Lý do lựa chọn:** Phương pháp đại diện cho hướng tổ chức tri thức theo cấu trúc phân cấp, phù hợp với các câu hỏi yêu cầu tổng hợp thông tin từ tài liệu dài.  
* **Bài báo:** [https\://proceedings.iclr.cc/paper\_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)  
* **Mã nguồn:** [https\://github.com/parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)

#### **3.2. Hướng 2: Truy hồi kết hợp đồ thị tri thức và vector**

**Paper 2: LightRAG: Simple and Fast Retrieval-Augmented Generation (Findings of EMNLP 2025).**

* **Ý tưởng:** LightRAG tích hợp cấu trúc đồ thị vào quá trình lập chỉ mục và truy hồi, kết hợp biểu diễn vector với việc khai thác thực thể và quan hệ. Phương pháp hỗ trợ truy hồi ở cả cấp độ thông tin cụ thể và cấp độ khái quát.  
* **Lý do lựa chọn:** Đây là hướng tiếp cận trực tiếp cho bài toán chatbot hỏi đáp dựa trên kho tri thức, đặc biệt khi câu trả lời phụ thuộc vào mối quan hệ giữa nhiều thực thể hoặc nhiều đoạn tài liệu.  
* **Bài báo:** [https\://aclanthology.org/2025.findings-emnlp.568/](https://aclanthology.org/2025.findings-emnlp.568/)  
* **Mã nguồn:** [https\://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

#### **3.3. Hướng 3: Truy hồi có xét đến cấu trúc và điều kiện của câu hỏi**

**Paper 3: Structure Guided Retrieval-Augmented Generation for Factual Queries (ACL 2026).**

* **Ý tưởng:** SG-RAG mô hình hóa quá trình truy hồi như một bài toán khớp cấu trúc đồ thị, sử dụng các quan hệ và cấu trúc được truy hồi để hướng dẫn mô hình ngôn ngữ tạo câu trả lời đáp ứng các điều kiện của câu hỏi.  
* **Lý do lựa chọn:** Phương pháp đại diện cho hướng nghiên cứu mới về truy hồi bằng chứng cho các câu hỏi thực tế có nhiều điều kiện, giúp nhóm nghiên cứu sâu hơn về độ chính xác và tính đầy đủ của thông tin được truy hồi.  
* **Bài báo:** [https\://aclanthology.org/2026.acl-long.1873/](https://aclanthology.org/2026.acl-long.1873/)  
* **Mã nguồn:** [https\://github.com/CAU-X-AI-Lab/SG-RAG](https://github.com/CAU-X-AI-Lab/SG-RAG)

Các paper trên đại diện cho ba cách tiếp cận khác nhau trong việc tổ chức và truy hồi tri thức. Nhóm sẽ kiểm tra khả năng triển khai, cấu hình thực nghiệm và yêu cầu tài nguyên trước khi chốt phạm vi tái lập.

### **4\. Mã nguồn và phương án triển khai**

Nhóm dự kiến sử dụng mã nguồn công khai của các tác giả làm cơ sở tái lập, đồng thời tự xây dựng thành phần đánh giá và giao diện chatbot để bảo đảm có phần triển khai riêng.

Các thành phần chính của hệ thống gồm:

1. **Tiền xử lý dữ liệu:** Tải dữ liệu, chuẩn hóa văn bản, xây dựng kho tài liệu và chuẩn bị các câu hỏi cùng đáp án tham chiếu.  
2. **Hệ thống truy hồi:** Cài đặt baseline BM25 hoặc RAG thông thường, sau đó tích hợp các phương pháp được lựa chọn.  
3. **Sinh câu trả lời:** Sử dụng cùng một mô hình sinh và cấu hình prompt trong các thí nghiệm so sánh để hạn chế ảnh hưởng từ những yếu tố ngoài phương pháp truy hồi.  
4. **Đánh giá:** Xây dựng quy trình chạy thực nghiệm thống nhất, lưu kết quả, thời gian xử lý và cấu hình phần cứng.  
5. **Demo:** Phát triển giao diện hỏi đáp bằng Streamlit hoặc Gradio, hiển thị câu trả lời cùng các đoạn tài liệu tham khảo.

Nhóm ưu tiên sử dụng checkpoint và mô hình có sẵn khi được phép, thay vì huấn luyện lại các mô hình lớn từ đầu. Nếu tài nguyên hạn chế, nhóm sẽ thu nhỏ tập dữ liệu hoặc kho tài liệu theo phương án lấy mẫu rõ ràng và ghi nhận đầy đủ các thay đổi so với thiết lập gốc.

### **5\. Bộ dữ liệu dự kiến**

Nhóm dự kiến khảo sát và lựa chọn các bộ dữ liệu hỏi đáp công khai sau:

* **HotpotQA:** Bộ dữ liệu hỏi đáp yêu cầu suy luận nhiều bước, trong đó một số câu hỏi cần kết hợp thông tin từ nhiều tài liệu.  
* **Natural Questions (NQ):** Bộ dữ liệu câu hỏi thực tế và câu trả lời dựa trên nguồn tri thức, phù hợp để nghiên cứu hệ thống hỏi đáp miền mở.  
* **2WikiMultiHopQA:** Bộ dữ liệu hỏi đáp nhiều bước, chú trọng việc kết hợp thông tin và quan hệ giữa các thực thể.  
* **ERQA:** Bộ dữ liệu được giới thiệu cùng SG-RAG, gồm các câu hỏi thực tế có nhiều điều kiện và được sử dụng để đánh giá phương pháp này.

Đường dẫn tham khảo:

* HotpotQA: [https\://hotpotqa.github.io/](https://hotpotqa.github.io/)  
* Natural Questions: [https\://github.com/google-research-datasets/natural-questions](https://github.com/google-research-datasets/natural-questions)  
* 2WikiMultiHopQA: [https\://github.com/Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)  
* ERQA và mã nguồn SG-RAG: [https\://github.com/CAU-X-AI-Lab/SG-RAG](https://github.com/CAU-X-AI-Lab/SG-RAG)

Nhóm sẽ ưu tiên các bộ dữ liệu được sử dụng trong paper gốc để tái lập kết quả. Đối với phần so sánh chung, nhóm sẽ lựa chọn một benchmark phù hợp với các phương pháp được triển khai, thống nhất cách chia dữ liệu, kho tài liệu, số lượng tài liệu truy hồi và các độ đo. Nếu không thể sử dụng cùng một bộ dữ liệu cho cả ba phương pháp, nhóm sẽ trình bày rõ phạm vi và giới hạn của phép so sánh.

### **6\. Phương pháp đánh giá và nội dung thực nghiệm**

Nhóm dự kiến thực hiện các nội dung sau:

* Tái lập kết quả của từng phương pháp trên bộ dữ liệu và thiết lập được paper công bố, hoặc trên tập con nếu tài nguyên tính toán không đáp ứng.  
* So sánh với baseline truyền thống như BM25 và hệ thống RAG thông thường.  
* Đánh giá chất lượng truy hồi bằng các độ đo phù hợp như Recall@k, MRR hoặc nDCG.  
* Đánh giá chất lượng câu trả lời bằng Exact Match (EM), token-level F1 và phân tích định tính khi cần thiết.  
* Ghi nhận thời gian lập chỉ mục, độ trễ truy hồi, độ trễ sinh câu trả lời, số tham số mô hình và cấu hình phần cứng sử dụng.  
* Phân tích các trường hợp trả lời sai, bao gồm truy hồi thiếu bằng chứng, truy hồi nhầm tài liệu, bỏ sót điều kiện của câu hỏi và sinh thông tin không được tài liệu hỗ trợ.

Kết quả tái lập sẽ được trình bày riêng với kết quả so sánh trên thiết lập chung, nhằm phân biệt sự khác biệt do phương pháp với sự khác biệt do dữ liệu, mô hình hoặc cấu hình thực nghiệm.

### **7\. Dự kiến kết quả và phần đóng góp của nhóm**

Sau khi hoàn thành, nhóm dự kiến đạt được các kết quả sau:

* Tái lập được một số phương pháp truy hồi tri thức hiện đại trong hệ thống RAG.  
* Xây dựng bảng so sánh thực nghiệm về chất lượng truy hồi, chất lượng câu trả lời và chi phí tính toán.  
* Phân tích ưu điểm, hạn chế và các dạng lỗi thường gặp của từng phương pháp.  
* Xây dựng một chatbot hỏi đáp dựa trên tri thức, có khả năng cung cấp câu trả lời kèm nguồn tài liệu tham khảo.  
* Thử nghiệm một cải tiến nhỏ, chẳng hạn kết hợp BM25 với truy hồi ngữ nghĩa hoặc điều chỉnh cách lựa chọn các đoạn tài liệu đưa vào mô hình sinh.

**Phần nhóm tự thực hiện** dự kiến bao gồm quy trình đánh giá thống nhất, phần tích hợp các phương pháp, phân tích lỗi, thực nghiệm cải tiến và giao diện demo. Các thành phần kế thừa từ mã nguồn của tác giả sẽ được ghi rõ trong báo cáo.

### **8\. Tài liệu tham khảo**

1. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. International Conference on Learning Representations (ICLR 2024).  
2. Guo, Z., Xia, L., Yu, Y., Ao, T., & Huang, C. (2025). *LightRAG: Simple and Fast Retrieval-Augmented Generation*. Findings of the Association for Computational Linguistics: EMNLP 2025\.  
3. Xie, M., Zhang, X., Li, Y., & Lv, C. (2026). *Structure Guided Retrieval-Augmented Generation for Factual Queries*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026).

