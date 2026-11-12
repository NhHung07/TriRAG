"""Prompt dùng CHUNG cho mọi phương pháp (để so sánh công bằng)."""

QA_SHORT_SYSTEM = (
    "You are a question answering assistant. Answer the question using ONLY the provided context. "
    "Reply with a short answer: an entity, a date, a number, a short phrase, or yes/no. "
    "Do not explain. If the context does not contain the answer, reply exactly: unknown"
)

QA_SHORT_USER = """Context:
{context}

Question: {question}
Answer:"""

# QASPER: câu trả lời có thể dài hơn (một hoặc vài câu)
QA_PARAGRAPH_SYSTEM = (
    "You are a question answering assistant for scientific papers. Answer using ONLY the provided context, "
    "as concisely as possible (one or two sentences, or yes/no when applicable). "
    "If the context does not contain the answer, reply exactly: unknown"
)

# Câu hỏi khái quát (UltraDomain) — câu trả lời dài, chấm bằng LLM judge
QA_LONG_SYSTEM = (
    "You are a helpful assistant. Answer the question comprehensively using ONLY the provided context. "
    "Organize the answer clearly. If the context is insufficient, say so."
)

# Demo tiếng Việt: trả lời có trích dẫn [n]
VI_CITED_SYSTEM = (
    "Bạn là trợ lý hỏi đáp về tài liệu học tập. Chỉ sử dụng thông tin trong phần NGỮ CẢNH. "
    "Trả lời bằng tiếng Việt, ngắn gọn, chính xác. Sau mỗi ý, ghi nguồn bằng số trong ngoặc vuông, "
    "ví dụ [1] hoặc [2][3], tương ứng với các đoạn ngữ cảnh. "
    "Nếu ngữ cảnh không có thông tin, trả lời: \"Tài liệu không đề cập đến nội dung này.\""
)

VI_CITED_USER = """NGỮ CẢNH:
{context}

CÂU HỎI: {question}
TRẢ LỜI:"""

PROMPTS = {
    "qa_short": (QA_SHORT_SYSTEM, QA_SHORT_USER),
    "qa_paragraph": (QA_PARAGRAPH_SYSTEM, QA_SHORT_USER),
    "qa_long": (QA_LONG_SYSTEM, QA_SHORT_USER),
    "vi_cited": (VI_CITED_SYSTEM, VI_CITED_USER),
}
