import os
from langchain_core.tools import tool
from function.embedding import embedding
from function.database import hybrid_search

POSTGRES_URI = os.getenv("POSTGRES_URI")


@tool
def search_document_knowledge(question: str, document_id: int) -> str:
    """
    Tìm kiếm các đoạn văn bản, kiến thức liên quan trực tiếp trong một tài liệu cụ thể dựa trên ID tài liệu.
    Sử dụng công cụ này khi người dùng đặt câu hỏi tra cứu thông tin, định nghĩa, số liệu
    hoặc yêu cầu giải thích các vấn đề nằm trong file tài liệu/giáo trình đã upload.
    """

    query_vector = embedding(question)
    if query_vector is None:
        return "Error: Cannot extract vector for this question."

    matched_chunks = hybrid_search(
        postgres_uri=POSTGRES_URI,
        document_id=document_id,
        query_text=question,
        query_vector=query_vector,
        top_k=3
    )

    if not matched_chunks:
        return "Cannot find any matches for this question in documents."

    context_str = "\n---\n".join([f"[Chunk {c['chunk_index']}]: {c['content']}" for c in matched_chunks])
    return context_str