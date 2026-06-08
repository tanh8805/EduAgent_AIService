from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage
from pydantic import Field, BaseModel
from workflow import graph

app = FastAPI(title="EduAgent Chat API Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: int = Field(..., description="ID của người dùng đang thực hiện chat")
    document_id: int = Field(..., description="ID của tài liệu giáo trình cần tra cứu ngữ cảnh")
    prompt: str = Field(..., description="Câu hỏi chữ thô từ phía người dùng gửi lên")

@app.post("/api/agent/chat")
async def agent_chat(payload: ChatRequest):
    try:
        user_id = int(payload.user_id)
        document_id = int(payload.document_id)
        prompt = payload.prompt

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Nội dung prompt không được để trống.")
        thread_id = f"user_{user_id}_doc_{document_id}"
        config = {"configurable": {"thread_id": thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "document_id": document_id
        }

        final_state = graph.invoke(inputs, config=config)

        last_message = final_state["messages"][-1]
        current_summary = final_state.get("summary", "")
        return {
            "status": "success",
            "thread_id": thread_id,
            "response": last_message.content,
            "summary_status": "Đã cập nhật" if current_summary else "Trống"
        }
    except Exception as e:
        print(f"Error executing LangGraph agent chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống trong quá trình Agent xử lý: {str(e)}")
