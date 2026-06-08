import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.params import Form
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from pydantic import Field, BaseModel
from workflow import graph

from function.read_file import read_pdf
from function.chunk_text import chunk_text
from function.embedding import embedding
from function.database import store_database, create_document

load_dotenv()

app = FastAPI(title="EduAgent AI Integrated Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POSTGRES_URI = os.getenv("POSTGRES_URI")
assert POSTGRES_URI is not None


class ChatRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    session_id: int = Field(..., description="Session ID")
    prompt: str = Field(..., description="User text prompt")


@app.post("/api/upload")
async def upload(
        file: UploadFile = File(...),
        session_id: int = Form(...),
        user_id: int = Form(...)
):
    raw_text = read_pdf(file.file)
    if not raw_text:
        return {"status": "error", "message": "Cannot extract text from PDF"}

    chunks = chunk_text(raw_text)
    if not chunks:
        return {"status": "error", "message": "No chunks generated from the text"}

    embeddings = []
    for chunk in chunks:
        vector = embedding(chunk)
        if vector is None:
            return {"status": "error", "message": "Cannot extract vector from chunk"}
        embeddings.append(vector)

    doc_id = create_document(POSTGRES_URI, session_id, user_id, file.filename)
    if doc_id is None:
        return {"status": "error", "message": "Cannot store document to database"}

    if store_database(POSTGRES_URI, doc_id, chunks, embeddings):
        return {"status": "success", "message": "Successfully stored document to database"}
    else:
        return {"status": "error", "message": "System Error"}


@app.post("/api/agent/chat")
async def agent_chat(payload: ChatRequest):
    try:
        user_id = int(payload.user_id)
        session_id = int(payload.session_id)
        prompt = payload.prompt

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt content cannot be empty")

        thread_id = f"user_{user_id}_session_{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": session_id
        }

        final_state = graph.invoke(inputs, config=config)
        last_message = final_state["messages"][-1]
        current_summary = final_state.get("summary", "")

        return {
            "status": "success",
            "thread_id": thread_id,
            "response": last_message.content,
            "summary_status": "Updated" if current_summary else "Empty"
        }
    except Exception as e:
        print(f"Error executing LangGraph agent chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"System error during agent processing: {str(e)}")