import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.params import Form

from function.read_file import read_pdf
from function.chunk_text import chunk_text
from function.embedding import embedding
from function.database import store_database, create_document

load_dotenv()

app = FastAPI(title="EduAgent AI Service - Python RAG")

POSTGRES_URI=os.getenv("POSTGRES_URI")

assert POSTGRES_URI is not None

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), user_id: int = Form(...)):
    raw_text = read_pdf(file.file)

    if not raw_text:
        return {
            "status": "error",
            "message": "Cannot extract text from PDF"
        }

    chunks = chunk_text(raw_text)
    if not chunks:
        return {
            "status": "error",
            "message": "No chunks generated from the text"
        }

    embeddings = []
    for chunk in chunks:
        vector = embedding(chunk)
        if vector is None:
            return {
                "status": "error",
                "message": "Cannot extract vector from chunk"
            }
        embeddings.append(vector)
    doc_id = create_document(POSTGRES_URI,user_id, file.filename)
    if doc_id is None:
        return {
            "status": "error",
            "message": "Cannot store document to database"
        }

    if store_database(POSTGRES_URI, doc_id, chunks, embeddings):
        return {
            "status": "success",
            "message": "Successfully stored document to database"
        }
    else:
        return {
            "status": "error",
            "message": "System Error"
        }