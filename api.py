import tempfile
from fastapi import FastAPI, UploadFile
from main import build_vectorstore, answer_question
 
app = FastAPI(title="RAG Document Analyzer API")
 
 
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")

def upload(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    session_id = build_vectorstore(tmp_path)

    return {"session_id": session_id, "message": "Document processed successfully"}


@app.post("/ask")
def ask(session_id: str, question: str):

    answer = answer_question(question, session_id)
    return {"answer": answer}
