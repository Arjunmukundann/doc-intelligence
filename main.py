import os, uuid, shutil
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.database import get_db, Document
from rag.pipeline import run_rag_pipeline
from rag.query_router import classify_query
from rag.summarizer import map_reduce_summarize


from ingestion import ingest_document
from vector_store.vector_store import (
    create_embeddings,
    create_faiss_index,
    search_similar_chunks
    
)
from vector_store.manager import (save_index,
    delete_index, get_all_chunks, search_all_documents)
from pipeline import generate_response

app = FastAPI(title="doc intelligence api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

doc_status = {}

def embed_document(doc_id: str, file_path: str):
    try:
        doc_status[doc_id] = "processing"
        chunks = ingest_document(file_path)
        embeddings = create_embeddings(chunks)
        index = create_faiss_index(embeddings)
        save_index(doc_id, index, chunks)
        doc_status[doc_id] = "completed"
        print(f"[background] Doc {doc_id} processed successfully.")
    except Exception as e:
        doc_status[doc_id] = "failed"
        print(f"[background] Error processing doc {doc_id}: {str(e)}")

@app.get("/")
def home():
    return {"message": "Doc Intelligence API Running"}

@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = Document(id=doc_id, filename=file.filename, file_path=file_path)
    db.add(doc)
    db.commit()
    doc_status[doc_id] = "processing"
    background_tasks.add_task(embed_document, doc_id, file_path)
    return {"doc_id": doc_id, "status": "processing"}

class QueryRequest(BaseModel):
    doc_id: str
    question: str
    session_id:str="default"

@app.post('/query')
async def query_document(req: QueryRequest,db:Session=Depends(get_db)):

    if req.doc_id not in doc_status:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc_status[req.doc_id] == 'processing':
        raise HTTPException(status_code=202, detail="Document is still processing")
    if doc_status[req.doc_id] == 'failed':
        raise HTTPException(status_code=500, detail='Document processing failed')
    
    query_type=classify_query(req.question)
    if query_type=="summarize":
        all_chunks=get_all_chunks(req.doc_id)
        summary=map_reduce_summarize(all_chunks)
        return {"answer":summary,"type":"summary","sources":[]}
    else:

        result = run_rag_pipeline( question=req.question,doc_id=req.doc_id,session_id=req.session_id)

    return result
    
class GlobalQueryRequest(BaseModel):
    question:str
    top_k: int =10
    session_id:str="default"
    

@app.post("/query/global")
async def query_all_documents(req: GlobalQueryRequest):
    raw_results=search_all_documents(req.question,k=10)

    return raw_results


@app.get("/documents")
def list_documents():
    return [
        {
            "doc_id": doc_id,
            "status": status
        }
        for doc_id, status in doc_status.items()
    ]

@app.get("/documents/{doc_id}")
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc.id, "status": doc.status}
@app.delete("/delete")
def delete_documents(doc_id: str,db: Session=Depends(get_db)):
    if doc_id not in doc_status:
        raise HTTPException(status_code=404, detail="document not found")
    delete_index(doc_id)
    del doc_status[doc_id]
    return {"message": f"Document {doc_id} deleted"}
