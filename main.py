from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path
import pypdf
import uuid

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import time

from pydantic import BaseModel
from typing import Optional 

import requests 
# Database setup
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, unique=True)
    rag_hardware_id = Column(String)  # Where RAG service runs (embedding/vector search)
    llm_hardware_id = Column(String)  # Where LLM inference runs
    model_name = Column(String)  # Which LLM model (e.g., llama3:8b, gpt-4o-mini)
    timestamp = Column(DateTime, default=datetime.utcnow)
    question = Column(Text)
    answer = Column(Text)
    num_chunks_retrieved = Column(Integer)
    avg_similarity_score = Column(Float)
    embedding_time_ms = Column(Float)
    vector_search_time_ms = Column(Float)
    llm_time_ms = Column(Float)
    total_time_ms = Column(Float)
    estimated_cost_usd = Column(Float)
    success = Column(Boolean)
    error_message = Column(Text, nullable=True)

class UploadLog(Base):
    __tablename__ = 'upload_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_id = Column(String, unique=True)
    rag_hardware_id = Column(String)  # Where RAG service runs (document processing)
    timestamp = Column(DateTime, default=datetime.utcnow)
    filename = Column(String)
    num_chunks = Column(Integer)
    file_size_bytes = Column(Integer)
    text_extraction_time_ms = Column(Float)
    chunking_time_ms = Column(Float)
    embedding_time_ms = Column(Float)
    vector_store_time_ms = Column(Float)
    total_time_ms = Column(Float)
    success = Column(Boolean)
    error_message = Column(Text, nullable=True)

# Logging API request models
class QueryLogRequest(BaseModel):
    query_id: str
    rag_hardware_id: str
    llm_hardware_id: str
    model_name: str
    question: str
    answer: str
    num_chunks_retrieved: int
    avg_similarity_score: float
    embedding_time_ms: float
    vector_search_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    estimated_cost_usd: float
    success: bool
    error_message: Optional[str] = None

class UploadLogRequest(BaseModel):
    upload_id: str
    rag_hardware_id: str
    filename: str
    num_chunks: int
    file_size_bytes: int
    text_extraction_time_ms: float
    chunking_time_ms: float
    embedding_time_ms: float
    vector_store_time_ms: float
    total_time_ms: float
    success: bool
    error_message: Optional[str] = None

# Create database (persisted in PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db/rag_logs.db")  # Fallback to SQLite for local dev
RAG_HARDWARE_ID = os.getenv("RAG_HARDWARE_ID", "local-dev")
LLM_HARDWARE_ID = os.getenv("LLM_HARDWARE_ID", "local-dev")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOG_SERVER_URL = os.getenv("LOG_SERVER_URL", None)

engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI()

# CORS - allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://adamlgent.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant = QdrantClient(host="qdrant", port=6333)
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLLECTION_NAME = "documents"


def init_collection():
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def send_log_to_server(endpoint: str, log_data: dict):
    """Send log data to remote logging server"""
    if not LOG_SERVER_URL:
        return False  # No server configured
    
    try:
        response = requests.post(
            f"{LOG_SERVER_URL}{endpoint}",
            json=log_data,
            timeout=5
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send log to server: {e}")
        return False

init_collection()

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Helper functions
def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 200):
    """Simple chunking by character count with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_file(file_path: str):
    """Extract text from PDF or text file"""
    if file_path.endswith('.pdf'):
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Endpoints
@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and add it to the vector store"""
    upload_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Save temporarily and get file size
        file_path = f"/tmp/{file.filename}"
        content = await file.read()
        file_size_bytes = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text
        extract_start = time.time()
        text = extract_text_from_file(file_path)
        text_extraction_time_ms = (time.time() - extract_start) * 1000
        
        # Chunk it
        chunk_start = time.time()
        chunks = chunk_text(text)
        chunking_time_ms = (time.time() - chunk_start) * 1000
        
        # Embed and store
        embed_start = time.time()
        points = []
        for i, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file.filename,
                    "chunk_id": i
                }
            )
            points.append(point)
        embedding_time_ms = (time.time() - embed_start) * 1000
        
        # Store in vector DB
        store_start = time.time()
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        vector_store_time_ms = (time.time() - store_start) * 1000
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Log to server
        log_data = {
            "upload_id": upload_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "file_size_bytes": file_size_bytes,
            "text_extraction_time_ms": text_extraction_time_ms,
            "chunking_time_ms": chunking_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "vector_store_time_ms": vector_store_time_ms,
            "total_time_ms": total_time_ms,
            "success": True,
            "error_message": None
        }
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - upload not logged!")
        elif not send_log_to_server("/api/log-upload", log_data):
            print(f"ERROR: Failed to send upload log to {LOG_SERVER_URL}")
        
        return {
            "message": f"Uploaded {file.filename} with {len(chunks)} chunks",
            "upload_id": upload_id,
            "timing": {
                "text_extraction_ms": round(text_extraction_time_ms, 2),
                "chunking_ms": round(chunking_time_ms, 2),
                "embedding_ms": round(embedding_time_ms, 2),
                "vector_store_ms": round(vector_store_time_ms, 2),
                "total_ms": round(total_time_ms, 2)
            }
        }
        
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        log_data = {
            "upload_id": upload_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "filename": file.filename if file.filename else "unknown",
            "num_chunks": 0,
            "file_size_bytes": 0,
            "text_extraction_time_ms": 0.0,
            "chunking_time_ms": 0.0,
            "embedding_time_ms": 0.0,
            "vector_store_time_ms": 0.0,
            "total_time_ms": total_time_ms,
            "success": False,
            "error_message": str(e)
        }
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - error not logged!")
        elif not send_log_to_server("/api/log-upload", log_data):
            print(f"ERROR: Failed to send error log to {LOG_SERVER_URL}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-documents")
def load_documents():
    """Load all PDFs and text files from data/ directory into Qdrant"""
    data_dir = Path("data")
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    if not files:
        raise HTTPException(status_code=404, detail="No PDF or TXT files found in data/ directory")
    
    points = []
    for file in files:
        # Extract text
        text = extract_text_from_file(str(file))
        
        # Chunk it
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file.name,
                    "chunk_id": i
                }
            )
            points.append(point)
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {
        "message": f"Loaded {len(files)} documents with {len(points)} chunks",
        "files": [f.name for f in files]
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the RAG system"""
    
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 1. Embed the query
        embed_start = time.time()
        query_vector = embed_model.encode(request.question).tolist()
        embedding_time_ms = (time.time() - embed_start) * 1000
        
        # 2. Search Qdrant
        search_start = time.time()
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        vector_search_time_ms = (time.time() - search_start) * 1000
        
        if not search_results:
            # Log failed query
            log_data = {
                "query_id": query_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "llm_hardware_id": LLM_HARDWARE_ID,
                "model_name": MODEL_NAME,
                "question": request.question,
                "answer": "",
                "num_chunks_retrieved": 0,
                "avg_similarity_score": 0.0,
                "embedding_time_ms": embedding_time_ms,
                "vector_search_time_ms": vector_search_time_ms,
                "llm_time_ms": 0.0,
                "total_time_ms": (time.time() - start_time) * 1000,
                "estimated_cost_usd": 0.0,
                "success": False,
                "error_message": "No relevant documents found"
            }
            
            if not LOG_SERVER_URL:
                print("WARNING: LOG_SERVER_URL not configured - failed query not logged!")
            elif not send_log_to_server("/api/log-query", log_data):
                print(f"ERROR: Failed to send log to {LOG_SERVER_URL}")
            
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # 3. Build context from retrieved chunks
        context = "\n\n".join([
            f"Source: {hit.payload['source']}\n{hit.payload['text']}"
            for hit in search_results
        ])
        
        # Calculate avg similarity score
        avg_similarity_score = sum(hit.score for hit in search_results) / len(search_results)
        
        # 4. Call OpenAI
        llm_start = time.time()
        prompt = f"""Answer the following question based only on the provided context.

        Context:
        {context}

        Question: {request.question}

        Answer:"""
        
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        llm_time_ms = (time.time() - llm_start) * 1000
        
        answer = response.choices[0].message.content
        
        # Calculate cost estimate (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        estimated_cost_usd = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        log_data = {
            "query_id": query_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "llm_hardware_id": LLM_HARDWARE_ID,
            "model_name": MODEL_NAME,
            "question": request.question,
            "answer": answer,
            "num_chunks_retrieved": len(search_results),
            "avg_similarity_score": avg_similarity_score,
            "embedding_time_ms": embedding_time_ms,
            "vector_search_time_ms": vector_search_time_ms,
            "llm_time_ms": llm_time_ms,
            "total_time_ms": total_time_ms,
            "estimated_cost_usd": estimated_cost_usd,
            "success": True,
            "error_message": None
        }
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - query not logged!")
        elif not send_log_to_server("/api/log-query", log_data):
            print(f"ERROR: Failed to send log to {LOG_SERVER_URL}")
        
        # 6. Format sources
        sources = [
            {
                "text": hit.payload['text'],
                "source": hit.payload['source'],
                "score": hit.score
            }
            for hit in search_results
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
            # Log unexpected errors
            total_time_ms = (time.time() - start_time) * 1000
            log_data = {
                "query_id": query_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "llm_hardware_id": LLM_HARDWARE_ID,
                "model_name": MODEL_NAME,
                "question": request.question,
                "answer": "",
                "num_chunks_retrieved": 0,
                "avg_similarity_score": 0.0,
                "embedding_time_ms": 0.0,
                "vector_search_time_ms": 0.0,
                "llm_time_ms": 0.0,
                "total_time_ms": total_time_ms,
                "estimated_cost_usd": 0.0,
                "success": False,
                "error_message": str(e)
            }
            
            if not LOG_SERVER_URL:
                print("WARNING: LOG_SERVER_URL not configured - error not logged!")
            elif not send_log_to_server("/api/log-query", log_data):
                print(f"ERROR: Failed to send error log to {LOG_SERVER_URL}")
            
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats(group_by: str = "rag_hardware"):
    """
    Get aggregated performance statistics
    
    group_by options:
    - "rag_hardware": Group by RAG hardware (embedding/vector search performance)
    - "llm_hardware": Group by LLM hardware (inference performance)
    - "model": Group by model name
    - "rag_llm_model": Group by RAG hardware + LLM hardware + model (all combinations)
    """
    db_session = SessionLocal()
    
    try:
        # Get all successful queries
        query_logs = db_session.query(QueryLog).filter(QueryLog.success == True).all()
        
        if not query_logs:
            return {
                "group_by": group_by,
                "groups": [],
                "message": "No successful queries logged yet"
            }
        
        # Group queries based on the group_by parameter
        from collections import defaultdict
        groups = defaultdict(list)
        
        for q in query_logs:
            if group_by == "rag_hardware":
                key = q.rag_hardware_id or "unknown"
            elif group_by == "llm_hardware":
                key = q.llm_hardware_id or "unknown"
            elif group_by == "model":
                key = q.model_name or "unknown"
            elif group_by == "rag_llm_model":
                key = f"{q.rag_hardware_id or 'unknown'}|{q.llm_hardware_id or 'unknown'}|{q.model_name or 'unknown'}"
            else:
                key = "all"
            
            groups[key].append(q)
        
        # Calculate stats for each group
        grouped_stats = []
        for key, queries in groups.items():
            # Parse the key back to components for rag_llm_model view
            if group_by == "rag_llm_model":
                rag_hw, llm_hw, model = key.split("|")
                group_label = {
                    "rag_hardware_id": rag_hw,
                    "llm_hardware_id": llm_hw,
                    "model_name": model
                }
            else:
                group_label = key
            
            stats = {
                "group": group_label,
                "total_queries": len(queries),
                "avg_embedding_time_ms": round(sum(q.embedding_time_ms for q in queries) / len(queries), 2),
                "avg_vector_search_time_ms": round(sum(q.vector_search_time_ms for q in queries) / len(queries), 2),
                "avg_llm_time_ms": round(sum(q.llm_time_ms for q in queries) / len(queries), 2),
                "avg_total_time_ms": round(sum(q.total_time_ms for q in queries) / len(queries), 2),
                "total_cost_usd": round(sum(q.estimated_cost_usd for q in queries), 6),
                "avg_chunks_retrieved": round(sum(q.num_chunks_retrieved for q in queries) / len(queries), 2),
                "avg_similarity_score": round(sum(q.avg_similarity_score for q in queries) / len(queries), 4)
            }
            grouped_stats.append(stats)
        
        # Sort by group name
        grouped_stats.sort(key=lambda x: str(x["group"]))
        
        # Upload stats (not grouped, just overall)
        upload_logs = db_session.query(UploadLog).filter(UploadLog.success == True).all()
        
        if not upload_logs:
            upload_stats = {
                "total_uploads": 0,
                "message": "No successful uploads logged yet"
            }
        else:
            upload_stats = {
                "total_uploads": len(upload_logs),
                "total_chunks_created": sum(u.num_chunks for u in upload_logs),
                "avg_text_extraction_time_ms": round(sum(u.text_extraction_time_ms for u in upload_logs) / len(upload_logs), 2),
                "avg_chunking_time_ms": round(sum(u.chunking_time_ms for u in upload_logs) / len(upload_logs), 2),
                "avg_embedding_time_ms": round(sum(u.embedding_time_ms for u in upload_logs) / len(upload_logs), 2),
                "avg_vector_store_time_ms": round(sum(u.vector_store_time_ms for u in upload_logs) / len(upload_logs), 2),
                "avg_total_time_ms": round(sum(u.total_time_ms for u in upload_logs) / len(upload_logs), 2)
            }
        
        # Recent queries (last 10, ungrouped)
        recent_queries = db_session.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(10).all()
        recent_query_data = [
            {
                "query_id": q.query_id,
                "rag_hardware_id": q.rag_hardware_id,
                "llm_hardware_id": q.llm_hardware_id,
                "model_name": q.model_name,
                "timestamp": q.timestamp.isoformat(),
                "question": q.question[:100] + "..." if len(q.question) > 100 else q.question,
                "success": q.success,
                "total_time_ms": round(q.total_time_ms, 2),
                "embedding_time_ms": round(q.embedding_time_ms, 2),
                "vector_search_time_ms": round(q.vector_search_time_ms, 2),
                "llm_time_ms": round(q.llm_time_ms, 2)
            }
            for q in recent_queries
        ]
        
        return {
            "group_by": group_by,
            "grouped_stats": grouped_stats,
            "upload_stats": upload_stats,
            "recent_queries": recent_query_data
        }
        
    finally:
        db_session.close()

@app.get("/stats-html")
def stats_html():
    """Serve the stats dashboard HTML page"""
    from fastapi.responses import FileResponse
    return FileResponse('stats.html')

@app.get("/documents")
def list_documents():
    """List what's in the vector store"""
    result = qdrant.count(collection_name=COLLECTION_NAME)
    return {
        "total_chunks": result.count,
        "collection": COLLECTION_NAME
    }


@app.post("/api/log-query")
def log_query_api(log_data: QueryLogRequest):
    """Receive query log from remote RAG client"""
    db_session = SessionLocal()
    
    try:
        # Create database entry
        log_entry = QueryLog(
            query_id=log_data.query_id,
            rag_hardware_id=log_data.rag_hardware_id,
            llm_hardware_id=log_data.llm_hardware_id,
            model_name=log_data.model_name,
            question=log_data.question,
            answer=log_data.answer,
            num_chunks_retrieved=log_data.num_chunks_retrieved,
            avg_similarity_score=log_data.avg_similarity_score,
            embedding_time_ms=log_data.embedding_time_ms,
            vector_search_time_ms=log_data.vector_search_time_ms,
            llm_time_ms=log_data.llm_time_ms,
            total_time_ms=log_data.total_time_ms,
            estimated_cost_usd=log_data.estimated_cost_usd,
            success=log_data.success,
            error_message=log_data.error_message
        )
        
        db_session.add(log_entry)
        db_session.commit()
        
        return {"status": "success", "query_id": log_data.query_id}
        
    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()


@app.post("/api/log-upload")
def log_upload_api(log_data: UploadLogRequest):
    """Receive upload log from remote RAG client"""
    db_session = SessionLocal()
    
    try:
        # Create database entry
        log_entry = UploadLog(
            upload_id=log_data.upload_id,
            rag_hardware_id=log_data.rag_hardware_id,
            filename=log_data.filename,
            num_chunks=log_data.num_chunks,
            file_size_bytes=log_data.file_size_bytes,
            text_extraction_time_ms=log_data.text_extraction_time_ms,
            chunking_time_ms=log_data.chunking_time_ms,
            embedding_time_ms=log_data.embedding_time_ms,
            vector_store_time_ms=log_data.vector_store_time_ms,
            total_time_ms=log_data.total_time_ms,
            success=log_data.success,
            error_message=log_data.error_message
        )
        
        db_session.add(log_entry)
        db_session.commit()
        
        return {"status": "success", "upload_id": log_data.upload_id}
        
    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()