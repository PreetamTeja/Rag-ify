from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import List, Dict, Any
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Document processing
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector database
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Document Q&A API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
embedding_model = None
llm_model = None
llm_tokenizer = None
pc_client = None
executor = ThreadPoolExecutor(max_workers=4)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-agents"
EMBEDDING_DIM = 384

class QueryRequest(BaseModel):
    question: str
    agent_name: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    agent_name: str

@app.on_event("startup")
async def startup_event():
    global embedding_model, llm_model, llm_tokenizer, pc_client
    
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(
        'BAAI/bge-small-en-v1.5',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    
    logger.info("Loading LLM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    
    logger.info("Initializing Pinecone...")
    pc_client = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    try:
        index_info = pc_client.describe_index(INDEX_NAME)
        logger.info(f"Index {INDEX_NAME} already exists")
    except:
        logger.info(f"Creating index {INDEX_NAME}")
        pc_client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for text chunks"""
    return embedding_model.encode(texts, normalize_embeddings=True)

def generate_answer(question: str, context: str) -> str:
    """Generate answer using LLM"""
    prompt = f"""You are a helpful assistant. Based on the following context, please answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    if torch.cuda.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id
        )
    
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    agent_name: str = Form(...)
):
    """Upload and process a document for a specific agent"""
    
    if not agent_name.strip():
        raise HTTPException(status_code=400, detail="Agent name is required")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(temp_file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(temp_file_path)
        elif file_extension == '.txt':
            text = extract_text_from_txt(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the document")
        
        # Split into chunks
        chunks = split_text_into_chunks(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Store in Pinecone
        index = pc_client.Index(INDEX_NAME)
        
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{agent_name}_{file.filename}_{i}_{uuid.uuid4().hex[:8]}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "filename": file.filename,
                    "chunk_index": i,
                    "agent_name": agent_name
                }
            })
        
        # Upsert vectors to Pinecone with namespace
        index.upsert(
            vectors=vectors_to_upsert,
            namespace=agent_name
        )
        
        logger.info(f"Successfully uploaded {file.filename} for agent {agent_name}")
        
        return {
            "message": f"Successfully uploaded {file.filename}",
            "agent_name": agent_name,
            "chunks_processed": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query a specific agent with a question"""
    
    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode([request.question], normalize_embeddings=True)[0]
        
        # Search in Pinecone namespace for the specific agent
        index = pc_client.Index(INDEX_NAME)
        
        search_results = index.query(
            vector=question_embedding.tolist(),
            top_k=5,
            include_metadata=True,
            namespace=request.agent_name
        )
        
        if not search_results.matches:
            return QueryResponse(
                answer="I cannot find any relevant information in the documents for this agent.",
                sources=[],
                agent_name=request.agent_name
            )
        
        # Prepare context from retrieved chunks
        context_chunks = []
        sources = set()

        for match in search_results.matches:
            if match.score > 0.7:
                context_chunks.append(match.metadata['text'])
                sources.add(match.metadata['filename'])

        # fallback: take best match if nothing passed threshold
        if not context_chunks and search_results.matches:
            best_match = search_results.matches[0]
            context_chunks.append(best_match.metadata['text'])
            sources.add(best_match.metadata['filename'])

        
        if not context_chunks:
            return QueryResponse(
                answer="I cannot find sufficiently relevant information to answer your question.",
                sources=[],
                agent_name=request.agent_name
            )
        
        context = "\n\n".join(context_chunks)
        
        # Generate answer using LLM
        answer = await asyncio.get_event_loop().run_in_executor(
            executor, generate_answer, request.question, context
        )
        
        return QueryResponse(
            answer=answer,
            sources=list(sources),
            agent_name=request.agent_name
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    try:
        index = pc_client.Index(INDEX_NAME)
        
        # Get index stats to see all namespaces
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        
        agents = []
        for namespace in namespaces:
            if namespace:  # Skip empty namespace
                namespace_stats = stats.namespaces[namespace]
                agents.append({
                    "name": namespace,
                    "document_count": namespace_stats.vector_count
                })
        
        return {"agents": agents}
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.delete("/agents/{agent_name}")
async def delete_agent(agent_name: str):
    """Delete an agent and all its documents"""
    try:
        index = pc_client.Index(INDEX_NAME)
        
        # Delete all vectors in the namespace
        index.delete(delete_all=True, namespace=agent_name)
        
        return {"message": f"Successfully deleted agent '{agent_name}' and all its documents"}
        
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "vector_db": "Pinecone"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)