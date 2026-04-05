# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import uvicorn
import os
import tempfile
from typing import List, Dict

from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata , WebLinkRequest, ClearContextResponse
from app.services.document_processor import DocumentProcessor
from app.services.web_scraper import WebScraperService
from app.services.rag_service import RAGService
from app.services.ai_service import AIService
from app.utils.config import settings

# app = FastAPI(title="Chat with Documents", version="1.0.0")
app = FastAPI(title="Chat with Documents", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize services
document_processor = DocumentProcessor()
rag_service = RAGService()
ai_service = AIService()
web_scraper = WebScraperService()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        # Validate file size
        content = await file.read()
        # file.size > 
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        file_extension = file.filename.lower().split('.')[-1]

        # Validate file type
        supported_extensions = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', 
                              '.md', '.markdown', '.pptx', '.ppt', '.html', '.htm']
        
        if f'.{file_extension}' not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            chunks = await document_processor.process_document(tmp_file_path, file.filename)
            
            # Add to RAG system
            rag_service.add_documents(chunks)
            
            return {"message": f"Successfully processed {file.filename} with {len(chunks)} chunks"}
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-link", response_model=Dict[str, str])
async def process_website_link(request: WebLinkRequest):
    """Process website link and add to context"""
    try:
        # Scrape website content
        chunks = await web_scraper.scrape_website(str(request.url))
        
        # Add to RAG system
        rag_service.add_documents(chunks)
        
        return {
            "message": f"Successfully processed website {request.url} with {len(chunks)} chunks",
            "url": str(request.url),
            "chunks_created": str(len(chunks))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the documents"""
    try:
        # Get relevant context
        context, sources = rag_service.get_context_for_query(request.query)
        print("context"+context+'context end')
        # print("sources"+sources+'sources end')
        # Generate response
        response = await ai_service.generate_response(request.query, context, sources)
        # print("response"+response+'response end')
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-context", response_model=ClearContextResponse)
async def clear_all_context():
    """Clear all documents and context"""
    try:
        result = rag_service.clear_all_context()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context-stats")
async def get_context_stats():
    """Get statistics about current context"""
    try:
        stats = rag_service.get_context_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents_indexed": rag_service.index.ntotal if rag_service.index else 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
