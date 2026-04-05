# app/models/schemas.py
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import re

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    WEBPAGE = "webpage"
    MARKDOWN = "markdown"     # New
    POWERPOINT = "powerpoint" # New
    HTML = "html"            # New

class DocumentMetadata(BaseModel):
    filename: str
    file_type: DocumentType
    file_size: Optional[int] = None
    upload_timestamp: datetime
    page_count: Optional[int] = None
    url: Optional[str] = None  # For web content
    
class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class WebLinkRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to scrape and process")
    
    @validator('url')
    def validate_url(cls, v):
        url_str = str(v)
        # Basic URL validation
        if not re.match(r'^https?://', url_str):
            raise ValueError('URL must start with http:// or https://')
        return v

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    session_id: str
    
class ChatSession(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = []
    created_at: datetime
    last_activity: datetime

class ClearContextResponse(BaseModel):
    message: str
    documents_cleared: int
    status: str