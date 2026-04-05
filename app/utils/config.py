# # app/utils/config.py
# from pydantic_settings import BaseSettings
# import os

# class Settings(BaseSettings):
#     # AI Model Configuration
#     OPENAI_API_KEY: str = ""
#     HUGGINGFACE_API_KEY: str = "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"
    
#     # Model Selection (Free options)
#     AI_MODEL: str = "microsoft/DialoGPT-medium"
#     # "google/gemma-2b-it" 
#      # Free Hugging Face model
#     EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
#     # FAISS Configuration
#     FAISS_INDEX_PATH: str = "data/faiss_index"
#     VECTOR_DIMENSION: int = 384
    
#     # Application Settings
#     MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
#     CHUNK_SIZE: int = 1000
#     CHUNK_OVERLAP: int = 200
#     MAX_SOURCES: int = 3
    
#     # Prompt Engineering
#     SYSTEM_PROMPT: str = """You are a helpful assistant that answers questions based ONLY on the provided context. 
#     If the question cannot be answered from the context, respond with "I cannot answer this question based on the provided documents."
#     Always cite the source of your information."""
    
#     class Config:
#         env_file = ".env"

# settings = Settings()


# app/utils/config.py
from typing import List
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # API Configuration
    HUGGINGFACE_API_KEY: str = ""
    
    # Use the working model from your test
    AI_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = "data/faiss_index"
    VECTOR_DIMENSION: int = 384
    
    # Application Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SOURCES: int = 3
    # Add format-specific settings
    
    MARKDOWN_EXTENSIONS: List[str] = ['tables', 'fenced_code', 'codehilite']
    HTML_PARSER: str = 'html.parser'  # or 'lxml' for better performance
    POWERPOINT_EXTRACT_NOTES: bool = True
    
    # File processing limits
    MAX_SLIDES_PER_PRESENTATION: int = 100
    MAX_HTML_SIZE: int = 5 * 1024 * 1024  # 5MB for HTML files

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
