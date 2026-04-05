# app/services/rag_service.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import pickle
import os
import logging
from app.models.schemas import DocumentChunk, QueryRequest, QueryResponse , ClearContextResponse
from app.utils.config import settings
import logging
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = None
        self.chunks_store = {}
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_path = f"{settings.FAISS_INDEX_PATH}.index"
        store_path = f"{settings.FAISS_INDEX_PATH}.pkl"
        
        if os.path.exists(index_path) and os.path.exists(store_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(store_path, 'rb') as f:
                    self.chunks_store = pickle.load(f)
                logger.info(f"Loaded existing index with {self.index.ntotal} documents")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(settings.VECTOR_DIMENSION)
        self.chunks_store = {}
        logger.info("Created new FAISS index")

    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to FAISS index"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        start_idx = len(self.chunks_store)
        for i, chunk in enumerate(chunks):
            self.chunks_store[start_idx + i] = chunk
        
        # Save index
        self.save_index()
        logger.info(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")
    
    def clear_all_context(self) -> ClearContextResponse:
        """Clear all documents and context from the system"""
        try:
            documents_count = self.index.ntotal if self.index else 0
            
            # Create new empty index
            self._create_new_index()
            
            # Remove saved files
            index_path = f"{settings.FAISS_INDEX_PATH}.index"
            store_path = f"{settings.FAISS_INDEX_PATH}.pkl"
            
            for path in [index_path, store_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info(f"Cleared {documents_count} documents from context")
            
            return ClearContextResponse(
                message="All documents and context have been cleared successfully",
                documents_cleared=documents_count,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return ClearContextResponse(
                message=f"Error clearing context: {str(e)}",
                documents_cleared=0,
                status="error"
            )

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context"""
        total_chunks = self.index.ntotal if self.index else 0
        
        # Count documents by type
        doc_types = {}
        sources = set()
        
        for chunk in self.chunks_store.values():
            source_type = chunk.metadata.get('source_type', 'document')
            doc_types[source_type] = doc_types.get(source_type, 0) + 1
            
            if 'url' in chunk.metadata:
                sources.add(chunk.metadata['url'])
            else:
                sources.add(chunk.metadata['filename'])
        
        return {
            "total_chunks": total_chunks,
            "document_types": doc_types,
            "unique_sources": len(sources),
            "sources": list(sources)
        }

    def save_index(self):
        """Save FAISS index and chunks store"""
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
        
        index_path = f"{settings.FAISS_INDEX_PATH}.index"
        store_path = f"{settings.FAISS_INDEX_PATH}.pkl"
        
        faiss.write_index(self.index, index_path)
        with open(store_path, 'wb') as f:
            pickle.dump(self.chunks_store, f)
    
    def clear_index(self):
        """Clear the FAISS index and chunks store, then save empty index"""
        self.index.reset()
        self.chunks_store = {}
        self.save_index()
    
    def search_similar_chunks(self, query: str, k: int = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using FAISS"""
        if k is None:
            k = settings.MAX_SOURCES
        
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.chunks_store:
                chunk = self.chunks_store[idx]
                results.append((chunk, float(score)))
        
        return results
    
    def get_context_for_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Get relevant context for query"""
        similar_chunks = self.search_similar_chunks(query)
        
        if not similar_chunks:
            return "", []
        
        # Build context
        context_parts = []
        sources = []
        
        for chunk, score in similar_chunks:
            source_info = chunk.metadata.get('url', chunk.metadata['filename'])
            context_parts.append(f"Source: {source_info}\nContent: {chunk.content}")
            
            sources.append({
                "filename": chunk.metadata['filename'],
                "url": chunk.metadata.get('url'),
                "chunk_index": chunk.metadata.get('chunk_index', 0),
                "confidence": score,
                "source_type": chunk.metadata.get('source_type', 'document')
            })
        
        context = "\n\n".join(context_parts)
        return context, sources
