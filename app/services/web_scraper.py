# app/services/web_scraper.py
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse
import hashlib
import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import DocumentChunk, DocumentMetadata, DocumentType
from app.utils.config import settings

logger = logging.getLogger(__name__)

class WebScraperService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 30
        self.max_content_length = 10 * 1024 * 1024  # 10MB limit
    
    async def scrape_website(self, url: str) -> List[DocumentChunk]:
        """Scrape website content and return chunks"""
        try:
            # Validate and clean URL
            cleaned_url = self._clean_url(url)
            
            # Fetch webpage content
            content = await self._fetch_webpage_content(cleaned_url)
            
            if not content:
                raise ValueError("No content could be extracted from the webpage")
            
            # Create metadata
            metadata = DocumentMetadata(
                filename=self._generate_filename_from_url(cleaned_url),
                file_type=DocumentType.WEBPAGE,
                upload_timestamp=datetime.now(),
                url=cleaned_url
            )
            
            # Create chunks
            chunks = self._create_chunks_from_content(content, metadata)
            
            logger.info(f"Successfully scraped {cleaned_url} - {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            raise Exception(f"Failed to scrape website: {str(e)}")
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URL"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    
    async def _fetch_webpage_content(self, url: str) -> str:
        """Fetch and extract clean text content from webpage"""
        try:
            # Make request with timeout
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content length
            if len(response.content) > self.max_content_length:
                raise ValueError("Webpage content too large")
            
            # Extract content using trafilatura (better than BeautifulSoup for articles)
            content = trafilatura.extract(response.text, include_comments=False, include_tables=True)
            
            if content:
                return content
            
            # Fallback to BeautifulSoup if trafilatura fails
            return self._extract_with_beautifulsoup(response.text)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch webpage: {str(e)}")
    
    def _extract_with_beautifulsoup(self, html: str) -> str:
        """Fallback content extraction using BeautifulSoup"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract text from main content areas
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content',
            '.entry-content', '.article-content', '.post-body'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = " ".join([elem.get_text(strip=True) for elem in elements])
                break
        
        # If no specific content area found, get all text
        if not content:
            content = soup.get_text(strip=True)
        
        # Clean up whitespace
        content = ' '.join(content.split())
        
        return content
    
    def _generate_filename_from_url(self, url: str) -> str:
        """Generate a filename from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/').replace('/', '_')
        
        if path:
            filename = f"{domain}_{path}"
        else:
            filename = domain
        
        # Clean filename
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        filename = filename[:100]  # Limit length
        
        return f"{filename}.webpage"
    
    def _create_chunks_from_content(self, content: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Create chunks from webpage content"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk_words = words[i:i + settings.CHUNK_SIZE]
            chunk_content = " ".join(chunk_words)
            
            chunk_id = hashlib.md5(f"{metadata.url}_{i}".encode()).hexdigest()
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                metadata={
                    "filename": metadata.filename,
                    "url": metadata.url,
                    "chunk_index": i // (settings.CHUNK_SIZE - settings.CHUNK_OVERLAP),
                    "word_count": len(chunk_words),
                    "source_type": "webpage"
                }
            )
            chunks.append(chunk)
        
        return chunks
