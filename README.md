
# Smart Chart V2 🚀

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.0-orange?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-1.11.0-yellow?logo=faiss)](https://faiss.ai)
[![Llama](https://img.shields.io/badge/Llama-3.3_70B-green?logo=llama)](https://llama.meta.com)

**AI-Powered Chat with Documents & Web**  
Smart Chart V2 is an advanced Retrieval-Augmented Generation (RAG) system that lets you upload documents (PDF, DOCX, images, PPTX, etc.), process websites, and ask natural language questions. Get accurate, cited answers with confidence scores—powered by FAISS vector search and Llama 3.3 70B.

## ✨ Features
- **Multi-Format Support**: PDF, DOCX, TXT, MD, PPTX, HTML, images (OCR via Tesseract)
- **Web Scraping**: Process URLs with clean content extraction (trafilatura)
- **RAG Pipeline**: Semantic search + LLM generation with source citations
- **Confidence Scoring**: Response reliability ratings
- **Real-time Stats**: Dashboard for chunks/sources/types
- **Responsive UI**: Dark-themed Bootstrap chat interface
- **Session Management**: Clear context anytime

## 📋 Supported Formats
| Type       | Examples                  | Processing                  |
|------------|---------------------------|-----------------------------|
| Documents | PDF, DOCX, TXT, MD       | Text extraction + chunking |
| Presentations | PPTX, PPT              | Slide + notes extraction   |
| Web       | URLs                     | Clean article parsing      |
| Images    | PNG, JPG, JPEG           | OCR text extraction        |
| HTML      | HTM, HTML files          | Structured parsing         |

**Limits**: 10MB/file, intelligent chunking (1000 words + 200 overlap)

## 🚀 Quick Start (1 Minute)
1. **Clone/Navigate**:
   ```
   cd c:/Users/vbhas/Documents/programming/project/smart-chart-2
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment** (create `.env`):
   ```
   HUGGINGFACE_API_KEY=your_together_ai_key_here
   ```
   - Get free key from [Together AI](https://api.together.ai) (uses Llama 3.3 70B Turbo Free)

4. **Run Server**:
   ```bash
   python app/main.py
   ```
   Or: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

5. **Open App**: http://localhost:8000
   - Upload files → Process URLs → Chat!

**Demo**: Upload a PDF, ask "What is the main topic?" → See cited answer!

## 🏗️ Architecture
```
Smart Chart V2
├── FastAPI Backend (app/main.py)
│   ├── Services: ai_service (LLM), rag_service (FAISS), document_processor, web_scraper
│   ├── Models: Pydantic schemas
│   └── Config: Chunking, models (all-MiniLM-L6-v2 embeddings)
├── FAISS Index (data/faiss_index)
├── Frontend: Bootstrap + Vanilla JS (index.html)
└── Dependencies: requirements.txt
```

**Flow**:
1. Upload → Extract → Chunk → Embed → Index (FAISS)
2. Query → Retrieve top chunks → LLM prompt → Answer + sources

**Endpoints**:
| Endpoint       | Method | Description                  |
|----------------|--------|------------------------------|
| `/`            | GET    | Chat UI                     |
| `/upload`      | POST   | File upload/process         |
| `/process-link`| POST   | URL scraping                |
| `/query`       | POST   | Ask question                |
| `/clear-context`| POST | Reset everything            |
| `/context-stats`| GET  | Stats                       |
| `/health`      | GET    | Health check                |

## 🔧 Environment Setup
Create `.env` in root:
```env
HUGGINGFACE_API_KEY=hf_xxxxxxxx  # From Together AI / Hugging Face
# Optional: AI_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
```

**Requirements** (key ones):
```
fastapi==0.116.0
sentence-transformers==5.0.0
faiss-cpu==1.11.0
PyMuPDF==1.26.3  # PDFs
pytesseract==0.3.13  # OCR
trafilatura==2.0.0  # Web
```

**Dev**: Python 3.8+, 4GB+ RAM recommended.

## 📖 Usage Guide
1. **Upload**: Drag files or select (multi-file OK)
2. **Web**: Paste URL → "Process"
3. **Chat**: Ask anything → Get answer + sources/confidence
4. **Stats**: View indexed chunks/document types
5. **Clear**: Reset session

**Tips**:
- Context persists across queries
- Sources shown as badges (file/URL + % confidence)
- "Cannot answer from docs" if no relevant context

## 📁 Project Structure
```
smart-chart-2/
├── README.md                 # 👈 You are here
├── requirements.txt
├── app/
│   ├── main.py              # FastAPI app
│   ├── services/            # AI/RAG/processing
│   ├── models/schemas.py
│   └── utils/config.py
├── frontend/templates/index.html
├── data/faiss_index.*       # Vector DB (auto-created)
└── TODO.md                  # Task tracking
```

## 🤝 Contributing
1. Fork & PR
2. Add tests: `python test.py`
3. Update docs
4. Follow PEP8

**Issues?** Check logs, ensure API key valid, file <10MB.

## 📈 Future Plans
- Multi-LLM support
- Local models (Ollama)
- Batch processing
- Auth/Users

**⭐ Star/Fork if useful!**

---
*Built with ❤️ using FastAPI, FAISS, & Llama | v2.0.0*

