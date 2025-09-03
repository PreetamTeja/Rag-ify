# 🤖 Rag-ify - No code RAG Pipeline

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![React](https://img.shields.io/badge/react-18.0+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*Transform your documents into intelligent AI agents that can answer questions, provide insights, and remember conversations.*

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation)

</div>

---

## 📖 Overview

The Multi-Agent Document Q&A System is a sophisticated RAG (Retrieval-Augmented Generation) platform that allows you to create specialized AI agents from your documents. Each agent can understand and answer questions about its specific document set, remember conversations, and maintain its own personality through customizable system prompts.

### 🎯 Key Highlights

- **🧠 Intelligent Document Processing**: Advanced extraction of text, tables, and images from PDFs, DOCX, and TXT files
- **🔍 Hybrid Search**: Combines dense vector embeddings with sparse TF-IDF for optimal retrieval accuracy
- **💬 Persistent Memory**: Chat sessions with toggleable conversation memory for contextual responses
- **🎭 Customizable Agents**: Each agent has its own personality via editable system prompts
- **📊 Multi-Modal Understanding**: Processes tables, images with OCR, and structured content
- **⚡ Real-Time Chat**: Interactive chat interface with session management

---

## ✨ Features

### 🗂️ Document Processing
- **Multi-Format Support**: PDF, DOCX, TXT file processing
- **Table Extraction**: Advanced table detection using Camelot and Tabula
- **Image Processing**: OCR text extraction and AI-generated image captions
- **Smart Chunking**: Intelligent text segmentation with overlap for better context

### 🤖 AI Agents
- **Custom System Prompts**: Define unique personalities and behaviors for each agent
- **Persistent Memory**: Remember conversations across sessions
- **Session Management**: Multiple concurrent chat sessions per agent
- **Agent Statistics**: Track document count, chat sessions, and creation dates

### 🔍 Advanced Retrieval
- **Hybrid Search**: Dense + Sparse retrieval for comprehensive results
- **Content-Aware**: Special handling for tables, images, and structured data
- **Contextual Understanding**: Maintains conversation context for better responses
- **Source Attribution**: Tracks and cites source documents in responses

### 💻 User Interface
- **Modern React UI**: Clean, responsive interface with Tailwind CSS
- **Real-Time Chat**: Instant messaging with typing indicators
- **Agent Builder**: Intuitive agent creation and management
- **Session History**: Browse and restore previous conversations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Pinecone API Key
- Optional: Tesseract OCR for image processing

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multi-agent-document-qa.git
cd multi-agent-document-qa
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Create .env file
echo "PINECONE_API_KEY=your_pinecone_api_key_here" > .env

# Start the backend
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm start
```

### 4. Access the Application
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🛠️ Installation

### Backend Dependencies
```bash
pip install fastapi uvicorn python-multipart
pip install sentence-transformers torch transformers
pip install pinecone-client python-dotenv
pip install PyPDF2 python-docx pandas pillow
pip install scikit-learn numpy==1.24.3

# Optional advanced features
pip install camelot-py[cv] tabula-py pytesseract PyMuPDF
```

### Frontend Dependencies
```bash
npm install react react-dom @types/react @types/react-dom
npm install lucide-react tailwindcss
npm install @vitejs/plugin-react vite
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript

# macOS
brew install tesseract ghostscript

# Windows
# Download and install from respective websites
```

---

## 📚 Usage

### Creating Your First Agent

1. **Upload a Document**
   - Navigate to the Agent Builder
   - Enter an agent name (e.g., "Legal Assistant")
   - Customize the system prompt to define behavior
   - Upload your PDF, DOCX, or TXT file
   - Click "Create Agent"

2. **Start Chatting**
   - Select your agent from the sidebar
   - Create a new chat session
   - Ask questions about your document
   - Enable/disable memory as needed

### Example System Prompts

```text
Legal Assistant:
"You are a legal expert who analyzes contracts and legal documents. 
Provide precise, professional responses and cite specific clauses when possible."

Research Helper:
"You are a research assistant who helps analyze academic papers. 
Summarize key findings, identify methodologies, and highlight important insights."

Technical Documentation Bot:
"You are a technical writer who explains complex procedures clearly. 
Break down steps, provide examples, and use simple language for clarity."
```

---

## 🔧 Configuration

### Environment Variables
```bash
# .env file in backend directory
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=document-agents
EMBEDDING_DIM=384
```

### Model Configuration
The system uses these models by default:
- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **LLM**: `microsoft/DialoGPT-small`
- **Vision**: `Salesforce/blip-image-captioning-base`

---

## 📡 API Documentation

### Core Endpoints

#### Create Agent
```http
POST /upload
Content-Type: multipart/form-data

file: [document file]
agent_name: "My Agent"
system_prompt: "Custom behavior description"
```

#### Query Agent
```http
POST /query
Content-Type: application/json

{
  "question": "What is this document about?",
  "agent_name": "My Agent",
  "session_id": "session_123",
  "memory_enabled": true
}
```

#### List Agents
```http
GET /agents
```

#### Update System Prompt
```http
PUT /agents/{agent_name}/system-prompt
Content-Type: application/json

{
  "system_prompt": "New behavior description"
}
```

#### Session Management
```http
GET /sessions                    # List all sessions
POST /sessions?agent_name=Agent  # Create new session
DELETE /sessions/{session_id}    # Delete session
```

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI       │    │   Pinecone      │
│                 │◄──►│   Backend       │◄──►│   Vector DB     │
│ • Agent Builder │    │ • Document      │    │ • Embeddings    │
│ • Chat Interface│    │   Processing    │    │ • Namespaces    │
│ • Session Mgmt  │    │ • Vector Search │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │ • Sentence      │
                       │   Transformers  │
                       │ • LLM           │
                       │ • Vision Model  │
                       └─────────────────┘
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Pinecone](https://www.pinecone.io/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) and [Tailwind CSS](https://tailwindcss.com/) for the frontend
- [Hugging Face](https://huggingface.co/) for pre-trained models

---

## 📞 Support

- 📧 **Email**: your-email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/multi-agent-document-qa/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/multi-agent-document-qa/discussions)

---

<div align="center">

**Made with ❤️ for the AI community**

[⭐ Star this repository](https://github.com/yourusername/multi-agent-document-qa) if you find it useful!

</div>
