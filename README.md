# Enhanced Law Firm Chatbot

An advanced RAG (Retrieval-Augmented Generation) system designed specifically for legal document analysis and querying. This chatbot provides intelligent legal analysis with document upload capabilities and smart query processing.

## 🚀 Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Legal Query Analysis**: Advanced legal keyword detection and analysis
- **Smart Response Generation**: Context-aware responses with confidence scoring
- **Professional Legal Analysis**: Tailored for legal terminology and concepts
- **Interactive API Documentation**: Built-in Swagger/OpenAPI docs
- **Health Monitoring**: Real-time server health checks

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Server**: Uvicorn ASGI server
- **Document Processing**: Multi-format document support
- **API Documentation**: Swagger/OpenAPI
- **Virtual Environment**: Python venv

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Windows/macOS/Linux

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd lawfirmChatbot
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv robi

# Activate virtual environment
# On Windows:
.\robi\Scripts\activate.bat
# On macOS/Linux:
source robi/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Server

**Option 1: Using the Batch File (Windows - Recommended)**
```bash
.\start_chatbot.bat
```

**Option 2: Using Python Directly**
```bash
python working_server.py
```

**Option 3: Using Uvicorn**
```bash
uvicorn working_server:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Application

Once the server is running, you can access:

- **Main API**: http://127.0.0.1:8000
- **Interactive API Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message and API information |
| GET | `/health` | Server health check |
| GET | `/docs` | Interactive API documentation |

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload-document` | Upload legal documents (PDF, DOCX, TXT) |
| POST | `/api/v1/query` | Query uploaded documents with legal analysis |

## 🔧 Usage Examples

### Upload a Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload-document" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your-legal-document.pdf"
```

### Query the Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main contractual obligations?"}'
```

## 📁 Project Structure

```
lawfirmChatbot/
├── app/                          # Application modules
│   ├── modules/
│   │   ├── lawfirmchatbot/      # Core chatbot functionality
│   │   ├── services/            # Business logic services
│   │   └── router.py            # API routing
├── core/                        # Core configuration
├── database/                    # Database configurations
├── robi/                       # Virtual environment
├── working_server.py           # Main server application (ACTIVE)
├── start_chatbot.bat          # Windows startup script (ACTIVE)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── main.py                   # Alternative server entry point
```

## 🧪 Testing the API

### Using the Interactive Documentation
1. Navigate to http://127.0.0.1:8000/docs
2. Try the `/health` endpoint to verify the server is running
3. Use the `/api/v1/upload-document` endpoint to upload a test document
4. Use the `/api/v1/query` endpoint to ask questions about your document

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "server": "running",
  "documents_uploaded": 0,
  "ready": true
}
```

## 🔍 Legal Analysis Features

The chatbot provides specialized legal analysis including:

- **Legal Keyword Detection**: Recognizes legal terminology and concepts
- **Query Complexity Analysis**: Determines if queries are simple or complex
- **Confidence Scoring**: Provides reliability estimates for responses
- **Professional Recommendations**: Suggests consulting with legal counsel when appropriate
- **Document Context**: Maintains context of uploaded legal documents

## 🚨 Troubleshooting

### Server Won't Start
1. Ensure the virtual environment is activated
2. Check that all dependencies are installed: `pip install -r requirements.txt`
3. Verify Python version: `python --version` (should be 3.8+)
4. Check if port 8000 is already in use

### Cannot Access Server
1. Verify the server is running (check console output)
2. Try accessing http://127.0.0.1:8000/health first
3. Check Windows Firewall settings
4. Ensure no other applications are using port 8000

### Document Upload Issues
1. Verify file format is supported (PDF, DOCX, TXT)
2. Check file size (large files may take longer to process)
3. Ensure the file is not corrupted

## 🔧 Development

### Adding New Features
1. Modify `working_server.py` for API changes
2. Update endpoints and business logic as needed
3. Test using the interactive documentation at `/docs`

### Environment Variables
Currently, the application runs with default settings. For production deployment, consider adding environment-specific configurations.

## 📝 License

This project is developed for legal document analysis and professional use.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the interactive API documentation at `/docs`
3. Verify all prerequisites are met

---

**Status**: ✅ Active and Running
**Last Updated**: $(Get-Date -Format "yyyy-MM-dd")
**Server**: http://127.0.0.1:8000
