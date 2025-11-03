# 🧠 Medical RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** based **Medical Chatbot** designed to provide accurate, context-aware, and reliable medical responses by combining **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** with a **retrieval system** that pulls relevant medical data from trusted sources.

---

## 🚀 Features

- 💬 **Context-Aware Medical Chat** — Understands user queries and provides relevant responses using advanced LLMs.  
- 📚 **Retrieval-Augmented Generation (RAG)** — Retrieves factual data from medical documents to ensure accuracy.  
- 🩺 **Medical Knowledge Base Integration** — Uses vectorized embeddings of medical texts for grounded answers.  
- 🔍 **Semantic Search** — Efficiently finds the most relevant medical context from a large corpus.  
- 🧩 **Scalable Design** — Modular structure for easy updates, additional sources, or LLM fine-tuning.  
- 🌐 **Interactive Frontend (Optional)** — Can be integrated with a simple web UI using Streamlit or FastAPI.

---

## 🏗️ Project Structure

Medical_RAG_Chatbot_/
│
├── .github/
│ └── workflows/ # GitHub Actions workflows for CI/CD
│
├── pycache/ # Python cache files (auto-generated)
│
├── data/ # Folder containing medical documents or datasets
│
├── vectorstore/
│ └── db_faiss/ # FAISS vector database for document embeddings
│
├── chatbot.py # Main RAG chatbot script
├── connect_memory_with_llm.py # Connects the retriever and LLM pipeline
├── create_memory_llm.py # Generates embeddings and builds memory database
├── flask_chatbot.py # Flask web API for chatbot interaction
├── dockerfile # Docker configuration for containerization
├── requirements.txt # Project dependencies
└── .gitignore # Ignored files and folders


---

## ⚙️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/<your-username>/Medical_RAG_Chatbot_.git
cd Medical_RAG_Chatbot_

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

3. Install Dependencies
pip install -r requirements.txt

4. Add Your API Keys

If you’re using OpenAI, Hugging Face, or other APIs, create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_api_key_here

🧩 How It Works

Document Loading:
Loads medical documents or research papers into the system.

Embedding Generation:
Converts text into numerical vector representations using embedding models.

Semantic Retrieval:
When a user asks a question, the retriever searches for the most relevant chunks.

LLM Response Generation:
The chatbot combines the retrieved context with the user query to generate accurate, fact-grounded answers.

💡 Example Query

User:

What are the symptoms of diabetes?

Chatbot:

Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, and slow-healing wounds. Consult a doctor for proper diagnosis and treatment.

🧠 Tech Stack

Python 3.10+

LangChain / LlamaIndex (for RAG pipeline)

FAISS / ChromaDB (for vector storage)

OpenAI GPT / Mistral / Llama 3 (as the LLM)

Streamlit / FastAPI (for frontend interface)

dotenv, transformers, PyPDF2, etc.

🧪 Run the Application
 Run via Command Line
python chabot.py


