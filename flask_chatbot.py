import os
import json
from flask import Flask, request, jsonify, render_template_string
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# load environment variables
load_dotenv(find_dotenv())

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"

# Simple manual cache for the vector store
_vectorstore_cache = {}

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Functions (Backend) ---

def get_vectorstore():
    """
    Loads the FAISS vector store. Caches the result to avoid reloading.
    """
    if 'db' not in _vectorstore_cache:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            _vectorstore_cache['db'] = db
            return db
        except Exception as e:
            print(f"Failed to load the medical knowledge base: {e}")
            return None
    return _vectorstore_cache['db']

def set_custom_prompt(custom_prompt_template):
    """
    Creates and returns a PromptTemplate instance.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# --- HTML Interface Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Medical Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
            color: #1f2937;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 1rem;
        }
        .chat-box {
            flex-grow: 1;
            overflow-y: auto;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .message {
            margin-bottom: 1rem;
            padding: 0.75rem 1rem;
            border-radius: 20px;
            max-width: 75%;
        }
        .user-message {
            background-color: #d1fae5;
            align-self: flex-end;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #e5e7eb;
            align-self: flex-start;
        }
        .input-form {
            display: flex;
            gap: 0.5rem;
        }
        .loading-dots:after {
            content: '.';
            animation: dots 1s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% {
                color: rgba(0,0,0,0);
                text-shadow:
                    .25em 0 0 rgba(0,0,0,0),
                    .5em 0 0 rgba(0,0,0,0);
            }
            40% {
                color: #555;
                text-shadow:
                    .25em 0 0 rgba(0,0,0,0),
                    .5em 0 0 rgba(0,0,0,0);
            }
            60% {
                text-shadow:
                    .25em 0 0 #555,
                    .5em 0 0 rgba(0,0,0,0);
            }
            80%, 100% {
                text-shadow:
                    .25em 0 0 #555,
                    .5em 0 0 #555;
            }
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="chat-container bg-white rounded-lg shadow-lg overflow-hidden md:h-[90vh] h-full flex flex-col">
        <header class="bg-green-600 text-white p-4 text-center rounded-t-lg shadow-md">
            <h1 class="text-xl md:text-2xl font-semibold">🩺 AI Medical Assistant</h1>
            <p class="text-sm mt-1">Chatbot for informational purposes only. Please consult a doctor for advice.</p>
        </header>

        <main id="chat-box" class="flex-grow p-4 overflow-y-auto space-y-4">
            <div class="assistant-message bg-gray-200 text-gray-800 self-start">
                Namaste! I'm your AI Medical Assistant. I can help answer your health-related questions.
                Please remember, I'm an AI and not a substitute for a qualified medical professional.
            </div>
        </main>

        <footer class="p-4 bg-gray-50 border-t border-gray-200">
            <form id="chat-form" class="input-form">
                <input type="text" id="user-input" placeholder="Type your message..." class="flex-grow p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500" required>
                <button type="submit" class="bg-green-600 text-white p-3 rounded-lg font-medium hover:bg-green-700 transition-colors duration-200">
                    Send
                </button>
            </form>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const chatForm = document.getElementById('chat-form');
            const userInput = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');

            function addMessage(role, content) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}-message`;
                messageDiv.innerHTML = content.replace(/\\n/g, '<br>');
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = userInput.value;
                if (!query) return;

                // Add user message to UI
                addMessage('user', query);
                userInput.value = '';

                // Add loading indicator
                const loadingMessage = document.createElement('div');
                loadingMessage.className = 'assistant-message bg-gray-200 text-gray-800 self-start';
                loadingMessage.innerHTML = 'Thinking<span class="loading-dots"></span>';
                chatBox.appendChild(loadingMessage);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await response.json();
                    
                    // Remove loading indicator
                    chatBox.removeChild(loadingMessage);

                    if (data.error) {
                        addMessage('assistant', `Error: ${data.error}`);
                    } else {
                        let responseContent = data.result;
                        if (data.sources && data.sources.length > 0) {
                            responseContent += '<br><br><strong>Sources:</strong><br>';
                            data.sources.forEach((source, index) => {
                                responseContent += `• Page ${source.page} from <code>${source.source}</code><br>`;
                            });
                        }
                        addMessage('assistant', responseContent);
                    }
                } catch (error) {
                    // Remove loading indicator and show error
                    chatBox.removeChild(loadingMessage);
                    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                    console.error('Fetch error:', error);
                }
            });
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route("/")
def home():
    """Serves the main chat interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat interaction."""
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    CUSTOM_PROMPT_TEMPLATE = """
    You are an AI Medical Assistant providing information based on the given context.
    Use the pieces of medical information provided in the context to accurately answer the user's health-related question.
    If the answer is not available in the provided context, state clearly that you don't have enough information to answer the question, and do not try to make up an answer.
    Focus on providing factual, evidence-based information from the context.
    Do not provide any personal medical advice, diagnosis, or treatment recommendations. Always advise the user to consult a qualified healthcare professional for specific health concerns.

    Context: {context}
    Question: {question}

    Begin your answer directly.
    """

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return jsonify({"error": "Failed to load vector store."}), 500

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return jsonify({"error": "GROQ_API_KEY not found."}), 500

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.0,
                groq_api_key=groq_api_key,
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': query})
        result = response["result"]
        source_documents = response["source_documents"]

        sources = []
        for doc in source_documents:
            sources.append({
                "page": doc.metadata.get('page', 'N/A'),
                "source": doc.metadata.get('source', 'Unknown Document')
            })

        return jsonify({
            "result": result,
            "sources": sources
        })

    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Using '0.0.0.0' makes the server accessible externally within a container
    app.run(host='0.0.0.0', port=5001)
