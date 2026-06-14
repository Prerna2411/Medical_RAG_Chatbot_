import os
import streamlit as st
from dotenv import load_dotenv
import traceback

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
# ------------------ CONFIG ------------------

load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="🧠 Medical RAG Chatbot", layout="wide")

st.title("🩺 AI Medical Assistant")
st.caption("For informational purposes only. Consult a doctor for medical advice.")

# ------------------ CACHE VECTORSTORE ------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# ------------------ PROMPT ------------------
def get_prompt():
    return ChatPromptTemplate.from_template("""
You are an AI Medical Assistant providing information based on the given context.

Rules:
- Use ONLY the context
- If answer not found → say "I don't have enough information"
- DO NOT give diagnosis or treatment
- Always suggest consulting a doctor

Context:
{context}

Question:
{question}

Answer:
""")

# ------------------ FORMAT DOCS ------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------ QA CHAIN (LCEL) ------------------
@st.cache_resource
def load_rag_chain():
    db = load_vectorstore()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found.")
        return None

  

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=groq_api_key,
    )

    retriever = db.as_retriever(search_kwargs={'k': 3})

    prompt = get_prompt()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# ------------------ CHAT MEMORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! I'm your AI Medical Assistant. Ask me any health-related question."
        }
    ]

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------
query = st.chat_input("Type your medical question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_chain, retriever = load_rag_chain()

                # Get docs separately for sources
                docs = retriever.invoke(query)

                result = rag_chain.invoke(query)

                # Format sources
                sources_text = ""
                if docs:
                    sources_text += "\n\n**Sources:**\n"
                    for doc in docs:
                        page = doc.metadata.get("page", "N/A")
                        source = doc.metadata.get("source", "Unknown")
                        sources_text += f"- Page {page} from `{source}`\n"

                final_response = result + sources_text

                st.markdown(final_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

            

            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
