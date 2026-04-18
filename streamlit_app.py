import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
# ------------------ CONFIG ------------------

load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"

# ------------------ STREAMLIT UI ------------------
st.set_page_config(
    page_title="🧠 Medical RAG Chatbot",
    layout="wide"
)

st.title("🩺 AI Medical Assistant")
st.caption("For informational purposes only. Consult a doctor for medical advice.")

# ------------------ CACHE VECTORSTORE ------------------
@st.cache_resource
def load_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

# ------------------ PROMPT ------------------
def get_prompt():
    template = """
    You are an AI Medical Assistant providing information based on the given context.
    Use the medical context to answer the question accurately.

    If the answer is not in context:
    → Say you don’t have enough information.

    DO NOT:
    - Give diagnosis
    - Give treatment advice

    Always suggest consulting a doctor.

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ------------------ QA CHAIN ------------------
@st.cache_resource
def load_qa_chain():
    db = load_vectorstore()
    if db is None:
        return None

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment.")
        return None

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            groq_api_key=groq_api_key,
        ),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': get_prompt()}
    )
    return qa_chain

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
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                qa_chain = load_qa_chain()

                if qa_chain is None:
                    st.error("QA system not initialized.")
                else:
                    response = qa_chain.invoke({"query": query})

                    result = response["result"]
                    source_documents = response["source_documents"]

                    # Format sources
                    sources_text = ""
                    if source_documents:
                        sources_text += "\n\n**Sources:**\n"
                        for doc in source_documents:
                            page = doc.metadata.get("page", "N/A")
                            source = doc.metadata.get("source", "Unknown")
                            sources_text += f"- Page {page} from `{source}`\n"

                    final_response = result + sources_text

                    st.markdown(final_response)

                    # Save response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })

            except Exception as e:
                st.error(f"Error: {str(e)}")
