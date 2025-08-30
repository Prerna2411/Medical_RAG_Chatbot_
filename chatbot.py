import os
import streamlit as st

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Environment variable management
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the .env file.
load_dotenv(find_dotenv())

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
APP_TITLE = "🩺 AI Medical Assistant (RAG Chatbot)"
INITIAL_MESSAGE = "Namaste! I'm your AI Medical Assistant. I can help answer your health-related questions based on the information I've been trained on. Please remember, I'm an AI and not a substitute for a qualified medical professional."

# --- Helper Functions ---

@st.cache_resource
def get_vectorstore():
    """
    Loads the FAISS vector store from the local directory.
    This function is cached to avoid reloading the vector store on every rerun.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load the medical knowledge base: {e}\n"
                 f"Please ensure the FAISS index exists at '{DB_FAISS_PATH}'.")
        return None

def set_custom_prompt(custom_prompt_template):
    """
    Creates and returns a PromptTemplate instance.
    This template is used to format the context and question for the LLM.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")

    # --- Sidebar ---
    with st.sidebar:
        st.header("About This Assistant")
        st.markdown(
            "This AI Medical Assistant uses **Retrieval-Augmented Generation (RAG)** "
            "to answer your health-related questions based on a specific knowledge base. "
            "It is powered by LangChain, Streamlit, HuggingFace Embeddings, and Groq LLMs."
        )
        
        st.markdown("---")
       

    st.title(APP_TITLE)

    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': INITIAL_MESSAGE}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    user_query = st.chat_input("Ask a health-related question (e.g., 'What are the symptoms of dengue?')")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(user_query)

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
            # Load vector store and check if it's available
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return # Stop execution if vector store failed to load

            # Retrieve the GROQ API key from the environment variables
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY not found in environment variables. Please set it in your `.env` file.")
                return

            # Initialize the RetrievalQA chain with the Groq model
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama3-8b-8192",  # A fast, free Groq-hosted model
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}), # Retrieve top 3 relevant chunks
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Invoke the chain with the user's query
            with st.spinner("Searching and synthesizing information..."):
                response = qa_chain.invoke({'query': user_query})
            
            result = response["result"]
            source_documents = response["source_documents"]

            # Format and display the response
            with st.chat_message('assistant'):
                st.markdown(result)
                if source_documents:
                    with st.expander("📖 **Sources**"):
                        for i, doc in enumerate(source_documents):
                            page = doc.metadata.get('page', 'N/A')
                            source_name = doc.metadata.get('source', 'Unknown Document')
                            st.markdown(f"**{i+1}.** **Document:** `{source_name}`, **Page:** `{page}`")
                            st.text(doc.page_content[:200] + "...") # Show a snippet of the content

            # Add assistant response to chat history
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.session_state.messages.append({'role': 'assistant', 'content': "Sorry, I encountered an error while processing your request. Please try again or rephrase your question."})

if __name__ == "__main__":
    main()

