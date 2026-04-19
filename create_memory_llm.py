from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

###STEP1:LOAD RAW PDFS

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

docs = load_pdf_file(DATA_PATH)
print("Documents loaded:", len(docs))

def create_chunks(extracted_data):
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    return textsplitter.split_documents(extracted_data)

text_chunks = create_chunks(docs)
print("Chunks created:", len(text_chunks))

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 64  # increase batch size
        }
    )

embedding_model = get_embedding_model()

os.makedirs(DB_FAISS_PATH, exist_ok=True)

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("FAISS DB created successfully!")