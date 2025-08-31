from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

###STEP1:LOAD RAW PDFS

DATA_PATH="data/"
def load_pdf_file(data):
    loader=DirectoryLoader(data,glob='*.pdf',
    loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

docs=load_pdf_file(data=DATA_PATH)
print(len(docs))

##step 2-create chunks

def create_chunks(extracted_data):
    textsplitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=textsplitter.split_documents(extracted_data)
    return text_chunks


text_chunks=create_chunks(extracted_data=docs)

print(len(text_chunks))


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)

