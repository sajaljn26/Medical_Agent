from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

#load the data

DATA_PATH = "data/"
def load_pdf_files(data):
    loader  = DirectoryLoader(
        data,
        glob = '*.pdf',
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
#print("Lenght of pdf pages", len(documents))

#step-2 create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
#print("Lenght of text chunks", len(text_chunks))

# step-3 get embeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

embedding_models = get_embedding_model()

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_models)
db.save_local(DB_FAISS_PATH)