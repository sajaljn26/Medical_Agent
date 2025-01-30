from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
print("Lenght of pdf pages", len(documents))