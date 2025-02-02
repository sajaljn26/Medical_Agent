import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

# Configuration - UPDATED VARIABLE NAME
HUGGINGFACEHUB_API_TOKEN = os.environ["HF_TOKEN"]
HUGGINGFACE_REPO_ID ="google/gemma-2-2b-it"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def load_llm(huggingface_rep_id):
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"max_length": 512},  
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )


prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={'k': 3})
llm = load_llm(HUGGINGFACE_REPO_ID)
chain = create_stuff_documents_chain(llm, prompt)

user_query = input("Write Query Here: ")
documents = retriever.invoke(user_query)

# Invoke the chain with both context and the user's question
response = chain.invoke({"context": documents, "question": user_query})

print("RESULT:", response)
print("Source", documents)