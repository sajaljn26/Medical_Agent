import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# step 1 LLM
HF_TOKEN = os.environ["HF_TOKEN"]
huggingface_rep_id = "mistralai/Mistral-7B-Instruct-v0.3"
# step 2 connect with memory and create chain
