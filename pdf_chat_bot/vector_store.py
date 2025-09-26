from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    model_name = "thenlper/gte-large"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
