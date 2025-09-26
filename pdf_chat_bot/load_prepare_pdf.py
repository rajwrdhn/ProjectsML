from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf():
    loader = PyPDFLoader('pdf_chat_bot/example.pdf')
    documents = loader.load()
    return documents

def get_chunks(docu):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=2000)
    chunks = text_splitter.split_documents(docu)
    return chunks
