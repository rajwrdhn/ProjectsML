from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

def build_rag_chain(papers):
    """Build a RAG pipeline using paper abstracts."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = []
    for paper in papers:
        for chunk in text_splitter.split_text(paper["abstract"]):
            docs.append(chunk)

    if not docs:
        raise ValueError("No documents to build the RAG chain.")

    embeddings = HuggingFaceEmbeddings(model="thenlper/gte-large")
    vectorstore = Chroma.from_texts(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI Research Copilot. Based on the following research abstracts:
{context}

Answer the question:
{question}

Provide:
1. Comparison of key approaches
2. Identified research gaps
3. Suggested experiment proposals
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
