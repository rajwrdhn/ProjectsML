import os
from dotenv import load_dotenv
load_dotenv()
google_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from load_prepare_pdf import get_chunks, load_pdf
from vector_store import create_embeddings, create_vector_store

# Load pdf
docu = load_pdf()
chunks = get_chunks(docu=docu)
embed = create_embeddings()
vector_store = create_vector_store(chunks, embed)
# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

# Create the retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

question = "What is the main theme of the document? i think it is data structures and algorithms."
answer = qa_chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {answer['result']}")