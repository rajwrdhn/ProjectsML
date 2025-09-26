from flask import Flask, render_template, request
from retriever import arxiv_retriever
from rag_pipeline import build_rag_chain

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    papers = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")
        papers = arxiv_retriever(query, 5)
        qa_chain = build_rag_chain(papers)
        result = qa_chain.invoke(query)

    return render_template("index.html", result=result, papers=papers, query=query)

if __name__ == "__main__":
    app.run(debug=True)

