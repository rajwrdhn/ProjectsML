"""Flask API + web interface for movie search using embeddings + Gemini LLM."""
import os
from flask import Flask, request, jsonify, render_template
from pgvector.sqlalchemy import Vector
from sqlalchemy import text, cast
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from MovieRecommenderSystem.models import engine  

app = Flask(__name__)

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gemini LLM wrapper
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

def refine_prompt_with_llm(user_query: str) -> str:
    """Use Gemini to extract a clean genre description from free text."""
    messages = [
        ("system", "You are an assistant that extracts movie genres and ratings."),
        ("human", f"Extract the genre(s) from this query: {user_query}. Respond only with the genre phrase (e.g., 'sci-fi thriller').")
    ]
    # Use predict_messages if invoke() does not work
    response = llm.invoke(messages)
    return response.content.strip()

def search_by_genres(query: str, top_k: int = 5):
    """Semantic search by genres + order by highest rating."""
    refined_genre = refine_prompt_with_llm(query)
    emb = embed_model.encode([refined_genre])[0].tolist()

    sql = """
    SELECT title, year, genres, average_rating
    FROM movies
    ORDER BY embedding <-> cast(:vec AS vector), average_rating DESC
    LIMIT :k
    """
    with engine.connect() as conn:
        results = conn.execute(text(sql), {"vec": emb, "k": top_k}).fetchall()

    return {
        "original_query": query,
        "refined_genre": refined_genre,
        "results": [dict(r._mapping) for r in results]
    }

# API endpoint
@app.route("/search", methods=["GET"])
def search_endpoint():
    query = request.args.get("prompt")
    if not query:
        return jsonify({"error": "Please provide a prompt parameter."}), 400
    top_k = int(request.args.get("top_k", 5))
    results = search_by_genres(query, top_k=top_k)
    return jsonify(results)

# Web interface
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
