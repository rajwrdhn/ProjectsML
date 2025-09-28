# 🎬 Movie Recommender System

A semantic search–powered **Movie Recommendation Engine** that combines **pgvector**, **HuggingFace embeddings**, and **Google Gemini LLM** to deliver top-rated movies based on natural language prompts.

Built with **Flask**, **SQLAlchemy**, and **PostgreSQL**.

---

## ✨ Features

* ✅ Store and query IMDB movies (1990–2025) using **pgvector**.
* ✅ Generate embeddings with [HuggingFace Sentence Transformers](https://www.sbert.net/).
* ✅ Refine user prompts with **Google Gemini (LLM)**.
* ✅ Perform **semantic search** on movie genres.
* ✅ Rank results by **highest rating + similarity**.
* ✅ Web interface with `index.html` for dynamic querying.
* ✅ REST API (`/search`) endpoint.

---

## 📂 Project Structure

```
MovieRecommenderSystem/
│── app.py                # Flask app entry point
│── models.py             # SQLAlchemy model with pgvector
│── ingest_embeddings.py  # Script to load movies + embeddings
│── templates/
│   └── index.html        # Frontend for search
│── imdb_data/            # Folder for CSV dataset (IMDB dump)
│── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/MovieRecommenderSystem.git
cd MovieRecommenderSystem
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** should include:

```txt
flask
sqlalchemy
psycopg2-binary
pgvector
sentence-transformers
tqdm
langchain-google-genai
python-dotenv
```

---

## 🗄️ Database Setup

### Install PostgreSQL with pgvector

On **macOS**:

```bash
brew install postgresql@15
brew install pgvector
brew services start postgresql@15
```

Enable pgvector in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Create Database

```bash
createdb moviesdb
```

### Initialize Tables

```bash
python -m MovieRecommenderSystem.models
```

---

## 📥 Data Ingestion

Download the [IMDB dataset](https://datasets.imdbws.com/), preprocess it to CSV (`movies_1990_2025_imdb.csv`), then run:

```bash
python -m MovieRecommenderSystem.ingest_embeddings
```

This will:

* Load movie metadata into Postgres
* Encode titles + genres into embeddings
* Store them in the `embedding` column

---

## 🚀 Running the App

### Flask server

```bash
export FLASK_APP=MovieRecommenderSystem.app
flask run
```

Server runs at:

```
http://127.0.0.1:5000
```

### Endpoints

* **API search**:

  ```
  GET /search?prompt=show me sci-fi movies&top_k=5
  ```

Response:

```json
{
  "original_query": "show me sci-fi movies",
  "refined_genre": "sci-fi",
  "results": [
    {"title": "Interstellar", "year": 2014, "genres": "Sci-Fi", "average_rating": 8.6},
    {"title": "The Matrix", "year": 1999, "genres": "Sci-Fi", "average_rating": 8.7}
  ]
}
```

---

## 🧠 How It Works

1. User enters a **natural language query**.
2. **Gemini LLM** extracts/cleans the genre.
3. **Sentence Transformer** encodes the refined query into embeddings.
4. **pgvector** performs similarity search with `<->` operator.
5. Results are ranked by **semantic similarity + rating**.

---

##
