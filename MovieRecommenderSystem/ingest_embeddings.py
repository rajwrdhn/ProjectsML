"""Ingest the movie embeddings into the vector database."""
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from models import SessionLocal, Movie

CSV_PATH = os.path.join("MovieRecommenderSystem/imdb_data", "imdb_data.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

def ingest_embeddings():
    """Ingest embeddings into the database in batches"""
    df = pd.read_csv(CSV_PATH)
    print(f"Read {len(df)} rows from {CSV_PATH}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Using model {MODEL_NAME} ({model.get_sentence_embedding_dimension()} dims)")

    session = SessionLocal()
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        texts = (batch["primaryTitle"].fillna("") + " " + batch["genres"].fillna("")).tolist()
        embeddings = model.encode(texts, show_progress_bar=False)

        movies = []
        for j, row in batch.iterrows():
            movie = Movie(
                title=row["primaryTitle"],
                year=int(row["startYear"]) if not pd.isna(row["startYear"]) else None,
                genres=row["genres"],
                average_rating=float(row["averageRating"]) if not pd.isna(row["averageRating"]) else None,
                num_votes=int(row["numVotes"]) if not pd.isna(row["numVotes"]) else None,
                embedding=embeddings[j - i].tolist()
            )
            movies.append(movie)

        session.bulk_save_objects(movies)  # faster than add()
        session.commit()
        print(f"Inserted {i + len(batch)} / {len(df)}")

    session.close()
    print("All embeddings ingested.")

if __name__ == "__main__":
    ingest_embeddings()
