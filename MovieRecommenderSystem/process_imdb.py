"""preprocess the imdb dataset."""
import os
import pandas as pd

basics_path = os.path.join("MovieRecommenderSystem/imdb_data", "title.basics.tsv.gz")
ratings_path = os.path.join("MovieRecommenderSystem/imdb_data", "title.ratings.tsv.gz")
output_path = os.path.join("MovieRecommenderSystem/imdb_data", "imdb_data.csv")

def load_basics(path):
    """load the basics data"""
    df = pd.read_csv(path, sep="\t", na_values="\\N", dtype={"tconst": str})
    df = df[["tconst", "primaryTitle", "startYear", "genres"]]
    df = df.dropna(subset=["startYear", "genres"])
    df["startYear"] = df["startYear"].astype(int)
    df = df[df["startYear"] >= 1900]
    return df

def load_ratings(path):
    """load the ratings data"""
    df = pd.read_csv(path, sep="\t", na_values="\\N", dtype={"tconst": str})
    df = df[["tconst", "averageRating", "numVotes"]]
    df = df.dropna(subset=["averageRating", "numVotes"])
    df["averageRating"] = df["averageRating"].astype(float)
    df["numVotes"] = df["numVotes"].astype(int)
    return df

def preprocess_data(basics_path, ratings_path, output_path):
    """preprocess the imdb data"""
    basics_df = load_basics(basics_path)
    ratings_df = load_ratings(ratings_path)
    
    # Merge datasets on tconst
    df = pd.merge(basics_df, ratings_df, on="tconst")
    
    # Filter for movies with at least 10 votes
    df = df[df["numVotes"] >= 10]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(basics_path, ratings_path, output_path)
    print("All files processed.")