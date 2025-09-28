"""script to download the imdb data from https://datasets.imdbws.com/"""
import os
import requests

BASE_URL = "https://datasets.imdbws.com/"
FILES = ["title.basics.tsv.gz", "title.ratings.tsv.gz"]

for file in FILES:
    """download the file if it does not exist"""
    url = BASE_URL + file
    path = os.path.join("MovieRecommenderSystem/imdb_data", file)
    if not os.path.exists(path):
        print(f"Downloading {file}...")
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file} to {path}")
    else:
        print(f"{file} already exists at {path}")

print("All files downloaded.")