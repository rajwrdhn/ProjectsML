"""Postgresql model vector extension support."""
from sqlalchemy import create_engine, Column, Integer, Text, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
import os
import dotenv
dotenv.load_dotenv()
PG_USER = os.getenv("PG_USER")

DATA_BASE_URL = f"postgresql+psycopg2://{PG_USER}@localhost:5432/moviesdb"

engine = create_engine(DATA_BASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Movie(Base):
    """Movie model with vector support."""
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, index=True)
    year = Column(Integer, index=True)
    genres = Column(Text, index=True)
    average_rating = Column(Float, index=True)
    num_votes = Column(Integer, index=True)
    embedding = Column(Vector(384))

Base.metadata.create_all(bind=engine)