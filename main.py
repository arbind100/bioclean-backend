from fastapi import FastAPI
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import os

app = FastAPI()

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define request model
class QueryRequest(BaseModel):
    query: str

# Dummy FAISS index (for now)
dimension = 512
index = faiss.IndexFlatL2(dimension)

# Dummy vectors (random)
for _ in range(5):
    vec = np.random.rand(dimension).astype('float32')
    index.add(np.array([vec]))

@app.get("/")
def read_root():
    return {"message": "FastAPI app with FAISS and OpenAI is running."}

@app.post("/query")
def query_vector(request: QueryRequest):
    # Generate dummy query vector (in real app, use embeddings)
    query_vec = np.random.rand(dimension).astype('float32')
    D, I = index.search(np.array([query_vec]), k=3)
    return {
        "your_query": request.query,
        "top_matches_indices": I.tolist(),
        "distances": D.tolist()
    }
    
