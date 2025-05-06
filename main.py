from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Sample texts
texts = [
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information in living organisms.",
    "Climate change is a result of increased greenhouse gases.",
    "Enzymes are biological catalysts that speed up reactions."
]

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set dimension
dim = 384

# Create FAISS index
index = faiss.IndexFlatL2(dim)

# Generate embeddings for all texts
embeddings = np.array([model.encode(text) for text in texts], dtype='float32')
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Get query embedding
    query_embedding = np.array([model.encode(query_text)], dtype='float32')

    # Search FAISS index
    distances, indices = index.search(query_embedding, k=3)

    # Fetch top matches
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
