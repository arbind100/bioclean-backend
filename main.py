from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import json

app = FastAPI()

# Sample data
texts = [
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information in living organisms.",
    "Climate change is a result of increased greenhouse gases.",
    "Enzymes are biological catalysts that speed up reactions."
]

# Dummy embeddings (in practice, use real embeddings)
np.random.seed(42)
dim = 1536
embeddings = np.random.rand(len(texts), dim).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Simulate query embedding
    query_embedding = np.random.rand(1, dim).astype('float32')  # Dummy

    distances, indices = index.search(query_embedding, k=3)

    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })

