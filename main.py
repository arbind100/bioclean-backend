from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load texts from file
with open("data.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Create FAISS index
dim = 384  # Dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dim)

# Generate embeddings and add to index
embeddings = model.encode(texts, convert_to_numpy=True)
index.add(np.array(embeddings).astype("float32"))

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Get embedding for query
    query_embedding = model.encode([query_text], convert_to_numpy=True).astype("float32")

    # Search in index
    distances, indices = index.search(query_embedding, k=3)
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
