from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load sample texts (you can replace these with your own)
texts = []
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(line)

# Create FAISS index
dim = 384  # Dimension of MiniLM model embeddings
index = faiss.IndexFlatL2(dim)

# Function to get Hugging Face embedding
def get_embedding(text: str):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype('float32')

# Generate embeddings for the sample texts
embeddings = np.array([get_embedding(text) for text in texts])
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Get embedding for the query
    query_embedding = get_embedding(query_text)

    # Perform FAISS search
    distances, indices = index.search(np.array([query_embedding]), k=3)

    # Get matched texts based on the indices
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
