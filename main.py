from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import os
from openai import OpenAI  # Updated import

# Initialize FastAPI app
app = FastAPI()

# Load OpenAI API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample data
texts = [
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information in living organisms.",
    "Climate change is a result of increased greenhouse gases.",
    "Enzymes are biological catalysts that speed up reactions."
]

# Create FAISS index
dim = 1536
index = faiss.IndexFlatL2(dim)

# Updated OpenAI embedding function
def get_openai_embedding(text: str):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype='float32')
    except Exception as e:
        print("OpenAI embedding error:", e)
        return np.zeros(dim, dtype='float32')  # fallback

# Generate embeddings
embeddings = np.array([get_openai_embedding(text) for text in texts])
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    query_embedding = get_openai_embedding(query_text)
    distances, indices = index.search(np.array([query_embedding]), k=3)
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
