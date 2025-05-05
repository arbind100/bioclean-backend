from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import openai
import os

# Initialize FastAPI app
app = FastAPI()

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sample data (Replace with real data you want to use)
texts = [
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information in living organisms.",
    "Climate change is a result of increased greenhouse gases.",
    "Enzymes are biological catalysts that speed up reactions."
]

# Create FAISS index
dim = 1536  # OpenAI's model uses 1536 dimensions for embeddings
index = faiss.IndexFlatL2(dim)

# Function to get OpenAI embeddings
def get_openai_embedding(text: str):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use OpenAI's Ada model for embeddings
        input=text
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding, dtype='float32')

# Generate embeddings for the sample texts
embeddings = np.array([get_openai_embedding(text) for text in texts])
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Get embedding for the query
    query_embedding = get_openai_embedding(query_text)

    # Perform FAISS search
    distances, indices = index.search(np.array([query_embedding]), k=3)

    # Get matched texts based on the indices
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
