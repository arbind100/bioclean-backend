from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data from data.txt
with open("data.txt", "r") as f:
    texts = [line.strip() for line in f if line.strip()]

# Get embeddings using the new OpenAI SDK
def get_openai_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"OpenAI embedding error: {e}")
        return None

# Create FAISS index
dim = 1536
index = faiss.IndexFlatL2(dim)

# Generate and add embeddings
embedding_list = []
valid_texts = []
for text in texts:
    embedding = get_openai_embedding(text)
    if embedding is not None:
        embedding_list.append(embedding)
        valid_texts.append(text)

if embedding_list:
    index.add(np.array(embedding_list))

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    query_embedding = get_openai_embedding(query_text)
    if query_embedding is None or index.ntotal == 0:
        return JSONResponse(content={"error": "Embedding generation or indexing failed"}, status_code=500)

    distances, indices = index.search(np.array([query_embedding]), k=3)
    matches = [valid_texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matches,
        "distances": distances[0].tolist()
    })
