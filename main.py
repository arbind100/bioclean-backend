from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer  # Import sentence transformers

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data (Replace with real data you want to use)
texts = [
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "The mitochondria is the powerhouse of the cell.",
    "DNA carries genetic information in living organisms.",
    "Climate change is a result of increased greenhouse gases.",
    "Enzymes are biological catalysts that speed up reactions."
]

# Create FAISS index
dim = 384  # Sentence transformers use a 384-dimensional embedding
index = faiss.IndexFlatL2(dim)

# Function to get sentence embedding using SentenceTransformers
def get_sentence_embedding(text: str):
    embedding = model.encode(text)  # Generate embedding
    return np.array(embedding, dtype='float32')

# Generate embeddings for the sample texts
embeddings = np.array([get_sentence_embedding(text) for text in texts])
index.add(embeddings)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_text = body.get("query", "")

    # Get embedding for the query
    query_embedding = get_sentence_embedding(query_text)

    # Perform FAISS search
    distances, indices = index.search(np.array([query_embedding]), k=3)

    # Get matched texts based on the indices
    matched_texts = [texts[i] for i in indices[0]]

    return JSONResponse(content={
        "query": query_text,
        "top_matches": matched_texts,
        "distances": distances[0].tolist()
    })
