import os
import faiss
import pickle
from flask import Flask, request, jsonify
from openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set up FAISS index path
FAISS_INDEX_PATH = "vector_index.faiss"
DOCS_PATH = "docs.pkl"

# Load embedding model
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Load or create FAISS index
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
        return index, docs
    else:
        index = faiss.IndexFlatL2(1536)  # 1536 for OpenAI embeddings
        return index, []

faiss_index, stored_texts = load_faiss_index()

# Save FAISS index and docs
def save_faiss_index():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(stored_texts, f)

# Route to add data to vector index
@app.route("/add", methods=["POST"])
def add_data():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    embedding = embedding_model.embed_query(text)
    faiss_index.add([embedding])
    stored_texts.append(text)
    save_faiss_index()

    return jsonify({"message": "Data added to index."})

# Route to search
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_embedding = embedding_model.embed_query(query)
    D, I = faiss_index.search([query_embedding], k=3)

    results = [stored_texts[i] for i in I[0] if i < len(stored_texts)]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
