from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import numpy as np
import faiss
import pickle

app = FastAPI()

# Load FAISS index
faiss_index = faiss.read_index("tds_discourse_index.faiss")

# Load metadata
with open("tds_discourse_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ✅ Safe loading of precomputed embeddings
embeddings = []
with open("output.jsonl", "r") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            if "embedding" not in data:
                print(f"[Line {i}] Missing 'embedding' key: {data}")
                continue
            embeddings.append(data["embedding"])
        except json.JSONDecodeError as e:
            print(f"[Line {i}] JSON decode error: {e}")

embedding_matrix = np.array(embeddings).astype("float32")

class QueryRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/query")
def query_docs(data: QueryRequest):
    from openai import OpenAI

    client = OpenAI()

    # Get query embedding
    response = client.embeddings.create(
        input=data.query,
        model="text-embedding-3-small"
    )
    print("Embedding response:", response)

    query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    # Search FAISS index
    top_k = 3
    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        result = {
            "content": metadata[idx]["text"],
            "source": metadata[idx].get("source", "Unknown")
        }
        results.append(result)

    return {"results": results}
