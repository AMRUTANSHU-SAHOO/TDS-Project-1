from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import os
import numpy as np
import faiss
import pickle
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index
faiss_index = faiss.read_index("tds_discourse_index.faiss")

# Load metadata
with open("tds_discourse_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load JSONL embeddings (not used directly in this endpoint)
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

# Request schema
class QueryRequest(BaseModel):
    query: str

# Endpoint
@app.post("/query")
async def query_handler(request: QueryRequest):
    user_query = request.query

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # Generate embedding
        response = client.embeddings.create(
            input=user_query,
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Embedding failed: {str(e)}"},
            status_code=500,
        )

    # Search FAISS
    top_k = 3
    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({
            "content": meta.get("content", "No content"),
            "source": meta.get("url", "Unknown")
        })

    return {
        "answer": results[0]["content"] if results else "No relevant answer found.",
        "links": [
            {"url": r["source"], "text": r["content"][:100] + "..."}
            for r in results if r["source"] != "Unknown"
        ]
    }
