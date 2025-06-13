from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
import faiss
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify their domain if known
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    documents: Optional[List[str]] = None

@app.api_route("/query", methods=["GET", "POST"])
async def query_docs(request: Request):
    if request.method == "GET":
        return JSONResponse(content={"message": "Use POST method with a JSON body"}, status_code=200)

    # POST request
    try:
        body = await request.json()
        data = QueryRequest(**body)
    except Exception as e:
        return JSONResponse(content={"error": f"Invalid request format: {str(e)}"}, status_code=400)

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
