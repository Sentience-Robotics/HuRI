# ingestion.py
import argparse
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

USER_ID_FILE = os.path.expanduser("~/.huri_user_id")


def get_user_id(provided_id: str = None) -> str:
    """Use provided ID, or load from file, or generate new one."""
    if provided_id:
        return provided_id
    if os.path.exists(USER_ID_FILE):
        with open(USER_ID_FILE) as f:
            return f.read().strip()
    new_id = str(uuid.uuid4())
    with open(USER_ID_FILE, "w") as f:
        f.write(new_id)
    return new_id


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--user-id", type=str, default=None, help="User ID (reads from ~/.huri_user_id if not provided)")
    parser.add_argument("--collection", type=str, default="documents")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    args = parser.parse_args()

    user_id = get_user_id(args.user_id)
    print(f"Ingesting for user_id: {user_id}")

    client = QdrantClient(url=args.qdrant_url)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    collections = [c.name for c in client.get_collections().collections]
    if args.collection not in collections:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Created collection: {args.collection}")

    docs = [
        {"text": "The company budget for 2026 is 2 million euros.", "source": "budget.pdf"},
        {"text": "The project deadline is June 15th 2026.", "source": "planning.pdf"},
        {"text": "The team consists of 5 developers and 2 designers.", "source": "team.pdf"},
        {"text": "The main office is located in Paris, France.", "source": "info.pdf"},
    ]

    points = []
    for doc in docs:
        vector = model.encode(doc["text"], normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "user_id": user_id,
            },
        ))

    client.upsert(collection_name=args.collection, points=points)
    print(f"Ingested {len(points)} documents for user {user_id}")


if __name__ == "__main__":
    main()