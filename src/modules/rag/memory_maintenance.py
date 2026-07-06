"""Periodic memory maintenance: decay-based pruning + consolidation.

Run every N days (cron/systemd timer):
    python -m src.modules.rag.memory_maintenance
Strong memories are kept, weak ones are merged into a consolidated memory,
dead ones are deleted. Decay itself is computed lazily at query time in
rag.py; this job only prunes and compresses.
"""
import argparse
import json
import uuid
from collections import defaultdict
from datetime import datetime

import httpx
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

try:
    from .qdrant_utils import make_qdrant_client
except ImportError:
    from qdrant_utils import make_qdrant_client

DELETE_BELOW = 0.05
CONSOLIDATE_BELOW = 0.30
HALF_LIFE_DAYS = 5.0


def strength(payload: dict) -> float:
    """Query-independent strength: recency * importance (no relevance term)."""
    importance = payload.get("importance", 3)
    half_life = max(HALF_LIFE_DAYS * (importance / 5.0), 0.5)
    try:
        last = datetime.fromisoformat(payload.get("last_accessed") or payload["created_at"])
        age_days = (datetime.now() - last).total_seconds() / 86400.0
    except Exception:
        age_days = 0.0
    recency = 0.5 ** (age_days / half_life)
    return recency * (importance / 10.0)


def embed(client: httpx.Client, url: str, model: str, text: str) -> list[float]:
    r = client.post(f"{url}/v1/embeddings", json={"model": model, "input": text})
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def llm(client: httpx.Client, url: str, model: str, prompt: str) -> str:
    r = client.post(f"{url}/api/chat", json={
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": 300},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="conversations")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embedding-model", default="bge-large-en-v1.5-gguf-Q4_K_M")
    ap.add_argument("--llm-model", default="mistral:7b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qdrant = make_qdrant_client(args.qdrant_url)
    http = httpx.Client(timeout=180.0)

    points, offset = [], None
    while True:
        batch, offset = qdrant.scroll(collection_name=args.collection, limit=200,
                                      offset=offset, with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            break
    print(f"{len(points)} memories in '{args.collection}'")

    to_delete, weak_by_user = [], defaultdict(list)
    for p in points:
        s = strength(p.payload)
        if s < DELETE_BELOW:
            to_delete.append(p)
        elif s < CONSOLIDATE_BELOW:
            weak_by_user[p.payload.get("_user_id", "anonymous")].append(p)
    print(f"delete: {len(to_delete)}, consolidate candidates: {sum(map(len, weak_by_user.values()))}")

    if args.dry_run:
        return

    for user, weak in weak_by_user.items():
        if len(weak) < 3:
            continue  # not worth merging yet; keep decaying
        texts = [p.payload["text"] for p in weak]
        merged = llm(http, args.ollama_url, args.llm_model,
                     "These are old memories about conversations with the same person. "
                     "Merge them into a single 3-5 sentence memory keeping only durable "
                     "facts, preferences and recurring themes. Drop one-off small talk.\n\n"
                     + "\n---\n".join(texts)).strip()
        vec = embed(http, args.ollama_url, args.embedding_model, merged)
        now = datetime.now().isoformat()
        imp = min(max(p.payload.get("importance", 3) for p in weak) + 1, 10)
        qdrant.upsert(collection_name=args.collection, points=[PointStruct(
            id=str(uuid.uuid4()), vector=vec, payload={
                "text": merged, "_user_id": user, "type": "conversation_consolidated",
                "created_at": now, "last_accessed": now, "access_count": 0,
                "importance": imp,
            })])
        to_delete.extend(weak)
        print(f"[{user}] consolidated {len(weak)} → 1 (importance={imp})")

    if to_delete:
        qdrant.delete(collection_name=args.collection,
                      points_selector=PointIdsList(points=[p.id for p in to_delete]))
        print(f"deleted {len(to_delete)} points")


if __name__ == "__main__":
    main()