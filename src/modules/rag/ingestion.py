import re
import argparse
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from semantic_chunker import SemanticChunker

USER_ID_FILE = os.path.expanduser("~/.huri_user_id")

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = []
    for s in sentences:
        parts = s.split("\n\n")
        result.extend(parts)
    return [s.strip() for s in result if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Fallback: fixed-size chunking by sentences.
    Used when --chunking=fixed.
    """
    sentences = _split_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            overlap_words = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_words += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_words >= overlap:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_words

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        pass

    print("ERROR: Install a PDF library: pip install pymupdf  OR  pip install pypdf")
    sys.exit(1)


def get_user_id(provided_id: str = None) -> str:
    if provided_id:
        return provided_id
    if os.path.exists(USER_ID_FILE):
        with open(USER_ID_FILE) as f:
            uid = f.read().strip()
            if uid:
                return uid
    new_id = str(uuid.uuid4())
    with open(USER_ID_FILE, "w") as f:
        f.write(new_id)
    print(f"Generated new user_id: {new_id}")
    return new_id


def ensure_collection(client: QdrantClient, collection: str, vector_size: int):
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection}")


def ingest_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    collection: str,
    chunks: list[str],
    _user_id: str,
    source: str,
    doc_type: str = "document",
):
    """Embed chunks and upsert into Qdrant."""
    points = []
    timestamp = datetime.now().isoformat()

    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk, normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "_user_id": _user_id,
                "source": source,
                "type": doc_type,
                "chunk_index": i,
                "timestamp": timestamp,
            },
        ))

    if points:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection, points=batch)

    return len(points)


def chunk_strat(text: str, args, model: SentenceTransformer) -> list[str]:
    """Pick the right chunking strategy based on args."""
    if args.chunking == "semantic":
        chunker = SemanticChunker(
            model=model,
            strategy=args.semantic_strategy,
        )
        return chunker.chunk(text)
    else:
        return chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)


def cmd_pdf(args, client, model, _user_id):
    """Ingest PDF files."""
    files = []
    for path in args.files:
        p = Path(path)
        if p.is_dir():
            files.extend(p.glob("**/*.pdf"))
        elif p.suffix.lower() == ".pdf":
            files.append(p)
        else:
            print(f"Skipping non-PDF: {path}")

    if not files:
        print("No PDF files found.")
        return

    sample = model.encode("test", normalize_embeddings=True)
    ensure_collection(client, args.collection, len(sample))

    total = 0
    for pdf_path in files:
        print(f"\nProcessing: {pdf_path}")
        text = extract_text_from_pdf(str(pdf_path))

        if not text.strip():
            print(f"  WARNING: No text extracted from {pdf_path}")
            continue

        chunks = chunk_strat(text, args, model)
        count = ingest_chunks(
            client, model, args.collection, chunks,
            _user_id, source=pdf_path.name, doc_type="pdf",
        )
        print(f"  -> {count} chunks ingested")
        total += count

    print(f"\nDone. Total: {total} chunks from {len(files)} PDF(s)")


def cmd_text(args, client, model, _user_id):
    """Ingest text files."""
    sample = model.encode("test", normalize_embeddings=True)
    ensure_collection(client, args.collection, len(sample))

    total = 0
    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"\nProcessing: {file_path}")
        text = p.read_text(encoding="utf-8")

        if not text.strip():
            print(f"  WARNING: File is empty: {file_path}")
            continue

        chunks = chunk_strat(text, args, model)
        count = ingest_chunks(
            client, model, args.collection, chunks,
            _user_id, source=p.name, doc_type="text",
        )
        print(f"  -> {count} chunks ingested")
        total += count

    print(f"\nDone. Total: {total} chunks from {len(args.files)} file(s)")


def cmd_write(args, client, model, _user_id):
    """Write text interactively and ingest it."""
    title = args.title or f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Write your text below (title: '{title}')")
    print("Press Ctrl+D (Linux/Mac) or Ctrl+Z then Enter (Windows) when done.")
    print("-" * 40)

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    text = "\n".join(lines).strip()

    if not text:
        print("Nothing to ingest.")
        return

    print(f"\n{'-' * 40}")
    print(f"Received {len(text)} characters")

    sample = model.encode("test", normalize_embeddings=True)
    ensure_collection(client, args.collection, len(sample))

    chunks = chunk_strat(text, args, model)
    count = ingest_chunks(
        client, model, args.collection, chunks,
        _user_id, source=title, doc_type="manual",
    )

    print(f"Done. Ingested {count} chunks as '{title}'")


def cmd_list(args, client, model, _user_id):
    """List what's in the database for this user."""

    try:
        info = client.get_collection(args.collection)
        print(f"Collection: {args.collection}")
        print(f"Total points: {info.points_count}")
    except Exception:
        print(f"Collection '{args.collection}' doesn't exist.")
        return

    results = client.scroll(
        collection_name=args.collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="_user_id", match=MatchValue(value=_user_id)),
        ]),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    points = results[0]
    if not points:
        print(f"No documents found for user {_user_id}")
        return

    sources = {}
    for p in points:
        source = p.payload.get("source", "unknown")
        doc_type = p.payload.get("type", "unknown")
        if source not in sources:
            sources[source] = {"count": 0, "type": doc_type}
        sources[source]["count"] += 1

    print(f"\nDocuments for user {_user_id}:")
    print(f"{'Source':<40} {'Type':<10} {'Chunks':<8}")
    print("-" * 60)
    for source, info in sorted(sources.items()):
        print(f"{source:<40} {info['type']:<10} {info['count']:<8}")
    print(f"\nTotal: {len(points)} chunks across {len(sources)} sources")


def cmd_delete(args, client, model, _user_id):
    """Delete documents by source name."""

    if not args.source:
        print("Specify --source to delete. Use 'list' command to see sources.")
        return

    filter_conditions = [
        FieldCondition(key="_user_id", match=MatchValue(value=_user_id)),
        FieldCondition(key="source", match=MatchValue(value=args.source)),
    ]

    client.delete(
        collection_name=args.collection,
        points_selector=Filter(must=filter_conditions),
    )
    print(f"Deleted all chunks from source '{args.source}' for user {_user_id}")



def main():
    parser = argparse.ArgumentParser(description="HuRI RAG Ingestion Tool")
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--collection", type=str, default="documents")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in words (fixed mode)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in words (fixed mode)")
    parser.add_argument("--chunking", type=str, default="fixed",
                        choices=["semantic", "fixed"],
                        help="Chunking strategy: 'semantic' (default) or 'fixed'")
    parser.add_argument("--semantic-strategy", type=str, default="percentile",
                        choices=["percentile", "threshold", "stddev"],
                        help="Semantic chunking strategy (default: percentile)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pdf = subparsers.add_parser("pdf", help="Ingest PDF files")
    p_pdf.add_argument("files", nargs="+", help="PDF files or directories")

    p_text = subparsers.add_parser("text", help="Ingest text files (.txt, .md)")
    p_text.add_argument("files", nargs="+", help="Text files")

    p_write = subparsers.add_parser("write", help="Write text interactively")
    p_write.add_argument("--title", type=str, default=None, help="Title/source name")

    p_list = subparsers.add_parser("list", help="List ingested documents")

    p_delete = subparsers.add_parser("delete", help="Delete documents by source")
    p_delete.add_argument("--source", type=str, required=True, help="Source name to delete")

    args = parser.parse_args()

    _user_id = get_user_id(args._user_id)
    print(f"User: {_user_id}")

    client = QdrantClient(url=args.qdrant_url)
    model = SentenceTransformer(args.embedding_model)

    commands = {
        "pdf": cmd_pdf,
        "text": cmd_text,
        "write": cmd_write,
        "list": cmd_list,
        "delete": cmd_delete,
    }
    commands[args.command](args, client, model, _user_id)


if __name__ == "__main__":
    """
    Ingestion tool for HuRI RAG.

    Usage:
        # Ingest a PDF
        python ingestion.py pdf report.pdf

        # Ingest multiple PDFs
        python ingestion.py pdf doc1.pdf doc2.pdf doc3.pdf

        # Ingest a whole folder of PDFs 
        # TODO: To verify and to add the support of hole paths
        python ingestion.py pdf ./my_documents/

        # Write text interactively (type, then Ctrl+D to save)
        python ingestion.py write --title "My meeting notes"

        # Ingest a text file
        python ingestion.py text notes.txt story.md

        # Specify a user ID (otherwise reads from ~/.huri_user_id)
        python ingestion.py --user-id "abc-123" pdf report.pdf

        # Use a different collection
        python ingestion.py --collection "my_docs" pdf report.pdf

        # Use a different ingestion strategy
        python src/modules/rag/ingestion.py --chunking semantic --semantic-strategy threshold pdf "EN.pdf"

    """
    main()
