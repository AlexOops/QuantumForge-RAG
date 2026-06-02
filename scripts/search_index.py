import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("vector_index/faiss")
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
INFO_PATH = INDEX_DIR / "index_info.json"


def load_index_data():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks metadata not found: {CHUNKS_PATH}")

    if not INFO_PATH.exists():
        raise FileNotFoundError(f"Index info not found: {INFO_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

    return index, chunks, info


def search(query: str, top_k: int):
    index, chunks, info = load_index_data()

    model = SentenceTransformer(info["model_name"])

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "score": float(score),
                "chunk": chunk,
            }
        )

    return results


def format_result(result: dict, position: int) -> str:
    chunk = result["chunk"]
    text = chunk["text"].replace("\n", " ")
    preview = text[:700]

    if len(text) > 700:
        preview += "..."

    return (
        f"\n[{position}] score={result['score']:.4f}\n"
        f"title: {chunk['title']}\n"
        f"source: {chunk['source_path']}\n"
        f"chunk_id: {chunk['chunk_id']}\n"
        f"position: {chunk['start_char']}..{chunk['end_char']}\n"
        f"text: {preview}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search FAISS vector index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return")

    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"Query: {args.query}")
    print(f"Top K: {args.top_k}")

    for index, result in enumerate(results, start=1):
        print(format_result(result, index))


if __name__ == "__main__":
    main()
