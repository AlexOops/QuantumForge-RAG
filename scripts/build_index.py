import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_DIR = Path("vector_index/faiss")

INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
INFO_PATH = INDEX_DIR / "index_info.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100


def read_markdown_documents() -> list[dict]:
    documents = []

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        title = extract_title(content, path)

        documents.append(
            {
                "source_path": str(path),
                "source_file": path.name,
                "title": title,
                "content": content,
            }
        )

    return documents


def extract_title(content: str, path: Path) -> str:
    first_line = content.splitlines()[0].strip() if content.splitlines() else ""

    if first_line.startswith("# "):
        return first_line.replace("# ", "", 1).strip()

    return path.stem.replace("-", " ").title()


def split_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []

    for document in documents:
        text = document["content"]
        split_texts = splitter.split_text(text)

        search_start = 0

        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue

            start_char = text.find(chunk_text[:80], search_start)

            if start_char == -1:
                start_char = text.find(chunk_text[:80])

            if start_char == -1:
                start_char = 0

            end_char = start_char + len(chunk_text)
            search_start = end_char

            chunk_id = f"{Path(document['source_file']).stem}::chunk-{chunk_index:04d}"

            chunks.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_index,
                    "source_path": document["source_path"],
                    "source_file": document["source_file"],
                    "title": document["title"],
                    "start_char": start_char,
                    "end_char": end_char,
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text,
                }
            )

    return chunks


def build_embeddings(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def save_index(embeddings: np.ndarray, chunks: list[dict], documents_count: int, duration_seconds: float) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embedding_dim = embeddings.shape[1]

    # Так как embeddings нормализованы, IndexFlatIP работает как cosine similarity.
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    info = {
        "model_name": MODEL_NAME,
        "embedding_dimension": embedding_dim,
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "documents_count": documents_count,
        "chunks_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine similarity through normalized embeddings + inner product",
        "index_path": str(INDEX_PATH),
        "chunks_path": str(CHUNKS_PATH),
        "build_time_seconds": round(duration_seconds, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    INFO_PATH.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    started_at = time.perf_counter()

    documents = read_markdown_documents()

    if not documents:
        raise RuntimeError("No markdown documents found in knowledge_base/")

    print(f"Documents found: {len(documents)}")

    chunks = split_documents(documents)

    if not chunks:
        raise RuntimeError("No chunks created")

    print(f"Chunks created: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(model, chunks)

    duration_seconds = time.perf_counter() - started_at

    save_index(
        embeddings=embeddings,
        chunks=chunks,
        documents_count=len(documents),
        duration_seconds=duration_seconds,
    )

    print("Index created successfully")
    print(f"Index path: {INDEX_PATH}")
    print(f"Chunks path: {CHUNKS_PATH}")
    print(f"Info path: {INFO_PATH}")
    print(f"Build time: {duration_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
