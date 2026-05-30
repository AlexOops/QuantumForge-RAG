import json
import re
from pathlib import Path


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
TERMS_MAP_PATH = Path("terms_map.json")
MIN_DOCUMENTS_COUNT = 30


def build_term_pattern(term: str) -> re.Pattern:
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])"
    )


def main() -> None:
    terms_map = json.loads(TERMS_MAP_PATH.read_text(encoding="utf-8"))
    documents = list(KNOWLEDGE_BASE_DIR.glob("*.md"))

    print(f"Documents found: {len(documents)}")

    if len(documents) < MIN_DOCUMENTS_COUNT:
        raise RuntimeError(f"Expected at least {MIN_DOCUMENTS_COUNT} documents")

    leaked_terms = []

    for document in documents:
        content = document.read_text(encoding="utf-8")

        for original_term in terms_map:
            pattern = build_term_pattern(original_term)

            if pattern.search(content):
                leaked_terms.append((document.name, original_term))

    if leaked_terms:
        print("Original terms found:")

        for filename, term in leaked_terms[:100]:
            print(f"- {filename}: {term}")

        raise RuntimeError("Knowledge base still contains original terms")

    print("Knowledge base validation passed")


if __name__ == "__main__":
    main()
