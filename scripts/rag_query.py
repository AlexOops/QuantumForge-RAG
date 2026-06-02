import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_bot import RagBot


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask QuantumForge RAG Bot")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    bot = RagBot()
    result = bot.answer(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result["answer"])

    if result["sources"]:
        print()
        print("Sources:")

        for source in result["sources"]:
            print(
                f"- {source['title']} | "
                f"{source['source_file']} | "
                f"chunk_id={source['chunk_id']} | "
                f"score={source['score']}"
            )


if __name__ == "__main__":
    main()
