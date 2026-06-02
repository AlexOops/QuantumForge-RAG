import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_bot import RagBot


def main() -> None:
    bot = RagBot()

    print("QuantumForge RAG Bot")
    print("Введите вопрос. Для выхода: exit / quit")
    print()

    while True:
        query = input("Q: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break

        if not query:
            continue

        result = bot.answer(query)

        print()
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

        print()


if __name__ == "__main__":
    main()
