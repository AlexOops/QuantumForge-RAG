import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("vector_index/faiss")
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
INFO_PATH = INDEX_DIR / "index_info.json"

DEFAULT_TOP_K = 4
MIN_RELEVANCE_SCORE = 0.32


@dataclass
class SearchResult:
    score: float
    chunk: dict[str, Any]


class RagBot:
    def __init__(self) -> None:
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(self.info["model_name"])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results: list[SearchResult] = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position == -1:
                continue

            results.append(
                SearchResult(
                    score=float(score),
                    chunk=self.chunks[index_position],
                )
            )

        return results

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        results = self.search(query, top_k)

        if not results or results[0].score < MIN_RELEVANCE_SCORE:
            return {
                "answer": "Я не знаю. В базе знаний нет достаточно релевантного фрагмента для ответа на этот вопрос.",
                "reasoning_steps": [
                    "Я преобразовал вопрос в embedding.",
                    "Я выполнил поиск по FAISS-индексу.",
                    "Лучший найденный фрагмент оказался недостаточно релевантным.",
                    "Поэтому я не буду придумывать ответ без опоры на базу знаний.",
                ],
                "sources": [],
                "prompt": self.build_prompt(query, results),
            }

        prompt = self.build_prompt(query, results)

        if os.getenv("RAG_LLM_PROVIDER") == "openai":
            answer = self.generate_with_openai(prompt)
        else:
            answer = self.generate_offline_answer(query, results)

        return {
            "answer": answer,
            "reasoning_steps": [
                "Я преобразовал вопрос в embedding той же моделью, которая использовалась при индексации.",
                "Я нашёл ближайшие чанки в FAISS-индексе.",
                "Я проверил релевантность найденных фрагментов.",
                "Я сформировал ответ только на основе найденного контекста.",
            ],
            "sources": self.format_sources(results),
            "prompt": prompt,
        }

    def build_prompt(self, query: str, results: list[SearchResult]) -> str:
        context = "\n\n".join(
            [
                (
                    f"[SOURCE {index}]\n"
                    f"title: {result.chunk['title']}\n"
                    f"file: {result.chunk['source_file']}\n"
                    f"chunk_id: {result.chunk['chunk_id']}\n"
                    f"score: {result.score:.4f}\n"
                    f"text:\n{result.chunk['text']}"
                )
                for index, result in enumerate(results, start=1)
            ]
        )

        return f"""
SYSTEM:
Ты корпоративный RAG-ассистент QuantumForge Software.

Правила:
1. Отвечай только на основе CONTEXT.
2. Если в CONTEXT нет ответа, напиши: "Я не знаю".
3. Не выдумывай факты.
4. В ответе указывай источники: название документа и chunk_id.
5. Используй короткие проверяемые шаги рассуждения, без скрытых догадок.

FEW-SHOT EXAMPLES:

Q: Who is Varn Kaldor?
A:
Шаги:
1. Я нашёл фрагменты, где упоминается Varn Kaldor.
2. В контексте указано, что он связан с Noctari и Aurelian Dominion.
3. Отвечаю только по найденным данным.

Ответ:
Varn Kaldor — персонаж из базы знаний, связанный с Noctari и Aurelian Dominion.
Источник: документ Varn Kaldor.

Q: What is the Void Core?
A:
Шаги:
1. Я нашёл фрагменты по термину Void Core.
2. В контексте указано, что это крупная технологическая/военная конструкция.
3. Отвечаю только по найденному фрагменту.

Ответ:
Void Core — крупный объект из базы знаний, описанный в соответствующем документе.
Источник: документ Void Core.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
""".strip()

    def generate_offline_answer(self, query: str, results: list[SearchResult]) -> str:
        best_result = results[0]
        best_chunk = best_result.chunk

        relevant_sentences = self.extract_relevant_sentences(
            query=query,
            text=best_chunk["text"],
        )

        if not relevant_sentences:
            relevant_sentences = self.first_sentences(best_chunk["text"], limit=3)

        source_line = (
            f"Источник: {best_chunk['title']}, "
            f"файл {best_chunk['source_file']}, "
            f"chunk_id={best_chunk['chunk_id']}."
        )

        return (
            "Шаги:\n"
            "1. Я нашёл наиболее релевантный фрагмент в векторном индексе.\n"
            f"2. Лучший фрагмент относится к документу «{best_chunk['title']}».\n"
            "3. Ниже даю ответ только на основе найденного контекста.\n\n"
            "Ответ:\n"
            f"{' '.join(relevant_sentences)}\n\n"
            f"{source_line}"
        )

    def extract_relevant_sentences(self, query: str, text: str) -> list[str]:
        query_terms = self.normalize_words(query)
        sentences = self.split_sentences(text)

        scored_sentences: list[tuple[int, str]] = []

        for sentence in sentences:
            sentence_terms = self.normalize_words(sentence)
            score = len(query_terms.intersection(sentence_terms))

            if score > 0:
                scored_sentences.append((score, sentence))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        return [sentence for _, sentence in scored_sentences[:3]]

    def normalize_words(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-zА-Яа-я0-9'-]+", text.lower())

        stop_words = {
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "and",
            "or",
            "кто",
            "что",
            "где",
            "когда",
            "как",
            "это",
            "на",
            "в",
            "и",
            "или",
        }

        return {word for word in words if word not in stop_words and len(word) > 2}

    def split_sentences(self, text: str) -> list[str]:
        clean_text = text.replace("\n", " ")
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]

    def first_sentences(self, text: str, limit: int = 3) -> list[str]:
        return self.split_sentences(text)[:limit]

    def format_sources(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "score": round(result.score, 4),
                "title": result.chunk["title"],
                "source_file": result.chunk["source_file"],
                "chunk_id": result.chunk["chunk_id"],
                "start_char": result.chunk["start_char"],
                "end_char": result.chunk["end_char"],
            }
            for result in results
        ]

    def generate_with_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI provider selected, but package 'openai' is not installed"
            ) from error

        client = OpenAI()
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        response = client.responses.create(
            model=model_name,
            input=prompt,
        )

        return response.output_text
