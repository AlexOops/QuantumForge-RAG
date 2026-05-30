# Project Template

## Задание 1. Исследование моделей и инфраструктуры

См. подробный отчёт: `docs/task-1-research.md`.

### Краткая рекомендация

Для учебного проекта выбран стек:

- LLM: OpenAI/YandexGPT API;
- Embeddings: Sentence-Transformers;
- Vector DB: FAISS;
- Backend: FastAPI;
- Runtime: Docker Compose;
- Server: 4 vCPU, 8 GB RAM, без GPU.

Причина выбора: конфигурация простая, дешёвая, не требует GPU и подходит для быстрого прототипа RAG-бота.