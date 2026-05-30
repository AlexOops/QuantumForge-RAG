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

## Задание 2. Подготовка базы знаний

Для подготовки базы знаний была выбрана вселенная Star Wars / Wookieepedia.

Чтобы LLM не отвечала по памяти, ключевые термины были заменены на вымышленные названия. Например:

- `Darth Vader` → `Varn Kaldor`
- `Death Star` → `Void Core`
- `The Force` → `The Lumen Field`
- `Jedi Order` → `Aurelian Circle`
- `Galactic Empire` → `Aurelian Dominion`

Итоговая база находится в папке `knowledge_base/`.

Дополнительные файлы:

- `terms_map.json` — словарь замен;
- `source_data/starwars_pages.json` — список исходных страниц;
- `scripts/build_knowledge_base.py` — скрипт скачивания, очистки и замены терминов;
- `scripts/validate_knowledge_base.py` — проверка финальной базы;
- `docs/task-2-knowledge-base.md` — описание подхода.