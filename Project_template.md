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


## Задание 3. Создание векторного индекса базы знаний

Для базы знаний из папки `knowledge_base/` был создан FAISS-индекс.

Используемая embedding-модель:

- `sentence-transformers/all-MiniLM-L6-v2`
- размер эмбеддинга: 384
- тип модели: локальная Sentence-Transformers embedding-модель

Разбиение документов:

- инструмент: `RecursiveCharacterTextSplitter`
- `chunk_size`: 1200 символов
- `chunk_overlap`: 200 символов

Результаты индексации:

- индекс: `vector_index/faiss/faiss.index`
- чанки и метаданные: `vector_index/faiss/chunks.json`
- информация об индексе: `vector_index/faiss/index_info.json`

Поиск по индексу выполняется через:

```bash
python scripts/search_index.py "Who is Varn Kaldor?" --top-k 3
```


## Задание 4. Реализация RAG-бота с техниками промптинга

Реализован консольный RAG-бот.

Основные файлы:

- `app/rag_bot.py` — загрузка FAISS-индекса, поиск, prompt, генерация ответа;
- `scripts/rag_repl.py` — REPL-интерфейс;
- `scripts/rag_query.py` — запуск одного запроса;
- `examples/task-4-dialogues.md` — примеры диалогов;
- `docs/task-4-rag-bot.md` — описание реализации.

Пайплайн:

1. пользователь вводит вопрос;
2. вопрос преобразуется в embedding моделью `sentence-transformers/all-MiniLM-L6-v2`;
3. выполняется поиск ближайших чанков в FAISS;
4. формируется prompt с найденным контекстом;
5. используется few-shot prompting;
6. используются короткие проверяемые шаги рассуждения;
7. бот возвращает ответ с источниками или честно пишет «Я не знаю».

Запуск:

```bash
python scripts/rag_repl.py
```

Пример одноразового запроса:

```bash
python scripts/rag_query.py "Who is Varn Kaldor?"
```
