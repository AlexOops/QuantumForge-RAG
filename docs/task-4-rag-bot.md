# Задание 4. Реализация RAG-бота с техниками промптинга

## Цель

Реализовать работающий RAG-бот, который:

- принимает пользовательский запрос;
- преобразует запрос в embedding;
- ищет релевантные чанки в FAISS-индексе;
- формирует prompt с найденным контекстом;
- использует few-shot prompting;
- использует проверяемые шаги рассуждения;
- возвращает ответ с источниками;
- отвечает «Я не знаю», если релевантного контекста нет.

## Используемый индекс

Индекс был создан в задании 3 и находится в папке:

`vector_index/faiss/`

Файлы:

- `faiss.index` — FAISS-индекс;
- `chunks.json` — чанки и метаданные;
- `index_info.json` — информация об embedding-модели и параметрах индекса.

## Embedding-модель

Используется та же модель, что и при построении индекса: `sentence-transformers/all-MiniLM-L6-v2`

Размер embedding: `384`

## Интерфейс

Выбран простой консольный интерфейс REPL.

Запуск:

`python scripts/rag_repl.py`

Также есть одноразовый запуск запроса:

`python scripts/rag_query.py "Who is Varn Kaldor?"`

## Few-shot prompting

В prompt добавлены примеры из той же предметной области:

- вопрос про `Varn Kaldor`;
- вопрос про `Void Core`.

Эти примеры показывают модели ожидаемый формат ответа:

- короткие шаги;
- ответ;
- источник.

## Chain-of-Thought / проверяемые шаги

В prompt добавлено требование показывать короткие проверяемые шаги:

- найти релевантные фрагменты;
- проверить, есть ли ответ в контексте;
- ответить только на основе найденных данных.

В реализации это оформлено как видимые reasoning steps без скрытых догадок и без выдумывания фактов.

## Обработка неизвестных вопросов

Если лучший найденный фрагмент имеет score ниже порога релевантности, бот отвечает:

`Я не знаю. В базе знаний нет достаточно релевантного фрагмента для ответа на этот вопрос.`

Порог:

`MIN_RELEVANCE_SCORE = 0.32`

## Примеры запуска

Успешные запросы:

```bash
python scripts/rag_query.py "Who is Varn Kaldor?"
python scripts/rag_query.py "What is the Void Core?"
python scripts/rag_query.py "What is The Lumen Field?"
python scripts/rag_query.py "What is the Aurelian Circle?"
```

Запросы, на которые бот должен ответить `«Я не знаю»`:

```bash
python scripts/rag_query.py "What is the vacation policy for QuantumForge employees?"
python scripts/rag_query.py "How do I reset production AWS credentials?"
```

## Результат

В результате реализован минимальный RAG-бот:

- `app/rag_bot.py `— основной модуль RAG;
- `scripts/rag_repl.py` — консольный интерфейс;
- `scripts/rag_query.py` — одноразовый запуск запроса;
- `examples/task-4-dialogues.md` — примеры диалогов.
