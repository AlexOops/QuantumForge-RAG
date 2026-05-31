# Задание 3. Создание векторного индекса базы знаний

## Выбранная embedding-модель

Для индексации используется локальная embedding-модель:

- Название: `sentence-transformers/all-MiniLM-L6-v2`
- Репозиторий: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Размер эмбеддинга: 384
- Назначение: semantic search, similarity search, clustering

Модель выбрана потому, что она компактная, работает локально на CPU и не требует отправки документов во внешний API.

## База знаний

Источник данных:

- папка: `knowledge_base/`
- формат документов: Markdown
- количество документов: см. `vector_index/faiss/index_info.json`

База знаний была подготовлена в задании 2. Исходная предметная область была обфусцирована через словарь `terms_map.json`, чтобы LLM не могла отвечать по памяти.

## Разбиение на чанки

Для разбиения документов используется `RecursiveCharacterTextSplitter`.

Параметры:

- `chunk_size`: 1200 символов
- `chunk_overlap`: 200 символов
- минимальная длина чанка: 100 символов

Для каждого чанка сохраняются метаданные:

- `id`
- `chunk_id`
- `source_path`
- `source_file`
- `title`
- `start_char`
- `end_char`
- `word_count`
- `text`

Это нужно, чтобы в будущем RAG-бот мог показывать источники и объяснять, на какой фрагмент документа он опирается.

## Создание индекса

Индекс создаётся скриптом:

```bash
python scripts/build_index.py
```

## Результат сохраняется в папку:

`vector_index/faiss/`

## Файлы индекса:
- faiss.index — FAISS-индекс
- chunks.json — тексты чанков и метаданные
- index_info.json — техническая информация об индексе

## Поиск по индексу

Пример запуска:

```bash 
python scripts/search_index.py "Who is Varn Kaldor?" --top-k 3
```

Дополнительные тестовые запросы:

```bash
python scripts/search_index.py "What is the Void Core?" --top-k 3
python scripts/search_index.py "What is The Lumen Field?" --top-k 3
```

## Итог
В результате был создан локальный FAISS-индекс по обфусцированной базе знаний.
Индекс можно использовать для поиска релевантных фрагментов текста по пользовательскому запросу и дальнейшей передачи найденного контекста в RAG-бота.