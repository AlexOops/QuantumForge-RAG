import json
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup


API_URL = "https://starwars.fandom.com/api.php"

PAGES_PATH = Path("source_data/starwars_pages.json")
TERMS_MAP_PATH = Path("terms_map.json")

RAW_DIR = Path("raw_knowledge_base")
OUTPUT_DIR = Path("knowledge_base")

MIN_DOCUMENTS_COUNT = 30


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "document"


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    ignored_fragments = (
        "advertisement",
        "sign in to edit",
        "view source",
        "history",
        "talk",
        "navigation",
        "categories",
        "community content is available",
        "fandom apps",
        "take your favorite fandoms",
    )

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        normalized = line.lower()

        if any(fragment in normalized for fragment in ignored_fragments):
            continue

        if len(line) < 3:
            continue

        lines.append(line)

    return "\n\n".join(lines).strip()


def fetch_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
        headers={"User-Agent": "QuantumForgeRAGBot/1.0"},
    )
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        extract = page.get("extract")

        if extract:
            return extract

    return None


def fetch_html_text(title: str) -> str:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "redirects": "1",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
        headers={"User-Agent": "QuantumForgeRAGBot/1.0"},
    )
    response.raise_for_status()

    html = response.json()["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    for selector in [
        "sup",
        "table",
        ".mw-editsection",
        ".portable-infobox",
        ".navbox",
        ".metadata",
        ".reference",
        ".references",
    ]:
        for element in soup.select(selector):
            element.decompose()

    return soup.get_text("\n")


def fetch_page_text(title: str) -> str:
    extract = fetch_extract(title)

    if extract:
        return clean_text(extract)

    return clean_text(fetch_html_text(title))


def build_replacement_pattern(terms_map: dict[str, str]) -> re.Pattern:
    escaped_terms = [re.escape(term) for term in sorted(terms_map, key=len, reverse=True)]

    # Граница не должна разрывать обычные слова.
    # Например, Force не должен матчиться внутри forced/forces.
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(escaped_terms) + r")(?![A-Za-z0-9_])")


def replace_terms(text: str, terms_map: dict[str, str], pattern: re.Pattern) -> str:
    def replace(match: re.Match) -> str:
        source = match.group(0)
        return terms_map.get(source, source)

    return pattern.sub(replace, text)


def build_document(title: str, text: str, terms_map: dict[str, str], pattern: re.Pattern) -> tuple[str, str]:
    new_title = replace_terms(title, terms_map, pattern)
    new_text = replace_terms(text, terms_map, pattern)

    document = f"# {new_title}\n\n{new_text}\n"
    filename = f"{slugify(new_title)}.md"

    return filename, document


def main() -> None:
    pages = load_json(PAGES_PATH)
    terms_map = load_json(TERMS_MAP_PATH)

    RAW_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    pattern = build_replacement_pattern(terms_map)
    created_documents = 0

    for index, title in enumerate(pages, start=1):
        print(f"[{index}/{len(pages)}] Processing: {title}")

        try:
            raw_text = fetch_page_text(title)
        except Exception as error:
            print(f"  Failed: {error}")
            continue

        if len(raw_text) < 500:
            print("  Skipped: text is too short")
            continue

        raw_path = RAW_DIR / f"{slugify(title)}.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        filename, document = build_document(title, raw_text, terms_map, pattern)
        output_path = OUTPUT_DIR / filename
        output_path.write_text(document, encoding="utf-8")

        created_documents += 1
        time.sleep(0.5)

    if created_documents < MIN_DOCUMENTS_COUNT:
        raise RuntimeError(
            f"Expected at least {MIN_DOCUMENTS_COUNT} documents, but created {created_documents}"
        )

    print(f"Done. Created documents: {created_documents}")


if __name__ == "__main__":
    main()
