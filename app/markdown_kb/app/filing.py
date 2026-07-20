import json
import os
import re
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .safe_io import atomic_write_regular_file, exclusive_write_regular_file


REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DIR = REPO_ROOT / "wiki"
ANSWERS_DIR = REPO_ROOT / ".kb" / "answers"
ANSWER_FILE_CREATE_ATTEMPTS = 100
ANSWER_FILENAME_RANDOM_BYTES = 8


def slugify(text: str, max_length: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_length].strip("-") or "answer")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_wiki_index(sections: Iterable, wiki_dir: Path | None = None) -> Path:
    wiki_dir = wiki_dir or WIKI_DIR
    output_path = wiki_dir / "index.md"

    grouped = defaultdict(list)
    for section in sections:
        grouped[section.file].append(section)

    lines = [
        "# Knowledge Base Wiki Index",
        "",
        "Generated from indexed Markdown sections. Use this as a human-readable map of the knowledge base.",
        "",
    ]
    for filename in sorted(grouped):
        lines.extend([f"## {filename}", ""])
        for section in sorted(grouped[filename], key=lambda item: item.id):
            heading = " > ".join(section.heading_path)
            preview = " ".join(section.content.split())[:160]
            lines.append(f"- [{section.id}] — {heading}")
            if preview:
                lines.append(f"  - {preview}")
        lines.append("")

    rendered_index = "\n".join(lines).rstrip() + "\n"
    atomic_write_regular_file(
        output_path,
        rendered_index.encode("utf-8"),
        create_parents=True,
    )
    return output_path


def file_answer(
    question: str,
    answer: str,
    sources: list[dict],
    answers_dir: Path | None = None,
    model: str | None = None,
) -> Path:
    answers_dir = answers_dir or ANSWERS_DIR
    created_at = utc_timestamp()
    filename_ts = created_at.replace(":", "").replace("-", "")
    filename_prefix = f"{filename_ts}-{slugify(question)}"
    payload = {
        "schema_version": 1,
        "created_at": created_at,
        "question": question,
        "answer": answer,
        "sources": sources,
        "model": model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-nano"),
    }
    serialized_payload = json.dumps(payload, indent=2, ensure_ascii=False)

    for _ in range(ANSWER_FILE_CREATE_ATTEMPTS):
        random_suffix = secrets.token_hex(ANSWER_FILENAME_RANDOM_BYTES)
        path = answers_dir / f"{filename_prefix}-{random_suffix}.json"
        try:
            exclusive_write_regular_file(
                path,
                serialized_payload.encode("utf-8"),
                create_parents=True,
            )
        except FileExistsError:
            continue
        return path

    raise FileExistsError(
        f"Unable to create a unique answer file after {ANSWER_FILE_CREATE_ATTEMPTS} attempts"
    )
