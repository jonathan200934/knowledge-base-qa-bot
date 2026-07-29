import re
from html.parser import HTMLParser
from pathlib import Path

from app import indexer


REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = REPO_ROOT / "app" / "markdown_kb" / "app" / "static"


class InputAttributeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, dict[str, str | None]] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "input":
            return
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.inputs[element_id] = attributes


def test_browser_index_action_sends_a_password_key_without_persisting_or_logging_it():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    parser = InputAttributeParser()
    parser.feed(html)

    key_input = parser.inputs["index-key-input"]
    assert key_input["type"] == "password"
    assert key_input["autocomplete"] == "off"
    assert 'document.querySelector("#index-key-input")' in javascript
    assert re.search(
        r'fetch\("/index",\s*\{.*?headers:\s*\{\s*"X-Index-Key":\s*indexKeyInput\.value\s*\}',
        javascript,
        flags=re.DOTALL,
    )
    for forbidden_storage_or_logging_api in (
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "console.",
    ):
        assert forbidden_storage_or_logging_api not in javascript


def test_canonical_readmes_export_the_index_key_and_document_the_browser_field():
    for relative_path in ("README.md", "app/README.md"):
        readme = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert re.search(r"(?m)^export INDEX_API_KEY=", readme)
        assert "password-style **Index API key** field" in readme
        assert "X-Index-Key" in readme
        assert "current page memory only" in readme


def test_canonical_corpus_has_exactly_13_documents_and_49_parent_sections():
    markdown_documents = sorted(indexer.DOCS_DIR.glob("*.md"))
    parent_sections = [
        section
        for document in markdown_documents
        for section in indexer.parse_markdown(document)
    ]

    assert (len(markdown_documents), len(parent_sections)) == (13, 49)


def test_documented_install_includes_direct_pinned_test_dependencies():
    requirements = (
        REPO_ROOT / "app" / "markdown_kb" / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert "pytest==8.3.4" in requirements
    assert "httpx==0.28.1" in requirements


def test_local_hermes_artifacts_are_ignored():
    ignore_patterns = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".hermes/" in ignore_patterns
