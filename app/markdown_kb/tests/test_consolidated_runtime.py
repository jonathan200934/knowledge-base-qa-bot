from pathlib import Path

from fastapi.routing import APIRoute

from app.routes import router
from app.security import require_index_api_key


REPO_ROOT = Path(__file__).resolve().parents[3]
STANDALONE_RUNTIME = REPO_ROOT / "app" / "vector_rag"
FORBIDDEN_MARKERS = (
    "app/vector_rag",
    "app\\vector_rag",
    ".kb/vector_rag",
    ".kb\\vector_rag",
    "vector_rag",
)
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".txt"}


def supported_runtime_files() -> list[Path]:
    """Return canonical production, configuration, and documentation files.

    OpenSpec source-of-truth and archived planning documents are deliberately
    outside this allowlist. Historical PROMPT.md files are not supported
    runtime documentation either.
    """

    files = [
        REPO_ROOT / ".gitignore",
        REPO_ROOT / "README.md",
        REPO_ROOT / "app" / "README.md",
        REPO_ROOT / "app" / "markdown_kb" / "requirements.txt",
    ]
    runtime_root = REPO_ROOT / "app" / "markdown_kb" / "app"
    files.extend(
        path
        for path in runtime_root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )
    return sorted(set(files))


def test_standalone_vector_rag_runtime_is_absent():
    assert not STANDALONE_RUNTIME.exists(), (
        "app/vector_rag is a retired standalone runtime; retain the pure-vector "
        "baseline only through app/markdown_kb POST /compare"
    )


def test_supported_runtime_and_docs_have_no_standalone_vector_rag_references():
    references: list[str] = []
    for path in supported_runtime_files():
        text = path.read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in text:
                references.append(f"{path.relative_to(REPO_ROOT)}: {marker}")

    assert references == [], (
        "supported runtime/configuration/documentation still references the "
        f"retired standalone runtime: {references}"
    )


def test_markdown_kb_router_owns_the_supported_api_and_protects_indexing():
    api_routes = {
        (method, route.path): route
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }

    expected = {
        ("GET", "/health"),
        ("POST", "/index"),
        ("POST", "/chat"),
        ("POST", "/compare"),
    }
    assert set(api_routes) == expected
    assert all(
        route.endpoint.__module__ == "app.routes" for route in api_routes.values()
    )
    index_route = api_routes[("POST", "/index")]
    assert require_index_api_key in [
        dependency.call for dependency in index_route.dependant.dependencies
    ]


def test_canonical_docs_define_the_complete_safe_runtime_contract():
    docs = {
        name: (REPO_ROOT / name).read_text(encoding="utf-8").lower()
        for name in ("README.md", "app/README.md")
    }
    for text in docs.values():
        assert "app/markdown_kb" in text

    combined = "\n".join(docs.values())
    required = (
        "index_api_key",
        "x-index-key",
        "503",
        "401",
        "file_answer=false",
        "file_answer=true",
        "2000",
        "untrusted context",
        "citation allowlist",
        "i cannot confirm from the knowledge base.",
        "sources: []",
        "tf-idf",
        "cosine",
        "embedding_cache.sqlite3",
        "index.faiss",
        "children.json",
        "manifest.json",
        "no pickle",
        "atomic",
        "rebuild",
        "bm25",
        "vector",
        "hybrid",
        "reranked",
        "pure-vector baseline",
    )
    missing = [marker for marker in required if marker not in combined]
    assert missing == [], f"canonical docs omit required behavior: {missing}"


def test_runtime_requirements_install_compatible_faiss_backend():
    requirements = (
        REPO_ROOT / "app" / "markdown_kb" / "requirements.txt"
    ).read_text(encoding="utf-8").lower().splitlines()
    assert "faiss-cpu==1.9.0.post1" in requirements
    assert "numpy==1.26.4" in requirements
