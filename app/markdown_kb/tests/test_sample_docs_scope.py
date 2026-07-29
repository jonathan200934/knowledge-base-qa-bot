from pathlib import Path


def test_sample_docs_include_enough_ecommerce_fake_knowledge():
    docs_dir = Path(__file__).resolve().parents[3] / "docs"
    docs = sorted(docs_dir.glob("*.md"))
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in docs)

    assert len(docs) >= 10
    for required_topic in [
        "shipping",
        "refund",
        "account",
        "faq",
        "policy",
        "user agreement",
        "payment",
        "privacy",
        "warranty",
        "loyalty",
    ]:
        assert required_topic in text
