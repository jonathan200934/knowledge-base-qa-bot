import json

from app import filing, indexer


def test_write_wiki_index_groups_topics_without_conversation_memory(tmp_path):
    sections = [
        indexer.Section(
            id="shipping_faq.md#standard-shipping",
            file="shipping_faq.md",
            heading="Standard Shipping",
            heading_path=["Shipping FAQ", "Standard Shipping"],
            content="Standard shipping usually takes 3-5 business days.",
            tokens=["shipping", "standard"],
        ),
        indexer.Section(
            id="refund_policy.md#refund-timeline",
            file="refund_policy.md",
            heading="Refund Timeline",
            heading_path=["Refund Policy", "Refund Timeline"],
            content="Approved refunds are processed within 5-7 business days.",
            tokens=["refund", "timeline"],
        ),
    ]

    output_path = filing.write_wiki_index(sections, wiki_dir=tmp_path / "wiki")

    text = output_path.read_text(encoding="utf-8")
    assert output_path.name == "index.md"
    assert "# Knowledge Base Wiki Index" in text
    assert "## shipping_faq.md" in text
    assert "[shipping_faq.md#standard-shipping]" in text
    assert "Shipping FAQ > Standard Shipping" in text
    assert "conversation" not in text.lower()
    assert "memory" not in text.lower()


def test_file_answer_writes_source_grounded_card_without_chat_history(tmp_path):
    sources = [
        {
            "source": "refund_policy.md#refund-timeline",
            "heading": "Refund Policy > Refund Timeline",
            "score": 2.1,
            "content": "Approved refunds are processed within 5-7 business days.",
        }
    ]

    answer_file = filing.file_answer(
        question="How long do refunds take?",
        answer="Refunds take 5-7 business days. [refund_policy.md#refund-timeline]",
        sources=sources,
        answers_dir=tmp_path / "answers",
        model="gpt-4.1-nano",
    )

    payload = json.loads(answer_file.read_text(encoding="utf-8"))
    assert payload["question"] == "How long do refunds take?"
    assert payload["answer"].startswith("Refunds take")
    assert payload["sources"] == sources
    assert payload["model"] == "gpt-4.1-nano"
    assert payload["created_at"].endswith("Z")
    assert {"question", "answer", "sources", "model", "created_at", "schema_version"}.issubset(payload)
    assert not ({"messages", "chat_history", "conversation_id", "session_id"} & set(payload))


def test_file_answer_never_overwrites_when_names_collide_within_one_second(
    monkeypatch, tmp_path
):
    fixed_timestamp = "2026-07-18T12:34:56Z"
    monkeypatch.setattr(filing, "utc_timestamp", lambda: fixed_timestamp)
    generated_tokens = iter(["collision", "collision", "retry-winner"])
    monkeypatch.setattr(filing.secrets, "token_hex", lambda length: next(generated_tokens))
    answers_dir = tmp_path / "answers"
    first_answer = "First answer. [Source: policy.md#answer]"
    second_answer = "Second answer. [Source: policy.md#answer]"

    first_path = filing.file_answer(
        question="Same question?",
        answer=first_answer,
        sources=[{"source": "policy.md#answer"}],
        answers_dir=answers_dir,
    )
    second_path = filing.file_answer(
        question="Same question?",
        answer=second_answer,
        sources=[{"source": "policy.md#answer"}],
        answers_dir=answers_dir,
    )

    assert first_path != second_path
    assert first_path.parent == second_path.parent == answers_dir
    assert first_path.name == "20260718T123456Z-same-question-collision.json"
    assert second_path.name == "20260718T123456Z-same-question-retry-winner.json"
    assert first_path.suffix == second_path.suffix == ".json"
    assert json.loads(first_path.read_text(encoding="utf-8"))["answer"] == first_answer
    assert json.loads(second_path.read_text(encoding="utf-8"))["answer"] == second_answer
    assert len(list(answers_dir.glob("*.json"))) == 2
