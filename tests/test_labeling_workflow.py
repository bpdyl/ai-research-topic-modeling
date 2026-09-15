import json
from pathlib import Path

import pytest

from nrtm.labeling.providers import Completion, ProviderError, complete
from nrtm.labeling.workflow import (load_evidence, new_run, generate_pending, parse_label,
                                    record_review, export_labels)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return {"key": "test:0", "model_name": "Example", "topic_id": 0, "source_run": "test",
            "top_words": ["student", "education"], "representative_documents": [
                {"doc_id": "paper-1", "title": "AI in education", "year": 2025, "weight": .9}],
            "prompt": "label this topic"}


def label_text(**updates):
    value = {"label": "AI in Education", "rationale": "Students and learning feature in the evidence.",
             "coherence": "coherent", "supporting_doc_ids": ["paper-1"]}
    value.update(updates)
    return json.dumps(value)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_native_api_contracts(provider):
    def transport(url, headers, body, timeout):
        assert timeout == 60
        assert "secret" not in url and "secret" not in json.dumps(body)
        if provider == "openai":
            assert url.endswith("/v1/responses") and headers["Authorization"] == "Bearer secret"
            assert body["input"] == "prompt" and body["store"] is False
            return {"status": "completed", "model": "resolved", "id": "r1", "usage": {"input_tokens": 5},
                    "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "label"}]}]}
        if provider == "anthropic":
            assert headers["x-api-key"] == "secret" and body["max_tokens"] == 2048
            assert body["messages"][0]["content"] == "prompt"
            return {"stop_reason": "end_turn", "model": "resolved", "id": "r1",
                    "content": [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "label"}]}
        assert headers["x-goog-api-key"] == "secret" and url.endswith("/models/model:generateContent")
        assert body["contents"][0]["parts"][0]["text"] == "prompt"
        return {"modelVersion": "resolved", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "private", "thought": True}, {"text": "label"}]}}]}
    result = complete(provider, "model", "secret", "prompt", transport=transport)
    assert result.text == "label" and result.model == "resolved"


@pytest.mark.parametrize("provider,response", [
    ("openai", {"status": "incomplete"}),
    ("openai", {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}),
    ("anthropic", {"stop_reason": "max_tokens"}),
    ("gemini", {"candidates": [{"finishReason": "MAX_TOKENS"}]}),
    ("gemini", {"promptFeedback": {"blockReason": "SAFETY"}}),
])
def test_incomplete_or_blocked_responses_rejected(provider, response):
    with pytest.raises(ProviderError):
        complete(provider, "model", "secret", "prompt", transport=lambda *args: response)


@pytest.mark.parametrize("text", ['not JSON', '[]', label_text(supporting_doc_ids=["invented"]), label_text(coherence="perfect")])
def test_invalid_labels_cannot_be_exported(evidence, text):
    with pytest.raises(ValueError):
        parse_label(text, evidence)


def test_checkpoint_resume_and_review(tmp_path, evidence):
    other = {**evidence, "topic_id": 1, "key": "test:1"}
    path, run = new_run(tmp_path, [evidence, other], "openai", "model", 2048)
    calls = []
    def failing_client(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise ProviderError("Provider returned HTTP 429.")
        return Completion(label_text(), "resolved", "r1", {})
    generate_pending(path, run, "secret-key", client=failing_client)
    saved = json.loads(path.read_text())
    assert [i["status"] for i in saved["items"]] == ["generated", "failed"]
    assert "secret-key" not in path.read_text()
    assert export_labels(saved)["_provenance"]["exported_topics"] == 0
    calls.clear()
    def successful_client(*args, **kwargs):
        calls.append(args)
        return Completion(label_text(), "resolved", "r2", {})
    generate_pending(path, saved, "secret-key", client=successful_client)
    assert len(calls) == 1
    assert saved["items"][0]["response_id"] == "r1"
    item = saved["items"][0]
    with pytest.raises(ValueError):
        record_review(item, "accept", "", "", "checked")
    record_review(item, "revise", "Educational AI", "Reviewer A", "Confirmed against representative titles.")
    assert export_labels(saved)["Example"] == {"0": "Educational AI"}
    record_review(item, "reject", "", "Reviewer A", "Insufficient support on closer reading.")
    assert len(item["review_history"]) == 2
    assert export_labels(saved)["_provenance"]["exported_topics"] == 0
    assert export_labels(saved, False)["_provenance"]["exported_topics"] == 2


def test_frozen_evidence_covers_all_39_topics():
    entries = load_evidence(ROOT)
    assert len(entries) == 39 and len({e["key"] for e in entries}) == 39
    assert all(e["archived_draft"] and e["archived_audited_label"] for e in entries)
    assert all(len(e["representative_documents"]) == (0 if e["model_name"] == "Random-Search LDA" else 4) for e in entries)
    assert all("abstract" not in d for e in entries for d in e["representative_documents"])


def test_streamlit_browse_and_credential_gate():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "apps/topic_labeler.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "39"
    assert next(b for b in app.button if b.label == "Generate selected labels").disabled
    next(s for s in app.selectbox if s.label == "Provider").set_value("gemini").run()
    assert not app.exception
