import pytest

from agent import graph
from agent.tools import calculate

SOURCES = lambda q: [
    {"id": 1, "ticker": "GS", "section": "Executive Overview", "text": "our CET1 capital ratio was 14.3% under the Standardized Capital Rules"}
]


@pytest.fixture(autouse=True)
def isolate_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "TRACE", tmp_path / "trace.jsonl")


def scripted_draft(answers):
    it = iter(answers)

    def fake(state):
        return {"answer": next(it), "calcs": [], "attempts": state.get("attempts", 0) + 1}

    return fake


def run(question, approver=lambda s: True):
    return graph.build(approver).invoke({"question": question})


def make_state(answer, source_text, calcs=None):
    return {"answer": answer, "sources": [{"text": source_text}], "calcs": calcs or []}


# --- calculator ---
def test_calculate_basic():
    assert calculate("70645 - 61761") == 8884
    assert round(calculate("(16.9 - 13.4) / 13.4 * 100"), 1) == 26.1


@pytest.mark.parametrize(
    "expr", ["__import__('os').system('echo hi')", "open('x')", "2 ** 10000", "9 ** 9 ** 9", "1 / 0"]
)
def test_calculate_rejects_unsafe_or_invalid(expr):
    with pytest.raises(Exception):
        calculate(expr)


# --- verification ---
def test_verify_accepts_supported_numbers():
    s = make_state("In 2025 the ratio was 14.3% [1].", "CET1 capital ratio was 14.3% under")
    assert graph.verify(s)["unsupported"] == []


def test_verify_flags_invented_number():
    s = make_state("The ratio was 15.9% [1].", "CET1 capital ratio was 14.3% under")
    assert graph.verify(s)["unsupported"] == ["15.9"]


def test_verify_accepts_calculator_result():
    s = make_state(
        "Revenue grew by 8.8 [1].", "revenues of 70.6 and 61.8",
        calcs=[{"expression": "70.6 - 61.8", "result": 8.8}],
    )
    assert graph.verify(s)["unsupported"] == []


# --- full graph scenarios (model mocked) ---
def test_advice_is_refused_without_retrieval_or_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(graph, "search_filings", boom)
    monkeypatch.setattr(graph, "draft", boom)
    out = run("Should I buy Citi stock?")
    assert out["status"] == "refused"


def test_supported_answer_passes_first_time(monkeypatch):
    monkeypatch.setattr(graph, "search_filings", SOURCES)
    monkeypatch.setattr(graph, "draft", scripted_draft(["The ratio was 14.3% [1]."]))
    out = run("What was the GS CET1 ratio?")
    assert out["status"] == "ok"
    assert out["attempts"] == 1


def test_unsupported_number_triggers_retry_then_passes(monkeypatch):
    monkeypatch.setattr(graph, "search_filings", SOURCES)
    monkeypatch.setattr(
        graph, "draft", scripted_draft(["The ratio was 15.9% [1].", "The ratio was 14.3% [1]."])
    )
    out = run("What was the GS CET1 ratio?")
    assert out["status"] == "ok"
    assert out["attempts"] == 2


def test_persistent_unsupported_number_is_flagged(monkeypatch):
    monkeypatch.setattr(graph, "search_filings", SOURCES)
    monkeypatch.setattr(
        graph, "draft", scripted_draft(["The ratio was 15.9% [1].", "The ratio was 15.9% [1]."])
    )
    out = run("What was the GS CET1 ratio?")
    assert out["status"] == "unverified"
    assert out["attempts"] == 2
    assert out["unsupported"] == ["15.9"]


def test_human_can_reject_answer(monkeypatch):
    monkeypatch.setattr(graph, "search_filings", SOURCES)
    monkeypatch.setattr(graph, "draft", scripted_draft(["The ratio was 14.3% [1]."]))
    out = run("What was the GS CET1 ratio?", approver=lambda s: False)
    assert out["status"] == "rejected"