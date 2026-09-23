import pytest

from dynajev.compile import DecideIn, compile_request, order_stages
from dynajev.engine import Dynajev
from dynajev.errors import CompileError
from test_engine import FakeBackend

CONTEXT = "The mug arrived smashed. I want my money back."


def _decide(questions, backend=None):
    backend = backend or FakeBackend()
    return backend, Dynajev(backend).decide(DecideIn.model_validate({"context": CONTEXT, "questions": questions}))


def _compile(questions):
    return compile_request(DecideIn.model_validate({"context": "x", "questions": questions}))


def test_stages_follow_dependencies_and_reject_cycles():
    jobs, notes = _compile(
        {
            "a": {"type": "noul", "instructions": "Is it broken?"},
            "b": {"type": "choice", "instructions": "Which?", "options": ["x", "y"], "depends_on": {"question": "a", "when": True}},
            "c": {"type": "noul", "instructions": "Why?", "include_answers": ["b"]},
            "d": {"type": "noul", "instructions": "Other?"},
        }
    )
    assert order_stages(jobs) == [["a", "d"], ["b"], ["c"]]
    assert any("3 stages" in note for note in notes)
    with pytest.raises(CompileError, match="cycle"):
        _compile(
            {
                "a": {"type": "noul", "instructions": "A?", "include_answers": ["b"]},
                "b": {"type": "noul", "instructions": "B?", "depends_on": {"question": "a", "when": True}},
            }
        )
    with pytest.raises(CompileError, match="itself"):
        _compile({"a": {"type": "noul", "instructions": "A?", "include_answers": ["a"]}})
    with pytest.raises(CompileError, match="unknown question"):
        _compile({"a": {"type": "noul", "instructions": "A?", "depends_on": {"question": "zzz"}}})
    with pytest.raises(CompileError, match="not one of the answers"):
        _compile(
            {
                "a": {"type": "choice", "instructions": "A?", "options": ["x", "y"]},
                "b": {"type": "noul", "instructions": "B?", "depends_on": {"question": "a", "when": "z"}},
            }
        )


def test_depends_on_skips_unless_the_parent_answer_matches():
    # The fake backend always leans No on a yes/no question.
    _, result = _decide(
        {
            "broken": {"type": "noul", "instructions": "Is it broken?"},
            "if_no": {"type": "choice", "instructions": "Which?", "options": ["red", "blue"], "depends_on": {"question": "broken", "when": False}},
            "if_yes": {"type": "score", "instructions": "How bad?", "levels": ["low", "high"], "depends_on": {"question": "broken", "when": "yes"}},
            "after_yes": {"type": "noul", "instructions": "Refund?", "depends_on": {"question": "if_yes", "when": ["low", "high"]}},
            "flags": {"type": "flags", "instructions": "Which apply?", "options": ["damage", "delay"]},
            "tag_dep": {"type": "noul", "instructions": "Escalate?", "depends_on": {"question": "flags", "when": ["damage", "delay"]}},
        }
    )
    answers = result["answers"]
    assert answers["broken"]["noul"] < 0.5
    assert "choice" in answers["if_no"]
    assert answers["if_yes"] == {"type": "score", "skipped": True}
    assert answers["after_yes"] == {"type": "noul", "skipped": True}
    skipped = {field["id"]: field for field in result["fields"] if field.get("skipped")}
    assert "if_yes was skipped" in skipped["after_yes"]["reason"]
    assert [entry.get("skipped") for entry in result["plan"]["fields"] if entry["id"] == "if_yes"] == [True]
    assert result["plan"]["stages"][0] == ["broken", "flags"]
    flags = answers["flags"]["flags"]
    assert (answers["tag_dep"].get("skipped") is True) == (not flags)


def test_include_answers_continues_the_parent_conversation_and_forks_its_cache():
    backend, result = _decide(
        {
            "queue": {"type": "choice", "instructions": "Where should this go?", "options": ["a billing desk", "a repair desk"]},
            "why": {"type": "noul", "instructions": "Is the customer angry?", "include_answers": ["queue"]},
        }
    )
    entry = next(e for e in result["plan"]["fields"] if e["id"] == "why")
    assert entry["continues"] == "queue"
    second = result["plan"]["reads"][1]
    queue_tokens = next(e for e in result["plan"]["fields"] if e["id"] == "queue")["tokens"][0]
    assert second["cached_tokens"] == queue_tokens
    assert second["processed_tokens"] < entry["tokens"][0]


def test_include_answers_from_flags_are_written_into_the_prompt():
    backend, result = _decide(
        {
            "flags": {"type": "flags", "instructions": "Which apply?", "options": ["damage", "delay"]},
            "next": {"type": "choice", "instructions": "Next step?", "options": ["refund", "replace"], "include_answers": ["flags"]},
        }
    )
    entry = next(e for e in result["plan"]["fields"] if e["id"] == "next")
    assert "continues" not in entry
    assert any(prompt.count("Earlier answers:\nQ: Which apply?\nA: ") == 1 for prompt in backend.prompts)


def test_batch_runs_dependent_questions_in_shared_mode_only():
    engine = Dynajev(FakeBackend())
    questions = {
        "broken": {"type": "noul", "instructions": "Is it broken?"},
        "how": {"type": "choice", "instructions": "How?", "options": ["red", "blue"], "depends_on": {"question": "broken", "when": True}},
    }
    batch = engine.decide_batch(["a", "b"], questions)
    assert batch["mode"] == "shared"
    assert all(entry["answers"]["how"].get("skipped") for entry in batch["results"])
    with pytest.raises(CompileError, match="shared mode"):
        engine.decide_batch(["a"], questions, mode="dense")
