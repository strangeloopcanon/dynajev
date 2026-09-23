import pytest

from dynajev.compile import DecideIn, compile_request
from dynajev.errors import CompileError


def test_bare_question_needs_a_type():
    with pytest.raises(CompileError, match="Set type"):
        compile_request(DecideIn(context="The refund went out Tuesday.", question="Has the refund already been sent?"))


def test_legacy_type_accepts_typed_names():
    jobs, _ = compile_request(DecideIn(context="x", question="Has the refund been sent?", type="noul"))
    assert jobs[0].kind == "boolean"
    jobs, _ = compile_request(DecideIn(context="x", question="Quote the refund line.", type="quote"))
    assert jobs[0].kind == "extract"


def test_options_route_to_a_choice():
    jobs, _ = compile_request(
        DecideIn(
            context="The mug arrived in pieces.",
            question="What does the customer need?",
            options=["a replacement", "a tracking update"],
        )
    )
    assert jobs[0].kind == "categorical"
    assert jobs[0].labels == ["a replacement", "a tracking update"]


def test_exclusive_false_routes_to_independent_flags():
    jobs, _ = compile_request(
        DecideIn(
            context="Legal sent a bill and a threat.",
            question="Which apply?",
            options=["legal", "billing"],
            exclusive=False,
        )
    )
    assert jobs[0].kind == "multilabel"


def test_open_question_is_declared():
    jobs, _ = compile_request(DecideIn(context="Anything.", question="What should we do next?", type="open"))
    assert [job.kind for job in jobs] == ["open"]


def test_schema_compiles_a_different_head_per_field():
    jobs, notes = compile_request(
        DecideIn.model_validate(
            {
                "context": "We sent the refund on Tuesday. The mug arrived broken.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "refund_sent": {"type": "boolean", "description": "Has the refund already been sent?"},
                        "need": {"enum": ["a replacement mug", "a tracking link"]},
                        "urgency": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                            "description": "How urgent is it?",
                        },
                        "flags": {
                            "type": "array",
                            "description": "Which flags apply?",
                            "items": {"enum": ["legal", "billing"]},
                        },
                        "quote": {"type": "string", "description": "Quote the sentence about the mug.", "x-readout": "extract"},
                        "summary": {"type": "string", "description": "Quote nothing; summarise."},
                        "note": {"type": "string", "description": "A short internal note."},
                    },
                },
            }
        )
    )
    kinds = [job.kind for job in jobs]
    assert kinds == ["boolean", "categorical", "ordinal", "multilabel", "extract", "generate", "generate"]
    assert jobs[2].score_tokens == ["0", "1", "2", "3"]
    assert any("share one prefill" in note for note in notes)


def test_typed_questions_are_compiled_without_inference():
    jobs, notes = compile_request(
        DecideIn.model_validate(
            {
                "context": "The mug arrived smashed. Refund issued Tuesday. Two stars.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Was a refund issued?"},
                    "queue": {"type": "choice", "instructions": "Where does this go?", "options": ["billing", "support"]},
                    "stars": {"type": "score", "instructions": "How many stars?", "levels": ["1 star", "2 stars", "3 stars"]},
                    "flags": {"type": "flags", "instructions": "Which apply?", "options": ["damage", "praise"]},
                    "why": {"type": "quote", "instructions": "Quote the damage."},
                    "next": {"type": "open", "instructions": "What should we do?"},
                },
            }
        )
    )
    assert [job.kind for job in jobs] == ["boolean", "categorical", "ordinal", "multilabel", "extract", "open"]
    assert jobs[2].score_tokens == ["1", "2", "3"]
    assert any("declared, not inferred" in note for note in notes)


def test_typed_question_ids_are_validated():
    with pytest.raises(CompileError, match="Question id"):
        compile_request(
            DecideIn.model_validate(
                {"context": "x", "questions": {"bad id!": {"type": "noul", "instructions": "Is it?"}}}
            )
        )
