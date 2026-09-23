import pytest

from readhead.compile import DecideIn, compile_request
from readhead.errors import CompileError


def test_auxiliary_question_routes_to_boolean():
    jobs, notes = compile_request(
        DecideIn(context="The refund went out Tuesday.", question="Has the refund already been sent?")
    )
    assert jobs[0].kind == "boolean"
    assert any("yes/no" in note for note in notes)


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


def test_open_question_routes_to_plain_generation():
    jobs, notes = compile_request(DecideIn(context="Anything.", question="What should we do next?"))
    assert [job.kind for job in jobs] == ["open"]
    assert any("ordinary chat turn" in note for note in notes)


def test_auxiliary_verb_still_wins_over_open():
    jobs, _ = compile_request(DecideIn(context="Anything.", question="Is this urgent?"))
    assert jobs[0].kind == "boolean"


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
                        "quote": {"type": "string", "description": "Quote the sentence about the mug."},
                        "note": {"type": "string", "description": "A short internal note."},
                    },
                },
            }
        )
    )
    kinds = [job.kind for job in jobs]
    assert kinds == ["boolean", "categorical", "ordinal", "multilabel", "extract", "generate"]
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
