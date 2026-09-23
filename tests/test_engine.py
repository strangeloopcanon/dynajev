import re

from readhead.compile import DecideIn
from readhead.engine import Readhead
from readhead.errors import CompileError
from readhead.trunk import common_prefix_len
import pytest


class FakeTrunk:
    def __init__(self):
        self.model_id = "fake"
        self.hidden_size = 8
        self.vocab_size = 1000
        self.ids: dict[str, int] = {}
        self.generated: list[dict] = []

    def sanitize(self, context: str) -> str:
        return context

    def _pieces(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.ids) + 1
        return self.ids[piece]

    def token_ids(self, text: str) -> list[int]:
        return [self._id(piece) for piece in self._pieces(text)]

    def single_token(self, text: str) -> int | None:
        pieces = self._pieces(text)
        if len(pieces) == 1:
            return self._id(pieces[0])
        return None

    def encode_prompt(self, user_content: str, assistant_prefix: str, system: str | None = None) -> list[int]:
        return self.token_ids(user_content + "\n" + assistant_prefix)

    def read(self, sequences: list[list[int]]):
        hiddens = []
        for seq in sequences:
            hidden = [0.0] * 8
            for index, token in enumerate(seq[-8:]):
                hidden[index % 8] += token / 100
            hiddens.append(hidden)
        return hiddens, common_prefix_len(sequences)

    def read_dense(self, sequences, chunk_rows=32):
        hiddens, _ = self.read(sequences)
        return hiddens

    def class_logits(self, hidden, groups):
        return [float(group[0]) for group in groups], 0.42

    def prototype_logits(self, hidden, pieces):
        return [float(sum(piece)) for piece in pieces]

    def as_vector(self, hidden):
        return list(hidden)

    def generate(self, prompt_ids, context, extractive, mode="short"):
        self.generated.append({"extractive": extractive, "context": context, "mode": mode})
        if mode == "open":
            return "Send a replacement mug and confirm the refund.", 9
        if extractive:
            return "the mug arrived broken", 4
        return "follow up tomorrow", 3


def test_schema_compiles_heterogeneous_heads_on_one_prefix():
    trunk = FakeTrunk()
    result = Readhead(trunk).decide(
        DecideIn.model_validate(
            {
                "context": "We sent the refund on Tuesday. The mug arrived broken.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "refund_sent": {"type": "boolean", "description": "Has the refund already been sent?"},
                        "color": {"enum": ["red", "blue", "green"], "description": "Which color is named?"},
                        "need": {"enum": ["a replacement mug", "a tracking link"]},
                        "urgency": {"type": "integer", "minimum": 0, "maximum": 3, "description": "How urgent is it?"},
                        "flags": {
                            "type": "array",
                            "description": "Which flags apply?",
                            "items": {"enum": ["legal", "billing"]},
                        },
                        "quote": {"type": "string", "description": "Quote the sentence about the mug."},
                    },
                },
            }
        )
    )
    heads = {field["id"]: field["head"] for field in result["fields"]}
    assert heads == {
        "refund_sent": "binary_margin",
        "color": "vocab_slice",
        "need": "letter_slice",
        "urgency": "ordinal_expectation",
        "flags": "multilabel_margin",
        "quote": "extractive_decode",
    }
    flags = next(field for field in result["fields"] if field["id"] == "flags")
    assert abs(sum(flags["probabilities"].values()) - 1) > 0.05
    choice = next(field for field in result["fields"] if field["id"] == "need")
    assert abs(sum(choice["probabilities"].values()) - 1) < 1e-6
    quote = next(field for field in result["fields"] if field["id"] == "quote")
    assert quote["answer"] == "the mug arrived broken"
    assert quote["generated_tokens"] == 4
    assert trunk.generated[0]["extractive"] is True
    assert result["shared_prefix_tokens"] > 8
    urgency = next(field for field in result["fields"] if field["id"] == "urgency")
    assert urgency["score"] is not None
    assert urgency["generated_tokens"] == 0


def test_margin_strategy_couples_options():
    result = Readhead(FakeTrunk()).decide(
        DecideIn(
            context="The mug arrived broken.",
            question="What is needed?",
            options=["a replacement", "a tracking update", "a password reset"],
            strategy="margin",
        )
    )
    field = result["fields"][0]
    assert field["head"] == "option_margin"
    assert field["sequences"] == 3
    assert abs(sum(field["probabilities"].values()) - 1) < 1e-3


def test_open_question_falls_through_to_chat():
    trunk = FakeTrunk()
    result = Readhead(trunk).decide(DecideIn(context="Anything.", question="What should we do next?"))
    field = result["fields"][0]
    assert field["head"] == "chat_decode"
    assert field["answer"] == "Send a replacement mug and confirm the refund."
    assert field["generated_tokens"] == 9
    assert trunk.generated[-1]["mode"] == "open"
    assert any("ordinary chat turn" in note for note in result["notes"])


def test_missing_question_still_raises():
    with pytest.raises(CompileError):
        Readhead(FakeTrunk()).decide(DecideIn(context="Anything."))


def test_fit_payload_is_attached_for_labeled_states():
    result = Readhead(FakeTrunk()).decide(
        DecideIn(
            context="The customer says thank you for the refund.",
            question="Is the customer grateful?",
            type="boolean",
            examples=[
                {"context": "Thanks so much.", "label": True},
                {"context": "This is unacceptable.", "label": False},
                {"context": "Really appreciate it.", "label": True},
                {"context": "I want a manager.", "label": False},
            ],
        )
    )
    assert result["fit"]["n_examples"] == 4
    assert result["fit"]["chosen"] in {"zero_shot", "affine", "ridge_probe"}


def test_batch_answers_match_single_answers():
    trunk = FakeTrunk()
    engine = Readhead(trunk)
    questions = {
        "ok": {"type": "noul", "instructions": "Is it fine?"},
        "color": {"type": "choice", "instructions": "Which color?", "options": ["red", "blue", "green"]},
        "level": {"type": "score", "instructions": "How bad?", "levels": ["low", "mid", "high"]},
        "tags": {"type": "flags", "instructions": "Which apply?", "options": ["a", "b"]},
    }
    contexts = ["The mug is red and fine.", "The mug is blue and broken.", "Nothing to report."]
    batch = engine.decide_batch(contexts, questions, chunk_rows=2)
    assert batch["states"] == 3 and batch["questions"] == 4 and batch["decisions"] == 12
    assert batch["generated_tokens"] == 0
    for context, entry in zip(contexts, batch["results"]):
        single = engine.decide(DecideIn.model_validate({"context": context, "questions": questions}))
        assert entry["answers"] == single["answers"]


def test_batch_refuses_open_questions():
    with pytest.raises(CompileError, match="closed readouts only"):
        Readhead(FakeTrunk()).decide_batch(["x"], {"q": {"type": "open", "instructions": "Why?"}})
