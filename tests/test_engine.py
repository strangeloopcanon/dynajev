import re

from dynajev.compile import DecideIn
from dynajev.engine import Dynajev
from dynajev.errors import CompileError
import pytest


class FakeBackend:
    def __init__(self):
        self.model_id = "fake"
        self.hidden_size = 8
        self.vocab_size = 1000
        self.ids: dict[str, int] = {}
        self.generated: list[dict] = []
        self.forwards: list[int] = []
        self.prompts: list[str] = []

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
        self.prompts.append(user_content)
        return self.token_ids(user_content + "\n" + assistant_prefix)

    def continue_prompt(self, prompt_ids, answer, user_content, assistant_prefix):
        self.prompts.append(user_content)
        return list(prompt_ids) + self.token_ids(answer + " | " + user_content + "\n" + assistant_prefix)

    num_layers = 4

    def _vector(self, seq, depth=4):
        hidden = [0.0] * 8
        for index, token in enumerate(seq[-8:]):
            hidden[index % 8] += token / 100
        hidden[0] += depth / 1000
        return hidden

    def fork(self, cache):
        return None if cache is None else list(cache)

    def prefill(self, tokens, cache=None, layers=None):
        seq = list(cache or []) + list(tokens)
        self.forwards.append(len(tokens))
        return {k: self._vector(seq, k) for k in (layers or (4,))}, seq

    def prefill_rows(self, cache, rows, layers=None, chunk_rows=64):
        self.forwards.append(sum(len(r) for r in rows))
        return [
            {k: self._vector(list(cache or []) + list(row), k) for k in (layers[i] if layers else (4,))}
            for i, row in enumerate(rows)
        ]

    def read_dense(self, sequences, chunk_rows=32):
        return [self._vector(seq) for seq in sequences]

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
    trunk = FakeBackend()
    result = Dynajev(trunk).decide(
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
                        "quote": {"type": "string", "description": "Quote the sentence about the mug.", "x-readout": "extract"},
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
    result = Dynajev(FakeBackend()).decide(
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
    trunk = FakeBackend()
    result = Dynajev(trunk).decide(DecideIn(context="Anything.", question="What should we do next?", type="open"))
    field = result["fields"][0]
    assert field["head"] == "chat_decode"
    assert field["answer"] == "Send a replacement mug and confirm the refund."
    assert field["generated_tokens"] == 9
    assert trunk.generated[-1]["mode"] == "open"


def test_missing_question_still_raises():
    with pytest.raises(CompileError):
        Dynajev(FakeBackend()).decide(DecideIn(context="Anything."))


def test_fit_payload_is_attached_for_labeled_states():
    result = Dynajev(FakeBackend()).decide(
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
    trunk = FakeBackend()
    engine = Dynajev(trunk)
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
        Dynajev(FakeBackend()).decide_batch(["x"], {"q": {"type": "open", "instructions": "Why?"}})


def test_response_carries_a_readable_plan():
    result = Dynajev(FakeBackend()).decide(
        DecideIn.model_validate(
            {
                "context": "The mug arrived smashed. Refund issued Tuesday.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Was a refund issued?"},
                    "flags": {"type": "flags", "instructions": "Which apply?", "options": ["damage", "praise", "delay"]},
                    "why": {"type": "quote", "instructions": "Quote the damage."},
                },
            }
        )
    )
    plan = result["plan"]
    assert isinstance(plan["summary"], str) and "branches" in plan["summary"]
    by_id = {entry["id"]: entry for entry in plan["fields"]}
    assert by_id["flags"]["branches"] == 3
    assert by_id["flags"]["combine"] == "3 independent sigmoids"
    assert by_id["refund"]["combine"] == "sigmoid(Yes - No)"
    assert "decode" in by_id["why"]
    assert plan["reads"][0]["branches"] == 4
    assert set(result["answers"]) == {"refund", "flags", "why"}


class KeywordBackend(FakeBackend):
    """Shallow layers see the tone word; the unembedding slice never does."""

    def _vector(self, seq, depth=4):
        hidden = super()._vector(seq, depth)
        if depth <= 2:
            hidden[1] += 5.0 * (self.ids.get("thanks", -1) in seq) - 5.0 * (self.ids.get("furious", -1) in seq)
        return hidden

    def class_logits(self, hidden, groups):
        return [0.0 for _ in groups], 0.42


def test_examples_can_pick_an_early_exit_and_the_field_runs_shallow():
    backend = KeywordBackend()
    examples = [{"context": f"{word} number {i}", "label": label} for i in range(4) for word, label in (("thanks", "grateful"), ("furious", "angry"))]
    result = Dynajev(backend).decide(
        DecideIn.model_validate(
            {
                "context": "thanks for the refund",
                "questions": {"tone": {"type": "choice", "instructions": "What is the tone?", "options": ["grateful", "angry"]}},
                "examples": examples,
            }
        )
    )
    fit = result["fit"]
    assert fit["exit_layer"] == 1
    assert fit["chosen"] == "ridge_probe"
    assert [row["layer"] for row in fit["layer_scan"]] == [1]
    field = result["fields"][0]
    assert field["depth"] == 1 and field["head"] == "ridge_probe"
    assert result["answers"]["tone"]["choice"] == "grateful"
    assert result["plan"]["fields"][0]["depth"] == "1/4 layers"


def test_criteria_judge_each_side_on_its_own_branch():
    result = Dynajev(FakeBackend()).decide(
        DecideIn.model_validate(
            {
                "context": "The reply apologised and offered a refund.",
                "questions": {
                    "resolved": {
                        "type": "noul",
                        "instructions": "Was the complaint resolved?",
                        "criteria": {"true": "A concrete remedy was offered.", "false": "The customer was left without a remedy."},
                    }
                },
            }
        )
    )
    field = result["fields"][0]
    assert field["head"] == "criteria_judgment"
    answer = result["answers"]["resolved"]
    assert set(answer["judgments"]) == {"true", "false"}
    assert isinstance(answer["ambiguous"], bool)
    assert 0.0 <= answer["noul"] <= 1.0
    entry = result["plan"]["fields"][0]
    assert entry["branches"] == 2
    assert entry["combine"] == "sigmoid(true margin - false margin), each judged on its own"


def test_criteria_are_for_noul_only():
    with pytest.raises(CompileError, match="noul questions only"):
        Dynajev(FakeBackend()).decide(
            DecideIn.model_validate(
                {
                    "context": "x",
                    "questions": {"q": {"type": "choice", "instructions": "Which?", "options": ["a", "b"], "criteria": {"true": "t"}}},
                }
            )
        )
