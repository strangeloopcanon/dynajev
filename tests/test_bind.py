import re

from dynajev.bind import bind_field
from dynajev.compile import FieldJob


class WordEncoder:
    def __init__(self):
        self.ids: dict[str, int] = {}

    def _pieces(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def token_ids(self, text: str) -> list[int]:
        return [self._id(piece) for piece in self._pieces(text)]

    def single_token(self, text: str) -> int | None:
        pieces = self._pieces(text)
        if len(pieces) == 1:
            return self._id(pieces[0])
        return None

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.ids) + 1
        return self.ids[piece]


def test_single_token_labels_slice_the_unembedding():
    bound = bind_field(
        FieldJob(id="answer", kind="categorical", question="Which color?", labels=["red", "blue", "green"], strategy="auto"),
        WordEncoder(),
    )
    assert bound.head == "vocab_slice"
    assert bound.nodes[0].groups[0] != bound.nodes[0].groups[1]


def test_phrases_become_a_letter_head():
    bound = bind_field(
        FieldJob(
            id="answer",
            kind="categorical",
            question="What is needed?",
            labels=["a replacement", "a tracking update"],
            strategy="auto",
        ),
        WordEncoder(),
    )
    assert bound.head == "letter_slice"
    assert bound.letters == {"a replacement": "A", "a tracking update": "B"}


def test_margin_strategy_is_one_boolean_per_option():
    bound = bind_field(
        FieldJob(
            id="answer",
            kind="categorical",
            question="What is needed?",
            labels=["a replacement", "a tracking update", "a reset"],
            strategy="margin",
        ),
        WordEncoder(),
    )
    assert bound.head == "option_margin"
    assert len(bound.nodes) == 3


def test_flags_do_not_share_a_softmax_node():
    bound = bind_field(
        FieldJob(id="flags", kind="multilabel", question="Which apply?", labels=["legal", "billing"]),
        WordEncoder(),
    )
    assert bound.head == "multilabel_margin"
    assert len(bound.nodes) == 2
