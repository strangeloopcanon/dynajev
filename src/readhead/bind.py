"""Bind field jobs to a tokenizer: pick the actual head and its token rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from readhead.compile import FieldJob
from readhead.errors import CompileError
from readhead.prompts import (
    ASSISTANT_PREFIX,
    OPEN_SYSTEM,
    SYSTEM,
    boolean_block,
    extract_block,
    generate_block,
    letter_block,
    open_block,
    option_margin_block,
    ordinal_block,
    slice_block,
    tag_block,
)

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx")
YES_FORMS = ("Yes", "yes")
NO_FORMS = ("No", "no")


class Encoder(Protocol):
    def token_ids(self, text: str) -> list[int]: ...

    def single_token(self, text: str) -> int | None: ...


@dataclass
class Node:
    """One answer boundary. A field may own several (one per option or flag)."""

    field_id: str
    block: str
    assistant_prefix: str = ASSISTANT_PREFIX
    system: str = SYSTEM
    # For a closed softmax/margin node: token ids per class, class order fixed.
    groups: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    # Prototype head: token pieces per label, same order as class_names.
    pieces: list[list[int]] | None = None
    mode: str = "classes"  # classes | prototype | decode
    extractive: bool = False
    # decode nodes: "short" stops at a phrase boundary, "open" runs to end of turn
    decode: str = "short"
    label: str | None = None
    ids: list[int] | None = None
    hidden: object | None = None
    logits: list[float] | None = None
    allowed_mass: float | None = None


@dataclass
class BoundField:
    id: str
    kind: str
    head: str
    reason: str
    labels: list[str]
    nodes: list[Node]
    letters: dict[str, str] | None = None
    origin: float = 0.0
    prompt_preview: str = ""


def bind_field(job: FieldJob, enc: Encoder) -> BoundField:
    if job.kind == "boolean":
        return _boolean(job, enc)
    if job.kind == "categorical":
        return _categorical(job, enc)
    if job.kind == "ordinal":
        return _ordinal(job, enc)
    if job.kind == "multilabel":
        return _multilabel(job, enc)
    if job.kind == "extract":
        node = Node(
            field_id=job.id,
            block=extract_block(job.question),
            mode="decode",
            extractive=True,
        )
        return BoundField(
            id=job.id,
            kind=job.kind,
            head="extractive_decode",
            reason=(
                "Open string marked as a quote. No closed row exists, so only this field decodes, briefly, "
                "and the result is checked against the state: verbatim or flagged."
            ),
            labels=[],
            nodes=[node],
            prompt_preview=node.block,
        )
    if job.kind == "generate":
        node = Node(field_id=job.id, block=generate_block(job.question), mode="decode", extractive=False)
        return BoundField(
            id=job.id,
            kind=job.kind,
            head="short_decode",
            reason=(
                "Open string. A closed head would invent a label the question did not allow, "
                "so only this field generates, capped at a short phrase."
            ),
            labels=[],
            nodes=[node],
            prompt_preview=node.block,
        )
    if job.kind == "open":
        node = Node(
            field_id=job.id,
            block=open_block(job.question),
            assistant_prefix="",
            system=OPEN_SYSTEM,
            mode="decode",
            extractive=False,
            decode="open",
        )
        return BoundField(
            id=job.id,
            kind=job.kind,
            head="chat_decode",
            reason=(
                "Open question with no answer set. There are no rows to read, so this is an ordinary chat "
                "completion: the model generates until it ends its turn. It pays full decoding cost."
            ),
            labels=[],
            nodes=[node],
            prompt_preview=node.block,
        )
    raise CompileError(f"Unknown field kind {job.kind}.")


def _boolean(job: FieldJob, enc: Encoder) -> BoundField:
    yes = _forms(enc, YES_FORMS)
    no = _forms(enc, NO_FORMS)
    if not yes or not no:
        raise CompileError("This tokenizer has no single token for Yes and No, so the boolean head cannot be sliced.")
    if set(yes) & set(no):
        raise CompileError("Yes and No collide in this tokenizer.")
    node = Node(
        field_id=job.id,
        block=boolean_block(job.question),
        groups=[no, yes],
        class_names=["no", "yes"],
    )
    return BoundField(
        id=job.id,
        kind="boolean",
        head="binary_margin",
        reason=(
            "Yes/no is a two-row head: log-sum-exp of the Yes tokens minus log-sum-exp of the No tokens, "
            "then a sigmoid. The rows are taken from the frozen unembedding. Nothing is generated."
        ),
        labels=["no", "yes"],
        nodes=[node],
        prompt_preview=node.block,
    )


def _categorical(job: FieldJob, enc: Encoder) -> BoundField:
    strategy = job.strategy
    direct = _direct_groups(enc, job.labels)
    if strategy == "margin":
        return _option_margins(job)
    if strategy == "prototype":
        return _prototype(job, enc)
    if strategy == "slice" or (strategy == "auto" and direct is not None):
        if direct is None:
            bound = _letters(job, enc)
            bound.reason = (
                "A direct slice was requested, but the labels are not unique single tokens. "
                "Fell back to a letter head: the options are written out, and the unembedding is sliced to the letter tokens."
            )
            return bound
        node = Node(
            field_id=job.id,
            block=slice_block(job.question, job.labels),
            groups=direct,
            class_names=list(job.labels),
        )
        return BoundField(
            id=job.id,
            kind="categorical",
            head="vocab_slice",
            reason=(
                "Every label is already one distinct token, so the head is those rows of the frozen unembedding. "
                "One forward, one position, softmax over the labels. No letter code and no decode."
            ),
            labels=list(job.labels),
            nodes=[node],
            prompt_preview=node.block,
        )
    return _letters(job, enc)


def _letters(job: FieldJob, enc: Encoder) -> BoundField:
    if len(job.labels) > len(LETTERS):
        raise CompileError("Too many labels for the letter head.")
    pairs: list[tuple[str, str]] = []
    groups: list[list[int]] = []
    used: set[int] = set()
    cursor = 0
    for label in job.labels:
        chosen: tuple[str, int] | None = None
        while cursor < len(LETTERS):
            letter = LETTERS[cursor]
            cursor += 1
            tid = enc.single_token(letter)
            if tid is None or tid in used:
                continue
            chosen = (letter, tid)
            used.add(tid)
            break
        if chosen is None:
            raise CompileError("Ran out of single-token letters in this tokenizer.")
        pairs.append((chosen[0], label))
        groups.append([chosen[1]])
    node = Node(
        field_id=job.id,
        block=letter_block(job.question, pairs),
        groups=groups,
        class_names=list(job.labels),
    )
    return BoundField(
        id=job.id,
        kind="categorical",
        head="letter_slice",
        reason=(
            "Labels are not unique single tokens, so each one is given a letter and the head is the letter rows "
            "of the unembedding, read at the JSON answer boundary. One forward scores every option. "
            "This is the Simple Jev / OpenJev choice head."
        ),
        labels=list(job.labels),
        nodes=[node],
        letters={label: letter for letter, label in pairs},
        prompt_preview=node.block,
    )


def _option_margins(job: FieldJob) -> BoundField:
    nodes = [
        Node(
            field_id=job.id,
            block=option_margin_block(job.question, label),
            label=label,
            mode="margin_option",
        )
        for label in job.labels
    ]
    return BoundField(
        id=job.id,
        kind="categorical",
        head="option_margin",
        reason=(
            "Each option is its own yes/no question on a shared prefix, and the yes-minus-no margins are softmaxed "
            "so the options compete. This is the Glance pick-one head: more branches than a letter slice, "
            "and the candidate text is judged instead of a code."
        ),
        labels=list(job.labels),
        nodes=nodes,
        prompt_preview=nodes[0].block,
    )


def _prototype(job: FieldJob, enc: Encoder) -> BoundField:
    pieces: list[list[int]] = []
    for label in job.labels:
        ids = enc.token_ids(label)
        if not ids:
            raise CompileError(f"Label {label!r} has no tokens.")
        pieces.append(ids)
    node = Node(
        field_id=job.id,
        block=slice_block(job.question, job.labels),
        class_names=list(job.labels),
        pieces=pieces,
        mode="prototype",
    )
    return BoundField(
        id=job.id,
        kind="categorical",
        head="prototype",
        reason=(
            "Prototype head: each label is the mean of its unembedding rows, dotted with the hidden state at the "
            "answer boundary. One forward, no new weights. Overlapping words between labels blur this head; "
            "a letter slice is the better default."
        ),
        labels=list(job.labels),
        nodes=[node],
        prompt_preview=node.block,
    )


def _ordinal(job: FieldJob, enc: Encoder) -> BoundField:
    digits = job.score_tokens or [str(i) for i in range(len(job.labels))]
    if len(digits) != len(job.labels):
        raise CompileError("Ordinal score tokens must line up with levels.")
    groups: list[list[int]] = []
    seen: set[int] = set()
    for digit in digits:
        tid = enc.single_token(digit)
        if tid is None:
            raise CompileError(f"Level token {digit!r} is not a single vocabulary token.")
        if tid in seen:
            raise CompileError(f"Level token {digit!r} collides with another level.")
        seen.add(tid)
        groups.append([tid])
    node = Node(
        field_id=job.id,
        block=ordinal_block(job.question, job.labels, digits),
        groups=groups,
        class_names=list(job.labels),
    )
    return BoundField(
        id=job.id,
        kind="ordinal",
        head="ordinal_expectation",
        reason=(
            "A rating is not an argmax. The head is the digit rows of the unembedding; softmax gives a distribution "
            "over levels, and the score is the expected level. Same construction as Glance's rubric readout."
        ),
        labels=list(job.labels),
        nodes=[node],
        origin=job.origin,
        prompt_preview=node.block,
    )


def _multilabel(job: FieldJob, enc: Encoder) -> BoundField:
    # Touch the encoder so a tokenizer without Yes/No fails here, once.
    _boolean(FieldJob(id=job.id, kind="boolean", question=job.question), enc)
    nodes = [
        Node(
            field_id=job.id,
            block=tag_block(job.question, label),
            label=label,
            mode="independent",
        )
        for label in job.labels
    ]
    return BoundField(
        id=job.id,
        kind="multilabel",
        head="multilabel_margin",
        reason=(
            "Flags do not compete. Each label is its own yes/no margin and its own sigmoid, on a shared prefix. "
            "A softmax here would force exactly one flag, which is a different question."
        ),
        labels=list(job.labels),
        nodes=nodes,
        prompt_preview=nodes[0].block,
    )


def _forms(enc: Encoder, forms: tuple[str, ...]) -> list[int]:
    found: list[int] = []
    for form in forms:
        tid = enc.single_token(form)
        if tid is not None and tid not in found:
            found.append(tid)
    return found


def _direct_groups(enc: Encoder, labels: list[str]) -> list[list[int]] | None:
    groups: list[list[int]] = []
    seen: set[int] = set()
    for label in labels:
        tid = enc.single_token(label)
        if tid is None or tid in seen:
            return None
        seen.add(tid)
        groups.append([tid])
    return groups
