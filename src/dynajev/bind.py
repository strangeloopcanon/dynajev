"""Lower field jobs to plan nodes: which head, which branches, which token rows."""

from __future__ import annotations

from typing import Protocol

from dynajev.compile import FieldJob
from dynajev.errors import CompileError
from dynajev.plan import Branch, Combine, Decode, FieldPlan, Read
from dynajev.prompts import (
    OPEN_SYSTEM,
    boolean_block,
    criterion_block,
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

_KIND_TO_TYPE = {
    "boolean": "noul",
    "categorical": "choice",
    "ordinal": "score",
    "multilabel": "flags",
    "extract": "quote",
    "open": "open",
    "generate": "open",
}


class Encoder(Protocol):
    def token_ids(self, text: str) -> list[int]: ...

    def single_token(self, text: str) -> int | None: ...


def bind_field(job: FieldJob, enc: Encoder) -> FieldPlan:
    if job.kind == "boolean":
        return _boolean(job, enc)
    if job.kind == "categorical":
        return _categorical(job, enc)
    if job.kind == "ordinal":
        return _ordinal(job, enc)
    if job.kind == "multilabel":
        return _multilabel(job, enc)
    if job.kind == "extract":
        return _decode_field(
            job,
            "extractive_decode",
            "Open string marked as a quote. No closed row exists, so only this field decodes, briefly, "
            "and the result is checked against the state: verbatim or flagged.",
            Branch(id=job.id, field=job.id, block=extract_block(job.question)),
            Decode(branch=job.id, mode="short", extractive=True),
        )
    if job.kind == "generate":
        return _decode_field(
            job,
            "short_decode",
            "Open string. A closed head would invent a label the question did not allow, "
            "so only this field generates, capped at a short phrase.",
            Branch(id=job.id, field=job.id, block=generate_block(job.question)),
            Decode(branch=job.id, mode="short"),
        )
    if job.kind == "open":
        return _decode_field(
            job,
            "chat_decode",
            "Open question with no answer set. There are no rows to read, so this is an ordinary chat "
            "completion: the model generates until it ends its turn. It pays full decoding cost.",
            Branch(id=job.id, field=job.id, block=open_block(job.question), assistant_prefix="", system=OPEN_SYSTEM),
            Decode(branch=job.id, mode="open"),
        )
    raise CompileError(f"Unknown field kind {job.kind}.")


def yes_no_groups(enc: Encoder) -> list[list[int]]:
    yes = _forms(enc, YES_FORMS)
    no = _forms(enc, NO_FORMS)
    if not yes or not no:
        raise CompileError("This tokenizer has no single token for Yes and No, so the boolean head cannot be sliced.")
    if set(yes) & set(no):
        raise CompileError("Yes and No collide in this tokenizer.")
    return [no, yes]


def _plan(job: FieldJob, head: str, reason: str, branches: list[Branch], **extra) -> FieldPlan:
    return FieldPlan(
        id=job.id,
        qtype=job.qtype or _KIND_TO_TYPE.get(job.kind, "open"),
        kind=job.kind,
        head=head,
        reason=reason,
        labels=list(job.labels),
        branches=branches,
        question=job.question,
        depends_on=job.depends_on,
        include_answers=list(job.include_answers),
        **extra,
    )


def _decode_field(job: FieldJob, head: str, reason: str, branch: Branch, decode: Decode) -> FieldPlan:
    return _plan(job, head, reason, [branch], decode=decode)


def _boolean(job: FieldJob, enc: Encoder) -> FieldPlan:
    groups = yes_no_groups(enc)
    if job.criteria:
        return _criteria(job, groups)
    branch = Branch(id=job.id, field=job.id, block=boolean_block(job.question))
    return _plan(
        job,
        "binary_margin",
        "Yes/no is a two-row head: log-sum-exp of the Yes tokens minus log-sum-exp of the No tokens, "
        "then a sigmoid. The rows are taken from the frozen unembedding. Nothing is generated.",
        [branch],
        reads=[Read(branch=branch.id, groups=groups)],
        combine=Combine(op="sigmoid", labels=["no", "yes"]),
        answer_tokens=["No", "Yes"],
    )


def _criteria(job: FieldJob, groups: list[list[int]]) -> FieldPlan:
    criteria = job.criteria or {}
    branches = [
        Branch(id=f"{job.id}/{side}", field=job.id, block=criterion_block(job.question, criteria[side]), label=side)
        for side in ("true", "false")
        if side in criteria
    ]
    return _plan(
        job,
        "criteria_judgment",
        "Each criterion is judged on its own branch as a yes/no margin, without seeing the other. "
        "The probability is the sigmoid of the true margin minus the false margin; the separate judgments are "
        "reported too, so 'both fit' or 'neither fits' shows up instead of being hidden in one margin.",
        branches,
        reads=[Read(branch=b.id, groups=groups) for b in branches],
        combine=Combine(op="criteria", labels=["no", "yes"]),
    )


def _categorical(job: FieldJob, enc: Encoder) -> FieldPlan:
    strategy = job.strategy
    direct = _direct_groups(enc, job.labels)
    if strategy == "margin":
        return _option_margins(job, enc)
    if strategy == "prototype":
        return _prototype(job, enc)
    if strategy == "slice" or (strategy == "auto" and direct is not None):
        if direct is None:
            plan = _letters(job, enc)
            plan.reason = (
                "A direct slice was requested, but the labels are not unique single tokens. "
                "Fell back to a letter head: the options are written out, and the unembedding is sliced to the letter tokens."
            )
            return plan
        branch = Branch(id=job.id, field=job.id, block=slice_block(job.question, job.labels))
        return _plan(
            job,
            "vocab_slice",
            "Every label is already one distinct token, so the head is those rows of the frozen unembedding. "
            "One forward, one position, softmax over the labels. No letter code and no decode.",
            [branch],
            reads=[Read(branch=branch.id, groups=direct)],
            combine=Combine(op="softmax", labels=list(job.labels)),
            answer_tokens=list(job.labels),
        )
    return _letters(job, enc)


def _letters(job: FieldJob, enc: Encoder) -> FieldPlan:
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
    branch = Branch(id=job.id, field=job.id, block=letter_block(job.question, pairs))
    return _plan(
        job,
        "letter_slice",
        "Labels are not unique single tokens, so each one is given a letter and the head is the letter rows "
        "of the unembedding, read at the JSON answer boundary. One forward scores every option. "
        "This is the Simple Jev / OpenJev choice head.",
        [branch],
        reads=[Read(branch=branch.id, groups=groups)],
        combine=Combine(op="softmax", labels=list(job.labels)),
        letters={label: letter for letter, label in pairs},
        answer_tokens=[letter for letter, _ in pairs],
    )


def _option_margins(job: FieldJob, enc: Encoder) -> FieldPlan:
    groups = yes_no_groups(enc)
    branches = [
        Branch(id=f"{job.id}/{i}", field=job.id, block=option_margin_block(job.question, label), label=label)
        for i, label in enumerate(job.labels)
    ]
    return _plan(
        job,
        "option_margin",
        "Each option is its own yes/no question on a shared prefix, and the yes-minus-no margins are softmaxed "
        "so the options compete. This is the Glance pick-one head: more branches than a letter slice, "
        "and the candidate text is judged instead of a code.",
        branches,
        reads=[Read(branch=b.id, groups=groups) for b in branches],
        combine=Combine(op="margin_softmax", labels=list(job.labels)),
    )


def _prototype(job: FieldJob, enc: Encoder) -> FieldPlan:
    pieces: list[list[int]] = []
    for label in job.labels:
        ids = enc.token_ids(label)
        if not ids:
            raise CompileError(f"Label {label!r} has no tokens.")
        pieces.append(ids)
    branch = Branch(id=job.id, field=job.id, block=slice_block(job.question, job.labels))
    return _plan(
        job,
        "prototype",
        "Prototype head: each label is the mean of its unembedding rows, dotted with the hidden state at the "
        "answer boundary. One forward, no new weights. Overlapping words between labels blur this head; "
        "a letter slice is the better default.",
        [branch],
        reads=[Read(branch=branch.id, op="prototype", pieces=pieces)],
        combine=Combine(op="softmax", labels=list(job.labels)),
    )


def _ordinal(job: FieldJob, enc: Encoder) -> FieldPlan:
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
    branch = Branch(id=job.id, field=job.id, block=ordinal_block(job.question, job.labels, digits))
    return _plan(
        job,
        "ordinal_expectation",
        "A rating is not an argmax. The head is the digit rows of the unembedding; softmax gives a distribution "
        "over levels, and the score is the expected level. Same construction as Glance's rubric readout.",
        [branch],
        reads=[Read(branch=branch.id, groups=groups)],
        combine=Combine(op="expectation", labels=list(job.labels), origin=job.origin),
        answer_tokens=list(digits),
    )


def _multilabel(job: FieldJob, enc: Encoder) -> FieldPlan:
    groups = yes_no_groups(enc)
    branches = [
        Branch(id=f"{job.id}/{i}", field=job.id, block=tag_block(job.question, label), label=label)
        for i, label in enumerate(job.labels)
    ]
    return _plan(
        job,
        "multilabel_margin",
        "Flags do not compete. Each label is its own yes/no margin and its own sigmoid, on a shared prefix. "
        "A softmax here would force exactly one flag, which is a different question.",
        branches,
        reads=[Read(branch=b.id, groups=groups) for b in branches],
        combine=Combine(op="flags", labels=list(job.labels)),
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
