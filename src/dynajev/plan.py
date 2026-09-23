"""The compiled plan: what gets built for one request.

A request compiles to a small graph over the frozen model:

- `Branch`: one prompt, as a token sequence that ends at an answer boundary.
  Branches of a request share prefixes; the reader prefills each shared
  segment once.
- `Read`: what to take at a branch's last position: rows of the unembedding
  (`rows`), a prototype per label (`prototype`), or the hidden state itself
  for a fitted probe (`hidden`), at a given depth.
- `Combine`: how a field turns its reads into an answer: a sigmoid of a
  Yes-minus-No margin, a softmax, an expected level, independent flags, a
  softmax over per-option margins, or two independent criterion judgments.
- `Decode`: a field that has no rows to read and generates instead.

`compile` builds a `Plan`; `executor.run_plan` runs it; `Plan.describe`
reports what was built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dynajev.prompts import ASSISTANT_PREFIX, SYSTEM

ReadOp = Literal["rows", "prototype", "hidden"]
CombineOp = Literal["sigmoid", "softmax", "expectation", "flags", "margin_softmax", "criteria"]


@dataclass
class Branch:
    id: str
    field: str
    block: str
    system: str = SYSTEM
    assistant_prefix: str = ASSISTANT_PREFIX
    label: str | None = None
    tokens: list[int] | None = None
    # Decoder layers to run for this branch. None runs the whole stack.
    depth: int | None = None
    # A dependent question continues this branch's conversation, so its cache
    # must be computed exactly and kept for the rest of the request.
    keep: bool = False
    continued_from: str | None = None


@dataclass
class Read:
    branch: str
    op: ReadOp = "rows"
    groups: list[list[int]] = field(default_factory=list)
    pieces: list[list[int]] | None = None
    layer: int | None = None


@dataclass
class Combine:
    op: CombineOp
    labels: list[str]
    origin: float = 0.0
    correction: Any = None


@dataclass
class Decode:
    branch: str
    mode: Literal["short", "open"] = "short"
    extractive: bool = False


@dataclass
class FieldPlan:
    id: str
    qtype: str
    kind: str
    head: str
    reason: str
    labels: list[str]
    branches: list[Branch]
    reads: list[Read] = field(default_factory=list)
    combine: Combine | None = None
    decode: Decode | None = None
    letters: dict[str, str] | None = None
    # What the model would write for each class at the boundary, used when a
    # dependent question continues this field's conversation.
    answer_tokens: list[str] | None = None
    question: str = ""
    depends_on: Any = None
    include_answers: list[str] = field(default_factory=list)
    signature: str | None = None
    head_source: Literal["readout", "fitted", "stored"] = "readout"

    @property
    def prompt(self) -> str:
        return self.branches[0].block if self.branches else ""

    @property
    def depth(self) -> int | None:
        depths = [b.depth for b in self.branches]
        if any(d is None for d in depths):
            return None
        return max(depths) if depths else None  # type: ignore[type-var]


@dataclass
class Plan:
    fields: list[FieldPlan]
    stages: list[list[str]]
    num_layers: int = 0

    def field(self, field_id: str) -> FieldPlan:
        for item in self.fields:
            if item.id == field_id:
                return item
        raise KeyError(field_id)

    def describe(self, reads: list[dict[str, Any]], skipped: list[str]) -> dict[str, Any]:
        """A compact account of the architecture built for this request."""

        layers = self.num_layers
        fields = []
        for item in self.fields:
            entry: dict[str, Any] = {
                "id": item.id,
                "type": item.qtype,
                "head": item.head,
                "branches": len(item.branches),
                "tokens": [len(b.tokens or []) for b in item.branches],
            }
            if item.combine is not None:
                entry["read"] = _describe_reads(item)
                entry["combine"] = _describe_combine(item)
                depth = item.depth
                entry["depth"] = f"{depth if depth is not None else layers}/{layers} layers" if layers else None
            if item.decode is not None:
                entry["decode"] = "chat completion" if item.decode.mode == "open" else (
                    "short quote, checked verbatim" if item.decode.extractive else "short phrase"
                )
            if item.head_source != "readout":
                entry["source"] = f"{item.head_source} head {item.signature}" if item.signature else item.head_source
            if item.depends_on is not None:
                entry["depends_on"] = {"question": item.depends_on.question, "when": item.depends_on.when}
            if item.include_answers:
                entry["include_answers"] = list(item.include_answers)
                continued = next((b.continued_from for b in item.branches if b.continued_from), None)
                if continued:
                    entry["continues"] = continued
            if item.id in skipped:
                entry["skipped"] = True
            fields.append(entry)
        processed = sum(r.get("processed_tokens", 0) for r in reads)
        naive = sum(r.get("naive_tokens", 0) for r in reads)
        forwards = sum(r.get("forwards", 0) for r in reads)
        branches = sum(r.get("branches", 0) for r in reads)
        shared = [seg for r in reads for seg in r.get("segments", []) if seg.get("rows", 0) > 1]
        summary = (
            f"{branches} branch{'es' if branches != 1 else ''} in {len(self.stages)} stage{'s' if len(self.stages) != 1 else ''}; "
            f"{len(shared)} shared segment{'s' if len(shared) != 1 else ''}; {forwards} forward{'s' if forwards != 1 else ''}; "
            f"{processed} of {naive} prompt tokens processed"
        )
        return {
            "summary": summary,
            "stages": self.stages,
            "fields": fields,
            "reads": reads,
        }


def _describe_reads(item: FieldPlan) -> str:
    first = item.reads[0] if item.reads else None
    if first is None:
        return ""
    at = f" at {len(item.reads)} positions" if len(item.reads) > 1 else ""
    if first.op == "hidden":
        where = "final layer" if first.layer is None else f"layer {first.layer}"
        return f"hidden state at {where}{at}"
    if first.op == "prototype":
        return f"mean rows of each label ({sum(len(p) for p in first.pieces or [])} tokens){at}"
    rows = sum(len(g) for g in first.groups)
    return f"{rows} unembedding rows{at}"


def _describe_combine(item: FieldPlan) -> str:
    combine = item.combine
    assert combine is not None
    n = len(combine.labels)
    text = {
        "sigmoid": "sigmoid(Yes - No)",
        "softmax": f"softmax over {n}",
        "expectation": f"softmax over {n} levels, expected level",
        "flags": f"{n} independent sigmoids",
        "margin_softmax": f"softmax over {n} yes/no margins",
        "criteria": "sigmoid(true margin - false margin), each judged on its own",
    }[combine.op]
    correction = combine.correction
    if correction is not None and getattr(correction, "chosen", "readout") != "readout":
        if correction.chosen == "affine":
            text += f" + fitted bias, T={correction.temperature:.2f}"
        elif correction.chosen == "ridge_probe":
            where = "final layer" if correction.layer is None else f"layer {correction.layer}"
            text = f"ridge probe on the {where} hidden state, softmax over {n}"
    return text
