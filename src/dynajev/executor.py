"""Run a compiled plan on a backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynajev.plan import Branch, FieldPlan, Plan
from dynajev.prompts import user_content
from dynajev.score import binary_from_logits, ordinal_expectation, softmax


@dataclass
class ReadOut:
    logits: list[float] | None = None
    mass: float | None = None
    hidden: Any = None


@dataclass
class RunResult:
    fields: list[dict[str, Any]]
    reads: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    prefill_tokens: int = 0
    shared_prefix_tokens: int = 0
    cached_prefix_tokens: int = 0


def encode_branch(backend: Any, branch: Branch, context: str) -> list[int]:
    return backend.encode_prompt(user_content(context, branch.block), branch.assistant_prefix, branch.system)


def state_anchor(backend: Any, context: str, branch: Branch) -> list[int]:
    """Token prefix that every question about this state shares: system text plus the state block."""

    return backend.encode_prompt(user_content(context, ""), branch.assistant_prefix, branch.system)


def run_plan(plan: Plan, backend: Any, reader: Any, context: str) -> RunResult:
    """Run the plan stage by stage; a stage sees every answer from the stages before it."""

    by_id = {item.id: item for item in plan.fields}
    results: dict[str, dict[str, Any]] = {}
    out = RunResult(fields=[])
    kept: dict[tuple[int, ...], Any] = {}
    for stage in plan.stages:
        items = []
        for field_id in stage:
            item = by_id[field_id]
            reason = _skip_reason(item, results)
            if reason:
                results[item.id] = skipped_field(item, reason)
                out.skipped.append(item.id)
                continue
            for branch in item.branches:
                branch.tokens = _encode(backend, item, branch, context, by_id, results)
            items.append(item)
        closed = [item for item in items if item.combine is not None]
        if closed:
            branches = [branch for item in closed for branch in item.branches]
            anchor = state_anchor(backend, context, branches[0])
            hiddens, record = reader.read(branches, anchor, kept)
            out.reads.append(record)
            out.prefill_tokens += record["processed_tokens"]
            out.shared_prefix_tokens = max(out.shared_prefix_tokens, record.get("shared_prefix_tokens", 0))
            out.cached_prefix_tokens += record.get("prefix_cache_hit_tokens", 0)
            for item in closed:
                reads = [readout(backend, read, hiddens[read.branch]) for read in item.reads]
                results[item.id] = combine_field(item, reads, backend)
        for item in items:
            if item.decode is None:
                continue
            branch = next(b for b in item.branches if b.id == item.decode.branch)
            text, count = backend.generate(branch.tokens, context, item.decode.extractive, item.decode.mode)
            out.prefill_tokens += len(branch.tokens or [])
            results[item.id] = decode_field(item, text, count, context)
    out.fields = [results[item.id] for item in plan.fields if item.id in results]
    return out


def continuable(item: FieldPlan) -> bool:
    """A dependent question can continue this question's conversation (and fork its cache)."""

    return item.combine is not None and len(item.branches) == 1 and bool(item.answer_tokens)


def matches(payload: dict[str, Any], when: Any) -> bool:
    values = when if isinstance(when, list) else [when]
    answer = payload.get("answer")
    if isinstance(answer, bool):
        wanted = {v if isinstance(v, bool) else str(v).strip().lower() in {"yes", "true"} for v in values}
        return answer in wanted
    if isinstance(answer, list):
        return bool({str(v).strip() for v in values} & set(answer))
    return str(answer).strip().lower() in {str(v).strip().lower() for v in values}


def render_answer(payload: dict[str, Any]) -> str:
    answer = payload.get("answer")
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, list):
        return ", ".join(answer) if answer else "none"
    return "" if answer is None else str(answer)


def _skip_reason(item: FieldPlan, results: dict[str, dict[str, Any]]) -> str | None:
    dep = item.depends_on
    if dep is None:
        return None
    parent = results.get(dep.question)
    if parent is None or parent.get("skipped"):
        return f"Skipped because {dep.question} was skipped."
    if not matches(parent, dep.when):
        return f"Skipped because {dep.question} answered {render_answer(parent)!r}, not {dep.when!r}."
    return None


def _encode(
    backend: Any, item: FieldPlan, branch: Branch, context: str, by_id: dict[str, FieldPlan], results: dict[str, dict[str, Any]]
) -> list[int]:
    if not item.include_answers:
        return encode_branch(backend, branch, context)
    parent_id = branch.continued_from
    parent = by_id.get(parent_id) if parent_id else None
    parent_result = results.get(parent_id) if parent_id else None
    others = [i for i in item.include_answers if i != parent_id]
    lines = []
    for other in others:
        result = results.get(other)
        if result is None or result.get("skipped"):
            continue
        lines.append(f"Q: {by_id[other].question}\nA: {render_answer(result)}")
    known = ("Earlier answers:\n" + "\n".join(lines) + "\n\n") if lines else ""
    if parent is not None and parent_result is not None and not parent_result.get("skipped"):
        answer = _boundary_answer(parent, parent_result)
        closing = '"}' if parent.branches[0].assistant_prefix else ""
        tokens = None
        if answer is not None and hasattr(backend, "continue_prompt"):
            tokens = backend.continue_prompt(
                parent.branches[0].tokens or [], answer + closing, known + branch.block, branch.assistant_prefix
            )
        if tokens is not None:
            return tokens
        lines.insert(0, f"Q: {parent.question}\nA: {render_answer(parent_result)}")
        known = "Earlier answers:\n" + "\n".join(lines) + "\n\n"
    branch.continued_from = None
    return backend.encode_prompt(user_content(context, known + branch.block), branch.assistant_prefix, branch.system)


def _boundary_answer(parent: FieldPlan, result: dict[str, Any]) -> str | None:
    """What the model would have written at the parent's boundary: Yes/No, the label, letter, or digit."""

    if not parent.answer_tokens:
        return None
    answer = result.get("answer")
    if isinstance(answer, bool):
        index = 1 if answer else 0
    elif answer in parent.labels:
        index = parent.labels.index(answer)
    else:
        return None
    return parent.answer_tokens[index]


def skipped_field(item: FieldPlan, reason: str) -> dict[str, Any]:
    return {
        "id": item.id,
        "type": item.qtype,
        "head": item.head,
        "kind": item.kind,
        "reason": reason,
        "skipped": True,
        "answer": None,
        "confidence": None,
        "probabilities": None,
        "score": None,
        "letters": None,
        "allowed_mass": None,
        "rows_scored": 0,
        "sequences": 0,
        "generated_tokens": 0,
        "prompt": item.prompt,
        "logits": None,
        "weak_reading": False,
        "warning": None,
        "depth": None,
        "head_source": item.head_source,
    }


def readout(backend: Any, read: Any, hidden: Any) -> ReadOut:
    if read.op == "hidden":
        return ReadOut(hidden=hidden)
    if read.op == "prototype":
        return ReadOut(logits=backend.prototype_logits(hidden, read.pieces), hidden=hidden)
    logits, mass = backend.class_logits(hidden, read.groups)
    return ReadOut(logits=list(logits), mass=mass, hidden=hidden)


def combine_field(item: FieldPlan, reads: list[ReadOut], backend: Any) -> dict[str, Any]:
    combine = item.combine
    assert combine is not None
    labels = item.labels
    correction = combine.correction
    head = item.head
    reason = item.reason
    first = reads[0]
    logits = first.logits
    if correction is not None and correction.chosen == "ridge_probe":
        logits = correction.ridge_logits(backend.as_vector(first.hidden))
        head = "ridge_probe"
        reason = correction.note
    elif correction is not None and correction.chosen == "affine":
        logits = correction.affine_logits(list(logits or []))
        head = f"{item.head}+affine"
        reason = correction.note
    masses = [r.mass for r in reads if r.mass is not None]
    mean_mass = sum(masses) / len(masses) if masses else None
    payload: dict[str, Any] = {
        "id": item.id,
        "type": item.qtype,
        "head": head,
        "kind": item.kind,
        "reason": reason,
        "answer": None,
        "confidence": None,
        "probabilities": None,
        "score": None,
        "letters": item.letters,
        "allowed_mass": None if mean_mass is None else round(mean_mass, 4),
        "rows_scored": sum(len(g) for read in item.reads for g in read.groups),
        "sequences": len(item.branches),
        "generated_tokens": 0,
        "prompt": item.prompt,
        "logits": None,
        "weak_reading": mean_mass is not None and mean_mass < 0.05,
        "warning": _weak_warning(mean_mass),
        "depth": item.depth,
        "head_source": item.head_source,
    }
    op = combine.op
    if op == "flags":
        probabilities = {
            label: binary_from_logits(r.logits[0], r.logits[1]) for label, r in zip(labels, reads) if r.logits
        }
        payload["answer"] = [label for label, p in probabilities.items() if p >= 0.5]
        payload["confidence"] = round(sum(abs(p - 0.5) * 2 for p in probabilities.values()) / max(len(probabilities), 1), 4)
        payload["probabilities"] = {label: round(p, 4) for label, p in probabilities.items()}
        payload["logits"] = {label: round(r.logits[1] - r.logits[0], 4) for label, r in zip(labels, reads) if r.logits}
        return payload
    if op == "criteria":
        margins = {b.label: r.logits[1] - r.logits[0] for b, r in zip(item.branches, reads) if r.logits}
        margin = margins.get("true", 0.0) - margins.get("false", 0.0)
        judgments = {side: round(binary_from_logits(0.0, m), 4) for side, m in margins.items()}
        logits = [0.0, margin]
        payload["judgments"] = judgments
        if len(judgments) == 2:
            payload["ambiguous"] = (judgments["true"] >= 0.5) == (judgments["false"] >= 0.5)
    if op == "margin_softmax":
        logits = [(r.logits or [0.0, 0.0])[1] - (r.logits or [0.0, 0.0])[0] for r in reads]
    logits = list(logits or [])
    probabilities = softmax(logits)
    payload["probabilities"] = {label: round(p, 4) for label, p in zip(labels, probabilities)}
    payload["logits"] = {label: round(v, 4) for label, v in zip(labels, logits)}
    if first.hidden is None or item.reads[0].op != "rows":
        payload["allowed_mass"] = None
        payload["weak_reading"] = False
        payload["warning"] = None
    if op in {"sigmoid", "criteria"}:
        yes = probabilities[1]
        payload["answer"] = bool(yes >= 0.5)
        payload["confidence"] = round(max(yes, 1 - yes), 4)
        return payload
    winner = labels[max(range(len(labels)), key=lambda i: probabilities[i])]
    payload["answer"] = winner
    payload["confidence"] = round(max(probabilities), 4)
    if op == "expectation":
        payload["score"] = round(ordinal_expectation(probabilities, origin=combine.origin), 4)
    return payload


def decode_field(item: FieldPlan, text: str, count: int, context: str) -> dict[str, Any]:
    assert item.decode is not None
    payload: dict[str, Any] = {
        "id": item.id,
        "type": item.qtype,
        "head": item.head,
        "kind": item.kind,
        "reason": item.reason,
        "answer": text,
        "confidence": None,
        "probabilities": None,
        "score": None,
        "letters": None,
        "allowed_mass": None,
        "rows_scored": 0,
        "sequences": 1,
        "generated_tokens": count,
        "prompt": item.prompt,
        "logits": None,
        "weak_reading": False,
        "warning": None,
        "depth": None,
        "head_source": item.head_source,
    }
    if item.decode.extractive:
        verbatim = _normalize(text) in _normalize(context) if text else False
        payload["verbatim"] = verbatim
        if not verbatim:
            payload["warning"] = "The quoted text does not appear verbatim in the state."
    return payload


def typed_answer(field: dict[str, Any]) -> dict[str, Any]:
    """The compact, typed view of one field, keyed the way the question was asked."""

    qtype = field.get("type") or "open"
    out: dict[str, Any] = {"type": qtype}
    if field.get("skipped"):
        out["skipped"] = True
        return out
    probabilities = field.get("probabilities")
    if qtype == "noul":
        out["noul"] = None if not probabilities else probabilities.get("yes")
        if field.get("judgments"):
            out["judgments"] = field["judgments"]
            if "ambiguous" in field:
                out["ambiguous"] = field["ambiguous"]
    elif qtype == "choice":
        out["choice"] = field.get("answer")
        out["probabilities"] = probabilities
        out["confidence"] = field.get("confidence")
    elif qtype == "score":
        out["level"] = field.get("answer")
        out["score"] = field.get("score")
        out["probabilities"] = probabilities
        out["confidence"] = field.get("confidence")
    elif qtype == "flags":
        out["flags"] = field.get("answer")
        out["probabilities"] = probabilities
    elif qtype == "quote":
        out["quote"] = field.get("answer")
        out["verbatim"] = field.get("verbatim")
    else:
        out["text"] = field.get("answer")
    if field.get("weak_reading"):
        out["weak_reading"] = True
    return out


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _weak_warning(mass: float | None) -> str | None:
    if mass is None or mass >= 0.05:
        return None
    return (
        "Less than 5% of the next-token distribution sat on the allowed answers. "
        "The probabilities are renormalized over that thin slice."
    )
