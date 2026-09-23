"""Turn a request into field jobs. No tokenizer and no model yet.

The job says what geometry the answer has. Binding a tokenizer later decides
whether that geometry is a slice of the unembedding, a letter code, a bundle
of yes/no margins, or a short decode.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynajev.errors import CompileError

Kind = Literal["boolean", "categorical", "ordinal", "multilabel", "extract", "generate", "open"]
Strategy = Literal["auto", "slice", "letter", "margin", "prototype"]

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx")


class ExampleIn(BaseModel):
    context: str
    label: str | bool | int | float | None = None
    answers: dict[str, Any] | None = None


QuestionType = Literal["noul", "choice", "score", "flags", "quote", "open"]

# Typed question -> internal field kind. The names follow Jev so callers who
# know that API can read this one: noul is a yes/no probability, choice picks
# one option, score picks one ordered level, flags picks any number of options.
_TYPE_TO_KIND: dict[str, str] = {
    "noul": "boolean",
    "choice": "categorical",
    "score": "ordinal",
    "flags": "multilabel",
    "quote": "extract",
    "open": "open",
}


class DependsOn(BaseModel):
    """Ask this question only when another question's answer matches `when` (a value or a list of values)."""

    question: str
    when: Any = True


class Criteria(BaseModel):
    """What makes a `noul` true and what makes it false, each judged on its own."""

    true: str | None = None
    false: str | None = None


class QuestionIn(BaseModel):
    """One typed question. The type is declared, not inferred."""

    type: QuestionType
    instructions: str
    options: list[str] | None = None
    levels: list[str] | None = None
    criteria: Criteria | None = None
    strategy: Strategy = "auto"
    depends_on: DependsOn | None = None
    # Earlier answers this question's prompt should see. The first one that is
    # a single-branch closed question is continued as a conversation, so this
    # question forks that question's cache.
    include_answers: list[str] | None = None


class DecideIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    context: str
    # Primary contract: a map of typed questions, answered in one call.
    questions: dict[str, QuestionIn] | None = None
    # Fallback contract: one question. The type is either given, or read off the
    # shape when the shape decides it (levels -> score, options with
    # exclusive=false -> flags, options -> choice). Wording is never parsed.
    question: str | None = None
    type: Literal["auto", "noul", "choice", "score", "flags", "quote", "open", "boolean", "categorical", "ordinal", "multilabel", "extract", "schema"] = "auto"
    options: list[str] | None = None
    levels: list[str] | None = None
    schema_def: dict[str, Any] | None = Field(default=None, alias="schema")
    exclusive: bool | None = None
    strategy: Strategy = "auto"
    examples: list[ExampleIn] | None = None
    # Keep heads fitted from `examples` in the head store, keyed by signature.
    save_heads: bool = False
    # Apply stored heads to fields whose signature matches.
    use_heads: bool = True


class FieldJob(BaseModel):
    id: str
    kind: Kind
    question: str
    labels: list[str] = []
    strategy: Strategy = "auto"
    # Tokens the ordinal head reads, parallel to labels. Digits, not the names.
    score_tokens: list[str] | None = None
    origin: float = 0.0
    qtype: str | None = None
    criteria: dict[str, str] | None = None
    depends_on: Any = None
    include_answers: list[str] = []


def compile_request(req: DecideIn) -> tuple[list[FieldJob], list[str]]:
    if req.context is None:
        raise CompileError("A state is required.")
    if len(req.context) > 8000:
        raise CompileError("State is limited to 8000 characters in this build.")
    notes: list[str] = []
    if req.questions is not None:
        if not req.questions:
            raise CompileError("questions must contain at least one entry.")
        if len(req.questions) > 32:
            raise CompileError("At most 32 questions per call.")
        jobs = [_typed_job(name, q) for name, q in req.questions.items()]
        stages = order_stages(jobs)
        notes.append(f"Compiled {len(jobs)} typed question{'s' if len(jobs) != 1 else ''}. Types were declared, not inferred.")
        if len(stages) > 1:
            notes.append(f"Dependent questions run in {len(stages)} stages; a question waits for the answers it depends on.")
        return jobs, notes
    if req.schema_def is not None:
        jobs = compile_schema(req.schema_def, req.strategy)
        notes.append(
            f"Compiled {len(jobs)} field heads from the schema. "
            "They share one prefill of the state and branch only at each answer boundary."
        )
        return jobs, notes

    kind, routed = _infer_kind(req)
    if routed:
        notes.append(routed)
    question = (req.question or "").strip()
    if not question:
        raise CompileError("A question is required when no schema is given.")
    strategy: Strategy = req.strategy
    if kind == "boolean":
        if strategy not in ("auto", "slice"):
            notes.append("A yes/no question always uses the Yes/No margin. The requested strategy does not apply.")
        return [FieldJob(id="answer", kind="boolean", question=question, labels=["no", "yes"])], notes
    if kind == "categorical":
        labels = _labels(req.options, "options")
        return [FieldJob(id="answer", kind="categorical", question=question, labels=labels, strategy=strategy)], notes
    if kind == "multilabel":
        labels = _labels(req.options, "options")
        if strategy not in ("auto", "margin"):
            notes.append("Multi-label flags are independent yes/no heads. They are not a single softmax.")
        return [FieldJob(id="answer", kind="multilabel", question=question, labels=labels)], notes
    if kind == "ordinal":
        levels = _labels(req.levels, "levels")
        if not 2 <= len(levels) <= 10:
            raise CompileError("A rating needs between 2 and 10 levels.")
        return [_ordinal_job("answer", question, levels, origin=0.0)], notes
    if kind == "extract":
        return [FieldJob(id="answer", kind="extract", question=question)], notes
    if kind == "open":
        if strategy != "auto":
            notes.append("An open question has no rows to slice. The requested strategy does not apply.")
        return [FieldJob(id="answer", kind="open", question=question)], notes
    raise CompileError(f"Cannot compile a {kind} question.")


def _typed_job(name: str, q: QuestionIn) -> FieldJob:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name):
        raise CompileError(f"Question id {name!r} must be 1-64 letters, digits, _ or -.")
    instructions = q.instructions.strip()
    if not instructions:
        raise CompileError(f"Question {name} has empty instructions.")
    kind = _TYPE_TO_KIND[q.type]
    if q.criteria is not None and kind != "boolean":
        raise CompileError(f"Question {name}: criteria apply to noul questions only.")
    if kind == "boolean":
        job = FieldJob(id=name, kind="boolean", question=instructions, labels=["no", "yes"])
        if q.criteria is not None:
            criteria = {k: v.strip() for k, v in q.criteria.model_dump().items() if v and v.strip()}
            if not criteria:
                raise CompileError(f"Question {name}: criteria need a true or a false description.")
            job.criteria = criteria
    elif kind == "categorical":
        job = FieldJob(id=name, kind="categorical", question=instructions, labels=_labels(q.options, f"options for {name}"), strategy=q.strategy)
    elif kind == "multilabel":
        job = FieldJob(id=name, kind="multilabel", question=instructions, labels=_labels(q.options, f"options for {name}"))
    elif kind == "ordinal":
        job = _ordinal_job(name, instructions, _labels(q.levels, f"levels for {name}"), origin=0.0)
    elif kind == "extract":
        job = FieldJob(id=name, kind="extract", question=instructions)
    else:
        job = FieldJob(id=name, kind="open", question=instructions)
    job.qtype = q.type
    job.depends_on = q.depends_on
    job.include_answers = list(dict.fromkeys(q.include_answers or []))
    return job


def order_stages(jobs: list[FieldJob]) -> list[list[str]]:
    """Group questions into stages: each stage only needs answers from earlier ones.

    Raises on unknown ids, self-references, unmatched `when` values, and cycles.
    """

    by_id = {job.id: job for job in jobs}
    parents: dict[str, set[str]] = {job.id: set() for job in jobs}
    for job in jobs:
        needs = list(job.include_answers)
        if job.depends_on is not None:
            needs.append(job.depends_on.question)
            _check_when(job, by_id.get(job.depends_on.question))
        for parent in needs:
            if parent == job.id:
                raise CompileError(f"Question {job.id} cannot depend on itself.")
            if parent not in by_id:
                raise CompileError(f"Question {job.id} refers to unknown question {parent!r}.")
            parents[job.id].add(parent)
    stages: list[list[str]] = []
    done: set[str] = set()
    while len(done) < len(jobs):
        ready = [job.id for job in jobs if job.id not in done and parents[job.id] <= done]
        if not ready:
            cycle = sorted(job.id for job in jobs if job.id not in done)
            raise CompileError(f"Questions {', '.join(cycle)} depend on each other in a cycle.")
        stages.append(ready)
        done.update(ready)
    return stages


def _check_when(job: FieldJob, parent: FieldJob | None) -> None:
    if parent is None or job.depends_on is None:
        return
    values = job.depends_on.when if isinstance(job.depends_on.when, list) else [job.depends_on.when]
    if not values:
        raise CompileError(f"Question {job.id}: depends_on.when is empty.")
    for value in values:
        if parent.kind == "boolean":
            if not isinstance(value, bool) and str(value).strip().lower() not in {"yes", "no", "true", "false"}:
                raise CompileError(f"Question {job.id}: {parent.id} is a yes/no question, so when must be true or false.")
        elif parent.kind in {"categorical", "multilabel", "ordinal"} and str(value).strip() not in parent.labels:
            raise CompileError(f"Question {job.id}: {value!r} is not one of the answers of {parent.id}.")


def compile_schema(schema: dict[str, Any], strategy: Strategy) -> list[FieldJob]:
    if not isinstance(schema, dict):
        raise CompileError("Schema must be a JSON object.")
    if schema.get("type") not in (None, "object"):
        raise CompileError("The schema root must be an object.")
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        raise CompileError("Schema has no properties to compile.")
    jobs: list[FieldJob] = []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            raise CompileError(f"Property {name} must be an object.")
        jobs.append(_property(str(name), spec, strategy))
    return jobs


def _infer_kind(req: DecideIn) -> tuple[str, str | None]:
    if req.type == "schema":
        raise CompileError("type 'schema' needs a schema object.")
    if req.type != "auto":
        return _TYPE_TO_KIND.get(req.type, req.type), None
    if req.levels:
        return "ordinal", "Read as a score because levels were given."
    if req.options and req.exclusive is False:
        return "multilabel", "Read as flags because options were given with exclusive=false."
    if req.options:
        return "categorical", "Read as a choice because options were given."
    if not (req.question or "").strip():
        raise CompileError("A question is required when no schema is given.")
    raise CompileError(
        "Set type (noul, choice, score, flags, quote, or open), or give options or levels. "
        "The question's wording is not used to guess its type."
    )


def _labels(values: list[str] | None, what: str) -> list[str]:
    if not values:
        raise CompileError(f"Provide {what}.")
    cleaned = [str(v).strip() for v in values]
    if any(not v for v in cleaned):
        raise CompileError(f"Empty {what} are not allowed.")
    if len(set(cleaned)) != len(cleaned):
        raise CompileError(f"Duplicate {what} are not allowed.")
    if len(cleaned) > len(_LETTERS):
        raise CompileError(f"At most {len(_LETTERS)} {what} can be compiled.")
    return cleaned


def _property(name: str, spec: dict[str, Any], strategy: Strategy) -> FieldJob:
    readout = spec.get("x-readout")
    question = spec.get("x-question") or spec.get("description") or f"What is the value of {name.replace('_', ' ')}?"
    question = str(question).strip()
    if readout == "extract":
        return FieldJob(id=name, kind="extract", question=question)
    if readout == "generate":
        return FieldJob(id=name, kind="generate", question=question)

    if spec.get("type") == "array":
        items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        enum = items.get("enum")
        if isinstance(enum, list) and enum:
            labels = _labels([str(v) for v in enum], f"items of {name}")
            q = spec.get("description") or f"Which of these apply as {name.replace('_', ' ')}?"
            return FieldJob(id=name, kind="multilabel", question=str(q), labels=labels)
        raise CompileError(f"{name} is an array without an enum of items. Set items.enum or x-readout.")

    if "enum" in spec:
        if not isinstance(spec["enum"], list) or not spec["enum"]:
            raise CompileError(f"{name} has an empty enum.")
        labels = _labels([str(v) for v in spec["enum"]], f"enum of {name}")
        q = str(spec.get("description") or f"Which option fits {name.replace('_', ' ')}?")
        if spec.get("type") == "integer" and _consecutive_ints(spec["enum"]):
            return _ordinal_job(name, q, [str(v) for v in spec["enum"]], origin=float(spec["enum"][0]))
        return FieldJob(id=name, kind="categorical", question=q, labels=labels, strategy=strategy)

    type_name = spec.get("type", "string")
    if type_name == "boolean" or readout == "boolean":
        q = str(spec.get("description") or f"Is {name.replace('_', ' ')} true?")
        return FieldJob(id=name, kind="boolean", question=q, labels=["no", "yes"])
    if type_name == "integer" or readout == "ordinal":
        lo = int(spec.get("minimum", 0))
        hi = int(spec.get("maximum", 4))
        if hi < lo or hi - lo > 9:
            raise CompileError(f"{name} must span 2 to 10 integer levels.")
        labels = [str(i) for i in range(lo, hi + 1)]
        q = str(spec.get("description") or f"Rate {name.replace('_', ' ')} from {lo} to {hi}.")
        return _ordinal_job(name, q, labels, origin=float(lo))
    if type_name == "string":
        return FieldJob(id=name, kind="generate", question=question)
    raise CompileError(f"Property {name} has unsupported type {type_name!r}.")


def _ordinal_job(name: str, question: str, labels: list[str], origin: float) -> FieldJob:
    if not 2 <= len(labels) <= 10:
        raise CompileError(f"{name} must have between 2 and 10 levels.")
    # Read the level symbol itself when it is already a single digit. When every
    # level opens with its own digit ("1 star", "2 stars"), read those digits, so
    # the code the model answers with is the number it sees, not a 0-based index.
    # Otherwise fall back to the index.
    leading = [re.match(r"([0-9])\b", label) for label in labels]
    if all(re.fullmatch(r"[0-9]", label) for label in labels):
        digits = labels
        origin = origin if origin else float(labels[0])
    elif all(leading) and _consecutive_ints([m.group(1) for m in leading if m]):
        digits = [m.group(1) for m in leading if m]
        origin = float(digits[0])
    else:
        digits = [str(i) for i in range(len(labels))]
        origin = 0.0
    return FieldJob(
        id=name,
        kind="ordinal",
        question=question,
        labels=labels,
        score_tokens=digits,
        origin=origin,
    )


def _consecutive_ints(values: list[Any]) -> bool:
    try:
        nums = [int(v) for v in values]
    except (TypeError, ValueError):
        return False
    return nums == list(range(nums[0], nums[0] + len(nums))) and all(0 <= n <= 9 for n in nums)
