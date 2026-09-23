"""Compile a request, run the frozen trunk, score each head."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from readhead.bind import BoundField, Node, bind_field
from readhead.compile import DecideIn, FieldJob, compile_request
from readhead.errors import CompileError
from readhead.fit import apply_affine, apply_ridge, probabilities_from_logits, select_fit
from readhead.prompts import user_content
from readhead.score import binary_from_logits, ordinal_expectation, softmax
from readhead.trunk import common_prefix_len


class Readhead:
    def __init__(self, trunk: Any):
        self.trunk = trunk

    def decide(self, req: DecideIn) -> dict[str, Any]:
        started = time.perf_counter()
        context = self.trunk.sanitize(req.context) if hasattr(self.trunk, "sanitize") else req.context
        req = req.model_copy(update={"context": context})
        jobs, notes = compile_request(req)
        if req.examples and any(job.kind in {"extract", "generate", "open"} for job in jobs):
            notes.append("Open fields are not fitted. Only closed heads can be replaced.")
        bound = [bind_field(job, self.trunk) for job in jobs]
        fields, fit_payload = self._execute(req, jobs, bound)
        for field in fields:
            if "_ids" in field:
                extractive = field.pop("_extractive", False)
                text, count = self.trunk.generate(field.pop("_ids"), req.context, extractive, field.pop("_decode", "short"))
                field["answer"] = text
                field["generated_tokens"] = count
                if extractive:
                    verbatim = _normalize(text) in _normalize(req.context) if text else False
                    field["verbatim"] = verbatim
                    if not verbatim:
                        field["warning"] = "The quoted text does not appear verbatim in the state."
        elapsed = (time.perf_counter() - started) * 1000
        shared = fields[0].pop("_shared") if fields else 0
        prefill = _prefill_tokens(bound, shared)
        if getattr(self.trunk, "last_read_cached", False):
            prefill = max(0, prefill - shared)
            notes.append(f"Prefix cache hit: the {shared}-token state prefill was reused from an earlier request.")
        for field in fields:
            field.pop("_shared", None)
            field.pop("_logits", None)
            field.pop("_hidden", None)
            field.pop("_ids", None)
            field.pop("_extractive", None)
            field.pop("_decode", None)
        return {
            "model": getattr(self.trunk, "model_id", "unknown"),
            "elapsed_ms": round(elapsed, 1),
            "hidden_size": int(getattr(self.trunk, "hidden_size", 0)),
            "vocab_size": int(getattr(self.trunk, "vocab_size", 0)),
            "shared_prefix_tokens": shared,
            "prefill_tokens": prefill,
            "generated_tokens": sum(int(field.get("generated_tokens") or 0) for field in fields),
            "fields": fields,
            "answers": {field["id"]: _typed_answer(field) for field in fields},
            "fit": fit_payload,
            "notes": notes,
        }

    def _execute(
        self, req: DecideIn, jobs: list[FieldJob], bound: list[BoundField]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        self._fill(bound, req.context)
        fields = [self._score_field(item) for item in bound]
        fit_payload = None
        examples = list(req.examples or [])
        closed = [(job, item, field) for job, item, field in zip(jobs, bound, fields) if _fittable(item)]
        if examples and closed:
            if len(examples) * len(closed) > 24:
                fields_notes = (
                    "Too many example forwards for this request, so no head was fitted. "
                    "Ask one closed question with at most 8 labeled states."
                )
                fit_payload = {"chosen": "zero_shot", "note": fields_notes, "n_examples": len(examples)}
            else:
                fit_payload = self._fit(req, examples, closed, fields)
        return fields, fit_payload

    def _fill(self, bound: list[BoundField], context: str) -> int:
        nodes = self._encode(bound, context)
        readable = [node for node in nodes if node.mode != "decode"]
        shared = 0
        if readable:
            # The state block tokenizes identically for every question about the
            # same state, so its length is the reuse point for the prefix cache.
            anchor = None
            if getattr(self.trunk, "prefix_cache_size", 0):
                empty = self.trunk.encode_prompt(user_content(context, ""), readable[0].assistant_prefix, readable[0].system)
                anchor = common_prefix_len([empty, *[node.ids for node in readable]])
            try:
                hiddens, shared = self.trunk.read([node.ids for node in readable], anchor)
            except TypeError:
                hiddens, shared = self.trunk.read([node.ids for node in readable])
            for node, hidden in zip(readable, hiddens):
                node.hidden = hidden
        self._score_nodes(bound, shared)
        return shared

    def _encode(self, bound: list[BoundField], context: str) -> list[Node]:
        nodes: list[Node] = [node for item in bound for node in item.nodes]
        for node in nodes:
            node.ids = self.trunk.encode_prompt(user_content(context, node.block), node.assistant_prefix, node.system)
        return nodes

    def _score_nodes(self, bound: list[BoundField], shared: int) -> None:
        for item in bound:
            for node in item.nodes:
                node.shared = shared  # type: ignore[attr-defined]
                if node.mode == "prototype":
                    node.logits = self.trunk.prototype_logits(node.hidden, node.pieces)
                    node.allowed_mass = None
                elif node.mode == "classes":
                    logits, mass = self.trunk.class_logits(node.hidden, node.groups)
                    node.logits = logits
                    node.allowed_mass = mass
        self._score_margin_nodes(bound)

    def decide_batch(
        self,
        contexts: list[str],
        questions: dict[str, Any],
        chunk_rows: int = 32,
        trace: bool = False,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """Many states, one question set, closed types only.

        Two ways to run it, and they win on different hardware:

        - "dense": every (state, question, branch) prompt is a row of a padded
          batch, one forward per chunk. About twice the tokens of "shared" because
          nothing is reused within a state, but the forward is a rectangle, which
          is what a GPU wants.
        - "shared": one state at a time, the state prefilled once and its fields
          branched off that cache in one batched forward. Fewer tokens, sequential.
          Wins on CPU, where throughput does not improve with batch size.

        "auto" picks dense on CUDA and shared on CPU. Measured numbers are in the README.
        """

        if mode == "auto":
            mode = "dense" if str(getattr(self.trunk, "device", "cpu")).startswith("cuda") else "shared"
        if mode not in {"dense", "shared"}:
            raise CompileError("mode must be auto, dense, or shared.")

        started = time.perf_counter()
        if not contexts:
            raise CompileError("contexts must contain at least one state.")
        if len(contexts) > 1024:
            raise CompileError("At most 1024 states per batch call.")
        clean = [self.trunk.sanitize(c) if hasattr(self.trunk, "sanitize") else c for c in contexts]
        probe = DecideIn.model_validate({"context": clean[0], "questions": questions})
        jobs, notes = compile_request(probe)
        open_kinds = [job.id for job in jobs if job.kind in {"extract", "generate", "open"}]
        if open_kinds:
            raise CompileError(
                f"Batch decisions are closed readouts only; {', '.join(open_kinds)} would need generation. "
                "Use /api/decide for quote and open questions."
            )
        per_state: list[list[BoundField]] = []
        readable: list[Node] = []
        for context in clean:
            bound = [bind_field(job, self.trunk) for job in jobs]
            nodes = self._encode(bound, context)
            readable.extend(node for node in nodes if node.mode != "decode")
            per_state.append(bound)
        prefill = 0
        if mode == "dense":
            hiddens = self.trunk.read_dense([node.ids for node in readable], chunk_rows)
            for node, hidden in zip(readable, hiddens):
                node.hidden = hidden
            prefill = sum(len(node.ids or []) for node in readable)
        else:
            for bound in per_state:
                nodes = [node for item in bound for node in item.nodes if node.mode != "decode"]
                hiddens, shared = self.trunk.read([node.ids for node in nodes])
                for node, hidden in zip(nodes, hiddens):
                    node.hidden = hidden
                prefill += _prefill_tokens(bound, shared)
        results = []
        for bound in per_state:
            self._score_nodes(bound, 0)
            fields = [self._score_field(item) for item in bound]
            for field in fields:
                for key in ("_shared", "_logits", "_hidden", "_ids", "_extractive", "_decode"):
                    field.pop(key, None)
            entry: dict[str, Any] = {"answers": {field["id"]: _typed_answer(field) for field in fields}}
            if trace:
                entry["fields"] = fields
            results.append(entry)
        elapsed = (time.perf_counter() - started) * 1000
        decisions = len(clean) * len(jobs)
        return {
            "model": getattr(self.trunk, "model_id", "unknown"),
            "states": len(clean),
            "questions": len(jobs),
            "decisions": decisions,
            "rows": len(readable),
            "mode": mode,
            "chunk_rows": chunk_rows if mode == "dense" else None,
            "elapsed_ms": round(elapsed, 1),
            "decisions_per_second": round(decisions / (elapsed / 1000), 1) if elapsed > 0 else None,
            "prefill_tokens": prefill,
            "generated_tokens": 0,
            "results": results,
            "notes": notes
            + [
                "Batched dense: one padded forward per chunk of rows, nothing shared between rows, nothing generated."
                if mode == "dense"
                else "Batched shared: each state prefilled once, its fields branched in one forward, nothing generated."
            ],
        }

    def _score_margin_nodes(self, bound: list[BoundField]) -> None:
        """Yes/no branches reuse the boolean verbalizer, scored per branch."""

        for item in bound:
            if item.head not in {"option_margin", "multilabel_margin"}:
                continue
            # The boolean binder already checked Yes/No. Rebuild groups from a sibling call
            # stored on the first boolean-capable encoder via class names no/yes.
            probe = bind_field(FieldJob(id=item.id, kind="boolean", question="probe"), self.trunk)
            groups = probe.nodes[0].groups
            names = probe.nodes[0].class_names
            for node in item.nodes:
                node.groups = groups
                node.class_names = names
                logits, mass = self.trunk.class_logits(node.hidden, groups)
                node.logits = logits
                node.allowed_mass = mass

    def _score_field(self, item: BoundField) -> dict[str, Any]:
        shared = getattr(item.nodes[0], "shared", 0) if item.nodes else 0
        if item.head in {"extractive_decode", "short_decode", "chat_decode"}:
            return self._decode_field(item, shared)
        if item.head == "multilabel_margin":
            return self._multilabel_field(item, shared)
        if item.head == "option_margin":
            return self._margin_field(item, shared)
        if item.head == "ordinal_expectation":
            return self._ordinal_field(item, shared)
        return self._softmax_field(item, shared)

    def _softmax_field(self, item: BoundField, shared: int) -> dict[str, Any]:
        node = item.nodes[0]
        logits = list(node.logits or [])
        probabilities = softmax(logits)
        pairs = list(zip(item.labels, probabilities))
        winner = max(pairs, key=lambda pair: pair[1])[0] if pairs else None
        payload = self._base(item, shared, logits, probabilities, winner)
        if item.head == "binary_margin":
            yes = probabilities[item.labels.index("yes")]
            payload["answer"] = bool(yes >= 0.5)
            payload["confidence"] = round(max(yes, 1 - yes), 4)
            payload["probabilities"] = {"no": round(probabilities[0], 4), "yes": round(probabilities[1], 4)}
        if item.head == "prototype":
            payload["rows_scored"] = sum(len(piece) for piece in (node.pieces or []))
            payload["allowed_mass"] = None
            payload["weak_reading"] = False
            payload["warning"] = None
        return payload

    def _ordinal_field(self, item: BoundField, shared: int) -> dict[str, Any]:
        node = item.nodes[0]
        logits = list(node.logits or [])
        probabilities = softmax(logits)
        pairs = list(zip(item.labels, probabilities))
        winner = max(pairs, key=lambda pair: pair[1])[0]
        score = ordinal_expectation(probabilities, origin=item.origin)
        payload = self._base(item, shared, logits, probabilities, winner)
        payload["score"] = round(score, 4)
        return payload

    def _margin_field(self, item: BoundField, shared: int) -> dict[str, Any]:
        margins = []
        masses = []
        for node in item.nodes:
            logits = list(node.logits or [0.0, 0.0])
            margins.append(logits[1] - logits[0])
            if node.allowed_mass is not None:
                masses.append(node.allowed_mass)
        probabilities = softmax(margins)
        pairs = list(zip(item.labels, probabilities))
        winner = max(pairs, key=lambda pair: pair[1])[0]
        payload = self._base(item, shared, margins, probabilities, winner)
        mean_mass = sum(masses) / len(masses) if masses else None
        payload["allowed_mass"] = None if mean_mass is None else round(mean_mass, 4)
        payload["weak_reading"] = mean_mass is not None and mean_mass < 0.05
        payload["warning"] = _weak_warning(mean_mass)
        payload["rows_scored"] = sum(len(group) for node in item.nodes for group in node.groups)
        payload["sequences"] = len(item.nodes)
        return payload

    def _multilabel_field(self, item: BoundField, shared: int) -> dict[str, Any]:
        probabilities: dict[str, float] = {}
        masses = []
        for label, node in zip(item.labels, item.nodes):
            logits = list(node.logits or [0.0, 0.0])
            probabilities[label] = binary_from_logits(logits[0], logits[1])
            if node.allowed_mass is not None:
                masses.append(node.allowed_mass)
        chosen = [label for label, prob in probabilities.items() if prob >= 0.5]
        confidence = sum(abs(p - 0.5) * 2 for p in probabilities.values()) / max(len(probabilities), 1)
        return {
            "id": item.id,
            "head": item.head,
            "kind": item.kind,
            "reason": item.reason,
            "answer": chosen,
            "confidence": round(confidence, 4),
            "probabilities": {label: round(prob, 4) for label, prob in probabilities.items()},
            "score": None,
            "letters": None,
            "allowed_mass": round(sum(masses) / len(masses), 4) if masses else None,
            "rows_scored": sum(len(group) for node in item.nodes for group in node.groups),
            "sequences": len(item.nodes),
            "generated_tokens": 0,
            "prompt": item.prompt_preview,
            "logits": {label: round(item.nodes[i].logits[1] - item.nodes[i].logits[0], 4) for i, label in enumerate(item.labels)},
            "weak_reading": bool(masses) and (sum(masses) / len(masses) < 0.05),
            "warning": _weak_warning(sum(masses) / len(masses) if masses else None),
            "_shared": shared,
            "_logits": None,
            "_hidden": None,
        }

    def _decode_field(self, item: BoundField, shared: int) -> dict[str, Any]:
        node = item.nodes[0]
        return {
            "id": item.id,
            "head": item.head,
            "kind": item.kind,
            "reason": item.reason,
            "answer": None,
            "confidence": None,
            "probabilities": None,
            "score": None,
            "letters": None,
            "allowed_mass": None,
            "rows_scored": 0,
            "sequences": 1,
            "generated_tokens": 0,
            "prompt": item.prompt_preview,
            "logits": None,
            "weak_reading": False,
            "warning": None,
            "_shared": shared,
            "_logits": None,
            "_hidden": None,
            "_extractive": node.extractive,
            "_decode": node.decode,
            "_ids": node.ids,
        }

    def _base(
        self,
        item: BoundField,
        shared: int,
        logits: list[float],
        probabilities: list[float],
        winner: str | None,
    ) -> dict[str, Any]:
        mass = item.nodes[0].allowed_mass
        hidden = self.trunk.as_vector(item.nodes[0].hidden) if item.nodes[0].hidden is not None else None
        return {
            "id": item.id,
            "head": item.head,
            "kind": item.kind,
            "reason": item.reason,
            "answer": winner,
            "confidence": round(max(probabilities), 4) if probabilities else None,
            "probabilities": {label: round(prob, 4) for label, prob in zip(item.labels, probabilities)},
            "score": None,
            "letters": item.letters,
            "allowed_mass": None if mass is None else round(mass, 4),
            "rows_scored": sum(len(group) for group in item.nodes[0].groups) if item.nodes[0].groups else 0,
            "sequences": 1,
            "generated_tokens": 0,
            "prompt": item.prompt_preview,
            "logits": {label: round(value, 4) for label, value in zip(item.labels, logits)},
            "weak_reading": mass is not None and mass < 0.05,
            "warning": _weak_warning(mass),
            "_shared": shared,
            "_logits": logits,
            "_hidden": hidden,
        }

    def _fit(
        self,
        req: DecideIn,
        examples: list[Any],
        closed: list[tuple[FieldJob, BoundField, dict[str, Any]]],
        fields: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # One fit record per closed field. The response keeps a single object when
        # there is one field, and a list under "fields" when a schema is fitted.
        records = []
        for job, _item, field in closed:
            hidden_rows = []
            logit_rows = []
            label_index = []
            skipped = 0
            single = len(closed) == 1
            for example in examples:
                raw = _example_raw(example, job.id, single)
                if raw is None:
                    skipped += 1
                    continue
                try:
                    index = _label_index(job, raw)
                except CompileError:
                    skipped += 1
                    continue
                bound = [bind_field(job, self.trunk)]
                example_context = self.trunk.sanitize(example.context) if hasattr(self.trunk, "sanitize") else example.context
                self._fill(bound, example_context)
                scored = self._score_field(bound[0])
                if scored.get("_hidden") is None or scored.get("_logits") is None:
                    skipped += 1
                    continue
                hidden_rows.append(scored["_hidden"])
                logit_rows.append(scored["_logits"])
                label_index.append(index)
            if len(label_index) < 3 or field.get("_hidden") is None or field.get("_logits") is None:
                records.append(
                    {
                        "field": job.id,
                        "chosen": "zero_shot",
                        "n_examples": len(label_index),
                        "note": "Not enough usable labeled states to fit this field. The compiled head stands.",
                    }
                )
                continue
            decision = select_fit(np.array(hidden_rows), np.array(logit_rows, dtype=np.float64), np.array(label_index))
            self._apply_fit(field, job, decision)
            records.append(
                {
                    "field": job.id,
                    "chosen": decision.chosen,
                    "temperature": round(decision.temperature, 4),
                    "bias": None
                    if decision.bias is None
                    else {label: round(float(v), 4) for label, v in zip(job.labels, decision.bias)},
                    "n_examples": len(label_index),
                    "skipped": skipped,
                    "zero_shot_loo_nll": round(decision.zero_shot_loo_nll, 4),
                    "affine_loo_nll": round(decision.affine_loo_nll, 4),
                    "ridge_loo_nll": None if decision.ridge_loo_nll is None else round(decision.ridge_loo_nll, 4),
                    "zero_shot_loo_accuracy": round(decision.zero_shot_loo_accuracy, 4),
                    "affine_loo_accuracy": round(decision.affine_loo_accuracy, 4),
                    "ridge_loo_accuracy": None
                    if decision.ridge_loo_accuracy is None
                    else round(decision.ridge_loo_accuracy, 4),
                    "note": decision.note,
                }
            )
        if len(records) == 1:
            return records[0]
        return {"fields": records, "chosen": "per_field", "note": "Each closed field was fitted on its own."}

    def _apply_fit(self, field: dict[str, Any], job: FieldJob, decision: Any) -> None:
        labels = job.labels
        if decision.chosen == "ridge_probe":
            logits = apply_ridge(np.array(field["_hidden"], dtype=np.float64), decision)
            field["head"] = "ridge_probe"
            field["reason"] = decision.note
        elif decision.chosen == "affine":
            logits = apply_affine(list(field["_logits"]), decision.temperature, decision.bias)
            field["head"] = f"{field['head']}+affine"
            field["reason"] = decision.note
        else:
            return
        probabilities = probabilities_from_logits(logits)
        field["probabilities"] = {label: round(prob, 4) for label, prob in zip(labels, probabilities)}
        field["logits"] = {label: round(value, 4) for label, value in zip(labels, logits)}
        winner = labels[int(np.argmax(probabilities))]
        if job.kind == "boolean":
            yes = probabilities[labels.index("yes")]
            field["answer"] = bool(yes >= 0.5)
            field["confidence"] = round(max(yes, 1 - yes), 4)
        elif job.kind == "ordinal":
            field["answer"] = winner
            field["confidence"] = round(max(probabilities), 4)
            field["score"] = round(ordinal_expectation(probabilities, origin=job.origin), 4)
        else:
            field["answer"] = winner
            field["confidence"] = round(max(probabilities), 4)

_KIND_TO_TYPE = {
    "boolean": "noul",
    "categorical": "choice",
    "ordinal": "score",
    "multilabel": "flags",
    "extract": "quote",
    "open": "open",
    "generate": "open",
}


def _typed_answer(field: dict[str, Any]) -> dict[str, Any]:
    """The compact, typed view of one field, keyed the way the question was asked."""

    kind = field.get("kind")
    qtype = _KIND_TO_TYPE.get(kind, "open")
    out: dict[str, Any] = {"type": qtype}
    probabilities = field.get("probabilities")
    if qtype == "noul":
        out["noul"] = None if not probabilities else probabilities.get("yes")
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


def _prefill_tokens(bound: list[BoundField], shared: int) -> int:
    """Tokens the trunk actually processed: the shared prefix once, then each branch's suffix.

    Decode nodes are forwarded in full by generate(), so they count whole.
    """

    nodes = [node for item in bound for node in item.nodes if node.ids is not None]
    readable = [node for node in nodes if node.mode != "decode"]
    decoded = [node for node in nodes if node.mode == "decode"]
    total = sum(len(node.ids or []) for node in decoded)
    if not readable:
        return total
    if shared <= 0 or len(readable) == 1:
        return total + sum(len(node.ids or []) for node in readable)
    return total + shared + sum(len(node.ids or []) - shared for node in readable)


def _weak_warning(mass: float | None) -> str | None:
    if mass is None or mass >= 0.05:
        return None
    return (
        "Less than 5% of the next-token distribution sat on the allowed answers. "
        "The probabilities are renormalized over that thin slice."
    )


def _example_raw(example: Any, field_id: str, single: bool) -> Any:
    if example.answers and field_id in example.answers:
        return example.answers[field_id]
    if example.label is None:
        return None
    if single or field_id == "answer":
        return example.label
    return None


def _fittable(item: BoundField) -> bool:
    return item.kind in {"boolean", "categorical", "ordinal"} and item.head not in {"option_margin", "prototype"}


def _label_index(job: FieldJob, raw: Any) -> int:
    if job.kind == "boolean":
        if isinstance(raw, bool):
            return 1 if raw else 0
        text = str(raw).strip().lower()
        if text in {"yes", "true", "1", "y"}:
            return 1
        if text in {"no", "false", "0", "n"}:
            return 0
        raise CompileError(f"Bad boolean label {raw!r}.")
    if job.kind == "ordinal":
        if isinstance(raw, bool):
            raise CompileError("A rating label cannot be a boolean.")
        if isinstance(raw, (int, float)) and str(int(raw)) in job.labels and float(raw) == int(raw):
            return job.labels.index(str(int(raw)))
        text = str(raw).strip()
        if text in job.labels:
            return job.labels.index(text)
        raise CompileError(f"Label {raw!r} is not one of the levels.")
    text = str(raw).strip()
    if text not in job.labels:
        raise CompileError(f"Label {text!r} is not one of the options.")
    return job.labels.index(text)
