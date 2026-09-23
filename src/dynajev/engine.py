"""Compile a request to a plan, run it on the frozen trunk, fit heads from labels."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dynajev.bind import bind_field
from dynajev.compile import DecideIn, FieldJob, compile_request, order_stages
from dynajev.errors import CompileError
from dynajev.executor import combine_field, continuable, encode_branch, readout, run_plan, typed_answer
from dynajev.fit import candidate_layers, choose_exit, select_fit
from dynajev.heads import HeadParams, HeadStore, task_signature
from dynajev.plan import FieldPlan, Plan
from dynajev.trie import TrieReader

_MAX_EXAMPLE_ROWS = 64
_KIND_TO_TYPE = {"boolean": "noul", "categorical": "choice", "ordinal": "score"}


class Dynajev:
    def __init__(self, backend: Any, prefix_cache: int = 0, overhead_tokens: int = 64, heads: HeadStore | None = None):
        self.backend = backend
        self.reader = TrieReader(backend, prefix_cache=prefix_cache, overhead_tokens=overhead_tokens)
        self.heads = heads if heads is not None else HeadStore()

    def prefix_cache_stats(self) -> dict[str, int]:
        return self.reader.store.stats()

    def compile(
        self,
        jobs: list[FieldJob],
        corrections: dict[str, HeadParams] | None = None,
        sources: dict[str, str] | None = None,
    ) -> Plan:
        fields = [bind_field(job, self.backend) for job in jobs]
        for item, job in zip(fields, jobs):
            if _fittable(item):
                item.signature = self.signature(job)
            correction = (corrections or {}).get(item.id)
            if correction is not None:
                _apply_correction(item, correction, (sources or {}).get(item.id, "fitted"))
        by_id = {item.id: item for item in fields}
        for item in fields:
            parent = next((by_id[i] for i in item.include_answers if continuable(by_id[i])), None)
            if parent is not None:
                parent.branches[0].keep = True
                for branch in item.branches:
                    branch.continued_from = parent.id
        return Plan(
            fields=fields,
            stages=order_stages(jobs),
            num_layers=int(getattr(self.backend, "num_layers", 0)),
        )

    def decide(self, req: DecideIn) -> dict[str, Any]:
        started = time.perf_counter()
        context = self._sanitize(req.context)
        req = req.model_copy(update={"context": context})
        jobs, notes = compile_request(req)
        corrections: dict[str, HeadParams] = {}
        sources: dict[str, str] = {}
        fit_payload = None
        if req.examples:
            if any(job.kind in {"extract", "generate", "open"} for job in jobs):
                notes.append("Open fields are not fitted. Only closed heads can be replaced.")
            corrections, fit_payload = self._fit(req, jobs)
            sources = {field_id: "fitted" for field_id in corrections}
            if req.save_heads:
                notes.extend(self._save(corrections, fit_payload))
        if req.use_heads:
            stored, stored_notes = self._stored(jobs, skip=set(corrections))
            corrections.update(stored)
            sources.update({field_id: "stored" for field_id in stored})
            notes.extend(stored_notes)
        plan = self.compile(jobs, corrections, sources)
        run = run_plan(plan, self.backend, self.reader, context)
        if run.cached_prefix_tokens:
            notes.append(
                f"Prefix cache hit: the {run.cached_prefix_tokens}-token state prefill was reused from an earlier request."
            )
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "model": getattr(self.backend, "model_id", "unknown"),
            "elapsed_ms": round(elapsed, 1),
            "hidden_size": int(getattr(self.backend, "hidden_size", 0)),
            "vocab_size": int(getattr(self.backend, "vocab_size", 0)),
            "shared_prefix_tokens": run.shared_prefix_tokens,
            "prefill_tokens": run.prefill_tokens,
            "generated_tokens": sum(int(field.get("generated_tokens") or 0) for field in run.fields),
            "fields": run.fields,
            "answers": {field["id"]: typed_answer(field) for field in run.fields},
            "fit": fit_payload,
            "plan": plan.describe(run.reads, run.skipped),
            "notes": notes,
        }

    def decide_batch(
        self,
        contexts: list[str],
        questions: dict[str, Any],
        chunk_rows: int = 32,
        trace: bool = False,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """Many states, one question set, closed types only.

        - "dense": every (state, branch) prompt is a row of a padded batch, one
          forward per chunk. Nothing is reused within a state, but the forward is
          a rectangle, which is what a GPU wants.
        - "shared": one state at a time, each state's branches read through the
          same prefix sharing as /api/decide. Fewer tokens, sequential. Wins on CPU.

        "auto" picks dense on CUDA and shared on CPU.
        """

        if mode not in {"auto", "dense", "shared"}:
            raise CompileError("mode must be auto, dense, or shared.")
        started = time.perf_counter()
        if not contexts:
            raise CompileError("contexts must contain at least one state.")
        if len(contexts) > 1024:
            raise CompileError("At most 1024 states per batch call.")
        clean = [self._sanitize(c) for c in contexts]
        probe = DecideIn.model_validate({"context": clean[0], "questions": questions})
        jobs, notes = compile_request(probe)
        open_kinds = [job.id for job in jobs if job.kind in {"extract", "generate", "open"}]
        if open_kinds:
            raise CompileError(
                f"Batch decisions are closed readouts only; {', '.join(open_kinds)} would need generation. "
                "Use /api/decide for quote and open questions."
            )
        staged = any(job.depends_on is not None or job.include_answers for job in jobs)
        if mode == "auto":
            cuda = str(getattr(self.backend, "device", "cpu")).startswith("cuda")
            mode = "dense" if cuda and not staged else "shared"
        if mode == "dense" and staged:
            raise CompileError("Dependent questions need the shared mode, which runs them in stages.")
        plans = [self.compile(jobs) for _ in clean]
        prefill = 0
        rows = 0
        results = []
        if mode == "dense":
            branches = []
            for plan, context in zip(plans, clean):
                for item in plan.fields:
                    for branch in item.branches:
                        branch.tokens = encode_branch(self.backend, branch, context)
                        branches.append(branch)
            hiddens = self.backend.read_dense([b.tokens for b in branches], chunk_rows)
            by_branch = {id(b): h for b, h in zip(branches, hiddens)}
            prefill = sum(len(b.tokens or []) for b in branches)
            rows = len(branches)
            for plan in plans:
                fields = []
                for item in plan.fields:
                    lookup = {b.id: by_branch[id(b)] for b in item.branches}
                    reads = [readout(self.backend, read, lookup[read.branch]) for read in item.reads]
                    fields.append(combine_field(item, reads, self.backend))
                results.append(_batch_entry(fields, trace))
        else:
            for plan, context in zip(plans, clean):
                run = run_plan(plan, self.backend, self.reader, context)
                prefill += run.prefill_tokens
                rows += sum(len(item.branches) for item in plan.fields)
                results.append(_batch_entry(run.fields, trace))
        elapsed = (time.perf_counter() - started) * 1000
        decisions = len(clean) * len(jobs)
        return {
            "model": getattr(self.backend, "model_id", "unknown"),
            "states": len(clean),
            "questions": len(jobs),
            "decisions": decisions,
            "rows": rows,
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
                else "Batched shared: each state prefilled once, its branches read off that prefill, nothing generated."
            ],
        }

    def signature(self, job: FieldJob) -> str:
        return task_signature(
            job.qtype or _KIND_TO_TYPE.get(job.kind, job.kind),
            job.question,
            job.labels,
            str(getattr(self.backend, "model_id", "unknown")),
            job.strategy,
            job.criteria,
        )

    def fit_heads(self, questions: dict[str, Any], examples: list[Any]) -> dict[str, Any]:
        """Fit heads for a question set from labeled states and keep them in the store."""

        req = DecideIn.model_validate({"context": "", "questions": questions, "examples": examples})
        jobs, notes = compile_request(req)
        corrections, fit_payload = self._fit(req, jobs)
        if fit_payload is None:
            raise CompileError("No closed single-branch question to fit (noul, choice, or score).")
        notes.extend(self._save(corrections, fit_payload))
        return {"heads": [head.summary() for head in corrections.values()], "fit": fit_payload, "notes": notes}

    def _save(self, corrections: dict[str, HeadParams], fit_payload: dict[str, Any] | None) -> list[str]:
        notes = []
        records = (fit_payload or {}).get("fields") or ([fit_payload] if fit_payload else [])
        for record in records:
            head = corrections.get(record.get("field", ""))
            if head is None:
                record["saved"] = False
                continue
            self.heads.put(head)
            record["saved"] = True
            record["signature"] = head.signature
            notes.append(f"Saved the fitted head for {head.qtype} field {record['field']} as {head.signature}.")
        if records and not corrections:
            notes.append("Nothing was saved: the readout head stood for every field.")
        return notes

    def _stored(self, jobs: list[FieldJob], skip: set[str]) -> tuple[dict[str, HeadParams], list[str]]:
        found: dict[str, HeadParams] = {}
        notes = []
        num_layers = int(getattr(self.backend, "num_layers", 0) or 0)
        for job in jobs:
            if job.id in skip or job.kind not in {"boolean", "categorical", "ordinal"} or job.strategy in {"margin", "prototype"} or job.criteria:
                continue
            head = self.heads.get(self.signature(job))
            if head is None or head.labels != list(job.labels):
                continue
            if head.layer is not None and num_layers and head.layer > num_layers:
                continue
            found[job.id] = head
            where = f", exit after layer {head.layer} of {num_layers}" if head.layer is not None else ""
            notes.append(f"Field {job.id} used stored head {head.signature} ({head.chosen.replace('_', ' ')}{where}).")
        return found, notes

    def _sanitize(self, context: str) -> str:
        return self.backend.sanitize(context) if hasattr(self.backend, "sanitize") else context

    def _fit(self, req: DecideIn, jobs: list[FieldJob]) -> tuple[dict[str, HeadParams], dict[str, Any] | None]:
        examples = list(req.examples or [])
        closed = [(job, item) for job in jobs if _fittable(item := bind_field(job, self.backend))]
        if not closed:
            return {}, None
        if len(examples) * len(closed) > _MAX_EXAMPLE_ROWS:
            return {}, {
                "chosen": "zero_shot",
                "note": "Too many example forwards for this request, so no head was fitted. "
                f"Keep labeled states times closed questions at or under {_MAX_EXAMPLE_ROWS}.",
                "n_examples": len(examples),
            }
        corrections: dict[str, HeadParams] = {}
        records = []
        single = len(closed) == 1
        num_layers = int(getattr(self.backend, "num_layers", 0) or 0)
        layers = tuple(candidate_layers(num_layers)) if num_layers else None
        for job, item in closed:
            label_index, sequences = [], []
            skipped = 0
            branch = item.branches[0]
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
                sequences.append(encode_branch(self.backend, branch, self._sanitize(example.context)))
                label_index.append(index)
            if len(label_index) < 3:
                records.append(
                    {
                        "field": job.id,
                        "chosen": "zero_shot",
                        "n_examples": len(label_index),
                        "note": "Not enough usable labeled states to fit this field. The compiled head stands.",
                    }
                )
                continue
            taps = self.backend.prefill_rows(None, sequences, [layers] * len(sequences) if layers else None, 8)
            final = [row[num_layers] if num_layers else next(iter(row.values())) for row in taps]
            outs = [readout(self.backend, item.reads[0], hidden) for hidden in final]
            labels = np.array(label_index)
            decision = select_fit(
                np.array([self.backend.as_vector(h) for h in final]), np.array([o.logits for o in outs], dtype=np.float64), labels
            )
            record = _fit_record(job, decision, len(label_index), skipped)
            record["fit_tokens"] = sum(len(seq) for seq in sequences)
            correction = HeadParams.from_decision(decision, job.labels) if decision.chosen != "zero_shot" else None
            if layers and len(layers) > 1:
                per_layer = {
                    k: np.array([self.backend.as_vector(row[k]) for row in taps]) for k in layers if k < num_layers
                }
                exit_choice = choose_exit(per_layer, labels, len(job.labels), decision, num_layers)
                record["num_layers"] = num_layers
                record["exit_layer"] = exit_choice.layer
                record["layer_scan"] = exit_choice.scan
                record["full_depth_loo"] = {
                    "accuracy": round(exit_choice.full_accuracy, 4),
                    "nll": round(exit_choice.full_nll, 4),
                }
                if exit_choice.layer is not None and exit_choice.probe is not None:
                    probe = exit_choice.probe
                    correction = HeadParams(
                        chosen="ridge_probe",
                        layer=exit_choice.layer,
                        weight=probe.weight,
                        ridge_bias=probe.bias,
                        mu=probe.mu,
                        sd=probe.sd,
                        scale=probe.scale,
                        note=exit_choice.note,
                        labels=list(job.labels),
                    )
                    record["chosen"] = "ridge_probe"
                    record["note"] = exit_choice.note
            if correction is not None:
                correction.signature = self.signature(job)
                correction.qtype = job.qtype or _KIND_TO_TYPE.get(job.kind, job.kind)
                correction.instructions = job.question
                correction.model_id = str(getattr(self.backend, "model_id", "unknown"))
                correction.n_examples = len(label_index)
                correction.metrics = {k: v for k, v in record.items() if k not in {"field", "note", "bias"}}
                corrections[job.id] = correction
            records.append(record)
        if len(records) == 1:
            return corrections, records[0]
        return corrections, {"fields": records, "chosen": "per_field", "note": "Each closed field was fitted on its own."}


def _apply_correction(item: FieldPlan, correction: HeadParams, source: str) -> None:
    if item.combine is None or correction.chosen == "readout":
        return
    item.combine.correction = correction
    item.head_source = source  # type: ignore[assignment]
    if correction.chosen == "ridge_probe":
        read = item.reads[0]
        read.op = "hidden"
        read.groups = []
        read.layer = correction.layer
        for branch in item.branches:
            branch.depth = correction.layer


def _fit_record(job: FieldJob, decision: Any, n: int, skipped: int) -> dict[str, Any]:
    return {
        "field": job.id,
        "chosen": decision.chosen,
        "temperature": round(decision.temperature, 4),
        "bias": None if decision.bias is None else {label: round(float(v), 4) for label, v in zip(job.labels, decision.bias)},
        "n_examples": n,
        "skipped": skipped,
        "zero_shot_loo_nll": round(decision.zero_shot_loo_nll, 4),
        "affine_loo_nll": round(decision.affine_loo_nll, 4),
        "ridge_loo_nll": None if decision.ridge_loo_nll is None else round(decision.ridge_loo_nll, 4),
        "zero_shot_loo_accuracy": round(decision.zero_shot_loo_accuracy, 4),
        "affine_loo_accuracy": round(decision.affine_loo_accuracy, 4),
        "ridge_loo_accuracy": None if decision.ridge_loo_accuracy is None else round(decision.ridge_loo_accuracy, 4),
        "note": decision.note,
    }


def _batch_entry(fields: list[dict[str, Any]], trace: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"answers": {field["id"]: typed_answer(field) for field in fields}}
    if trace:
        entry["fields"] = fields
    return entry


def _example_raw(example: Any, field_id: str, single: bool) -> Any:
    if example.answers and field_id in example.answers:
        return example.answers[field_id]
    if example.label is None:
        return None
    if single or field_id == "answer":
        return example.label
    return None


def _fittable(item: FieldPlan) -> bool:
    return (
        item.kind in {"boolean", "categorical", "ordinal"}
        and len(item.branches) == 1
        and bool(item.reads)
        and item.reads[0].op == "rows"
    )


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
