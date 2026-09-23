"""Head parameters: the trainable part of a compiled head, and where they are kept.

The default head is a pure readout of the frozen unembedding and has no
parameters. A fit on labeled states can replace it with an affine correction
on the sliced logits, or with a ridge probe on the hidden state at some layer
(which may be an early exit). Fitted heads are keyed by a task signature, so a
later request asking the same question gets the same head without labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynajev.fit import FitDecision, apply_affine
from dynajev.prompts import TEMPLATE_VERSION

_ARRAYS = ("weight", "ridge_bias", "mu", "sd")


def task_signature(
    qtype: str,
    instructions: str,
    labels: list[str],
    model_id: str,
    strategy: str = "auto",
    criteria: dict[str, Any] | None = None,
) -> str:
    """A stable key for "this question, with these answers, on this model".

    Instructions are compared after collapsing whitespace and case. Labels are
    compared exactly and in order, because they decide the token rows and the
    class order of any fitted weights. The prompt template version is part of
    the key: a head fitted on one template is not valid on another.
    """

    payload = {
        "v": TEMPLATE_VERSION,
        "model": model_id,
        "type": qtype,
        "instructions": " ".join(instructions.split()).lower(),
        "labels": [str(label).strip() for label in labels],
        "strategy": strategy,
        "criteria": {k: " ".join(str(v).split()).lower() for k, v in sorted((criteria or {}).items()) if v},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class HeadParams:
    chosen: str = "readout"  # readout | affine | ridge_probe
    temperature: float = 1.0
    bias: list[float] | None = None
    # Ridge probe. `layer` is the number of decoder layers run before the read;
    # None means the full stack (after the final norm).
    layer: int | None = None
    weight: np.ndarray | None = None
    ridge_bias: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    scale: float = 1.0
    note: str = ""
    signature: str | None = None
    qtype: str | None = None
    labels: list[str] = field(default_factory=list)
    n_examples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    model_id: str = ""
    created: float = field(default_factory=time.time)

    @classmethod
    def from_decision(cls, decision: FitDecision, labels: list[str]) -> "HeadParams":
        return cls(
            chosen="readout" if decision.chosen == "zero_shot" else decision.chosen,
            temperature=float(decision.temperature),
            bias=None if decision.bias is None else [float(v) for v in decision.bias],
            weight=decision.weight,
            ridge_bias=decision.ridge_bias,
            mu=decision.mu,
            sd=decision.sd,
            note=decision.note,
            labels=list(labels),
        )

    def affine_logits(self, logits: list[float]) -> list[float]:
        bias = None if self.bias is None else np.asarray(self.bias, dtype=np.float64)
        return apply_affine(logits, self.temperature, bias)

    def ridge_logits(self, hidden: Any) -> list[float]:
        if self.weight is None or self.ridge_bias is None or self.mu is None or self.sd is None:
            raise ValueError("ridge weights are missing")
        z = (np.asarray(hidden, dtype=np.float64) - self.mu) / self.sd
        return (self.scale * (self.weight @ z + self.ridge_bias)).tolist()

    def summary(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "type": self.qtype,
            "instructions": self.instructions,
            "labels": list(self.labels),
            "chosen": self.chosen,
            "exit_layer": self.layer,
            "temperature": round(self.temperature, 4),
            "n_examples": self.n_examples,
            "model": self.model_id,
            "created": round(self.created, 1),
        }

    def to_json(self) -> dict[str, Any]:
        meta = {
            "chosen": self.chosen,
            "temperature": self.temperature,
            "bias": self.bias,
            "layer": self.layer,
            "scale": self.scale,
            "note": self.note,
            "signature": self.signature,
            "qtype": self.qtype,
            "labels": self.labels,
            "n_examples": self.n_examples,
            "metrics": self.metrics,
            "instructions": self.instructions,
            "model_id": self.model_id,
            "created": self.created,
        }
        return json.loads(json.dumps(meta, default=_plain))

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _ARRAYS if getattr(self, name) is not None}

    @classmethod
    def from_json(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> "HeadParams":
        head = cls(**{k: v for k, v in meta.items() if k in cls.__dataclass_fields__})
        for name, value in (arrays or {}).items():
            if name in _ARRAYS:
                setattr(head, name, np.asarray(value, dtype=np.float64))
        return head


class HeadStore:
    """Fitted heads by signature: an in-memory LRU, optionally backed by a directory.

    With a directory, each head is `<signature>.json` (metadata, affine
    parameters, scores) plus `<signature>.npz` (probe weights, if any).
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None, capacity: int = 256):
        self.directory = Path(directory) if directory else None
        self.capacity = capacity
        self._memory: OrderedDict[str, HeadParams] = OrderedDict()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, signature: str) -> HeadParams | None:
        head = self._memory.get(signature)
        if head is not None:
            self._memory.move_to_end(signature)
            return head
        head = self._load(signature)
        if head is not None:
            self._remember(head)
        return head

    def put(self, head: HeadParams) -> None:
        if not head.signature:
            raise ValueError("a stored head needs a signature")
        self._remember(head)
        if self.directory is not None:
            (self.directory / f"{head.signature}.json").write_text(json.dumps(head.to_json(), indent=1))
            arrays = head.arrays()
            npz = self.directory / f"{head.signature}.npz"
            if arrays:
                np.savez(npz, **arrays)
            elif npz.exists():
                npz.unlink()

    def list(self) -> list[dict[str, Any]]:
        heads = {sig: head.summary() for sig, head in self._memory.items()}
        if self.directory is not None:
            for path in sorted(self.directory.glob("*.json")):
                if path.stem not in heads:
                    try:
                        heads[path.stem] = HeadParams.from_json(json.loads(path.read_text())).summary()
                    except (OSError, ValueError, TypeError):
                        continue
        return sorted(heads.values(), key=lambda item: -(item.get("created") or 0))

    def _remember(self, head: HeadParams) -> None:
        assert head.signature is not None
        self._memory[head.signature] = head
        self._memory.move_to_end(head.signature)
        while len(self._memory) > self.capacity:
            self._memory.popitem(last=False)

    def _load(self, signature: str) -> HeadParams | None:
        if self.directory is None:
            return None
        path = self.directory / f"{signature}.json"
        if not path.exists():
            return None
        meta = json.loads(path.read_text())
        npz = self.directory / f"{signature}.npz"
        arrays = dict(np.load(npz)) if npz.exists() else None
        return HeadParams.from_json(meta, arrays)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
