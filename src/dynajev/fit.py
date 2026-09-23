"""Fit a replacement head at request time, in closed form, and keep it only if it wins.

The zero-shot head is a slice of the pretrained unembedding. With a handful of
labeled states for the same question, two adjustments are possible:

- an affine correction on the sliced logits: one temperature and one bias per
  class. The bias is what moves a decision boundary, so it is the part that can
  fix a model that says "yes" too often. Glance's labeled `fit` is this step.
- a ridge probe from the answer-position hidden state to the classes
  (OpenJev's per-task head, without an offline training run)

Both are scored by leave-one-out log loss and leave-one-out accuracy. A change
is kept only when its held-out loss is lower and its held-out accuracy is not.

A few correct examples say nothing about how confident the model should be, so
the affine fit carries a prior toward the identity and the temperature is
capped at 2x sharpening. That stops a perfectly ranked six-example set from
driving every probability to 0 or 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynajev.score import nll

_MIN_T = 0.5
_MAX_T = 5.0
_PRIOR_LOG_T = 0.25
_PRIOR_BIAS = 0.1
_STEPS = 400
_LR = 0.3


@dataclass
class FitDecision:
    chosen: str
    temperature: float
    bias: np.ndarray | None
    zero_shot_loo_nll: float
    affine_loo_nll: float
    ridge_loo_nll: float | None
    zero_shot_loo_accuracy: float
    affine_loo_accuracy: float
    ridge_loo_accuracy: float | None
    weight: np.ndarray | None
    ridge_bias: np.ndarray | None
    mu: np.ndarray | None
    sd: np.ndarray | None
    note: str


def select_fit(hidden: np.ndarray, logits: np.ndarray, labels: np.ndarray, lam: float = 1.0) -> FitDecision:
    hidden = np.asarray(hidden, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, n_classes = logits.shape
    if hidden.shape[0] != n or labels.shape[0] != n:
        raise ValueError("hidden, logits, and labels must have the same length")
    base_nll = _mean_nll(logits, labels)
    base_acc = _accuracy(logits, labels)
    if n < 3:
        return FitDecision(
            chosen="zero_shot",
            temperature=1.0,
            bias=None,
            zero_shot_loo_nll=base_nll,
            affine_loo_nll=base_nll,
            ridge_loo_nll=None,
            zero_shot_loo_accuracy=base_acc,
            affine_loo_accuracy=base_acc,
            ridge_loo_accuracy=None,
            weight=None,
            ridge_bias=None,
            mu=None,
            sd=None,
            note="Fewer than 3 labeled states. The unembedding slice is unchanged.",
        )

    affine_nll, affine_acc = _loo_affine(logits, labels)
    # The correction we deploy is fit on every labeled row. The numbers we compare
    # are leave-one-out, so an in-sample gain that does not survive a held-out row
    # does not move the head.
    temperature, bias = _fit_affine(logits, labels)
    affine_wins = affine_nll < base_nll - 1e-3 and affine_acc >= base_acc
    if not affine_wins:
        temperature, bias = 1.0, None

    ridge_nll = ridge_acc = None
    weight = ridge_bias = mu = sd = None
    if n >= max(4, n_classes):
        ridge_nll, ridge_acc, weight, ridge_bias, mu, sd = _ridge_full_and_loo(hidden, labels, n_classes, lam)

    chosen = "zero_shot"
    note = "Leave-one-out did not beat the frozen unembedding slice, so that slice stayed the head."
    if affine_wins:
        chosen = "affine"
        flipped = _flips(logits, labels, temperature, bias)
        moved = f"It changed {flipped} of {n} answers on the labeled states." if flipped else "It changed no labeled answer."
        note = (
            f"A bias per class and a temperature of {temperature:.2f} on the sliced logits lowered leave-one-out loss. "
            f"{moved} The rows of the head are still the pretrained unembedding."
        )
    ridge_wins = (
        ridge_nll is not None
        and ridge_acc is not None
        and ridge_nll + 1e-4 < min(base_nll, affine_nll if affine_wins else base_nll)
        and ridge_acc >= max(base_acc, affine_acc if affine_wins else base_acc)
    )
    if ridge_wins:
        chosen = "ridge_probe"
        note = (
            "A ridge probe on the answer-position hidden state beat the unembedding slice under leave-one-out, "
            "so this request swaps the head. The trunk stays frozen. Small labeled sets still overfit; "
            "the gate is leave-one-out, not training loss."
        )
    return FitDecision(
        chosen=chosen,
        temperature=temperature if chosen == "affine" else 1.0,
        bias=bias if chosen == "affine" else None,
        zero_shot_loo_nll=base_nll,
        affine_loo_nll=affine_nll,
        ridge_loo_nll=ridge_nll,
        zero_shot_loo_accuracy=base_acc,
        affine_loo_accuracy=affine_acc,
        ridge_loo_accuracy=ridge_acc,
        weight=weight if chosen == "ridge_probe" else None,
        ridge_bias=ridge_bias if chosen == "ridge_probe" else None,
        mu=mu if chosen == "ridge_probe" else None,
        sd=sd if chosen == "ridge_probe" else None,
        note=note,
    )


def apply_affine(logits: list[float], temperature: float, bias: np.ndarray | None) -> list[float]:
    row = np.asarray(logits, dtype=np.float64) / temperature
    if bias is not None:
        row = row + bias
    return row.tolist()


def _mean_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, bias: np.ndarray | None = None) -> float:
    total = 0.0
    for row, label in zip(logits, labels):
        total += nll(apply_affine(row.tolist(), temperature, bias), int(label), 1.0)
    return total / len(labels)


def _accuracy(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, bias: np.ndarray | None = None) -> float:
    hits = 0
    for row, label in zip(logits, labels):
        hits += int(np.argmax(apply_affine(row.tolist(), temperature, bias)) == label)
    return hits / len(labels)


def _flips(logits: np.ndarray, labels: np.ndarray, temperature: float, bias: np.ndarray | None) -> int:
    changed = 0
    for row in logits:
        before = int(np.argmax(row))
        after = int(np.argmax(apply_affine(row.tolist(), temperature, bias)))
        changed += int(before != after)
    return changed


def _fit_affine(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """MAP fit of log-temperature and per-class bias by gradient descent.

    The problem is tiny (n <= 24 rows, K <= 50 classes), so a fixed number of
    plain steps is enough. The prior pulls toward T=1, b=0.
    """
    n, n_classes = logits.shape
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), labels] = 1.0
    u = 0.0
    b = np.zeros(n_classes)
    lo, hi = np.log(_MIN_T), np.log(_MAX_T)
    for _ in range(_STEPS):
        scale = np.exp(-u)
        z = logits * scale + b
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad_z = (p - onehot) / n
        grad_u = float(np.sum(grad_z * (-logits * scale))) + 2.0 * _PRIOR_LOG_T * u
        grad_b = grad_z.sum(axis=0) + 2.0 * _PRIOR_BIAS * b
        u = float(np.clip(u - _LR * grad_u, lo, hi))
        b = b - _LR * grad_b
        b = b - b.mean()
    return float(np.exp(u)), b


def _loo_affine(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    n = len(labels)
    total = 0.0
    hits = 0
    for held in range(n):
        train = np.ones(n, dtype=bool)
        train[held] = False
        temperature, bias = _fit_affine(logits[train], labels[train])
        row = apply_affine(logits[held].tolist(), temperature, bias)
        total += nll(row, int(labels[held]), 1.0)
        hits += int(np.argmax(row) == labels[held])
    return total / n, hits / n


def _ridge_full_and_loo(
    hidden: np.ndarray, labels: np.ndarray, n_classes: int, lam: float
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = hidden.mean(axis=0)
    sd = hidden.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z = (hidden - mu) / sd
    n = len(labels)
    total = 0.0
    hits = 0
    for held in range(n):
        train = np.ones(n, dtype=bool)
        train[held] = False
        weight, bias = _solve(z[train], labels[train], n_classes, lam)
        pred = weight @ z[held] + bias
        total += nll(pred.tolist(), int(labels[held]), 1.0)
        hits += int(np.argmax(pred) == labels[held])
    weight, bias = _solve(z, labels, n_classes, lam)
    return total / n, hits / n, weight, bias, mu, sd


def _solve(z: np.ndarray, labels: np.ndarray, n_classes: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression onto one-hot targets with an unpenalized intercept.

    Solved in the dual, (Zc Zc^T + lam I), which is n x n: a request carries
    a handful of labeled states and the hidden state has thousands of dims.
    """

    n = z.shape[0]
    targets = np.zeros((n, n_classes), dtype=np.float64)
    targets[np.arange(n), labels] = 1.0
    z_mean = z.mean(axis=0)
    t_mean = targets.mean(axis=0)
    zc = z - z_mean
    alpha = np.linalg.solve(zc @ zc.T + lam * np.eye(n), targets - t_mean)
    weight = (zc.T @ alpha).T
    bias = t_mean - weight @ z_mean
    return weight, bias


@dataclass
class Probe:
    """A ridge probe whose outputs are scaled into logits."""

    weight: np.ndarray
    bias: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    scale: float
    loo_nll: float
    loo_accuracy: float


_SCALES = np.geomspace(1.0, 16.0, 17)


def ridge_probe(hidden: np.ndarray, labels: np.ndarray, n_classes: int, lam: float = 1.0) -> Probe:
    """Fit a ridge probe and score it leave-one-out.

    Ridge regression onto one-hot targets gives outputs near 0 and 1, which are
    poor logits. One scale is chosen to minimize the leave-one-out log loss of
    the scaled outputs; that makes the probe's loss comparable with the sliced
    head's. The scale is picked on the same held-out predictions it is scored
    on, so the reported loss is slightly optimistic; it is one parameter, and
    it is capped at 16 so a perfectly separated handful of states does not
    become certainty.
    """

    hidden = np.asarray(hidden, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mu = hidden.mean(axis=0)
    sd = hidden.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z = (hidden - mu) / sd
    n = len(labels)
    preds = np.zeros((n, n_classes))
    for held in range(n):
        train = np.ones(n, dtype=bool)
        train[held] = False
        weight, bias = _solve(z[train], labels[train], n_classes, lam)
        preds[held] = weight @ z[held] + bias
    accuracy = float(np.mean(np.argmax(preds, axis=1) == labels))
    losses = [float(np.mean([nll((scale * row).tolist(), int(y), 1.0) for row, y in zip(preds, labels)])) for scale in _SCALES]
    best = int(np.argmin(losses))
    weight, bias = _solve(z, labels, n_classes, lam)
    return Probe(weight, bias, mu, sd, float(_SCALES[best]), losses[best], accuracy)


def candidate_layers(num_layers: int) -> list[int]:
    """Every fourth layer and the last one (every layer on a stack shorter than 8)."""

    step = 4 if num_layers >= 8 else 1
    return sorted({*range(step, num_layers, step), num_layers})


# How much worse (in nats of leave-one-out log loss) a shallower probe may be
# than the full-depth head and still be chosen.
EXIT_NLL_TOLERANCE = 0.1


@dataclass
class ExitChoice:
    layer: int | None
    probe: Probe | None
    full_accuracy: float
    full_nll: float
    scan: list[dict[str, float]]
    note: str


def choose_exit(
    per_layer: dict[int, np.ndarray], labels: np.ndarray, n_classes: int, full: FitDecision, num_layers: int
) -> ExitChoice:
    """The shallowest layer whose probe holds up against the full-depth head.

    A layer is accepted when its probe's leave-one-out accuracy is at least the
    full-depth head's and its leave-one-out loss is at most
    `EXIT_NLL_TOLERANCE` nats worse. The full-depth head is whatever
    `select_fit` chose: the sliced unembedding, its affine correction, or a
    ridge probe on the final layer.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if full.chosen == "affine":
        full_acc, full_nll = full.affine_loo_accuracy, full.affine_loo_nll
    elif full.chosen == "ridge_probe" and full.ridge_loo_accuracy is not None and full.ridge_loo_nll is not None:
        full_acc, full_nll = full.ridge_loo_accuracy, full.ridge_loo_nll
    else:
        full_acc, full_nll = full.zero_shot_loo_accuracy, full.zero_shot_loo_nll
    scan: list[dict[str, float]] = []
    if len(labels) < max(4, n_classes):
        return ExitChoice(None, None, full_acc, full_nll, scan, "Too few labeled states to compare layers.")
    for layer in sorted(k for k in per_layer if k < num_layers):
        probe = ridge_probe(per_layer[layer], labels, n_classes)
        scan.append({"layer": layer, "loo_accuracy": round(probe.loo_accuracy, 4), "loo_nll": round(probe.loo_nll, 4)})
        if probe.loo_accuracy >= full_acc and probe.loo_nll <= full_nll + EXIT_NLL_TOLERANCE:
            return ExitChoice(
                layer,
                probe,
                full_acc,
                full_nll,
                scan,
                f"A ridge probe after layer {layer} of {num_layers} matched the full-depth head under leave-one-out "
                f"(accuracy {probe.loo_accuracy:.2f} vs {full_acc:.2f}, loss {probe.loo_nll:.3f} vs {full_nll:.3f}), "
                f"so this field runs {layer} layers and reads that probe.",
            )
    return ExitChoice(None, None, full_acc, full_nll, scan, "No shallower layer matched the full-depth head; the full stack runs.")
