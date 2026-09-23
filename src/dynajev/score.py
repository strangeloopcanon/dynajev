"""Probability geometry for closed readouts.

These are the heads, minus the weights. A boolean is a margin between two
verbalizers. A choice is a softmax over a closed set. A rating is the
expectation of that softmax. None of them generate a token.
"""

from __future__ import annotations

import math


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_from_logits(no_logit: float, yes_logit: float) -> float:
    """P(yes) = sigmoid(yes − no). Equivalent to a 2-way softmax."""

    return sigmoid(yes_logit - no_logit)


def ordinal_expectation(probabilities: list[float], origin: float = 0.0, step: float = 1.0) -> float:
    return sum((origin + i * step) * p for i, p in enumerate(probabilities))


def nll(logits: list[float], label: int, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [v / temperature for v in logits]
    probabilities = softmax(scaled)
    prob = min(max(probabilities[label], 1e-12), 1.0)
    return -math.log(prob)
