import numpy as np

from readhead.fit import apply_affine, apply_ridge, select_fit


def test_ridge_replaces_a_head_with_no_logit_signal():
    rng = np.random.default_rng(0)
    hidden = np.array(
        [
            [2.0, 0.1],
            [1.6, -0.2],
            [1.8, 0.0],
            [-2.0, 0.2],
            [-1.5, -0.1],
            [-1.7, 0.3],
        ]
    )
    logits = rng.normal(scale=0.01, size=(6, 2))
    labels = np.array([0, 0, 0, 1, 1, 1])
    decision = select_fit(hidden, logits, labels, lam=0.1)
    assert decision.chosen == "ridge_probe"
    assert decision.ridge_loo_nll is not None
    assert decision.ridge_loo_nll < decision.zero_shot_loo_nll
    scored = apply_ridge(hidden[0], decision)
    assert scored[0] > scored[1]


def test_bias_moves_a_boundary_the_slice_gets_wrong():
    # The slice leans "yes" by a constant offset: every "no" state still has the
    # yes logit ahead. Only a bias can fix that; a temperature never changes argmax.
    rng = np.random.default_rng(2)
    hidden = rng.normal(size=(8, 12))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0 = no, 1 = yes
    logits = np.array(
        [
            [0.0, 0.6],
            [0.0, 0.8],
            [0.0, 0.7],
            [0.0, 0.5],
            [0.0, 2.6],
            [0.0, 2.9],
            [0.0, 2.7],
            [0.0, 3.1],
        ]
    )
    decision = select_fit(hidden, logits, labels, lam=50.0)
    assert decision.chosen == "affine"
    assert decision.zero_shot_loo_accuracy == 0.5
    assert decision.affine_loo_accuracy == 1.0
    assert decision.bias is not None
    corrected = apply_affine([0.0, 0.7], decision.temperature, decision.bias)
    assert corrected[0] > corrected[1]
    corrected = apply_affine([0.0, 2.8], decision.temperature, decision.bias)
    assert corrected[1] > corrected[0]


def test_correct_examples_do_not_collapse_confidence():
    # Six states the slice already ranks correctly at about 87% confidence. That
    # is not evidence for certainty, so the fit must not drive probabilities to 0/1.
    rng = np.random.default_rng(3)
    hidden = rng.normal(size=(6, 12))
    labels = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([[1.9, 0.0]] * 3 + [[0.0, 1.9]] * 3)
    decision = select_fit(hidden, logits, labels, lam=50.0)
    assert decision.temperature >= 0.5
    if decision.chosen == "affine":
        row = apply_affine([1.9, 0.0], decision.temperature, decision.bias)
        margin = row[0] - row[1]
        assert margin < 1.9 * 2 + 1e-6


def test_three_examples_are_required():
    decision = select_fit(np.zeros((2, 3)), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
    assert decision.chosen == "zero_shot"
    assert decision.weight is None
