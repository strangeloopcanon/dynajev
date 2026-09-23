import math

from dynajev.score import binary_from_logits, group_logit, logsumexp, ordinal_expectation, softmax
from dynajev.trunk import trim_repetition


def test_logsumexp_matches_two_equal_values():
    assert logsumexp([2.0, 2.0]) == 2.0 + math.log(2)


def test_group_softmax_is_a_union_probability():
    # Two spellings of yes against one no. Renormalizing the token masses
    # must match softmax of the group log-sum-exp scores.
    yes = group_logit([2.0, 2.0])
    no = group_logit([0.0])
    probs = softmax([no, yes])
    token_probs = softmax([2.0, 2.0, 0.0])
    assert abs(probs[1] - (token_probs[0] + token_probs[1])) < 1e-9
    assert abs(probs[1] - binary_from_logits(no, yes)) < 1e-9


def test_repeated_extract_is_cut_at_the_first_cycle():
    looped = "smashed, photo, pieces, arrived, smashed, photo, pieces, arrived, smashed, photo"
    assert trim_repetition(looped) == "smashed, photo, pieces, arrived"


def test_ordinal_expectation():
    assert ordinal_expectation([0.25, 0.25, 0.5]) == 1.25
    assert ordinal_expectation([0.0, 1.0], origin=1) == 2.0
