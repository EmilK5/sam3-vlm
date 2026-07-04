import os

import numpy as np

from verifier.queries import QuerySet, Query, load_query_set
from verifier import vip

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


def _clean_answers(query_set, class_name):
    """The noise-free answers a MockOracle(noise=0) would give for `class_name`:
    each answer is exactly that class's template value."""
    return np.array([q.templates[class_name] for q in query_set.queries], dtype=int)


def _toy_query_set():
    """3 classes, a few informative queries plus one all-zero-template query."""
    classes = ["c0", "c1", "c2"]
    queries = [
        Query("q0", "", {"c0": 1, "c1": -1, "c2": 0}),
        Query("q1", "", {"c0": 1, "c1": 0, "c2": -1}),
        Query("q2", "", {"c0": -1, "c1": 1, "c2": 0}),
        Query("q3", "", {"c0": 0, "c1": -1, "c2": 1}),
        Query("qZ", "", {"c0": 0, "c1": 0, "c2": 0}),  # uninformative for all
    ]
    return QuerySet(concept="toy", classes=classes, epsilon=0.15, queries=queries)


# (a) clean template answers for class k -> posterior(k) > 0.95 within max_q
def test_posterior_recovers_true_class():
    qs = load_query_set(GREEN_CITRUS)
    table = vip.likelihood_table(qs)
    for k, cls in enumerate(qs.classes):
        answers = _clean_answers(qs, cls)
        result = vip.run_ip(answers, table, prior=None, stop_eps=0.05, max_q=10)
        assert result["verdict_idx"] == k, f"class {cls} misclassified"
        assert result["posterior"][k] > 0.95


# (b) a query with template 0 for all classes is never selected
def test_zero_template_query_never_selected():
    qs = _toy_query_set()
    table = vip.likelihood_table(qs)
    zero_idx = [i for i, q in enumerate(qs.queries) if set(q.templates.values()) == {0}][0]

    # Zero-template query has exactly zero information gain from the empty set.
    assert vip.info_gain(zero_idx, [], np.zeros(len(qs.queries), dtype=int), table) == 0.0

    answers = _clean_answers(qs, "c0")
    result = vip.run_ip(answers, table, prior=None, stop_eps=0.05, max_q=len(qs.queries))
    selected = [m for m, _a in result["chain"]]
    assert zero_idx not in selected


# (c) chain length strictly shorter than M for clean answers
def test_chain_shorter_than_M_for_clean_answers():
    qs = load_query_set(GREEN_CITRUS)
    table = vip.likelihood_table(qs)
    M = len(qs.queries)
    answers = _clean_answers(qs, "target")
    result = vip.run_ip(answers, table, prior=None, stop_eps=0.05, max_q=M)
    assert 0 < len(result["chain"]) < M


# (d) permutation-invariance: posterior over full S equals product regardless of order
def test_posterior_permutation_invariant():
    qs = _toy_query_set()
    table = vip.likelihood_table(qs)
    answers = _clean_answers(qs, "c1")

    S1 = [0, 1, 2, 3]
    S2 = [3, 1, 0, 2]
    p1 = vip.posterior(answers, S1, table)
    p2 = vip.posterior(answers, S2, table)
    assert np.allclose(p1, p2)

    # Also equals the directly-computed normalized prior * product of likelihoods.
    K = table.shape[0]
    manual = np.full(K, 1.0 / K)
    for m in S1:
        manual = manual * table[:, m, answers[m] + 1]
    manual = manual / manual.sum()
    assert np.allclose(p1, manual)


# --- additional hardening tests for the V-IP math ---

def test_likelihood_rows_are_valid_distributions():
    qs = load_query_set(GREEN_CITRUS)
    table = vip.likelihood_table(qs)
    assert table.shape == (len(qs.classes), len(qs.queries), 3)
    assert np.all(table > 0.0)  # every entry positive -> log-space is safe
    assert np.allclose(table.sum(axis=2), 1.0)  # each P(a | class) sums to 1


def test_uninformative_template_gives_uniform_row():
    # tau == 0 -> uniform over the three answer values, for any epsilon.
    assert np.allclose(vip._template_answer_dist(0, 0.15), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(vip._template_answer_dist(0, 0.4), [1 / 3, 1 / 3, 1 / 3])


def test_info_gain_is_non_negative():
    qs = _toy_query_set()
    table = vip.likelihood_table(qs)
    answers = _clean_answers(qs, "c0")
    for m in range(len(qs.queries)):
        assert vip.info_gain(m, [], answers, table) >= -1e-12


def test_posterior_with_empty_S_equals_prior():
    qs = _toy_query_set()
    table = vip.likelihood_table(qs)
    answers = _clean_answers(qs, "c0")
    prior = np.array([0.2, 0.5, 0.3])
    assert np.allclose(vip.posterior(answers, [], table, prior), prior)


def test_run_ip_respects_max_q_cap():
    # With a tiny stop_eps the confidence rule can't trigger, so the budget caps it.
    qs = load_query_set(GREEN_CITRUS)
    table = vip.likelihood_table(qs)
    answers = _clean_answers(qs, "target")
    result = vip.run_ip(answers, table, prior=None, stop_eps=1e-6, max_q=3)
    assert len(result["chain"]) <= 3


def test_nonuniform_prior_shifts_verdict():
    # All-unsure answers carry no evidence, so the verdict follows the prior.
    qs = _toy_query_set()
    table = vip.likelihood_table(qs)
    unsure = np.zeros(len(qs.queries), dtype=int)
    result = vip.run_ip(unsure, table, prior=np.array([0.1, 0.1, 0.8]), stop_eps=0.05,
                        max_q=len(qs.queries))
    assert result["verdict_idx"] == 2  # c2 has the dominant prior
