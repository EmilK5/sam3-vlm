"""
verifier/vip.py

Training-free Information Pursuit core for the FM+V-IP verifier. Pure numpy,
no model calls. Given a query set's per-class answer templates tau_{k,m} and a
fixed noise rate epsilon, this builds a factorized naive-Bayes likelihood and
computes the closed-form posterior, greedy conditional-mutual-information query
selection, and the standard IP stopping rule.

Conventions
-----------
- Classes are indexed 0..K-1 in `query_set.classes` order.
- Queries are indexed 0..M-1 in `query_set.queries` order.
- Answers take values in {-1, 0, +1} = (no, unsure/undecidable, yes).
- The likelihood table axis of length 3 is indexed by answer value via
  idx = answer + 1, i.e. -1->0, 0->1, +1->2 (see ANSWER_VALUES).
- All entropies are in nats (natural log); the choice is internally consistent
  and does not affect argmax selection.
"""

import numpy as np

# Ordered answer values; position i corresponds to table[..., i].
ANSWER_VALUES = (-1, 0, 1)


def _answer_index(answer) -> int:
    """Map an answer value in {-1, 0, +1} to a table column index in {0, 1, 2}."""
    return int(answer) + 1


def _template_answer_dist(tau: int, eps: float) -> np.ndarray:
    """P(answer | class) over (a=-1, a=0, a=+1) for a single template value tau.

    Noise model (proposal): answer agrees with template -> 1-eps, contradicts ->
    eps, uninformative (answer 0 or template 0) -> 1/2. The raw values are then
    normalized over the three answer values so the row is a proper distribution.
    """
    raw = np.empty(3, dtype=float)
    for i, a in enumerate(ANSWER_VALUES):
        prod = a * tau
        if prod > 0:
            raw[i] = 1.0 - eps
        elif prod < 0:
            raw[i] = eps
        else:  # a == 0 or tau == 0 -> uninformative
            raw[i] = 0.5
    return raw / raw.sum()


def likelihood_table(query_set) -> np.ndarray:
    """Build the (K, M, 3) likelihood table P(answer | class) from templates + eps."""
    classes = query_set.classes
    queries = query_set.queries
    eps = query_set.epsilon
    K, M = len(classes), len(queries)

    table = np.empty((K, M, 3), dtype=float)
    for k, cls in enumerate(classes):
        for m, query in enumerate(queries):
            tau = query.templates[cls]
            table[k, m, :] = _template_answer_dist(tau, eps)
    return table


def _as_prior(prior, K: int) -> np.ndarray:
    """Return a length-K prior array; None -> uniform."""
    if prior is None:
        return np.full(K, 1.0 / K)
    return np.asarray(prior, dtype=float)


def posterior(answers, S, table: np.ndarray, prior=None) -> np.ndarray:
    """P(class | answers restricted to index set S), computed in log space.

    answers : array-like length M of observed answers in {-1, 0, +1}.
    S       : iterable of query indices whose answers are conditioned on.
    Returns a length-K probability vector.
    """
    K = table.shape[0]
    log_post = np.log(_as_prior(prior, K))
    for m in S:
        a_idx = _answer_index(answers[m])
        log_post = log_post + np.log(table[:, m, a_idx])
    log_post -= log_post.max()  # stabilize before exponentiating
    p = np.exp(log_post)
    return p / p.sum()


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats of a distribution p (entries assumed > 0)."""
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p * np.log(p)))


def info_gain(m: int, S, answers, table: np.ndarray, prior=None) -> float:
    """Closed-form conditional mutual information I(a_m ; v | a_S).

    I = H(a_m | a_S) - sum_k P(k | a_S) H(a_m | v=k). This is the *expected*
    information gain of querying m next; it does not depend on the (unobserved)
    value answers[m], only on the current posterior over classes given a_S.
    """
    post_k = posterior(answers, S, table, prior)  # P(k | a_S), length K
    dist_mk = table[:, m, :]                       # P(a_m | k), shape (K, 3)

    # Marginal predictive over a_m given a_S: sum_k P(k|a_S) P(a_m | k).
    marginal = post_k @ dist_mk                     # length 3
    h_marginal = _entropy(marginal)

    # Expected conditional entropy sum_k P(k|a_S) H(a_m | k).
    h_conditional = float(sum(post_k[k] * _entropy(dist_mk[k]) for k in range(len(post_k))))

    return h_marginal - h_conditional


def run_ip(answers, table: np.ndarray, prior=None, stop_eps: float = 0.10,
           max_q: int = 10) -> dict:
    """Greedy Information Pursuit over a precomputed answer vector.

    Selects queries by maximum expected CMI until the posterior is confident
    (max_k P >= 1 - stop_eps), the budget max_q is reached, or no remaining
    query is informative. The post-hoc chain over a batched answer vector is
    identical to the sequential variant.

    Returns dict with:
        verdict_idx : MAP class index.
        posterior   : length-K posterior over classes at termination.
        chain       : list of (query_idx, observed_answer) in selection order.
    """
    K, M, _ = table.shape
    S = []
    chain = []
    post = posterior(answers, S, table, prior)

    while len(S) < max_q:
        if post.max() >= 1.0 - stop_eps:
            break

        # Pick the unselected query with the largest expected information gain.
        best_m, best_gain = None, 0.0
        for m in range(M):
            if m in S:
                continue
            g = info_gain(m, S, answers, table, prior)
            if g > best_gain:
                best_gain = g
                best_m = m

        # No informative query remains (e.g. only all-zero-template queries left).
        if best_m is None or best_gain <= 1e-12:
            break

        S.append(best_m)
        chain.append((best_m, int(answers[best_m])))
        post = posterior(answers, S, table, prior)

    return {
        "verdict_idx": int(post.argmax()),
        "posterior": post,
        "chain": chain,
    }
