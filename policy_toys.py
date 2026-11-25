"""Small, exact toys behind Lecture 14.1: policy gradients, PPO clipping, preferences, DPO and GRPO.

The module is the single source of every number in the lecture. It needs numpy only. Everything is synthetic.

Part 1 learns a *policy* directly on the customer model of Lecture 12.1 (new, active, at risk; churned is the end).
Part 2 is one prompt with a handful of candidate answers: a reward model fitted to preferences, the effect of keeping
the trained model close to the supervised one (a KL penalty), direct preference optimisation, and group-relative
advantages with a rule-based reward.

There is deliberately no value iteration and no Q-learning here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

STATES = ("new", "active", "at risk")
GAMMA = 0.9
HORIZON = 60
END = 3  # churned

# margin (thousand tenge) of a month in each state under (nothing, discount)
REWARDS = np.array([[4.0, 2.0], [10.0, 7.0], [3.0, 1.0]])
# P[state, action, next] with next in (new, active, at risk, churned)
TRANSITIONS = np.array([
    [[0.0, 0.4, 0.3, 0.3], [0.0, 0.6, 0.3, 0.1]],
    [[0.0, 0.8, 0.2, 0.0], [0.0, 0.85, 0.15, 0.0]],
    [[0.0, 0.2, 0.4, 0.4], [0.0, 0.5, 0.4, 0.1]],
])


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ------------------------------------------------------------------------------------------------ Part 1: policies

def policy_value(p_discount: Sequence[float]) -> np.ndarray:
    """Exact value of the policy that offers a discount in state ``s`` with probability ``p_discount[s]``."""
    probs = np.asarray(p_discount, dtype=float)
    mix = np.stack([1 - probs, probs], axis=1)
    p_pi = np.einsum("sa,sat->st", mix, TRANSITIONS)[:, :3]
    r_pi = np.einsum("sa,sa->s", mix, REWARDS)
    return np.linalg.solve(np.eye(3) - GAMMA * p_pi, r_pi)


@dataclass(frozen=True)
class Batch:
    """Months from several customers followed with one fixed policy, flattened.

    Attributes:
        states: State of each month.
        actions: 0 nothing, 1 discount.
        returns: Discounted return from that month to the end of the customer's story.
        weights: ``gamma ** month`` (the month's position in its story).
        n_customers: Number of customers followed.
    """

    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    weights: np.ndarray
    n_customers: int


def sample_batch(theta: np.ndarray, n_customers: int, rng: np.random.Generator) -> Batch:
    """Follow ``n_customers`` new customers (until they churn or ``HORIZON`` months) with policy ``theta``."""
    probs = sigmoid(theta)
    cum = np.cumsum(TRANSITIONS, axis=2)
    cum[:, :, -1] = 1.0
    states, actions, returns, weights = [], [], [], []
    for _ in range(n_customers):
        s, episode = 0, []
        for _month in range(HORIZON):
            a = int(rng.random() < probs[s])
            nxt = int(np.searchsorted(cum[s, a], rng.random(), side="right"))
            episode.append((s, a, REWARDS[s, a]))
            if nxt == END:
                break
            s = nxt
        g = 0.0
        tail = []
        for s_t, a_t, r_t in reversed(episode):
            g = r_t + GAMMA * g
            tail.append((s_t, a_t, g))
        for t, (s_t, a_t, g_t) in enumerate(reversed(tail)):
            states.append(s_t)
            actions.append(a_t)
            returns.append(g_t)
            weights.append(GAMMA**t)
    return Batch(np.array(states), np.array(actions), np.array(returns), np.array(weights), n_customers)


def policy_gradient(theta: np.ndarray, batch: Batch, baseline: np.ndarray) -> np.ndarray:
    """REINFORCE gradient (Sutton and Barto, section 13.4): average of ``gamma^t (G_t - b(s_t)) grad log pi``."""
    probs = sigmoid(theta)
    score = batch.actions - probs[batch.states]  # d log pi(a|s) / d theta_s for a Bernoulli policy
    terms = batch.weights * (batch.returns - baseline[batch.states]) * score
    return np.bincount(batch.states, weights=terms, minlength=3) / batch.n_customers


def update_baseline(baseline: np.ndarray, batch: Batch, rate: float) -> np.ndarray:
    """Move the baseline of every visited state towards the mean return seen from it in this batch."""
    counts = np.bincount(batch.states, minlength=3)
    sums = np.bincount(batch.states, weights=batch.returns, minlength=3)
    seen = counts > 0
    new = baseline.copy()
    new[seen] += rate * (sums[seen] / counts[seen] - baseline[seen])
    return new


@dataclass(frozen=True)
class TrainRun:
    """Exact value of a new customer after every update, and the final discount probabilities."""

    values: np.ndarray
    p_discount: np.ndarray


def reinforce(seed: int | np.random.SeedSequence, n_updates: int, n_customers: int, lr: float, use_baseline: bool,
              baseline_rate: float = 0.2) -> TrainRun:
    """Train with REINFORCE; with ``use_baseline`` the returns are compared with a running estimate of ``v(s)``."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)
    baseline = np.zeros(3)
    values = [float(policy_value(sigmoid(theta))[0])]
    for _ in range(n_updates):
        batch = sample_batch(theta, n_customers, rng)
        b = baseline if use_baseline else np.zeros(3)
        theta = theta + lr * policy_gradient(theta, batch, b)
        baseline = update_baseline(baseline, batch, baseline_rate)
        values.append(float(policy_value(sigmoid(theta))[0]))
    return TrainRun(np.array(values), sigmoid(theta))


def gradient_spread(n_batches: int, n_customers: int, seed: int | np.random.SeedSequence) -> dict[str, np.ndarray]:
    """Mean and standard deviation of the gradient estimate at the 50/50 policy, with and without a baseline.

    The baseline is the exact value of that policy, so the two estimators have the same expectation.
    """
    theta = np.zeros(3)
    exact_v = policy_value(sigmoid(theta))
    rng = np.random.default_rng(seed)
    plain, centred = [], []
    for _ in range(n_batches):
        batch = sample_batch(theta, n_customers, rng)
        plain.append(policy_gradient(theta, batch, np.zeros(3)))
        centred.append(policy_gradient(theta, batch, exact_v))
    return {"plain_mean": np.mean(plain, axis=0), "plain_sd": np.std(plain, axis=0, ddof=1),
            "base_mean": np.mean(centred, axis=0), "base_sd": np.std(centred, axis=0, ddof=1)}


# ------------------------------------------------------------------------------------------------ PPO

def surrogate_steps(theta_old: np.ndarray, batch: Batch, advantages: np.ndarray, lr: float, n_steps: int,
                    clip: float | None) -> np.ndarray:
    """``n_steps`` gradient steps on one batch of the surrogate ``mean(ratio * A)``, optionally clipped as in PPO.

    PPO's clipped objective is ``mean(min(r A, clip(r, 1-eps, 1+eps) A))``; its gradient is that of ``r A`` where the
    unclipped term is the smaller one, and zero elsewhere.
    """
    p_old = sigmoid(theta_old)
    prob_old = np.where(batch.actions == 1, p_old[batch.states], 1 - p_old[batch.states])
    theta = theta_old.astype(float).copy()
    for _ in range(n_steps):
        p = sigmoid(theta)
        prob = np.where(batch.actions == 1, p[batch.states], 1 - p[batch.states])
        ratio = prob / prob_old
        active = np.ones_like(ratio, dtype=bool)
        if clip is not None:
            active = ratio * advantages <= np.clip(ratio, 1 - clip, 1 + clip) * advantages
        grad_ratio = ratio * (batch.actions - p[batch.states])
        terms = np.where(active, advantages * grad_ratio, 0.0)
        theta = theta + lr * np.bincount(batch.states, weights=terms, minlength=3) / len(ratio)
    return theta


def max_ratio(theta_old: np.ndarray, theta_new: np.ndarray) -> float:
    """Largest factor by which the probability of some action in some state changed (up or down)."""
    p0, p1 = sigmoid(theta_old), sigmoid(theta_new)
    ratios = np.concatenate([p1 / p0, (1 - p1) / (1 - p0)])
    return float(np.max(np.maximum(ratios, 1 / ratios)))


# ------------------------------------------------------------------------------------------------ Part 2: preferences

@dataclass(frozen=True)
class Answer:
    """One candidate answer to a fixed question: whether it is right and how long it is (1 = short)."""

    name: str
    correct: int
    length: int
    p_sft: float


ANSWERS: tuple[Answer, ...] = (
    Answer("short, right", 1, 1, 0.30),
    Answer("short, wrong", 0, 1, 0.30),
    Answer("long, right", 1, 2, 0.20),
    Answer("long, wrong", 0, 2, 0.19),
    Answer("very long, wrong", 0, 6, 0.01),
)


def sft_probs() -> np.ndarray:
    """How often the supervised model gives each candidate answer."""
    p = np.array([a.p_sft for a in ANSWERS])
    return p / p.sum()


def true_utility() -> np.ndarray:
    """What the firm wants: right answers. Length is worth nothing."""
    return np.array([2.0 * a.correct for a in ANSWERS])


def features() -> np.ndarray:
    """Columns: correct, length."""
    return np.array([[a.correct, a.length] for a in ANSWERS], dtype=float)


LENGTH_CAP = 2  # labelers like longer answers, but only up to "long"
SEEN = 4  # the common answers; the very long one never reaches a labeler


def labeler_utility(length_bias: float) -> np.ndarray:
    """What labelers reward: right answers and, with ``length_bias``, longer ones up to ``LENGTH_CAP``."""
    return true_utility() + length_bias * np.minimum(features()[:, 1], LENGTH_CAP)


def sample_preferences(n_pairs: int, seed: int | np.random.SeedSequence, length_bias: float) -> np.ndarray:
    """``(winner, loser)`` index pairs among the ``SEEN`` common answers, drawn as often as the model writes them.

    A pair is judged by the Bradley-Terry model: ``Pr(i beats j) = sigmoid(u_i - u_j)``.
    """
    rng = np.random.default_rng(seed)
    utility = labeler_utility(length_bias)
    p = sft_probs()[:SEEN] / sft_probs()[:SEEN].sum()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i, j = rng.choice(SEEN, size=2, p=p)
        if i == j:
            continue
        if rng.random() < sigmoid(utility[i] - utility[j]):
            pairs.append((i, j))
        else:
            pairs.append((j, i))
    return np.array(pairs)


def fit_reward_model(pairs: np.ndarray, x: np.ndarray, steps: int = 4000, lr: float = 0.5) -> np.ndarray:
    """Weights ``w`` maximising the Bradley-Terry likelihood of the preferences with scores ``x @ w``."""
    diff = x[pairs[:, 0]] - x[pairs[:, 1]]
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        w = w + lr * diff.T @ (1.0 - sigmoid(diff @ w)) / len(pairs)
    return w


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p || q)`` in nats."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def kl_regularised(reward: np.ndarray, beta: float) -> np.ndarray:
    """The distribution maximising ``E_p[reward] - beta * KL(p || p_sft)``: ``p_sft * exp(reward / beta)``, normalised."""
    logits = np.log(sft_probs()) + reward / beta
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def dpo(pairs: np.ndarray, beta: float, steps: int = 20000, lr: float = 2.0) -> np.ndarray:
    """Minimise the DPO loss over a free policy on the candidates, starting from the supervised model."""
    ref = np.log(sft_probs())
    theta = ref.copy()
    for _ in range(steps):
        logp = theta - np.log(np.sum(np.exp(theta)))
        margin = beta * (logp - ref)
        delta = margin[pairs[:, 0]] - margin[pairs[:, 1]]
        weight = beta * (1.0 - sigmoid(delta)) / len(pairs)
        grad_logp = np.zeros(len(theta))
        np.add.at(grad_logp, pairs[:, 0], weight)
        np.add.at(grad_logp, pairs[:, 1], -weight)
        grad = grad_logp - grad_logp.sum() * np.exp(logp)  # chain rule through the softmax
        theta = theta + lr * grad
    p = np.exp(theta - theta.max())
    return p / p.sum()


def tabular_reward(pairs: np.ndarray) -> np.ndarray:
    """A reward model with one free score per candidate, fitted to the preferences (the two-step route)."""
    return fit_reward_model(pairs, np.eye(len(ANSWERS)), steps=20000, lr=2.0)


# ------------------------------------------------------------------------------------------------ GRPO

def group_advantages(rewards: Sequence[float]) -> np.ndarray:
    """GRPO's advantage within one group of answers to the same question: ``(r - mean) / std`` (zero if all equal)."""
    r = np.asarray(rewards, dtype=float)
    sd = r.std()
    if sd == 0:
        return np.zeros_like(r)
    return (r - r.mean()) / sd


def grpo_train(seed: int | np.random.SeedSequence, n_steps: int, group: int, lr: float, p_start: Sequence[float],
               correct: int) -> np.ndarray:
    """Probability of the correct answer after every step, training with a rule-based reward (1 if right, else 0).

    A policy-gradient update with group-relative advantages and no value model. The full method also keeps a clipped
    ratio and a KL penalty; with one gradient step per group the ratio is 1, so they are left out of this toy.
    """
    rng = np.random.default_rng(seed)
    theta = np.log(np.asarray(p_start, dtype=float))
    curve = []
    for _ in range(n_steps + 1):
        p = np.exp(theta - theta.max())
        p /= p.sum()
        curve.append(p[correct])
        answers = rng.choice(len(p), size=group, p=p)
        adv = group_advantages((answers == correct).astype(float))
        grad = np.zeros_like(theta)
        for a, g in zip(answers, adv, strict=True):
            grad += g * (np.eye(len(p))[a] - p)
        theta = theta + lr * grad / group
    return np.array(curve)


def lora_counts(d_in: int, d_out: int, rank: int) -> dict[str, int]:
    """Trainable numbers of one weight matrix: all of it, or two low-rank factors."""
    return {"full": d_in * d_out, "lora": rank * (d_in + d_out)}
