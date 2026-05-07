"""Dependency-light strategy blueprints for ELSA federated experiments.

These functions are intentionally small and framework-agnostic. They are not a
replacement for Flower strategies; they capture weighting/selection rules that
can be moved into `federated_elsa_robotics.server_app` once an experiment needs
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ClientSignal:
    """Server-visible client summary used for selection or weighting."""

    client_id: str
    num_examples: int
    train_loss: float
    eval_loss: float | None = None
    local_steps: int | None = None
    success_rate: float | None = None


def normalize_weights(raw_weights: Sequence[float], eps: float = 1.0e-12) -> list[float]:
    """Normalize non-negative weights; fall back to uniform if all are zero."""
    if not raw_weights:
        return []
    if any(weight < 0 for weight in raw_weights):
        raise ValueError("weights must be non-negative")
    total = sum(raw_weights)
    if total <= eps:
        return [1.0 / len(raw_weights)] * len(raw_weights)
    return [float(weight) / total for weight in raw_weights]


def simplex_project(values: Sequence[float]) -> list[float]:
    """Project values onto the probability simplex.

    This is the Wang-Carreira-Perpinan sorting projection used by AFL-style
    mixture weights: output entries are non-negative and sum to one.
    """
    if not values:
        return []
    sorted_values = sorted((float(value) for value in values), reverse=True)
    cumulative = 0.0
    theta = 0.0
    for index, value in enumerate(sorted_values, start=1):
        cumulative += value
        candidate = (cumulative - 1.0) / index
        if value - candidate > 0.0:
            theta = candidate
    projected = [max(float(value) - theta, 0.0) for value in values]
    total = sum(projected)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in projected]


def afl_update_lambdas(
    lambdas: Sequence[float],
    losses: Sequence[float],
    *,
    lambda_lr: float,
) -> list[float]:
    """Agnostic FL dual update over client/domain mixture weights."""
    if len(lambdas) != len(losses):
        raise ValueError("lambdas and losses must have the same length")
    if lambda_lr < 0:
        raise ValueError("lambda_lr must be non-negative")
    return simplex_project(
        [float(lam) + lambda_lr * float(loss) for lam, loss in zip(lambdas, losses)]
    )


def fedprox_mu_grid() -> list[float]:
    """Small grid that keeps FedAvg as the zero-prox baseline."""
    return [0.0, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]


def power_of_choice_candidates(
    clients: list[ClientSignal],
    *,
    d: int,
    metric: str = "train_loss",
) -> list[ClientSignal]:
    """Return the highest-loss clients among a candidate set.

    Use this for diagnostic rounds only unless the evaluation metric also
    reports unbiased random-client performance.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if metric not in {"train_loss", "eval_loss"}:
        raise ValueError("metric must be 'train_loss' or 'eval_loss'")

    def score(client: ClientSignal) -> float:
        value = getattr(client, metric)
        if value is None:
            return float("-inf")
        return float(value)

    return sorted(clients, key=score, reverse=True)[:d]


def qffl_weight(loss: float, q: float, eps: float = 1.0e-12) -> float:
    """Unnormalized q-FFL style loss exponent.

    This is only the numerator coefficient. Full q-FedAvg also needs the
    denominator returned by `qffl_h`, otherwise it is just a loss-weighted
    diagnostic rather than the paper's dynamic step-size rule.
    """
    if q < 0:
        raise ValueError("q must be non-negative")
    return max(float(loss), eps) ** q


def qffl_h(
    loss: float,
    q: float,
    grad_norm_sq: float,
    *,
    base_lr: float,
    eps: float = 1.0e-10,
) -> float:
    """q-FedAvg denominator term for one client.

    The fair_flearn reference computes
    q * loss^(q-1) * ||grad||^2 + (1 / lr) * loss^q. The `grad_norm_sq`
    argument is the squared L2 norm of the pseudo-gradient/update.
    """
    if q < 0:
        raise ValueError("q must be non-negative")
    if grad_norm_sq < 0:
        raise ValueError("grad_norm_sq must be non-negative")
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    clipped_loss = max(float(loss), eps)
    curvature = 0.0
    if q > 0.0:
        curvature = q * (clipped_loss ** (q - 1.0)) * float(grad_norm_sq)
    return curvature + (clipped_loss**q) / float(base_lr)


def qffl_server_scale(h_values: Sequence[float], eps: float = 1.0e-12) -> float:
    """Server multiplier for the summed q-FedAvg deltas."""
    if any(value < 0 for value in h_values):
        raise ValueError("h values must be non-negative")
    total = sum(h_values)
    if total <= eps:
        raise ValueError("sum of h values must be positive")
    return 1.0 / total


def maxfl_appeal_weight(
    loss: float,
    threshold: float,
    *,
    temperature: float = 10.0,
) -> float:
    """Smooth MaxFL-style weight around a client-specific requirement.

    The weight is largest near the threshold, where a small update can change
    whether a client finds the global model acceptable.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = temperature * (float(loss) - float(threshold))
    # Numerically stable sigmoid derivative.
    if z >= 0:
        exp_neg = math.exp(-z)
        sigmoid = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(z)
        sigmoid = exp_pos / (1.0 + exp_pos)
    return sigmoid * (1.0 - sigmoid) * temperature


def maxfl_server_lr(
    appeal_weights: Sequence[float],
    *,
    base_lr: float,
    eps: float = 1.0e-12,
) -> float:
    """MaxFL adaptive server LR from the sum of appeal weights."""
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if any(weight < 0 for weight in appeal_weights):
        raise ValueError("appeal weights must be non-negative")
    return float(base_lr) / (sum(appeal_weights) + eps)


def fednova_normalizer(local_steps: int, num_examples: int) -> float:
    """Simple normalizer for heterogeneous local update lengths."""
    if local_steps <= 0:
        raise ValueError("local_steps must be positive")
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    return float(num_examples) / float(local_steps)


def fednova_effective_tau(
    local_steps: Sequence[int],
    weights: Sequence[float],
) -> float:
    """Weighted effective local step count used by FedNova."""
    if len(local_steps) != len(weights):
        raise ValueError("local_steps and weights must have the same length")
    if any(step <= 0 for step in local_steps):
        raise ValueError("local steps must be positive")
    normalized = normalize_weights(weights)
    return sum(float(step) * weight for step, weight in zip(local_steps, normalized))


def fedexp_server_lr(
    average_delta_norm: float,
    client_delta_norms: list[float],
    *,
    eps: float = 1.0e-12,
    min_lr: float = 1.0,
    max_lr: float = 3.0,
) -> float:
    """Bounded FedExP server step-size from client and average delta norms."""
    if average_delta_norm < 0 or any(value < 0 for value in client_delta_norms):
        raise ValueError("delta norms must be non-negative")
    if max_lr < min_lr:
        raise ValueError("max_lr must be >= min_lr")
    if not client_delta_norms:
        return min_lr
    if average_delta_norm <= eps:
        return min_lr
    client_norm_sq_sum = sum(value * value for value in client_delta_norms)
    raw = client_norm_sq_sum / (
        2.0 * len(client_delta_norms) * average_delta_norm * average_delta_norm + eps
    )
    return max(min_lr, min(max_lr, raw))


def _flatten_numbers(values: object) -> Iterable[float]:
    if isinstance(values, Mapping):
        for key in sorted(values, key=str):
            yield from _flatten_numbers(values[key])
    elif isinstance(values, (list, tuple)):
        for item in values:
            yield from _flatten_numbers(item)
    else:
        yield float(values)


def squared_l2_norm(values: object) -> float:
    """Squared L2 norm for nested numeric Python containers."""
    return sum(value * value for value in _flatten_numbers(values))
