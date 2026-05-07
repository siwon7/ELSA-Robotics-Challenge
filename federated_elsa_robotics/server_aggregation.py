from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SERVER_STRATEGIES = {
    "fedavg",
    "fedexp",
    "fednova",
    "qfedavg",
    "afl",
    "maxfl",
}


@dataclass
class ClientUpdate:
    arrays: list[np.ndarray]
    num_examples: int
    metrics: dict[str, Any]
    client_key: str


def _cfg_section(config, section: str):
    return getattr(config, section, None) if config is not None else None


def _cfg_float(config, key: str, default: float) -> float:
    federated_cfg = _cfg_section(config, "federated")
    value = getattr(federated_cfg, key, None) if federated_cfg is not None else None
    if value in (None, ""):
        return float(default)
    return float(value)


def _cfg_bool(config, key: str, default: bool) -> bool:
    federated_cfg = _cfg_section(config, "federated")
    value = getattr(federated_cfg, key, None) if federated_cfg is not None else None
    if value in (None, ""):
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _model_float(config, key: str, default: float) -> float:
    model_cfg = _cfg_section(config, "model")
    value = getattr(model_cfg, key, None) if model_cfg is not None else None
    if value in (None, ""):
        return float(default)
    return float(value)


def _metric_float(metrics: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = metrics.get(key, default)
    if value in (None, ""):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _client_loss(metrics: dict[str, Any], default: float = 1.0) -> float:
    for key in ("pre_train_loss", "train_loss", "post_train_loss"):
        value = _metric_float(metrics, key)
        if value is not None:
            return max(value, 1.0e-12)
    return float(default)


def client_key_from_metrics(metrics: dict[str, Any], fallback: str) -> str:
    partition = metrics.get("partition_id")
    env_id = metrics.get("env_id")
    if partition not in (None, "") and env_id not in (None, ""):
        return f"partition_{partition}_env_{env_id}"
    if partition not in (None, ""):
        return f"partition_{partition}"
    if env_id not in (None, ""):
        return f"env_{env_id}"
    return fallback


def _zeros_like(reference: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros_like(array, dtype=np.float64) for array in reference]


def _weighted_sum(array_sets: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
    if not array_sets:
        raise ValueError("Cannot aggregate an empty update set")
    if len(array_sets) != len(weights):
        raise ValueError("array_sets and weights must have the same length")
    result = _zeros_like(array_sets[0])
    for arrays, weight in zip(array_sets, weights):
        for index, array in enumerate(arrays):
            result[index] += float(weight) * array.astype(np.float64, copy=False)
    return result


def _normalize_weights(raw_weights: list[float], eps: float = 1.0e-12) -> list[float]:
    if not raw_weights:
        return []
    clipped = [max(0.0, float(weight)) for weight in raw_weights]
    total = sum(clipped)
    if total <= eps:
        return [1.0 / len(raw_weights)] * len(raw_weights)
    return [weight / total for weight in clipped]


def _weighted_average(array_sets: list[list[np.ndarray]], raw_weights: list[float]) -> list[np.ndarray]:
    return _weighted_sum(array_sets, _normalize_weights(raw_weights))


def _deltas(
    updates: list[ClientUpdate],
    previous_arrays: list[np.ndarray],
) -> list[list[np.ndarray]]:
    return [
        [
            updated.astype(np.float64, copy=False) - previous.astype(np.float64, copy=False)
            for updated, previous in zip(update.arrays, previous_arrays, strict=True)
        ]
        for update in updates
    ]


def _add_scaled(
    previous_arrays: list[np.ndarray],
    delta: list[np.ndarray],
    scale: float,
) -> list[np.ndarray]:
    return [
        previous + float(scale) * update.astype(previous.dtype, copy=False)
        for previous, update in zip(previous_arrays, delta, strict=True)
    ]


def _scale_delta(delta: list[np.ndarray], scale: float) -> list[np.ndarray]:
    return [float(scale) * array for array in delta]


def _sum_delta(deltas: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
    return _weighted_sum(deltas, weights)


def _squared_l2(arrays: list[np.ndarray]) -> float:
    return float(sum(float(np.sum(array.astype(np.float64, copy=False) ** 2)) for array in arrays))


def _project_simplex(values: list[float]) -> list[float]:
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


def _sigmoid_derivative_weight(loss: float, threshold: float, temperature: float) -> float:
    z = float(temperature) * (float(loss) - float(threshold))
    if z >= 0:
        exp_neg = math.exp(-z)
        sigmoid = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(z)
        sigmoid = exp_pos / (1.0 + exp_pos)
    return sigmoid * (1.0 - sigmoid) * float(temperature)


class ServerSideAggregator:
    """Stateful server-side aggregation methods for trainable parameter arrays."""

    def __init__(self, strategy_name: str, config=None):
        normalized = str(strategy_name or "fedavg").lower()
        if normalized not in SERVER_STRATEGIES:
            raise ValueError(
                f"Unsupported server strategy: {strategy_name}. "
                f"Expected one of {sorted(SERVER_STRATEGIES)}"
            )
        self.strategy_name = normalized
        self.config = config
        self.afl_lambdas: dict[str, float] = {}

    def aggregate(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        if not updates:
            raise ValueError("Cannot aggregate an empty update set")
        strategy = self.strategy_name
        if strategy == "fedavg":
            return self._fedavg(updates)
        if strategy == "fedexp":
            return self._fedexp(updates, previous_arrays)
        if strategy == "fednova":
            return self._fednova(updates, previous_arrays)
        if strategy == "qfedavg":
            return self._qfedavg(updates, previous_arrays)
        if strategy == "afl":
            return self._afl(updates, previous_arrays)
        if strategy == "maxfl":
            return self._maxfl(updates, previous_arrays)
        raise AssertionError(f"unreachable strategy={strategy}")

    def _base_metrics(self, updates: list[ClientUpdate]) -> dict[str, float | int | str]:
        losses = [_client_loss(update.metrics) for update in updates]
        return {
            "server_strategy": self.strategy_name,
            "client_loss_min": min(losses),
            "client_loss_max": max(losses),
            "client_loss_mean": sum(losses) / len(losses),
        }

    def _fedavg(self, updates: list[ClientUpdate]) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        arrays = _weighted_average(
            [update.arrays for update in updates],
            [float(update.num_examples) for update in updates],
        )
        metrics = self._base_metrics(updates)
        metrics["server_lr"] = 1.0
        return arrays, metrics

    def _fedexp(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        fedavg_arrays, _ = self._fedavg(updates)
        avg_delta = [
            current.astype(np.float64, copy=False) - previous.astype(np.float64, copy=False)
            for current, previous in zip(fedavg_arrays, previous_arrays, strict=True)
        ]
        avg_delta_norm = math.sqrt(max(_squared_l2(avg_delta), 0.0))
        client_deltas = _deltas(updates, previous_arrays)
        client_norms = [
            math.sqrt(
                max(
                    _metric_float(update.metrics, "delta_norm_sq", _squared_l2(delta))
                    or 0.0,
                    0.0,
                )
            )
            for update, delta in zip(updates, client_deltas)
        ]
        min_lr = _cfg_float(self.config, "fedexp_min_lr", 1.0)
        max_lr = _cfg_float(self.config, "fedexp_max_lr", 3.0)
        if max_lr < min_lr:
            max_lr = min_lr
        if avg_delta_norm <= 1.0e-12:
            server_lr = min_lr
        else:
            # FedExP uses sum_i ||Delta_i||^2 / (2M ||Delta_bar||^2 + eps).
            # Our deltas have the opposite sign from the paper's Delta_i, which
            # does not affect the squared norms or the final update direction.
            client_norm_sq_sum = sum(norm * norm for norm in client_norms)
            raw_lr = client_norm_sq_sum / (
                2.0 * len(client_norms) * (avg_delta_norm * avg_delta_norm) + 1.0e-12
            )
            server_lr = max(min_lr, min(max_lr, raw_lr))
        server_lr *= _cfg_float(self.config, "server_learning_rate", 1.0)
        arrays = _add_scaled(previous_arrays, avg_delta, server_lr)
        metrics = self._base_metrics(updates)
        metrics.update(
            {
                "server_lr": float(server_lr),
                "fedexp_avg_delta_norm": float(avg_delta_norm),
                "fedexp_client_delta_norm_mean": float(sum(client_norms) / len(client_norms)),
            }
        )
        return arrays, metrics

    def _fednova(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        deltas = _deltas(updates, previous_arrays)
        examples = [max(1.0, float(update.num_examples)) for update in updates]
        weights = _normalize_weights(examples)
        local_steps = [
            max(1.0, _metric_float(update.metrics, "local_steps", 1.0) or 1.0)
            for update in updates
        ]
        normalized_deltas = [
            _scale_delta(delta, 1.0 / tau)
            for delta, tau in zip(deltas, local_steps)
        ]
        effective_tau = sum(weight * tau for weight, tau in zip(weights, local_steps))
        normalized_delta = _sum_delta(normalized_deltas, weights)
        server_lr = _cfg_float(self.config, "server_learning_rate", 1.0)
        arrays = _add_scaled(previous_arrays, normalized_delta, server_lr * effective_tau)
        metrics = self._base_metrics(updates)
        metrics.update(
            {
                "server_lr": float(server_lr),
                "fednova_effective_tau": float(effective_tau),
                "fednova_tau_min": float(min(local_steps)),
                "fednova_tau_max": float(max(local_steps)),
            }
        )
        return arrays, metrics

    def _qfedavg(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        local_lr = _cfg_float(
            self.config,
            "qffl_learning_rate",
            _model_float(self.config, "learning_rate", 1.0e-3),
        )
        local_lr = max(local_lr, 1.0e-12)
        q_param = max(0.0, _cfg_float(self.config, "qffl_q", 1.0))
        deltas = _deltas(updates, previous_arrays)
        losses = [_client_loss(update.metrics) for update in updates]
        dynamic_step = _cfg_bool(self.config, "qffl_dynamic_step", True)
        if not dynamic_step:
            weights = _normalize_weights([loss ** q_param for loss in losses])
            q_delta = _sum_delta(deltas, weights)
            server_lr = _cfg_float(self.config, "server_learning_rate", 1.0)
            arrays = _add_scaled(previous_arrays, q_delta, server_lr)
            metrics = self._base_metrics(updates)
            metrics.update(
                {
                    "server_lr": float(server_lr),
                    "qffl_q": float(q_param),
                    "qffl_dynamic_step": int(0),
                    "qffl_delta_norm": float(math.sqrt(max(_squared_l2(q_delta), 0.0))),
                }
            )
            return arrays, metrics

        scaled_deltas = _zeros_like(previous_arrays)
        h_sum = 0.0
        for update, delta, loss in zip(updates, deltas, losses):
            grad_norm_sq = (
                (_metric_float(update.metrics, "delta_norm_sq", _squared_l2(delta)) or 0.0)
                / (local_lr * local_lr)
            )
            loss_q = loss ** q_param
            curvature = 0.0
            if q_param > 0.0:
                curvature = q_param * (loss ** (q_param - 1.0)) * grad_norm_sq
            h_sum += curvature + loss_q / local_lr
            for index, array in enumerate(delta):
                scaled_deltas[index] += loss_q * array / local_lr

        if h_sum <= 1.0e-12:
            return self._fedavg(updates)
        q_delta = [array / h_sum for array in scaled_deltas]

        fedavg_arrays, _ = self._fedavg(updates)
        fedavg_delta = [
            current.astype(np.float64, copy=False) - previous.astype(np.float64, copy=False)
            for current, previous in zip(fedavg_arrays, previous_arrays, strict=True)
        ]
        q_norm = math.sqrt(max(_squared_l2(q_delta), 0.0))
        avg_norm = math.sqrt(max(_squared_l2(fedavg_delta), 0.0))
        max_multiplier = max(0.0, _cfg_float(self.config, "qffl_max_delta_multiplier", 2.0))
        clipped = 0
        raw_q_norm = q_norm
        if max_multiplier > 0.0 and avg_norm > 1.0e-12 and q_norm > max_multiplier * avg_norm:
            q_delta = _scale_delta(q_delta, (max_multiplier * avg_norm) / (q_norm + 1.0e-12))
            clipped = 1
            q_norm = math.sqrt(max(_squared_l2(q_delta), 0.0))

        server_lr = _cfg_float(self.config, "server_learning_rate", 1.0)
        arrays = _add_scaled(previous_arrays, q_delta, server_lr)
        metrics = self._base_metrics(updates)
        metrics.update(
            {
                "server_lr": float(server_lr),
                "qffl_q": float(q_param),
                "qffl_dynamic_step": int(1),
                "qffl_h_sum": float(h_sum),
                "qffl_delta_norm": float(q_norm),
                "qffl_delta_norm_raw": float(raw_q_norm),
                "qffl_clipped": int(clipped),
            }
        )
        return arrays, metrics

    def _afl(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        for update in updates:
            self.afl_lambdas.setdefault(update.client_key, 1.0)
        lambda_lr = max(0.0, _cfg_float(self.config, "afl_lambda_lr", 0.1))
        for update in updates:
            self.afl_lambdas[update.client_key] += lambda_lr * _client_loss(update.metrics)

        keys = sorted(self.afl_lambdas)
        projected = _project_simplex([self.afl_lambdas[key] for key in keys])
        self.afl_lambdas = {key: weight for key, weight in zip(keys, projected)}
        current_weights = _normalize_weights(
            [self.afl_lambdas.get(update.client_key, 0.0) for update in updates]
        )
        deltas = _deltas(updates, previous_arrays)
        aggregate_delta = _sum_delta(deltas, current_weights)
        server_lr = _cfg_float(self.config, "server_learning_rate", 1.0)
        arrays = _add_scaled(previous_arrays, aggregate_delta, server_lr)
        all_lambdas = list(self.afl_lambdas.values())
        entropy = -sum(weight * math.log(max(weight, 1.0e-12)) for weight in all_lambdas)
        metrics = self._base_metrics(updates)
        metrics.update(
            {
                "server_lr": float(server_lr),
                "afl_lambda_lr": float(lambda_lr),
                "afl_known_clients": int(len(self.afl_lambdas)),
                "afl_lambda_max": float(max(all_lambdas)),
                "afl_lambda_entropy": float(entropy),
            }
        )
        return arrays, metrics

    def _maxfl(
        self,
        updates: list[ClientUpdate],
        previous_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
        losses = [_client_loss(update.metrics) for update in updates]
        default_threshold = sum(losses) / len(losses)
        threshold = _cfg_float(self.config, "maxfl_loss_threshold", default_threshold)
        if threshold <= 0.0:
            threshold = default_threshold
        temperature = max(1.0e-6, _cfg_float(self.config, "maxfl_temperature", 10.0))
        appeal_weights = [
            _sigmoid_derivative_weight(loss, threshold, temperature)
            for loss in losses
        ]
        if sum(appeal_weights) <= 1.0e-12:
            appeal_weights = [loss for loss in losses]
        deltas = _deltas(updates, previous_arrays)
        aggregate_delta = _sum_delta(deltas, _normalize_weights(appeal_weights))
        server_lr = _cfg_float(self.config, "server_learning_rate", 1.0)
        arrays = _add_scaled(previous_arrays, aggregate_delta, server_lr)
        metrics = self._base_metrics(updates)
        metrics.update(
            {
                "server_lr": float(server_lr),
                "maxfl_loss_threshold": float(threshold),
                "maxfl_temperature": float(temperature),
                "maxfl_weight_max": float(max(appeal_weights)),
            }
        )
        return arrays, metrics
