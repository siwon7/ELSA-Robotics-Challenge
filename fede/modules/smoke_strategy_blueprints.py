"""Smoke checks for dependency-light FL strategy helpers."""

from __future__ import annotations

from strategy_blueprints import (
    ClientSignal,
    afl_update_lambdas,
    fedexp_server_lr,
    fednova_effective_tau,
    maxfl_appeal_weight,
    maxfl_server_lr,
    normalize_weights,
    power_of_choice_candidates,
    qffl_h,
    qffl_server_scale,
    qffl_weight,
    simplex_project,
    squared_l2_norm,
)


def _approx_equal(left: float, right: float, tol: float = 1.0e-9) -> None:
    if abs(left - right) > tol:
        raise AssertionError(f"{left} != {right}")


def main() -> None:
    weights = normalize_weights([2.0, 1.0, 1.0])
    _approx_equal(sum(weights), 1.0)
    assert weights[0] == 0.5

    projected = simplex_project([1.5, -0.5, 0.2])
    _approx_equal(sum(projected), 1.0)
    assert min(projected) >= 0.0

    lambdas = afl_update_lambdas([0.5, 0.5], [0.1, 2.0], lambda_lr=0.2)
    _approx_equal(sum(lambdas), 1.0)
    assert lambdas[1] > lambdas[0]

    h_values = [
        qffl_h(0.5, 1.0, grad_norm_sq=4.0, base_lr=0.1),
        qffl_h(1.5, 1.0, grad_norm_sq=2.0, base_lr=0.1),
    ]
    assert qffl_weight(2.0, 2.0) == 4.0
    assert qffl_server_scale(h_values) > 0.0

    appeal = [
        maxfl_appeal_weight(loss=0.9, threshold=1.0),
        maxfl_appeal_weight(loss=2.0, threshold=1.0),
    ]
    assert appeal[0] > appeal[1]
    assert maxfl_server_lr(appeal, base_lr=1.0) > 0.0

    tau_eff = fednova_effective_tau([2, 4], [1.0, 3.0])
    _approx_equal(tau_eff, 3.5)

    lr = fedexp_server_lr(average_delta_norm=1.0, client_delta_norms=[2.0, 4.0])
    assert 1.0 <= lr <= 3.0

    norm = squared_l2_norm({"b": [3.0], "a": [1.0, 2.0]})
    _approx_equal(norm, 14.0)

    clients = [
        ClientSignal("a", 10, train_loss=0.5),
        ClientSignal("b", 10, train_loss=1.5),
        ClientSignal("c", 10, train_loss=0.1),
    ]
    selected = power_of_choice_candidates(clients, d=2)
    assert [client.client_id for client in selected] == ["b", "a"]


if __name__ == "__main__":
    main()
