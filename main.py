"""Tune a motor torque PI controller with the Firefly Algorithm."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from firefly import Firefly, search
from motor import (
    EvaluationResponse,
    LEVER_ARM_M,
    LOAD_FORCE_N,
    SETPOINT_NM,
    SETTLING_BAND_NM,
    evaluation_response,
    objective,
)


LOWER = np.array([0.0, 0.1])
UPPER = np.array([120.0, 120.0])
population = 30
generations = 100
gamma = 1.0
alpha = 0.01
seed = 1


def plot_results(
    swarm: list[Firefly], history: list[Firefly], evaluation: EvaluationResponse
) -> None:
    best = history[-1]
    positions = np.array([fly.position for fly in swarm])
    recovery_times = np.array([-fly.brightness[0] for fly in swarm])
    convergence = np.array([-fly.brightness[0] for fly in history])
    convergence[~np.isfinite(convergence)] = np.nan

    fig, (space, progress, response) = plt.subplots(
        1, 3, figsize=(14, 4.5), layout="constrained"
    )
    points = space.scatter(
        *positions.T, c=np.ma.masked_invalid(recovery_times), plotnonfinite=True,
        cmap="viridis_r", edgecolors="black", linewidths=0.4,
    )
    space.scatter(*best.position, s=130, facecolors="none", edgecolors="tab:red")
    space.set(
        xlim=(LOWER[0], UPPER[0]), ylim=(LOWER[1], UPPER[1]),
        xlabel="Kp", ylabel="Ki", title="PI gain search",
    )
    fig.colorbar(points, ax=space, label="Settling time (s)")

    progress.plot(convergence)
    progress.set(
        xlabel="Generation", ylabel="Best settling time (s)", title="Convergence"
    )
    progress.grid(alpha=0.3)

    response.axhspan(
        SETPOINT_NM - SETTLING_BAND_NM, SETPOINT_NM + SETTLING_BAND_NM,
        alpha=0.15, label="±2% band",
    )
    response.plot(evaluation.time_s, evaluation.torque_nm, label="Net shaft torque")
    response.axvline(
        evaluation.step_settling_s, color="tab:green", linestyle="--", label="Step settled",
    )
    response.axvline(
        evaluation.disturbance_time_s, color="0.5", linestyle=":",
        label=f"{LOAD_FORCE_N:g} N load",
    )
    response.axvline(
        evaluation.disturbance_time_s + evaluation.load_settling_s,
        color="tab:red", linestyle="--", label="Load settled",
    )
    response.set(
        xlim=(0.0, float(evaluation.time_s[-1])),
        xlabel="Time (s)", ylabel="Torque (N·m)",
        title="Step response & disturbance rejection",
    )
    response.legend()
    plt.show()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Run without opening a plot.")
    parser.add_argument(
        "--alpha", type=float, default=alpha,
        help="Initial random step width in normalized coordinates (default: %(default)s).",
    )
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha < 0.0:
        parser.error("--alpha must be a finite, nonnegative number")

    swarm, history = search(
        objective, LOWER, UPPER,
        population=population, generations=generations, gamma=gamma, alpha=args.alpha, seed=seed,
    )
    best = history[-1]
    if not np.isfinite(best.brightness[0]):
        raise SystemExit("No candidate settled within the simulation window.")

    kp, ki = best.position
    settling_s, overshoot_pct = (-value for value in best.brightness)
    print(f"Best PI gains: Kp={kp:.6g}, Ki={ki:.6g}")
    print(f"Load step: {LOAD_FORCE_N:g} N at {LEVER_ARM_M:.2f} m")
    print(f"Settling time (±2%): {settling_s:.3f} s after the load step")
    print(f"Rebound overshoot: {overshoot_pct:.3f}%")

    if not args.no_plot:
        evaluation = evaluation_response(kp, ki)
        assert evaluation is not None
        plot_results(swarm, history, evaluation)


if __name__ == "__main__":
    main()
