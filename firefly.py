"""Firefly Algorithm for bounded search with a supplied objective."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Brightness = tuple[float, float]
Position = NDArray[np.float64]


@dataclass(frozen=True)
class Firefly:
    position: Position
    brightness: Brightness


def search(
    objective: Callable[[Position], Brightness],
    lower: Position,
    upper: Position,
    *,
    population: int = 24,
    generations: int = 40,
    seed: int = 1,
    gamma: float = 1.0,
    alpha: float = 0.18,
    tolerance: float = 1e-6,
    patience: int | None = None,
) -> tuple[list[Firefly], list[Firefly]]:
    """Return the final swarm and the best firefly seen through each generation.

    Alpha sets the initial random step width in normalized coordinates.
    When patience is set, stop early after that many consecutive generations
    without a brightness improvement larger than tolerance.
    """
    rng = np.random.default_rng(seed)
    span = upper - lower
    dimensions = lower.size

    # Normalize coordinates so all dimensions have the same distance scale.
    swarm = [
        Firefly(position, objective(lower + span * position))
        for position in rng.random((population, dimensions))
    ]
    swarm.sort(key=lambda fly: fly.brightness, reverse=True)
    history = [swarm[0]]
    stagnant = 0

    for _ in range(generations):
        prev_best = history[-1].brightness
        for i in range(population):
            fly = swarm[i]
            for j in range(i):
                other = swarm[j]
                if other.brightness > fly.brightness:
                    direction = other.position - fly.position
                    distance_squared = np.dot(direction, direction)
                    attractiveness = np.exp(-gamma * distance_squared)
                    random_step = alpha * (rng.random(dimensions) - 0.5)
                    position = np.clip(
                        fly.position + attractiveness * direction + random_step, 0.0, 1.0
                    )
                    fly = Firefly(position, objective(lower + span * position))
                    swarm[i] = fly

        swarm.sort(key=lambda fly: fly.brightness, reverse=True)
        history.append(swarm[0])
        alpha *= 0.93

        if patience is not None:
            best = swarm[0].brightness
            improved = any(
                np.isfinite(cur) and (not np.isfinite(prev) or cur - prev > tolerance)
                for cur, prev in zip(best, prev_best)
            )
            stagnant = 0 if improved else stagnant + 1
            if stagnant >= patience:
                break

    def with_gains(fly: Firefly) -> Firefly:
        return Firefly(lower + span * fly.position, fly.brightness)

    return [with_gains(fly) for fly in swarm], [with_gains(fly) for fly in history]
