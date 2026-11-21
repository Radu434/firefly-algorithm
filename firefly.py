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
) -> tuple[list[Firefly], list[Firefly]]:
    """Return the final swarm and the best firefly seen through each generation.

    Alpha sets the initial random step width in normalized coordinates.
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

    for _ in range(generations):
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

    def with_gains(fly: Firefly) -> Firefly:
        return Firefly(lower + span * fly.position, fly.brightness)

    return [with_gains(fly) for fly in swarm], [with_gains(fly) for fly in history]
