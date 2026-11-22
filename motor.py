"""Setpoint tracking and load rejection for a first-order, linear motor torque model."""

from dataclasses import dataclass

import control as ct
import numpy as np
from numpy.typing import NDArray


SETPOINT_NM = 5.0
MOTOR_TIME_CONSTANT_S = 1.0
LOAD_FORCE_N = 20.0
LEVER_ARM_M = 0.10
SETTLING_THRESHOLD = 0.02
SETTLING_BAND_NM = SETTLING_THRESHOLD * SETPOINT_NM

MOTOR = ct.tf([1.0], [MOTOR_TIME_CONSTANT_S, 1.0])


@dataclass(frozen=True)
class EvaluationResponse:
    time_s: NDArray[np.float64]
    torque_nm: NDArray[np.float64]
    step_settling_s: float
    disturbance_time_s: float
    load_settling_s: float


def load_rejection_system(kp: float, ki: float) -> ct.TransferFunction | None:
    """Return a transfer function whose unit step is net torque after the load hits."""
    # Characteristic polynomial is tau*s^2 + (1 + kp)*s + ki; Routh-Hurwitz requires
    # both (1 + kp) > 0 and ki > 0 (with tau > 0) for stability and zero steady-state error.
    if ki <= 0.0 or 1.0 + kp <= 0.0:
        return None

    tau = MOTOR_TIME_CONSTANT_S
    r = SETPOINT_NM
    d = LOAD_FORCE_N * LEVER_ARM_M
    # Closed form of r - d * S(s), where S(s) = 1 / (1 + MOTOR * pi):
    #   ((r - d)*tau*s^2 + (r*(1 + kp) - d)*s + r*ki) / (tau*s^2 + (1 + kp)*s + ki)
    return ct.tf([(r - d) * tau, r * (1.0 + kp) - d, r * ki], [tau, 1.0 + kp, ki])


def step_metrics(system: ct.TransferFunction) -> tuple[float, float]:
    """Evaluate settling time (s) and overshoot (%) with python-control's step_info."""
    try:
        info = ct.step_info(system, SettlingTimeThreshold=SETTLING_THRESHOLD)
    except (IndexError, ValueError):
        return np.inf, np.inf

    settling_s = float(info["SettlingTime"])
    overshoot_pct = float(info["Overshoot"])
    if not (np.isfinite(settling_s) and np.isfinite(overshoot_pct)):
        return np.inf, np.inf
    return settling_s, overshoot_pct


def evaluation_response(kp: float, ki: float) -> EvaluationResponse | None:
    """Simulate a setpoint step from 0 to SETPOINT_NM, then apply the opposing load."""
    load_sys = load_rejection_system(kp, ki)
    if load_sys is None:
        return None

    pi = ct.tf([kp, ki], [1.0, 0.0])
    setpoint_sys = SETPOINT_NM * ct.feedback(MOTOR * pi, 1.0)
    sensitivity = ct.feedback(1.0, MOTOR * pi)

    step_settling_s, _ = step_metrics(setpoint_sys)
    load_settling_s, _ = step_metrics(load_sys)
    if not (np.isfinite(step_settling_s) and np.isfinite(load_settling_s)):
        return None

    disturbance_time_s = 2.5 * step_settling_s
    post_load_s = 2.5 * load_settling_s

    dt = min(disturbance_time_s, post_load_s) / 400.0
    n_before = max(1, int(np.round(disturbance_time_s / dt)))
    n_after = max(1, int(np.round(post_load_s / dt)))
    if n_before + n_after > 20_000:
        dt = (disturbance_time_s + post_load_s) / 4_000.0
        n_before = max(1, int(np.round(disturbance_time_s / dt)))
        n_after = max(1, int(np.round(post_load_s / dt)))

    disturbance_time_s = n_before * dt
    t_all = dt * np.arange(n_before + n_after + 1)
    t_after = dt * np.arange(n_after + 1)

    y_step = np.asarray(ct.step_response(setpoint_sys, t_all).outputs)
    y_dist = np.asarray(ct.step_response(sensitivity, t_after).outputs)
    load_torque_nm = LOAD_FORCE_N * LEVER_ARM_M

    time_s = np.concatenate([t_all[: n_before + 1], t_all[n_before:]])
    torque_nm = np.concatenate(
        [y_step[: n_before + 1], y_step[n_before:] - load_torque_nm * y_dist]
    )
    return EvaluationResponse(
        time_s=time_s,
        torque_nm=torque_nm,
        step_settling_s=step_settling_s,
        disturbance_time_s=disturbance_time_s,
        load_settling_s=load_settling_s,
    )


def objective(gains: NDArray[np.float64]) -> tuple[float, float]:
    """Maximize brightness using settling time first and overshoot second."""
    load_sys = load_rejection_system(*gains)
    if load_sys is None:
        return -np.inf, -np.inf

    settling_s, overshoot_pct = step_metrics(load_sys)
    if not np.isfinite(settling_s):
        return -np.inf, -np.inf

    return -settling_s, -overshoot_pct
