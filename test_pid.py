"""Regression tests for the PI disturbance-rejection tuning example."""

import unittest

import control as ct
import numpy as np

from firefly import search
from motor import (
    LEVER_ARM_M,
    LOAD_FORCE_N,
    MOTOR,
    SETPOINT_NM,
    evaluation_response,
    load_rejection_system,
    objective,
    step_metrics,
)


class PITest(unittest.TestCase):
    def test_closed_form_load_system_matches_feedback_sensitivity(self) -> None:
        kp, ki = 5.0, 10.0
        load_sys = load_rejection_system(kp, ki)
        assert load_sys is not None
        pi = ct.tf([kp, ki], [1.0, 0.0])
        expected = SETPOINT_NM - (LOAD_FORCE_N * LEVER_ARM_M) * ct.feedback(1.0, MOTOR * pi)
        t = np.linspace(0.0, 4.0, 200)
        np.testing.assert_allclose(
            ct.step_response(load_sys, t).outputs,
            ct.step_response(expected, t).outputs,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_opposing_load_step_from_steady_state(self) -> None:
        load_sys = load_rejection_system(5.0, 10.0)
        self.assertIsNotNone(load_sys)
        assert load_sys is not None
        response = ct.step_response(load_sys)
        torque_nm = np.asarray(response.outputs)
        self.assertAlmostEqual(torque_nm[0], SETPOINT_NM - LOAD_FORCE_N * LEVER_ARM_M)
        self.assertAlmostEqual(torque_nm[-1], SETPOINT_NM, places=2)

    def test_step_metrics_adapts_time_grid_to_controller_speed(self) -> None:
        fast_sys = load_rejection_system(120.0, 120.0)
        slow_sys = load_rejection_system(0.0, 0.1)
        assert fast_sys is not None and slow_sys is not None

        fast_settling_s, fast_overshoot = step_metrics(fast_sys)
        slow_settling_s, slow_overshoot = step_metrics(slow_sys)

        self.assertGreater(fast_settling_s, 0.0)
        self.assertLess(fast_settling_s, 0.03)
        self.assertEqual(fast_overshoot, 0.0)
        self.assertGreater(slow_settling_s, 12.0)
        self.assertTrue(np.isfinite(slow_settling_s))
        self.assertEqual(slow_overshoot, 0.0)

    def test_invalid_or_unstable_candidate_has_lowest_brightness(self) -> None:
        self.assertEqual(objective(np.array([5.0, 0.0])), (-np.inf, -np.inf))
        self.assertEqual(objective(np.array([-5.0, 1.0])), (-np.inf, -np.inf))

    def test_evaluation_response_steps_to_setpoint_then_applies_disturbance(self) -> None:
        ev = evaluation_response(5.0, 10.0)
        self.assertIsNotNone(ev)
        assert ev is not None

        self.assertAlmostEqual(ev.time_s[0], 0.0)
        self.assertAlmostEqual(ev.torque_nm[0], 0.0)
        self.assertGreater(ev.disturbance_time_s, ev.step_settling_s)

        # At disturbance_time_s the response has settled near SETPOINT_NM, then
        # drops immediately by LOAD_FORCE_N * LEVER_ARM_M when the load hits.
        dist_indices = np.flatnonzero(ev.time_s == ev.disturbance_time_s)
        self.assertEqual(dist_indices.size, 2)
        before_idx, after_idx = int(dist_indices[0]), int(dist_indices[1])
        self.assertAlmostEqual(ev.torque_nm[before_idx], SETPOINT_NM, delta=0.1)
        self.assertAlmostEqual(
            ev.torque_nm[before_idx] - ev.torque_nm[after_idx],
            LOAD_FORCE_N * LEVER_ARM_M,
            places=6,
        )
        self.assertAlmostEqual(ev.torque_nm[-1], SETPOINT_NM, delta=0.1)

    def test_search_ranks_settling_time_before_overshoot(self) -> None:
        def brightness(gains: np.ndarray) -> tuple[float, float]:
            kp, ki = gains
            return (-(1.0 if kp > 0.5 else 2.0), -(50.0 + ki if kp > 0.5 else 0.0))

        lower, upper = np.zeros(2), np.ones(2)
        initial_points = np.random.default_rng(1).random((24, 2))
        _, history = search(brightness, lower, upper, generations=2)
        self.assertEqual(history[0].brightness, max(map(brightness, initial_points)))
        self.assertEqual(history[-1].brightness[0], -1.0)
        self.assertGreaterEqual(history[-1].brightness, history[0].brightness)

    def test_search_compares_all_peers_using_current_brightness(self) -> None:
        # The second firefly dims after moving toward the first, then moves
        # toward the third. The third later sees the second's improved score.
        scores = iter([3.0, 2.0, 1.0, 3.0, 0.0, 5.0, 4.0, 6.0])
        evaluations = []

        def brightness(gains: np.ndarray) -> tuple[float, float]:
            evaluations.append(gains)
            return next(scores), 0.0

        swarm, history = search(
            brightness, np.zeros(2), np.ones(2),
            population=3, generations=1, gamma=0.0, alpha=0.0,
        )
        self.assertEqual(len(evaluations), 8)
        np.testing.assert_allclose(
            evaluations[3:], [evaluations[index] for index in (0, 0, 2, 0, 2)]
        )
        self.assertEqual([fly.brightness[0] for fly in swarm], [6.0, 5.0, 3.0])
        np.testing.assert_array_equal(history[-1].position, evaluations[-1])

    def test_search_random_walk_uses_fixed_alpha_and_preserves_best(self) -> None:
        scores = iter([4.0, 3.0, 2.0, 1.0])
        alpha = 0.4
        rng = np.random.default_rng(1)
        initial_position = rng.random(1)
        expected_position = initial_position
        for _ in range(3):
            expected_position = np.clip(
                expected_position + alpha * (rng.random(1) - 0.5), 0.0, 1.0
            )

        swarm, history = search(
            lambda _: (next(scores), 0.0), np.zeros(1), np.ones(1),
            population=1, generations=3, alpha=alpha,
        )

        np.testing.assert_allclose(swarm[0].position, expected_position)
        self.assertEqual(swarm[0].brightness, (1.0, 0.0))
        self.assertEqual([fly.brightness for fly in history], [(4.0, 0.0)] * 4)
        np.testing.assert_array_equal(history[-1].position, initial_position)

    def test_search_stops_early_when_stagnant(self) -> None:
        _, history = search(
            lambda _: (-1.0, 0.0),
            np.zeros(2),
            np.ones(2),
            population=5,
            generations=50,
            patience=3,
            tolerance=1e-6,
        )
        # Initial best + 3 stagnant generations = 4 entries.
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
