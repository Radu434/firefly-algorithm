import numpy as np
import control as ct

SETPOINT_NM = 5.0
MOTOR_TIME_CONSTANT_S = 1.0
LOAD_TORQUE_NM = 20.0 * 0.10          # 2 Nm constant disturbance
SETTLING_BAND_NM = 0.02 * SETPOINT_NM

MOTOR = ct.tf([1.0], [MOTOR_TIME_CONSTANT_S, 1.0])

wc = 20.0                              # desired closed-loop bandwidth, rad/s
T_des = ct.tf([wc], [1, wc])

# Guillemin-Truxal: C = T / (G (1 - T))
C = ct.minreal(T_des / (MOTOR * (1 - T_des)))   # = wc*(s+1)/s

Kp = wc * MOTOR_TIME_CONSTANT_S
Ki = wc
C_pi = ct.tf([Kp, Ki], [1, 0])

T_cl = ct.feedback(C_pi * MOTOR, 1)
t = np.linspace(0, 2, 2000)
t, y = ct.step_response(SETPOINT_NM * T_cl, t)

print(f"Kp={Kp}, Ki={Ki}")
print(f"overshoot = {max(0, y.max() - SETPOINT_NM):.4f} Nm")
print(f"ss error  = {SETPOINT_NM - y[-1]:.2e} Nm")

# Load disturbance at the plant input: y/d = G / (1 + G C)
Gd = ct.feedback(MOTOR, C_pi)
_, yd = ct.step_response(LOAD_TORQUE_NM * 0 + 2.0 * Gd, t)
print(f"peak load-induced deviation = {np.abs(yd).max():.3f} Nm")