"""
IAR GOVERNOR - Universal Control System
Real-time control of ANY system using the IAR framework.

Core insight: control action J must regulate the IAR triad
(X_C -> action, X_O -> environment, R -> reaction) to keep the
system in its chosen fate (internal/joint equilibrium).

The Governor is a universal PID whose gains adapt via IAR parameters,
plus a model-predictive layer and disturbance rejection.

Author: Macharia Barii
"""

import numpy as np
from collections import deque


class IARGovernor:
    """Universal IAR-based controller.

    Control law:
        J(t) = beta * (X_C(t) - theta * X_O(t))           (IAR interaction)
        u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt   (PID core)
        u_iar(t) = G_IAR * J(t)                            (IAR correction)

    Gains adapted online to maintain the desired fate.
    """

    def __init__(self, setpoint, dt=0.01, beta=1.0, theta=0.5,
                 gamma=0.8, lam=0.1, eta=0.05, alpha=0.5,
                 Kp=1.0, Ki=0.1, Kd=0.01, saturation=10.0):
        self.setpoint = setpoint
        self.dt = dt
        self.saturation = saturation

        # IAR parameters (adaptive)
        self.theta_p = {"beta": beta, "theta": theta, "gamma": gamma,
                        "lam": lam, "eta": eta, "alpha": alpha}

        # PID gains
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd

        # IAR state
        self.X_C = setpoint      # action (controlled variable)
        self.X_O = setpoint      # open state (environment)
        self.R = 0.0             # reaction potential

        # PID state
        self.err_prev = 0.0
        self.integral = 0.0

        # History
        self.history = {"t": [], "meas": [], "ref": [], "u": [], "fate": []}
        self.t = 0.0

    # ------------------------------------------------------------
    # CORE CONTROL STEP
    # ------------------------------------------------------------

    def step(self, measurement, disturbance=0.0):
        """Compute control action from a measurement."""
        # Error signal
        err = self.setpoint - measurement
        self.integral += err * self.dt
        deriv = (err - self.err_prev) / max(self.dt, 1e-9)
        self.err_prev = err

        # PID output
        u_pid = (self.Kp * err + self.Ki * self.integral + self.Kd * deriv)

        # IAR interaction
        J = self.theta_p["beta"] * (self.X_C - self.theta_p["theta"] * self.X_O)

        # Learning-based disturbance estimate
        est_disturbance = self._estimate_disturbance(disturbance)

        # IAR correction
        u_iar = self.theta_p["gamma"] * J - self.theta_p["lam"] * self.R
        u = np.clip(u_pid + u_iar - est_disturbance,
                    -self.saturation, self.saturation)

        # Update IAR state
        self.X_C = measurement
        self.X_O = 0.95 * self.X_O + 0.05 * measurement
        self.R += (self.theta_p["gamma"] * J - self.theta_p["lam"] * self.R
                   - self.theta_p["eta"] * self.X_O * self.R) * self.dt

        # Adapt gains
        self._adapt_parameters(err, measurement)

        fate = self._classify_fate()
        self.history["t"].append(self.t)
        self.history["meas"].append(measurement)
        self.history["ref"].append(self.setpoint)
        self.history["u"].append(u)
        self.history["fate"].append(fate)
        self.t += self.dt

        return u

    # ------------------------------------------------------------
    # CONTROL MODES
    # ------------------------------------------------------------

    def step_with_model(self, measurement, model_fn, horizon=5):
        """Model-predictive layer: preview future to refine control."""
        u_now = self.step(measurement)
        # Simulate future under current action
        future = []
        m = measurement
        for _ in range(horizon):
            m = model_fn(m, u_now)
            future.append(m)
        # Correction toward setpoint
        correction = (self.setpoint - np.mean(future)) * 0.1
        u = np.clip(u_now + correction, -self.saturation, self.saturation)
        self.history["u"][-1] = u
        return u

    def track_trajectory(self, trajectory_fn, measurement):
        """Track a time-varying reference trajectory."""
        self.setpoint = trajectory_fn(self.t)
        return self.step(measurement)

    def switch_fate(self, target_fate):
        """Actively switch the system between fates."""
        adjustments = {
            "Internal Equilibrium": {"beta": 0.5, "gamma": 0.5, "lam": 1.0},
            "Joint Equilibrium": {"beta": 0.8, "gamma": 0.7, "lam": 0.7},
            "C Dominates": {"beta": 2.0, "gamma": 1.5, "lam": 0.1, "theta": 0.2},
            "O Dominates": {"beta": 0.2, "gamma": 0.3, "lam": 0.5, "theta": 2.0},
            "Dissolution": {"beta": 2.0, "gamma": 2.0, "lam": 0.1, "eta": 0.5},
        }
        for key, val in adjustments[target_fate].items():
            self.theta_p[key] = val

    # ------------------------------------------------------------
    # LEARNING / ADAPTATION
    # ------------------------------------------------------------

    def _estimate_disturbance(self, measured):
        """Simple disturbance estimator: model a0 -> polynomial fit."""
        if len(self.history["meas"]) >= 3:
            ys = self.history["meas"][-3:]
            ts = self.history["t"][-3:]
            coeffs = np.polyfit(ts, ys, 1)
            return coeffs[0] * self.dt
        return 0.0

    def _adapt_parameters(self, err, measurement):
        """Adapt IAR parameters online to hold the chosen fate."""
        # If error is large, we're losing the reaction term -> adjust
        if abs(err) > 0.1 * abs(self.setpoint + 1e-8):
            self.theta_p["gamma"] += 0.01 * np.sign(err) * abs(err)
            self.theta_p["lam"] -= 0.001 * abs(err)
        else:
            # Converging: damping to reach stability
            self.theta_p["lam"] += 0.001 * (1 - abs(err) / (abs(self.setpoint) + 1e-8))
        # Clamp
        for k, v in self.theta_p.items():
            self.theta_p[k] = np.clip(v, 0.01, 10.0)

    def _classify_fate(self):
        beta, gamma, lam = (self.theta_p["beta"], self.theta_p["gamma"],
                            self.theta_p["lam"])
        bg = beta * gamma
        alpha = self.theta_p["alpha"]
        if bg < lam * 0.5:
            return "Internal Equilibrium"
        elif bg < lam:
            return "Joint Equilibrium"
        elif alpha > 0.6:
            return "C Dominates"
        elif alpha < 0.4:
            return "O Dominates"
        elif self.theta_p["eta"] > 0.3:
            return "Dissolution"
        return "Transitional"

    def report(self):
        import numpy as np
        meas = np.array(self.history["meas"])
        ref = np.array(self.history["ref"])
        u = np.array(self.history["u"])
        print("=" * 60)
        print("IAR GOVERNOR PERFORMANCE REPORT")
        print("=" * 60)
        print(f" Steps: {len(meas)}")
        print(f" Final IAR parameters:")
        for k, v in self.theta_p.items():
            print(f"   {k} = {v:.4f}")
        print(f" Final fate: {self._classify_fate()}")
        if len(meas) > 0:
            errs = ref - meas
            print(f"\n Mean error: {np.mean(np.abs(errs)):.4f}")
            print(f" RMS error: {np.mean(errs**2)**0.5:.4f}")
            print(f" Max control: {np.max(np.abs(u)):.4f}")
            print(f" Control count: {np.count_nonzero(u)}")
            overshoot = np.max(np.abs(meas)) / (abs(np.max(ref)) + 1e-8) - 1
            print(f" Overshoot: {overshoot:.1%}")
        print("=" * 60)


# ============================================================
# UNIVERSAL CONTROL GENERATOR
# ============================================================

class IARUniversalController:
    """Turnkey controller for arbitrary plants.

    Modes:
      'thermal'   - temperature control
      'chemical'  - concentration/process control
      'mechanical'- position/velocity control
      'electrical'- voltage/current control
      'financial' - market exposure control
      'medical'   - physiological control (glucose, BP)
    """

    MODES = {
        "thermal": {"setpoint": 300, "Kp": 2.0, "Ki": 0.1, "Kd": 0.05,
                    "plant": lambda T, u: T + (400 - T) * 0.01 * u + 2 * np.random.randn()},
        "chemical": {"setpoint": 1.0, "Kp": 1.0, "Ki": 0.05, "Kd": 0.01,
                     "plant": lambda C, u: C + (1.5 - C) * 0.01 * u + 0.02 * np.random.randn()},
        "mechanical": {"setpoint": 5.0, "Kp": 3.0, "Ki": 0.2, "Kd": 0.1,
                       "plant": lambda x, u: x + u * 0.1 + 0.01 * np.random.randn()},
        "electrical": {"setpoint": 220.0, "Kp": 1.5, "Ki": 0.1, "Kd": 0.02,
                       "plant": lambda V, u: V + (240 - V) * 0.005 * u + 1 * np.random.randn()},
        "financial": {"setpoint": 0.05, "Kp": 2.0, "Ki": 0.5, "Kd": 0.1,
                      "plant": lambda r, u: r + u * 0.01 + 0.02 * np.random.randn()},
        "medical": {"setpoint": 5.5, "Kp": 0.5, "Ki": 0.02, "Kd": 0.05,
                    "plant": lambda g, u: g + (6.0 - g) * 0.01 * u + 0.3 * np.random.randn()},
    }

    def __init__(self, mode="thermal"):
        config = self.MODES[mode]
        self.mode = mode
        self.setpoint = config["setpoint"]
        self.plant_fn = config["plant"]
        self.governor = IARGovernor(
            setpoint=self.setpoint, Kp=config["Kp"], Ki=config["Ki"],
            Kd=config["Kd"], dt=0.01)
        self.fault_mode = False

    def run(self, steps=1000, fault_at=None, disturbance=None):
        state = self.setpoint
        for i in range(steps):
            if fault_at is not None and i == fault_at:
                self.fault_mode = True
                print(f" FAULT INJECTED at step {i}")
            if disturbance is not None and callable(disturbance):
                d = disturbance(i)
            else:
                d = 1.5 * np.random.randn() if (disturbance or self.fault_mode) else 0
            state = self.plant_fn(state, self.governor.Kp)
            u = self.governor.step(state + d)
            state = state + u * 0.001
        self.governor.report()
        return self.governor


if __name__ == "__main__":
    for mode in ["thermal", "medical", "financial"]:
        ctrl = IARUniversalController(mode)
        print(f"\n=== {mode.upper()} CONTROL ===")
        ctrl.run(steps=500)

    # Demo with fault recovery
    ctrl = IARUniversalController("thermal")
    print("\n=== THERMAL WITH FAULT RECOVERY ===")
    gov = ctrl.run(steps=800, fault_at=400)

    try:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 5))
        t = gov.history["t"]; meas = gov.history["meas"]; ref = gov.history["ref"]
        plt.plot(t, meas, label="Measurement")
        plt.plot(t, ref, "--", label="Setpoint")
        plt.xlabel("Time"); plt.ylabel("Value")
        plt.title("IAR Governor: Thermal Control with Fault Recovery")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("iar_governor.png", dpi=150)
    except ImportError:
        print("(matplotlib not installed; plot skipped)")