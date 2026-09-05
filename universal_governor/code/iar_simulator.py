"""
IAR PHENOMENOLOGICAL SIMULATOR
Universal digital-twin / what-if simulation engine.

The simulator reproduces ANY phenomenon as the IAR triad
(X_C, X_O, R) and lets the user run what-if scenarios:
perturb parameters, swap fates, inject interventions,
and observe the resulting trajectory.

Author: Macharia Barii
"""

import numpy as np
from collections import namedtuple

Scenario = namedtuple("Scenario", ["name", "params", "initial", "intervention"])

DOMAINS = {
    "epidemiology": {
        "title": "Epidemiology", "sims": [
            Scenario("COVID-19", {"beta": 0.6, "theta": 0.5, "gamma": 1.2,
                                  "lam": 0.3, "eta": 0.1, "alpha": 0.6},
                     {"X_C": 0.01, "X_O": 0.001, "R": 0.0}, None),
            Scenario("Control: lock-down", {"beta": 0.6, "theta": 0.5, "gamma": 1.2,
                                            "lam": 0.3, "eta": 0.1, "alpha": 0.6},
                     {"X_C": 0.01, "X_O": 0.001, "R": 0.0},
                     {"t_start": 50, "param": "lam", "value": 1.2})],
    },
    "finance": {
        "title": "Financial Markets", "sims": [
            Scenario("S&P-500", {"beta": 0.4, "theta": 0.6, "gamma": 1.0,
                                 "lam": 0.2, "eta": 0.05, "alpha": 0.48},
                     {"X_C": 0.5, "X_O": 0.5, "R": 0.0}, None),
            Scenario("Crash intervention", {"beta": 0.4, "theta": 0.6, "gamma": 1.2,
                                            "lam": 0.2, "eta": 0.05, "alpha": 0.48},
                     {"X_C": 0.5, "X_O": 0.5, "R": 0.0},
                     {"t_start": 100, "param": "eta", "value": 0.5})],
    },
    "climate": {
        "title": "Climate", "sims": [
            Scenario("Global warming", {"beta": 0.3, "theta": 0.5, "gamma": 0.9,
                                        "lam": 0.1, "eta": 0.03, "alpha": 0.45},
                     {"X_C": 0.3, "X_O": 0.3, "R": 0.0}, None),
            Scenario("Emission cuts", {"beta": 0.3, "theta": 0.5, "gamma": 0.9,
                                       "lam": 0.1, "eta": 0.03, "alpha": 0.45},
                     {"X_C": 0.3, "X_O": 0.3, "R": 0.0},
                     {"t_start": 100, "param": "beta", "value": 0.1})],
    },
}


class IARPhenomenologicalSimulator:
    """Universal what-if simulator for any phenomenon."""

    def __init__(self, dt=0.01):
        self.dt = dt
        self.result_stack = []

    def step(self, params, state):
        X_C, X_O, R = state["X_C"], state["X_O"], state["R"]
        beta, theta, gamma, lam, eta, alpha = (params["beta"], params["theta"],
                                               params["gamma"], params["lam"],
                                               params["eta"], params["alpha"])
        J = beta * (X_C - theta * X_O)
        dX_C = (1 - alpha) * J
        dX_O = alpha * R
        dR = gamma * J - lam * R - eta * X_O * R
        return {"X_C": X_C + dX_C * self.dt,
                "X_O": X_O + dX_O * self.dt,
                "R": R + dR * self.dt}

    def run(self, scenario, T=500):
        params = dict(scenario.params)
        state = dict(scenario.initial)
        trajectory = []
        params_hist = []
        for t in range(T):
            if (scenario.intervention is not None
                    and t == scenario.intervention["t_start"]):
                params[scenario.intervention["param"]] = (
                    scenario.intervention["value"])
            state = self.step(params, state)
            trajectory.append((t, state["X_C"], state["X_O"], state["R"]))
            params_hist.append(dict(params))
        self.result_stack.append({"name": scenario.name,
                                  "params": params_hist,
                                  "traj": trajectory})
        return trajectory

    def run_what_if(self, domain, base_sim, variation_fn, horizon=500):
        """Run a baseline + variations and compare outcomes."""
        base = self.run(base_sim, horizon)
        variants = []
        for label, modified in variation_fn(base_sim.params):
            mod = Scenario(f"{base_sim.name} | {label}", modified,
                           base_sim.initial, base_sim.intervention)
            variants.append((label, self.run(mod, horizon)))
        return {"base": base, "variants": variants}

    def classify_fate(self, params):
        beta, gamma, lam, alpha, eta = (params["beta"], params["gamma"],
                                        params["lam"], params["alpha"],
                                        params["eta"])
        bg = beta * gamma
        if bg < lam * 0.5:
            return "Internal Equilibrium"
        elif bg < lam:
            return "Joint Equilibrium"
        elif alpha > 0.6:
            return "C Dominates"
        elif alpha < 0.4:
            return "O Dominates"
        elif eta > 0.3:
            return "Dissolution"
        return "Transitional"

    def measure_signals(self, traj):
        """Extract quantitative signal parameters from a trajectory."""
        t = np.array([p[0] for p in traj]); X_C = np.array([p[1] for p in traj])
        peak_idx = np.argmax(X_C)
        peak_time = t[peak_idx]
        peak_value = X_C[peak_idx]
        final = X_C[-1]
        variance = np.var(X_C)
        # Oscillation measure: zero-crossings of derivative
        diff = np.diff(X_C)
        crossings = ((diff[:-1] * diff[1:]) < 0).sum()
        return {"peak_time": peak_time, "peak_value": peak_value,
                "final": final, "variance": variance,
                "oscillations": crossings}

    def compare_scenarios(self):
        """Compare results (peak, final, variance, oscillations)."""
        summary = []
        for step, result in enumerate(self.result_stack):
            sig = self.measure_signals(result["traj"])
            summary.append({"step": step, "name": result["name"], **sig})
        return summary


def print_sim_summary(engine, domain):
    results = engine.compare_scenarios()
    print(f"\n{'='*60}\n{domain.upper()} WHAT-IF SIMULATIONS\n{'='*60}")
    for r in results:
        entry = next((res for res in engine.result_stack
                      if res["name"] == r["name"]), None)
        fate = engine.classify_fate(entry["params"][-1]) if entry else "n/a"
        print(f"\n {r['name']}:")
        print(f"   Peak: {r['peak_value']:.3f} @ t={r['peak_time']}")
        print(f"   Final: {r['final']:.3f}   Variance: {r['variance']:.4f}")
        print(f"   Oscillations: {r['oscillations']:d}   Fate: {fate}")


if __name__ == "__main__":
    engine = IARPhenomenologicalSimulator(dt=0.01)

    for domain, cfg in DOMAINS.items():
        for sim in cfg["sims"]:
            engine.run(sim, T=500)
        print_sim_summary(engine, domain)

    try:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (domain, cfg) in zip(axes, DOMAINS.items()):
            for sim in cfg["sims"]:
                res = next((r for r in engine.result_stack
                            if r["name"] == sim.name), None)
                if res:
                    traj = res["traj"]
                    ax.plot([p[0] for p in traj], [p[1] for p in traj],
                            label=sim.name)
            ax.set_title(cfg["title"])
            ax.set_xlabel("Time"); ax.set_ylabel("X_C")
            ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("iar_simulator.png", dpi=150)
    except ImportError:
        print("(matplotlib not installed; plot skipped)")

    print("\n=== ENGINEERING WHAT-IF EXAMPLE: MACHINE HEALTH ===")
    base = Scenario("Healthy motor", {"beta": 0.5, "theta": 0.5, "gamma": 0.7,
                                      "lam": 0.6, "eta": 0.05, "alpha": 0.5},
                    {"X_C": 0.5, "X_O": 0.5, "R": 0.0}, None)
    fatigue = Scenario("Progressive wear", {"beta": 1.2, "theta": 0.5, "gamma": 1.0,
                                            "lam": 0.4, "eta": 0.02, "alpha": 0.5},
                       {"X_C": 0.5, "X_O": 0.5, "R": 0.0}, None)
    overload = Scenario("Overload event", {"beta": 1.5, "theta": 0.5, "gamma": 1.2,
                                           "lam": 0.3, "eta": 0.4, "alpha": 0.5},
                        {"X_C": 0.5, "X_O": 0.5, "R": 0.0}, None)
    for s in (base, fatigue, overload):
        engine.run(s, T=500)
    print_sim_summary(engine, "Machine Health")