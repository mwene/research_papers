"""
PRINCIPLE OF OPEN SYSTEMS: Universal IAR Engine
Open-Form Implementation (No Closed-Form ODEs)
Author: Macharia Barii
License: MIT

Dependencies: numpy, scipy (both available on any scientific Python).
Optional: matplotlib (only for the plotting demo).
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ring_laplacian(n):
    """Graph Laplacian of a periodic ring graph (numpy only)."""
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2.0
        L[i, (i - 1) % n] -= 1.0
        L[i, (i + 1) % n] -= 1.0
    return L


# ============================================================
# OPEN-FORM IAR ENGINE
# ============================================================

class OpenIAR:
    """Universal open-form Interaction -> Action -> Reaction engine.

    Key features:
    - No ODE solver; everything is step-by-step algorithmic
    - Parameters learned online via gradient-free optimization
    - Spatial diffusion on arbitrary graphs (ring by default)
    - Replaceable action and reaction functions
    - Lightweight surrogate (local polynomial) for prediction
    - Handles heavy-tailed noise (Student-t)
    - External forcing at any time step
    """

    def __init__(self, n_nodes=100, n_features=3, spatial_graph=None):
        self.n_nodes = n_nodes
        self.n_features = n_features

        # State variables (open, not fixed)
        self.X_C = np.random.rand(n_nodes) * 0.5 + 0.25
        self.X_O = np.random.rand(n_nodes) * 0.5 + 0.25
        self.R = np.zeros(n_nodes)

        # Parameters (learned online, not fixed)
        self.theta = {
            "beta": 0.5,    # coupling strength
            "theta": 1.0,   # exchange ratio
            "gamma": 0.5,   # reaction accumulation rate
            "lam": 0.1,     # reaction decay rate
            "eta": 0.3,     # reaction release rate
            "alpha": 0.5,   # influence fraction (0.5 = symmetric)
        }

        # Spatial diffusion (graph Laplacian)
        if spatial_graph is None:
            self.L = _ring_laplacian(n_nodes)   # 1D ring, periodic
        else:
            self.L = spatial_graph

        # Diffusion coefficients (can be learned)
        self.D_C = 0.01
        self.D_O = 0.05
        self.D_R = 0.001

        # Open noise parameters (heavy-tailed)
        self.sigma_C = 0.1
        self.sigma_O = 0.1
        self.sigma_R = 0.01
        self.noise_df = 3  # Student-t degrees of freedom

        # Replaceable functions
        self.action_function = self.default_action
        self.reaction_function = self.default_reaction
        self.loss_function = self.default_loss

        # Memory / history (for online learning)
        self.history = {"C": [], "O": [], "R": [], "J": [], "time": []}
        self.step_count = 0
        self._surrogate = None   # fitted polynomial surrogate
        self.trained = False

        # Application metadata
        self.app_name = "Generic"
        self.app_params = {}

    # --------------------------------------------------------
    # DEFAULT FUNCTIONS (replaceable by user)
    # --------------------------------------------------------

    def default_action(self, X_C, J):
        """C responds to interaction J."""
        return (1 - self.theta["alpha"]) * J + 0.1 * X_C * (1 - X_C)

    def default_reaction(self, J, X_O, R):
        """R accumulates from J and decays; any memory kernel may replace it."""
        return (self.theta["gamma"] * J - self.theta["lam"] * R
                - self.theta["eta"] * X_O * R)

    def default_loss(self, pred, actual):
        """Mean squared error."""
        return np.mean((pred - actual) ** 2)

    # --------------------------------------------------------
    # CORE METHODS
    # --------------------------------------------------------

    def interaction(self):
        """J(t) = beta * (X_C - theta * X_O)"""
        return self.theta["beta"] * (self.X_C - self.theta["theta"] * self.X_O)

    def step(self, external_forcing=None, observation=None, learning_rate=0.01):
        """One open-form update step.

        external_forcing : dict or None with keys 'C','O','R' (external inputs).
        observation      : dict or None with keys 'C','O','R' (for learning).
        """
        # 1. Interaction
        J = self.interaction()

        # 2. Action (C responds to J)
        action = self.action_function(self.X_C, J)

        # 3. Reaction potential
        reaction = self.reaction_function(J, self.X_O, self.R)
        self.R += reaction

        # 4. Spatial diffusion (graph Laplacian)
        diff_C = self.D_C * self.L @ self.X_C
        diff_O = self.D_O * self.L @ self.X_O
        diff_R = self.D_R * self.L @ self.R

        # 5. Open noise (heavy-tailed Student-t)
        noise_C = self.sigma_C * self.X_C * np.random.standard_t(
            df=self.noise_df, size=self.n_nodes)
        noise_O = self.sigma_O * self.X_O * np.random.standard_t(
            df=self.noise_df, size=self.n_nodes)
        noise_R = self.sigma_R * self.R * np.random.standard_t(
            df=self.noise_df, size=self.n_nodes)

        # 6. Update states
        self.X_C += action + diff_C + noise_C
        self.X_O += self.theta["alpha"] * self.R + diff_O + noise_O
        self.R += reaction + diff_R + noise_R

        # 7. External forcing (open input)
        if external_forcing is not None:
            self.X_C += external_forcing.get("C", 0)
            self.X_O += external_forcing.get("O", 0)
            self.R += external_forcing.get("R", 0)

        # 8. Enforce non-negativity (soft constraint)
        self.X_C = np.maximum(self.X_C, 0)
        self.X_O = np.maximum(self.X_O, 0)
        self.R = np.maximum(self.R, 0)

        # 9. Store history
        self.history["C"].append(self.X_C.copy())
        self.history["O"].append(self.X_O.copy())
        self.history["R"].append(self.R.copy())
        self.history["J"].append(J.copy())
        self.history["time"].append(self.step_count)
        self.step_count += 1

        # 10. Online parameter learning (if observation provided)
        if observation is not None:
            self.learn_parameters(observation, lr=learning_rate)

        return self.X_C, self.X_O, self.R, J

    def learn_parameters(self, observation, lr=0.01):
        """Online parameter update via gradient-free finite differences."""
        for param in ["beta", "theta", "gamma", "lam", "eta", "alpha"]:
            orig = self.theta[param]
            grad = 0.0
            for delta in [0.001, -0.001]:
                self.theta[param] = orig + delta
                J_new = self.interaction()
                pred_C = self.X_C + self.action_function(self.X_C, J_new)
                pred_O = self.X_O + self.theta["alpha"] * self.R
                loss = (self.loss_function(pred_C, observation.get("C", self.X_C))
                        + self.loss_function(pred_O, observation.get("O", self.X_O)))
                grad += (loss
                         - self.loss_function(self.X_C, observation.get("C", self.X_C))
                         - self.loss_function(self.X_O, observation.get("O", self.X_O))) / (2 * delta)
            self.theta[param] = orig - lr * np.clip(grad, -1, 1)

        # Update surrogate if enough data
        if len(self.history["C"]) > 20:
            self._update_surrogate()

    def _update_surrogate(self):
        """Fit a lightweight polynomial surrogate to the mean trajectory."""
        n = len(self.history["C"])
        if n < 20:
            return
        t = np.array(self.history["time"]).astype(float)
        y_C = np.array(self.history["C"]).mean(axis=1)
        y_O = np.array(self.history["O"]).mean(axis=1)
        y_R = np.array(self.history["R"]).mean(axis=1)
        try:
            pC = np.polyfit(t, y_C, deg=3)
            pO = np.polyfit(t, y_O, deg=3)
            pR = np.polyfit(t, y_R, deg=3)
            res_C = np.std(y_C - np.polyval(pC, t))
            res_O = np.std(y_O - np.polyval(pO, t))
            res_R = np.std(y_R - np.polyval(pR, t))
            self._surrogate = {"pC": pC, "pO": pO, "pR": pR,
                               "sC": max(res_C, 1e-6),
                               "sO": max(res_O, 1e-6),
                               "sR": max(res_R, 1e-6)}
            self.trained = True
        except np.linalg.LinAlgError:
            pass

    def predict_future(self, n_steps=50, return_uncertainty=True):
        """Open-form prediction using the surrogate (not ODEs)."""
        if not self.trained or self._surrogate is None:
            raise RuntimeError("Surrogate not trained. Run at least 20 steps "
                               "with observations.")
        s = self._surrogate
        ta = self.step_count
        ts = np.arange(ta, ta + n_steps)
        mean_C = np.maximum(np.polyval(s["pC"], ts), 0)
        mean_O = np.maximum(np.polyval(s["pO"], ts), 0)
        mean_R = np.maximum(np.polyval(s["pR"], ts), 0)

        if return_uncertainty:
            unc_C = np.full(n_steps, s["sC"]) * (np.arange(n_steps) + 1) / n_steps
            unc_O = np.full(n_steps, s["sO"]) * (np.arange(n_steps) + 1) / n_steps
            unc_R = np.full(n_steps, s["sR"]) * (np.arange(n_steps) + 1) / n_steps
            return (mean_C, mean_O, mean_R, unc_C, unc_O, unc_R)
        return mean_C, mean_O, mean_R

    def classify_fate(self):
        """Classify the current state into one of the four fates."""
        beta = self.theta["beta"]; gamma = self.theta["gamma"]
        lam = self.theta["lam"]; alpha = self.theta["alpha"]; eta = self.theta["eta"]
        bg = beta * gamma
        if bg < 0.5 * lam:
            return "Internal Equilibrium"
        if bg < lam:
            return "Joint Equilibrium"
        if alpha > 0.6:
            return "C Dominates (Asymmetric Dominance)"
        if alpha < 0.4:
            return "O Dominates (Asymmetric Dominance)"
        if eta > 0.3:
            return "Dissolution"
        return "Transitional"

    def get_state(self):
        """Return current open state (not a closed solution)."""
        return {
            "X_C": self.X_C.copy(),
            "X_O": self.X_O.copy(),
            "R": self.R.copy(),
            "parameters": self.theta.copy(),
            "step": self.step_count,
            "history_length": len(self.history["C"]),
            "trained": self.trained,
            "app_name": self.app_name,
            "fate": self.classify_fate(),
        }

    # --------------------------------------------------------
    # APPLICATIONS
    # --------------------------------------------------------

    APPLICATIONS = {
        "Sun-Earth":          {"init_C": 0.8, "init_O": 0.3, "beta": 0.5, "theta": 2.0, "alpha": 0.5},
        "Enzyme-Bath":        {"init_C": 0.5, "init_O": 0.5, "beta": 0.4, "theta": 0.5, "alpha": 0.5},
        "Cell-ECF":           {"init_C": 0.7, "init_O": 0.4, "beta": 0.6, "theta": 1.5, "alpha": 0.5},
        "Predator-Prey":      {"init_C": 0.3, "init_O": 0.7, "beta": 0.7, "theta": 0.8, "alpha": 0.5},
        "Pathogen-Host":      {"init_C": 0.2, "init_O": 0.8, "beta": 0.5, "theta": 0.3, "alpha": 0.5},
        "Individual-Social":  {"init_C": 0.6, "init_O": 0.4, "beta": 0.4, "theta": 0.7, "alpha": 0.5},
        "Firm-Market":        {"init_C": 0.4, "init_O": 0.6, "beta": 0.5, "theta": 0.6, "alpha": 0.5},
        "Subculture-Main":    {"init_C": 0.3, "init_O": 0.7, "beta": 0.3, "theta": 1.0, "alpha": 0.5},
        "State-Global":       {"init_C": 0.5, "init_O": 0.5, "beta": 0.4, "theta": 1.2, "alpha": 0.5},
        "Tech-Culture":       {"init_C": 0.2, "init_O": 0.3, "beta": 0.3, "theta": 0.4, "alpha": 0.5},
    }

    def apply_to_application(self, app_name, **kwargs):
        """Configure engine for a specific application domain."""
        self.app_name = app_name
        config = self.APPLICATIONS.get(app_name, {})
        if config:
            self.theta["beta"] = config.get("beta", self.theta["beta"])
            self.theta["theta"] = config.get("theta", self.theta["theta"])
            self.theta["alpha"] = config.get("alpha", self.theta["alpha"])
            self.X_C = np.full(self.n_nodes, config.get("init_C", 0.5)) + 0.05 * np.random.randn(self.n_nodes)
            self.X_O = np.full(self.n_nodes, config.get("init_O", 0.5)) + 0.05 * np.random.randn(self.n_nodes)
            self.R = np.zeros(self.n_nodes)
            self.history = {"C": [], "O": [], "R": [], "J": [], "time": []}
            self.step_count = 0
            self.trained = False
            self._surrogate = None
        return self

    def run_demo(self, n_steps=200, n_nodes=50, verbose=True):
        """Run the engine for all 10 applications with the identical engine."""
        apps = list(self.APPLICATIONS.keys())
        results = []
        for idx, app in enumerate(apps):
            iar = OpenIAR(n_nodes=n_nodes).apply_to_application(app)
            for step in range(n_steps):
                forcing = None
                if np.random.rand() < 0.02:
                    forcing = {"C": 0.1 * np.random.standard_t(df=3, size=n_nodes),
                               "O": 0.1 * np.random.standard_t(df=3, size=n_nodes)}
                obs = None
                if step > 50 and step % 10 == 0:
                    obs = {"C": iar.X_C + 0.05 * np.random.randn(n_nodes),
                           "O": iar.X_O + 0.05 * np.random.randn(n_nodes)}
                iar.step(external_forcing=forcing, observation=obs)
            state = iar.get_state()
            results.append({"app": app, "fate": state["fate"],
                            "alpha": state["parameters"]["alpha"],
                            "X_C_mean": float(state["X_C"].mean()),
                            "X_O_mean": float(state["X_O"].mean()),
                            "trained": state["trained"]})
            if verbose:
                print(f"  {app:20s} | {state['fate']:45s} | "
                      f"alpha={state['parameters']['alpha']:.2f}")
        return results

    def plot_demo(self, results_dir=None):
        """Optional matplotlib figure (skipped when matplotlib absent)."""
        if not HAS_MPL:
            print("[plot_demo] matplotlib not installed; skipping figure.")
            return None
        apps = list(self.APPLICATIONS.keys())
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for idx, app in enumerate(apps):
            iar = OpenIAR(n_nodes=50).apply_to_application(app)
            for step in range(200):
                obs = None
                if step > 50 and step % 10 == 0:
                    obs = {"C": iar.X_C + 0.05 * np.random.randn(50),
                           "O": iar.X_O + 0.05 * np.random.randn(50)}
                iar.step(observation=obs)
            state = iar.get_state()
            ax = axes[idx]
            ax.plot(state["X_C"], "b-", alpha=0.8, label="C (Action)")
            ax.plot(state["X_O"], "r-", alpha=0.8, label="O (Open)")
            ax.plot(state["R"], "g-", alpha=0.8, label="R (Reaction)")
            ax.set_title(f"{app}\n{state['fate']} (alpha={state['parameters']['alpha']:.2f})", fontsize=10)
            ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
        plt.suptitle("Open-Form IAR: All 10 Applications Run with Identical Engine", fontsize=16)
        plt.tight_layout()
        import os
        out = os.path.join(results_dir or ".",
                           "open_form_IAR_all_applications.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out


if __name__ == "__main__":
    results = OpenIAR(n_nodes=50).run_demo()
    print("\nDone: 10 applications run with the identical open-form engine.")
    OpenIAR().plot_demo()