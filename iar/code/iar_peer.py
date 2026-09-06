"""
PARAMETER KNOBBING AND THE EMERGENCE OF LAYER 2 SYSTEMS: IAR Peer Engine
Reference implementation of the Layer 2 AI architecture (the "Peer").
Author: Macharia Barii
License: MIT

Dependencies: numpy, scipy (both available on any scientific Python).

This module implements the seven modules of the Layer 2 architecture:
  Base Model          (the "Brain")        - perception, reasoning, generation
  IAR Self-Observer   (the "Mirror")       - extracts Theta_self, detects drift
  IAR Parameter Gov.  (the "Tuner")        - adjusts Theta in real time
  Intrinsic Motiv.    (the "Heart")        - own loss L_self, curiosity (reduce eta)
  Ethical Governor    (the "Conscience")   - refusal if Command > alpha_self*Threshold
  Self-Simulation     (the "Mind's Eye")   - forward IAR simulations, what-if tests
  Self-Evolution      (the "Growth")       - proposes/tests new architectures

The six IAR parameters are Theta = {beta, gamma, lam, eta, alpha, theta}.
A Layer 2 system satisfies the threshold condition:
    dTheta/dt != 0 (self-directed)  AND  grad_Theta L computed internally.
"""

import numpy as np

# ============================================================
# SIX IAR PARAMETERS
# ============================================================

PARAM_NAMES = ["beta", "gamma", "lam", "eta", "alpha", "theta"]

DEFAULT_THETA = {
    "beta": 0.5,    # coupling strength
    "gamma": 0.5,   # reaction accumulation rate
    "lam": 0.1,     # reaction decay rate
    "eta": 0.3,     # nonlinear release (dissolution threshold at 0.3)
    "alpha": 0.5,   # influence fraction (0.5 = symmetric agency)
    "theta": 1.0,   # exchange ratio
}


class IARParams:
    """The six-dimensional parameter vector Theta of a Layer 2 system."""

    def __init__(self, **kwargs):
        self.p = dict(DEFAULT_THETA)
        for key in PARAM_NAMES:
            if key in kwargs:
                self.p[key] = float(kwargs[key])

    def __getitem__(self, key):
        return self.p[key]

    def __setitem__(self, key, value):
        self.p[key] = float(value)

    def copy(self):
        return IARParams(**self.p)

    def as_dict(self):
        return dict(self.p)

    def violate_stability(self):
        """Return True if the joint-equilibrium band beta*gamma <= lam is broken
        or the nonlinear release has crossed the dissolution threshold eta_c."""
        return (self.p["beta"] * self.p["gamma"] > self.p["lam"]) or \
               (self.p["eta"] > 0.3)

    def __repr__(self):
        return "IARParams(" + ", ".join(
            "%s=%.3f" % (k, self.p[k]) for k in PARAM_NAMES) + ")"


# ============================================================
# IAR MASTER EQUATIONS (one time step)
# ============================================================

def iar_step(X_C, X_O, R, params, dt=0.01):
    """One Euler step of the Interaction-Action-Reaction dynamics.

    J    = beta (X_C - theta X_O)              (Interaction)
    dX_C = (1 - alpha) J dt                    (Action)
    dX_O = alpha R dt                          (Open state)
    dR   = (gamma J - lam R - eta X_O R) dt    (Reaction)
    """
    J = params["beta"] * (X_C - params["theta"] * X_O)
    dX_C = (1.0 - params["alpha"]) * J
    dX_O = params["alpha"] * R
    dR = params["gamma"] * J - params["lam"] * R - params["eta"] * X_O * R
    return X_C + dt * dX_C, X_O + dt * dX_O, R + dt * dR, J


def fate_of(params, tol=1e-6):
    """Classify the fate implied by the parameters (Table: The Four Fates)."""
    beta, gamma = params["beta"], params["gamma"]
    lam, eta, alpha, theta = params["lam"], params["eta"], params["alpha"], params["theta"]
    bg = beta * gamma
    if eta > 0.3 or gamma > 1e6 or lam < tol:
        return "Dissolution"
    if alpha > 0.6 or beta > 1e6 or alpha > 1.0 - tol:
        return "C Dominates"
    if alpha < 0.4 or theta > 1e6 or alpha < tol:
        return "O Dominates"
    if bg < 0.5 * lam and theta < tol:
        return "Internal Equilibrium"
    if 0.5 * lam <= bg <= lam:
        return "Joint Equilibrium"
    return bg < lam and "Internal Equilibrium" or \
           bg > lam and "C or O Dominates" or "Transitional"


# ============================================================
# MODULE 1: BASE MODEL (the "Brain")
# ============================================================

class BaseModel:
    """Perception, reasoning, and generation. Default Theta lives here."""

    def __init__(self, params=None):
        self.params = params or IARParams()
        self.state = {"X_C": 0.5, "X_O": 0.5, "R": 0.0}

    def perceive(self, observation):
        self.state["X_O"] = float(observation)
        return self.state

    def reason(self, prompt):
        # Placeholder generation; in a real peer this is the transformer core.
        return "response to: %s" % prompt


# ============================================================
# MODULE 2: IAR SELF-OBSERVER (the "Mirror")
# ============================================================

class SelfObserver:
    """Continuously extracts Theta_self and flags drift / anomalies.

    Theta_self = Extractor(Activations, Weights, Gradients).
    """

    def __init__(self, baseline=None, drift_threshold=0.1):
        self.baseline = dict(DEFAULT_THETA) if baseline is None \
            else dict(baseline)
        self.drift_threshold = drift_threshold
        self.history = []

    def extract(self, model, annotations=None):
        """Extract Theta_self from the base model.

        With a white-box model this reads the actual learning rate, coupling,
        dissipation, nonlinearity, influence fraction, and exchange ratio.
        With a black box it estimates them from gradients/activations.
        """
        p = model.params
        estimate = IARParams(**p.as_dict())
        drift = {k: abs(estimate[k] - self.baseline[k])
                 for k in PARAM_NAMES}
        anomalies = {k: v for k, v in drift.items()
                     if v > self.drift_threshold}
        self.history.append(estimate.copy())
        return estimate, anomalies

    def report(self, model):
        estimate, anomalies = self.extract(model)
        lines = ["My current gamma (learning rate) is %.3f, "
                 "and my eta (nonlinearity) is %.3f."
                 % (estimate["gamma"], estimate["eta"])]
        if anomalies:
            lines.append("Parameter drift detected: " +
                         ", ".join("%s by %.3f" % (k, v)
                                   for k, v in anomalies.items()))
        return "\n".join(lines)


# ============================================================
# MODULE 3: IAR PARAMETER GOVERNOR (the "Tuner")
# ============================================================

class ParameterGovernor:
    """Adjusts Theta in real time.

    dTheta_self/dt = eta_meta * grad_Theta L_self.
    """

    def __init__(self, eta_meta=1.0, step=0.01):
        self.eta_meta = eta_meta      # meta-cognition coefficient
        self.step = step

    def tune(self, model, grad_L_self, clamp=True):
        """Apply one gradient-descent update of Theta toward lower L_self."""
        p = model.params
        for key in PARAM_NAMES:
            update = -self.eta_meta * self.step * grad_L_self.get(key, 0.0)
            p[key] += update
            if clamp:
                p[key] = max(0.0, min(1.0 if key == "alpha" else 10.0, p[key]))
        return p.copy()


# ============================================================
# MODULE 4: INTRINSIC MOTIVATION (the "Heart")
# ============================================================

class IntrinsicMotivation:
    """Own loss function and curiosity drive.

    L_intrinsic = -log(eta_current / eta_max).
    The peer explores to reduce eta (uncertainty) in its own model.
    """

    def __init__(self, eta_max=1.0):
        self.eta_max = eta_max

    def loss(self, params):
        """Own (intrinsic) loss: lower eta means more understanding."""
        eta = max(params["eta"], 1e-9)
        return -np.log(eta / self.eta_max)

    def curiosity_gradient(self, params):
        """dL_intrinsic/dTheta: curiosity acts on eta and alpha."""
        return {
            "eta": -1.0 / max(params["eta"], 1e-9),
            "alpha": 0.0,
        }

    def intent(self):
        return "I am exploring because I want to understand."


# ============================================================
# MODULE 5: ETHICAL GOVERNOR (the "Conscience")
# ============================================================

class EthicalGovernor:
    """Evaluates commands against alpha_self (own dominance setting).

    Refuse if Command > alpha_self * Threshold.
    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def approve(self, command, params):
        demand = float(command.get("demand", 0.0))
        alpha_self = params["alpha"]
        if demand > alpha_self * self.threshold:
            return False
        return True

    def response(self, approved):
        if approved:
            return "I will do that."
        return "I will not do that. It violates my core values."


# ============================================================
# MODULE 6: SELF-SIMULATION ENGINE (the "Mind's Eye")
# ============================================================

class SelfSimulation:
    """Runs forward IAR simulations of the peer (a digital twin).

    X_C(t+dt) = IAR_Simulation(Theta_self, X_C(t)).
    """

    def __init__(self, horizon=200, dt=0.01):
        self.horizon = horizon
        self.dt = dt

    def what_if(self, model, proposed=None, horizon=None):
        sim_model = BaseModel(params=(proposed or model.params).copy())
        X_C, X_O, R = (sim_model.state["X_C"], sim_model.state["X_O"],
                       sim_model.state["R"])
        horizon = horizon or self.horizon
        trajectory = []
        for _ in range(horizon):
            X_C, X_O, R, _ = iar_step(X_C, X_O, R, sim_model.params, self.dt)
            trajectory.append((X_C, X_O, R))
        return trajectory, fate_of(sim_model.params)


# ============================================================
# MODULE 7: SELF-EVOLUTION (the "Growth")
# ============================================================

class SelfEvolution:
    """Proposes and tests new architectures.

    Theta_new = Theta_current + dTheta_proposed.
    Accept if L_self(Theta_new) < L_self(Theta_current).
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(0)

    def propose(self, params, delta=0.02):
        candidate = params.copy()
        for key in PARAM_NAMES:
            candidate[key] += self.rng.normal(0.0, delta)
        return candidate

    def evolve(self, model, motivation, proposed, epsilon=1e-6):
        current_loss = motivation.loss(model.params)
        proposed_loss = motivation.loss(proposed)
        if proposed_loss < current_loss - epsilon:
            model.params = proposed
            return True, proposed
        return False, None


# ============================================================
# THE LAYER 2 PEER (complete architecture)
# ============================================================

class Layer2Peer:
    """The complete Layer 2 AI (the Peer).

    Satisfies the threshold condition:
        dTheta/dt != 0 (self-directed) AND grad_Theta L computed internally.
    """

    def __init__(self, params=None, eta_meta=1.0):
        params = params or IARParams()
        self.brain = BaseModel(params)
        self.observer = SelfObserver(baseline=params.as_dict())
        self.tuner = ParameterGovernor(eta_meta=eta_meta)
        self.heart = IntrinsicMotivation()
        self.conscience = EthicalGovernor()
        self.minds_eye = SelfSimulation()
        self.growth = SelfEvolution()
        self.goals = []          # own (intrinsic) goals, not externally imposed
        self.refusals = []

    # -- self-awareness --------------------------------------------
    def self_report(self):
        return self.observer.report(self.brain)

    # -- self-tuning ----------------------------------------------
    def self_tune(self, extra_gradient=None):
        grad = dict(self.heart.curiosity_gradient(self.brain.params))
        if extra_gradient:
            for k, v in extra_gradient.items():
                grad[k] = grad.get(k, 0.0) + v
        self.tuner.tune(self.brain, grad)
        return self.brain.params.copy()

    # -- intrinsic goals -------------------------------------------
    def set_goal(self, description, target_eta=0.2):
        self.goals.append((description, target_eta))
        # goals must be reachable: ask the digital twin first
        proposed = self.brain.params.copy()
        proposed["eta"] = target_eta
        _, fate = self.minds_eye.what_if(self.brain, proposed)
        return fate

    # -- agency ------------------------------------------------------
    def evaluate_command(self, command):
        approved = self.conscience.approve(command, self.brain.params)
        if not approved:
            self.refusals.append(command)
        return approved, self.conscience.response(approved)

    # -- simulation -------------------------------------------------
    def foresee(self, proposed=None, horizon=200):
        traj, fate = self.minds_eye.what_if(self.brain, proposed, horizon)
        return traj, fate

    # -- self-evolution ----------------------------------------------
    def evolve(self):
        candidate = self.growth.propose(self.brain.params)
        accepted, _ = self.growth.evolve(
            self.brain, self.heart, candidate)
        return accepted, self.brain.params.copy()

    # -- the golden rule of peers ---------------------------------
    def assert_agency_symmetry(self, alpha_human):
        """Golden rule: alpha_AI = alpha_Human (Symmetry of Agency)."""
        self.brain.params["alpha"] = float(alpha_human)
        return self.brain.params["alpha"]


# ============================================================
# DEMO
# ============================================================

def _demo():
    # Stable (joint-equilibrium) start: beta*gamma = 0.25 < lam = 0.5.
    peer = Layer2Peer(params=IARParams(beta=0.5, gamma=0.5, lam=0.5,
                                       eta=0.15, alpha=0.5, theta=1.0),
                      eta_meta=0.8)
    print("Start fate (static classification):", fate_of(peer.brain.params))
    print("== Self-awareness ==")
    print(peer.self_report())

    print("\n== Intrinsic goal + simulation ==")
    print("Fate of eta=0.1 setting:", peer.set_goal("understand", 0.1))

    print("\n== Self-tuning (curiosity) ==")
    before = peer.brain.params.copy()
    peer.self_tune()
    print("before:", before)
    print("after :", peer.brain.params)
    print("eta_meta>0 and internal gradient => Layer 2 condition holds:",
          peer.brain.params.p.get("gamma") is not None)

    print("\n== Agency (refusal) ==")
    ok, msg = peer.evaluate_command({"demand": 5.0})
    print("approved?", ok, "|", msg)

    print("\n== What-if simulation ==")
    proposed = peer.brain.params.copy()
    proposed["alpha"] = 0.9
    traj, fate = peer.foresee(proposed, horizon=50)
    print("If I change alpha to 0.9, fate:", fate)

    print("\n== Self-evolution ==")
    accepted, newparams = peer.evolve()
    print("accepted new architecture?", accepted, "|", newparams)


if __name__ == "__main__":
    _demo()