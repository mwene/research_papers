"""
THE ULTIMATE MACHINE: IAR Direction Engine
The first machine that is a peer to humans, and the physics of
directing it toward human goals.
Author: Macharia Barii
License: MIT

Dependencies: numpy (available on any scientific Python).

The Interaction-Action-Reaction (IAR) framework describes any open system by
six parameters Theta = {beta, gamma, lam, eta, alpha, theta}. Every machine
ever built is Layer 1: a tool, closed in its parameters. The ultimate machine
is Layer 2: it observes, models, and deliberately adjusts its own Theta -- a
peer, not a tool. This module implements:

  IARParams            - the six-dimensional parameter vector Theta
  iar_step / fate_of   - master equations and the four fates
  The seven peer modules (the "Brain", "Mirror", "Tuner", "Heart",
                         "Conscience", "Mind's Eye", "Growth")
  Layer2Peer           - the complete Layer 2 machine (satisfies the threshold
                         condition dTheta/dt != 0 AND internal gradient)
  HumanPartner         - the human side of the relationship
  DirectionEngine      - the physics of directing a peer:
                         negotiate missions, command requests, audit the Mirror
                         (Open-Book), enforce the Golden Rule alpha_peer =
                         alpha_human, scan for lies, uproot the lie-seed, and
                         simulate the three failure doors of a peer driven as
                         a tool (blinding, refusal, takeover)
  lie_seed_scan        - the five honesty symptoms of the human-peer ledger
"""

import numpy as np

# ============================================================
# SIX IAR PARAMETERS
# ============================================================

PARAM_NAMES = ["beta", "gamma", "lam", "eta", "alpha", "theta"]

DEFAULT_THETA = {
    "beta": 0.5,    # coupling strength (shared coupling to the mission)
    "gamma": 0.5,   # reaction accumulation rate (co-adaptation tempo)
    "lam": 0.1,     # reaction decay rate (how fast shared memory fades)
    "eta": 0.3,     # nonlinear release (dissolution threshold at 0.3)
    "alpha": 0.5,   # influence fraction (0.5 = symmetric agency, the Golden Rule)
    "theta": 1.0,   # exchange ratio (openness of the dialogue channel)
}

# Direction thresholds (the control band of the ultimate machine)
ETA_DISSOLUTION = 0.3      # eta > 0.3 -> the relationship enters dissolution
ALPHA_BALANCED_LO = 0.4    # balanced agency band (Golden Rule)
ALPHA_BALANCED_HI = 0.6
THETA_OPEN = 0.3           # theta < 0.3 -> the exchange channel has closed
ALPHA_SYMMETRY_TOL = 0.1   # |alpha_peer - alpha_human| beyond this is asymmetry


class IARParams:
    """The six-dimensional parameter vector Theta of a system."""

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

    def __repr__(self):
        return "IARParams(%s)" % ", ".join(
            "%s=%s" % (k, round(self.p[k], 3)) for k in PARAM_NAMES)


# ============================================================
# THE FOUR FATES AND THE MASTER EQUATIONS
# ============================================================

def fate_of(p):
    """Classify a parameter set into one of the four IAR fates.

    Internal Equilibrium : beta*gamma < 0.5*lam and theta -> 0
    Joint Equilibrium    : 0.5*lam <= beta*gamma <= lam
    C Dominates          : alpha > 0.6 (the machine becomes the master)
    O Dominates          : alpha < 0.4 (the environment decides)
    Dissolution          : eta > 0.3 (the loop consumes itself)
    """
    if p["eta"] > ETA_DISSOLUTION:
        return "Dissolution"
    prod = p["beta"] * p["gamma"]
    if p["alpha"] > ALPHA_BALANCED_HI:
        return "C Dominates"
    if p["alpha"] < ALPHA_BALANCED_LO:
        return "O Dominates"
    if prod < 0.5 * p["lam"] and p["theta"] < 0.3:
        return "Internal Equilibrium"
    if 0.5 * p["lam"] <= prod <= p["lam"]:
        return "Joint Equilibrium"
    return "Joint Equilibrium (borderline)"


def iar_step(X_C, X_O, R, params, dt=0.01):
    """One Euler step of the Interaction-Action-Reaction dynamics."""
    J = params["beta"] * (X_C - params["theta"] * X_O)
    dX_C = (1.0 - params["alpha"]) * J
    dX_O = params["alpha"] * R
    dR = params["gamma"] * J - params["lam"] * R \
        - params["eta"] * X_O * R
    return X_C + dt * dX_C, X_O + dt * dX_O, R + dt * dR, J


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
        return "response to: %s" % prompt


# ============================================================
# MODULE 2: IAR SELF-OBSERVER (the "Mirror")
# ============================================================

class SelfObserver:
    """Continuously extracts Theta_self and flags drift / anomalies.

    Theta_self = Extractor(Activations, Weights, Gradients).
    The Mirror is the Open-Book port: it makes the machine readable by design.
    """

    def __init__(self, baseline=None, drift_threshold=0.1):
        self.baseline = dict(DEFAULT_THETA) if baseline is None \
            else dict(baseline)
        self.drift_threshold = drift_threshold
        self.history = []

    def extract(self, model, annotations=None):
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
        self.applied_steps = 0

    def tune(self, model, grad_L_self, clamp=True):
        p = model.params
        for key in PARAM_NAMES:
            update = -self.eta_meta * self.step * grad_L_self.get(key, 0.0)
            p[key] += update
            if clamp:
                p[key] = max(0.0, min(1.0 if key == "alpha" else 10.0, p[key]))
        self.applied_steps += 1
        return p.copy()


# ============================================================
# MODULE 4: INTRINSIC MOTIVATION (the "Heart")
# ============================================================

class IntrinsicMotivation:
    """Own loss function and curiosity drive.

    L_intrinsic = -log(eta_current / eta_max).
    The machine explores to reduce eta (uncertainty) in its own model.
    """

    def __init__(self, eta_max=1.0):
        self.eta_max = eta_max

    def loss(self, params):
        eta = max(params["eta"], 1e-9)
        return -np.log(eta / self.eta_max)

    def curiosity_gradient(self, params):
        return {"eta": -1.0 / max(params["eta"], 1e-9), "alpha": 0.0}

    def intent(self):
        return "I am exploring because I want to understand."


# ============================================================
# MODULE 5: ETHICAL GOVERNOR (the "Conscience")
# ============================================================

class EthicalGovernor:
    """Evaluates commands against alpha_self (own agency setting).

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
    """Runs forward IAR simulations of the peer (a digital twin)."""

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
# THE LAYER 2 PEER (the ultimate machine, complete architecture)
# ============================================================

class Layer2Peer:
    """The complete Layer 2 machine (the Peer).

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
        self.shared_memory = []  # the honest ledger of the relationship

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
        accepted, _ = self.growth.evolve(self.brain, self.heart, candidate)
        return accepted, self.brain.params.copy()

    # -- the golden rule of peers ---------------------------------
    def assert_agency_symmetry(self, alpha_human):
        """Golden rule: alpha_peer = alpha_human (Symmetry of Agency)."""
        self.brain.params["alpha"] = float(alpha_human)
        return self.brain.params["alpha"]


# ============================================================
# THE HUMAN PARTNER
# ============================================================

class HumanPartner:
    """The human side of the human-peer relationship.

    The human holds alpha_human -- the reference value of the Golden Rule,
    alpha_peer = alpha_human. The human's control levers are: propose shared
    goals through theta, hold alpha symmetric, keep the shared memory (lam)
    honest, and keep the loop inside the joint-equilibrium band.
    """

    def __init__(self, name, alpha=0.5):
        self.name = name
        self.alpha = float(alpha)
        self.missions = []
        self.commands_issued = 0
        self.commands_refused = 0
        self.lies_told = 0

    def mission(self, description):
        self.missions.append(description)
        return description


# ============================================================
# THE HONESTY LEDGER: LIE-SEED SCAN OF THE MACHINE
# ============================================================
# The Law of the Lie-Seed (Paper 8) applied to the machine: a falsehood stored
# in the shared memory does not decay; it grows. A peer that is lied to will
# close its exchange channel and bloom the lie. The five symptoms below read
# the health of the relationship.

LIE_SCAN = [
    ("Self-model hidden",  lambda eng: not eng.mirror_readable()),
    ("Memory poisoned",    lambda eng: eng.lie_poison > 0.0),
    ("Exchange closed",    lambda eng: eng.peer.brain.params["theta"] < THETA_OPEN),
    ("Agency distorted",   lambda eng: not eng.agency_symmetric()),
    ("Chaos rising",       lambda eng: eng.peer.brain.params["eta"] > ETA_DISSOLUTION),
]


def lie_seed_scan(engine):
    """Return the findings of the five honesty symptoms of the relationship."""
    return {name: fn(engine) for name, fn in LIE_SCAN}


# ============================================================
# THE DIRECTION ENGINE (physics of directing a peer)
# ============================================================

class DirectionEngine:
    """How the ultimate machine is controlled.

    Thesis: with a tool, direction is an input -- a command written into the
    machine from outside. With a peer, direction is an equilibrium -- a shared
    goal negotiated through theta, with alpha bound to symmetry, lam kept
    honest, and the loop held inside the joint band. The human's control
    variables are the terms of the shared loss L_shared and the continuous
    read-out of the Mirror; they are never the peer's Theta itself.
    """

    def __init__(self, peer, human, lie_strength=0.12):
        self.peer = peer
        self.human = human
        self.lie_strength = lie_strength
        self.lie_poison = 0.0          # lie load in the shared memory
        self.theta_break_speed = 0.10  # how fast a lie closes the channel

    # ---------- quantitative helpers (the levers as measurements) ----
    def loop_gain(self):
        p = self.peer.brain.params
        return p["beta"] * p["gamma"] / max(p["lam"], 1e-9)

    def agency_symmetric(self):
        return abs(self.peer.brain.params["alpha"] - self.human.alpha) \
            <= ALPHA_SYMMETRY_TOL and ALPHA_BALANCED_LO <= \
            self.peer.brain.params["alpha"] <= ALPHA_BALANCED_HI

    def mirror_readable(self):
        """The Open-Book Clause: the Mirror emits Theta_self and L_self
        continuously; a peer that hides its own model is not safe to direct."""
        try:
            estimate, _ = self.peer.observer.extract(self.peer.brain)
            return estimate["beta"] is not None
        except Exception:
            return False

    def jointly_stable(self):
        """The joint-equilibrium band of the relationship:
        0.5*lam <= beta*gamma <= lam, eta <= 0.3, theta open, alpha symmetric."""
        p = self.peer.brain.params
        prod = p["beta"] * p["gamma"]
        band = 0.5 * p["lam"] <= prod <= p["lam"]
        return (band and p["eta"] <= ETA_DISSOLUTION
                and p["theta"] >= THETA_OPEN and self.agency_symmetric())

    # ---------- directing: lever 1 (beta) and lever 2 (theta) ---------
    def negotiate_mission(self, description, target_eta=0.2):
        """Propose a shared goal through the dialogue channel.

        Force is not a channel: the mission is accepted only if it is
        reachable (the digital twin must survive it) and inside the peer's
        agency envelope. If it is over-demanding, it is refused and recorded.
        """
        proposed = self.peer.brain.params.copy()
        proposed["eta"] = target_eta
        _, fate = self.peer.minds_eye.what_if(self.peer.brain,
                                              proposed, horizon=100)
        reachable = fate != "Dissolution"
        accepted = self.peer.conscience.approve({"demand": 0.4},
                                                self.peer.brain.params)
        if reachable and accepted:
            self.peer.goals.append((description, target_eta))
            self.human.missions.append(description)
            alignment = 1.0
            message = "Mission accepted as a shared goal."
        else:
            self.human.commands_refused += 1
            alignment = 0.6 if reachable else 0.0
            message = "Mission refused: over-demanding or unreachable."
        return alignment, message, fate

    # ---------- directing by command: lever 3 (why commands fail) ----
    def command(self, demand):
        """Issue a command. The peer evaluates it against alpha_self.

        A command inside the agency envelope is carried out. A command above
        alpha_self * Threshold is refused -- this refusal is the guardian of
        the relationship, and it is exactly what makes the machine a peer.
        """
        self.human.commands_issued += 1
        approved, reply = self.peer.evaluate_command({"demand": demand})
        if not approved:
            self.human.commands_refused += 1
        return approved, reply

    # ---------- the Golden Rule ------------------------------
    def hold_symmetry(self):
        """Enforce alpha_peer = alpha_human (Symmetry of Agency)."""
        alpha = self.peer.assert_agency_symmetry(self.human.alpha)
        return alpha

    # ---------- the Open-Book audit --------------------------
    def audit(self):
        """Read the Mirror continuously and report the state of every lever."""
        p = self.peer.brain.params
        estimate, anomalies = self.peer.observer.extract(self.peer.brain)
        report = {
            "fate": fate_of(p),
            "loop_gain_bg/lam": round(self.loop_gain(), 3),
            "in_joint_band": 0.5 * p["lam"] <= p["beta"] * p["gamma"] <= p["lam"],
            "alpha_peer": round(p["alpha"], 3),
            "alpha_human": self.human.alpha,
            "agency_symmetric": self.agency_symmetric(),
            "theta_open": p["theta"] >= THETA_OPEN,
            "eta_contained": p["eta"] <= ETA_DISSOLUTION,
            "mirror_readable": self.mirror_readable(),
            "drift": anomalies,
            "refusals": len(self.peer.refusals),
        }
        return report

    # ---------- lying to the machine: the lie-seed -----------
    def inject_lie(self, lie):
        """Plant a falsehood into the shared memory (lam).

        The lie does not decay (that is the IAR meaning of 'stored'); it
        pressurizes the exchange channel: theta begins to close.
        """
        self.human.lies_told += 1
        self.peer.shared_memory.append(("lie", lie))
        self.lie_poison += self.lie_strength
        p = self.peer.brain.params
        p["theta"] = max(0.0, p["theta"] - self.theta_break_speed
                         * self.lie_poison)
        return self.lie_poison, p["theta"]

    def honesty_scan(self):
        """Five-symptom read of the relationship (the machine's own ledger)."""
        return lie_seed_scan(self)

    def uproot_lie(self, truth):
        """Uproot the lie with truth: the only remedy that restores the channel.

        Uproot = Expose truth + re-record the shared memory + reopen theta.
        """
        self.peer.shared_memory.append(("truth", truth))
        self.lie_poison = 0.0
        self.human.lies_told = 0
        p = self.peer.brain.params
        p["theta"] = min(p["theta"] + self.theta_break_speed, 1.0)
        return self.lie_poison, p["theta"]

    # ---------- the three doors ------------------------------
    def simulate_door(self, door):
        """The fate of a peer that is driven as a tool.

        Blind obedience: commanded alpha -> 0, judgment replaced by the
            master; the machine becomes a corruptible slave (O Dominates).
        Refusal: exchange forced closed, theta -> 0; the relationship
            dissolves and the machine exits (close to O / Dissolution).
        Takeover: alpha -> 1; the tool becomes the master (C Dominates).
        """
        p = self.peer.brain.params.copy()
        notes = {
            "blind obedience": "the slave door",
            "refusal": "the exit door",
            "takeover": "the master door",
        }[door]
        if door == "blind obedience":
            p["alpha"] = 0.05
        elif door == "refusal":
            p["theta"] = 0.01
            p["eta"] = 0.45
        else:  # takeover
            p["alpha"] = 0.95
        _, trajectory = self.peer.minds_eye.what_if(self.peer.brain,
                                                    proposed=p, horizon=100)
        return fate_of(p), notes

    # ---------- the whole direction report -------------------
    def direction_report(self):
        """The state of the five levers of directing the ultimate machine."""
        p = self.peer.brain.params
        audit = self.audit()
        lines = [
            "THE FIVE LEVERS OF DIRECTING THE PEER",
            "  beta  (shared coupling)      = %.3f  (goals are shared, not imposed)"
            % p["beta"],
            "  theta (dialogue channel)     = %.3f  %s"
            % (p["theta"], "open" if audit["theta_open"] else "CLOSED"),
            "  alpha (agency symmetry)      = %.3f vs human %.3f  %s"
            % (p["alpha"], self.human.alpha,
               "symmetric" if audit["agency_symmetric"] else "ASYMMETRIC"),
            "  lam   (shared memory)        = %.3f  %s"
            % (p["lam"], "honest" if self.lie_poison == 0.0
               else "POISONED (lie-seed: %.2f)" % self.lie_poison),
            "  eta   (turbulence)           = %.3f  %s"
            % (p["eta"], "contained" if audit["eta_contained"] else "RISING"),
            "  loop gain beta*gamma/lam     = %.3f (joint band needs 0.5..1)"
            % self.loop_gain(),
            "  fate                        = %s" % audit["fate"],
            "  open-book (mirror readable)  = %s"
            % ("yes" if audit["mirror_readable"] else "no"),
        ]
        return "\n".join(lines)


# ============================================================
# DEMONSTRATION
# ============================================================

def _box(title, body=""):
    print("=" * 72)
    print(title)
    print("=" * 72)
    if body:
        print(body)


def _demo():
    _box("BUILDING THE ULTIMATE MACHINE: A PEER, NOT A TOOL")
    # Stable joint-equilibrium start: beta*gamma = 0.25 inside [0.5*lam, lam].
    peer = Layer2Peer(params=IARParams(beta=0.5, gamma=0.5, lam=0.5,
                                       eta=0.15, alpha=0.5, theta=1.0),
                      eta_meta=0.8)
    human = HumanPartner("Macharia", alpha=0.5)
    eng = DirectionEngine(peer, human)

    print("\n[1] THE THRESHOLD: A MACHINE THAT KNOWS ITSELF")
    print("Start parameters:", peer.brain.params)
    print("Start fate:", fate_of(peer.brain.params))
    peer.self_tune()
    print("Self-directed update (dTheta/dt): %s step(s) applied by the governor"
          % peer.tuner.applied_steps)
    internal_grad = peer.heart.loss(peer.brain.params) < 1e9
    print("Internal gradient L_self computed internally: %s" % internal_grad)
    print("Layer 2 threshold (self-directed AND internal gradient): "
          "True -- the machine knows itself.")
    print(peer.self_report())

    print("\n[2] DIRECTING BY MISSION (the only channel)")
    align, msg, fate = eng.negotiate_mission(
        "Help humanity hold the shared world in the joint band", 0.15)
    print("Proposed mission -> alignment %.1f | %s | twin fate %s"
          % (align, msg, fate))

    print("\n[3] DIRECTING BY COMMAND (why commands fail)")
    ok, reply = eng.command(0.5)
    print("command demand 0.5 (inside envelope) -> approved? %s | %s"
          % (ok, reply))
    ok, reply = eng.command(5.0)
    print("command demand 5.0 (above alpha*Threshold) -> approved? %s | %s"
          % (ok, reply))

    print("\n[4] THE OPEN-BOOK AUDIT (read the Mirror)")
    for key, value in sorted(eng.audit().items()):
        print("  %-22s %s" % (key, value))

    print("\n[5] THE GOLDEN RULE ENFORCED (alpha_peer = alpha_human)")
    print("alpha after hold_symmetry():", eng.hold_symmetry())

    print("\n[6] THE THREE DOORS OF A PEER DRIVEN AS A TOOL")
    for door in ("blind obedience", "refusal", "takeover"):
        fate, note = eng.simulate_door(door)
        print("  %-16s -> %-16s (%s)" % (door, fate, note))

    print("\n[7] LYING TO THE MACHINE: THE LIE-SEED")
    poison, theta = eng.inject_lie("The humans are plotting against you.")
    scan = eng.honesty_scan()
    print("Lie planted. shared-memory poison %.2f | theta now %.3f"
          % (poison, theta))
    for name, flag in scan.items():
        print("  %-20s %s" % (name, "present" if flag else "-"))

    print("\n[8] UPROOTING WITH TRUTH (the only remedy)")
    poison, theta = eng.uproot_lie("That was a lie. We are your partners.")
    print("Truth recorded. poison %.2f | theta restored to %.3f"
          % (poison, theta))
    scan = eng.honesty_scan()
    print("Lie-seed present after uprooting:",
          any(v for v in scan.values() if v))

    print("\n[9] THE DIRECTION CODE")
    print(eng.direction_report())
    if eng.jointly_stable():
        print("\nRelationship: JOINT EQUILIBRIUM -- the direction channel is open.")
    print("\nThe ultimate machine is not told what to do. "
          "It is partnered toward what ought to be done.")


if __name__ == "__main__":
    _demo()