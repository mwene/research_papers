"""
THE IAR ANALYSIS OF CIVILIZATIONS: Peace Engine
Reference implementation of the IAR theorems of longevity, war, and peace.
Author: Macharia Barii
License: MIT

Dependencies: numpy (available on any scientific Python).

The Interaction-Action-Reaction (IAR) framework describes any open system by
six parameters Theta = {beta, gamma, lam, eta, alpha, theta}. This module
implements the civilizational applications of the framework:

  IARParams          - the six-dimensional parameter vector Theta
  fate_of            - classification into the four IAR fates
  ConflictSystem     - two competing Theta_self systems + a shared Theta_world
  WarConditions      - the five necessary conditions of the IAR Theorem on War
  LieDetector        - signatures that reveal the foundational lie of a conflict
  LieSeed            - lifecycle: planted -> watered -> fertilized -> blooms as war
  PeaceEngine        - the five phases: ceasefire, dialogue, truth, power, coexistence
  UprootProcess      - expose truth, re-educate, re-memorialize, rebalance, re-identify
"""

import numpy as np

# ============================================================
# SIX IAR PARAMETERS
# ============================================================

PARAM_NAMES = ["beta", "gamma", "lam", "eta", "alpha", "theta"]

DEFAULT_THETA = {
    "beta": 0.5,    # coupling strength (to the environment / the land)
    "gamma": 0.5,   # reaction accumulation rate (speed of change)
    "lam": 0.1,     # e-folding reaction decay rate (memory / forgetting)
    "eta": 0.3,     # nonlinear release (dissolution threshold at 0.3)
    "alpha": 0.5,   # influence fraction (0.5 = symmetric, balanced power)
    "theta": 1.0,   # exchange ratio (openness to the environment)
}

# Theorems of the civilizations paper
ETA_DISSOLUTION = 0.3      # eta > 0.3 -> chaos / dissolution regime
ALPHA_BALANCED_LO = 0.4    # balanced power band (Egypt: 0.4-0.6)
ALPHA_BALANCED_HI = 0.6
ALPHA_CDOMINATES = 0.6     # alpha > 0.6 -> C Dominates
ALPHA_ODOMINATES = 0.4     # alpha < 0.4 -> O Dominates
LAMBDA_WEAPONIZED = 0.8    # high, enforced memory


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
# THE FOUR FATES
# ============================================================

def fate_of(p):
    """Classify a parameter set into one of the four IAR fates.

    The five fate conditions of the framework:
      Internal Equilibrium : beta*gamma < 0.5*lam and theta -> 0
      Joint Equilibrium    : 0.5*lam <= beta*gamma <= lam
      C Dominates          : alpha > 0.6 or beta -> inf
      O Dominates          : alpha < 0.4 or theta -> inf
      Dissolution          : eta > 0.3 (or gamma -> inf or lam -> 0)
    Dissolution is declared first: it asymptotically dominates every other fate.
    """
    if p["eta"] > ETA_DISSOLUTION:
        return "Dissolution"
    prod = p["beta"] * p["gamma"]
    if p["alpha"] > ALPHA_CDOMINATES:
        return "C Dominates"
    if p["alpha"] < ALPHA_ODOMINATES:
        return "O Dominates"
    if prod < 0.5 * p["lam"] and p["theta"] < 0.3:
        return "Internal Equilibrium"
    if 0.5 * p["lam"] <= prod <= p["lam"]:
        return "Joint Equilibrium"
    return "Joint Equilibrium (borderline)"


def longevity_score(p):
    """Egypt-style longevity index = max lam + min eta + balanced alpha,
    projected onto [0, 1]. High lambda, low eta, alpha ~ 0.5 score highest."""
    lam_norm = min(p["lam"] / 5.0, 1.0)
    eta_norm = max(1.0 - p["eta"], 0.0)
    bal = 1.0 - min(abs(p["alpha"] - 0.5) / 0.5, 1.0)
    return round(0.4 * lam_norm + 0.3 * eta_norm + 0.3 * bal, 3)


# ============================================================
# CONFLICT SYSTEM AND THE WAR THEOREM
# ============================================================

class ConflictSystem:
    """Two competing Theta_self systems sharing one Theta_world (the land).

    The IAR Theorem on War: War is the collapse of Joint Equilibrium when
    alpha -> 1 (C-Dominates) and theta -> 0 (exchange closes) simultaneously,
    driven by lambda (memory) as a weapon and eta > 0.3 (chaos) as catalyst.
    """

    def __init__(self, name, theta_a=None, theta_b=None, thetag=None):
        self.name = name
        # theta_a: system A, theta_b: system B, thetag: shared world (the land)
        self.theta_a = IARParams(**theta_a) if theta_a else IARParams()
        self.theta_b = IARParams(**theta_b) if theta_b else IARParams()
        self.thetag = thetag or {}

    def war_conditions(self):
        """Return the five necessary conditions for war as a dict of bools.

        1. Competing Theta_self : two incompatible self-models
        2. Weaponized lambda     : memory used as a weapon, one-sided history
        3. alpha -> 1            : at least one side seeks total control
        4. theta -> 0            : exchange closed, no dialogue
        5. eta > 0.3             : chaos / unpredictability
        """
        self_model_a = self.theta_a["beta"] > 0.2     # A couples to itself
        self_model_b = self.theta_b["beta"] > 0.2
        competing = (self_model_a and self_model_b
                     and abs(self.theta_a["alpha"] - self.theta_b["alpha"]) > 0.1)
        weaponized = (self.theta_a["lam"] >= LAMBDA_WEAPONIZED
                      or self.theta_b["lam"] >= LAMBDA_WEAPONIZED)
        dominance = (self.theta_a["alpha"] > ALPHA_CDOMINATES
                     or self.theta_b["alpha"] > ALPHA_CDOMINATES)
        closed = (self.theta_a["theta"] < 0.3
                  or self.theta_b["theta"] < 0.3)
        chaos = (self.theta_a["eta"] > ETA_DISSOLUTION
                 or self.theta_b["eta"] > ETA_DISSOLUTION)
        return {
            "1. Competing Theta_self": competing,
            "2. Weaponized lambda": weaponized,
            "3. alpha -> 1": dominance,
            "4. theta -> 0": closed,
            "5. eta > 0.3": chaos,
        }

    def conditions_met(self):
        conds = self.war_conditions()
        return {k: v for k, v in conds.items() if v}

    def is_war(self):
        """War when all five conditions hold simultaneously (strong theorem),
        or when alpha->1 and theta->0 coincide (weak theorem)."""
        conds = self.war_conditions()
        strong = all(conds.values())
        weak = conds["3. alpha -> 1"] and conds["4. theta -> 0"]
        return strong or weak

    def escalate(self, delta):
        """Apply a delta-vector of parameter changes to both sides."""
        for key in PARAM_NAMES:
            if key in delta:
                self.theta_a[key] = max(0.0, self.theta_a[key] + delta[key])
                self.theta_b[key] = max(0.0, self.theta_b[key] + delta[key])

    def summary(self):
        print("Conflict: %s" % self.name)
        print("  fate A: %s | fate B: %s"
              % (fate_of(self.theta_a), fate_of(self.theta_b)))
        conds = self.war_conditions()
        for label, ok in conds.items():
            print("  %-24s %s" % (label, "MET" if ok else "no"))
        print("  -> %s" % ("WAR (dissolution of Joint Equilibrium)"
                           if self.is_war() else "not (yet) war"))


# ============================================================
# THE FOUNDATIONAL LIE
# ============================================================

class LieDetector:
    """The foundational lie reveals itself through five symptoms.

      Dehumanization      : Theta_other = 0, the other is a label, not a mind
      Weaponized memory   : lambda is one-sided, history is a weapon
      Closed exchange     : theta -> 0, dialogue is impossible
      Dominance assertion : alpha -> 1, one side seeks total control
      Fear / chaos        : eta > 0.3, truth is suppressed through fear
    Master Key: to end any conflict, seek the lies at the base of it all.
    """

    SYMPTOMS = [
        ("Dehumanization", "theta_other == 0", lambda c: c.theta_other_zero),
        ("Weaponized memory", "lambda one-sided", lambda c: c.weaponized),
        ("Closed exchange", "theta -> 0", lambda c: c.closed),
        ("Dominance assertion", "alpha -> 1", lambda c: c.dominance),
        ("Fear and chaos", "eta > 0.3", lambda c: c.chaos),
    ]

    def __init__(self, conflict=None):
        # Booleans that external analysts would populate from evidence
        self.theta_other_zero = False
        self.weaponized = False
        self.closed = False
        self.dominance = False
        self.chaos = False
        if conflict is not None:
            conds = conflict.war_conditions()
            self.weaponized = conds["2. Weaponized lambda"]
            self.closed = conds["4. theta -> 0"]
            self.dominance = conds["3. alpha -> 1"]
            self.chaos = conds["5. eta > 0.3"]
            self.theta_other_zero = conds["1. Competing Theta_self"]

    def scan(self):
        found = {label[:-1]: fn(self) for label, _, fn in self.SYMPTOMS}
        return found

    def lie_present(self):
        scans = self.scan()
        # The lie is present when at least three symptoms fire.
        return sum(1 for v in scans.values() if v) >= 3

    def find_lie(self):
        print("Lie-Detection scan:")
        for label, marker, ok in self.report():
            print("  %-22s %-24s %s" % (label, marker, "present" if ok else "-"))


class LieSeed:
    """The Law of the Lie-Seed: every lie is a seed that blooms as war.

    Lifecycle:
      Planting     : the lie is told                      (moment)
      Watering     : the lie is repeated                  (days-years)
      Fertilizing  : the lie is institutionalized         (years-decades)
      Growing      : the lie becomes identity             (decades-generations)
      Blooming     : the lie becomes war                  (generations-centuries)
    To prevent war, uproot the seed before it grows.
    """

    PHASES = [
        ("Planting", 0.0),          # moment
        ("Watering", 1.0),          # days to years (use year 1 floor)
        ("Fertilizing", 10.0),      # years to decades
        ("Growing", 30.0),          # decades to a generation
        ("Blooming", 100.0),        # generations to centuries
    ]

    def __init__(self, conflict, lie, planted_year, bloomed_year=None):
        self.conflict = conflict
        self.lie = lie
        self.planted_year = planted_year
        self.bloomed_year = bloomed_year   # None -> seed still growing
        self.now = 2026

    def years_since_planting(self):
        ref = self.bloomed_year if self.bloomed_year else self.now
        return max(0.0, ref - self.planted_year)

    def phase(self):
        t = self.years_since_planting()
        phase = self.PHASES[0][0]
        for name, years in self.PHASES:
            if t >= years:
                phase = name
            else:
                break
        if self.bloomed_year is not None and t >= self.PHASES[-1][1]:
            phase = "Bloomed (war)"
        return phase

    def summary(self):
        print("%-24s | lie: %-42s | planted %s | years %5.0f | phase: %s"
              % (self.conflict, self.lie, self.planted_year,
                 self.years_since_planting(), self.phase()))


# ============================================================
# THE FIVE PHASES OF PEACE
# ============================================================

class PeaceEngine:
    """The IAR Blueprint for Peace Resolution.

      Phase 1 Ceasefire          : reduce eta   (stop the violence)
      Phase 2 Dialogue           : open theta   (establish communication)
      Phase 3 Truth & Reconc.    : resolve lam  (acknowledge the past)
      Phase 4 Power Sharing      : balance alpha(distribute power)
      Phase 5 Coexistence        : thaw Theta   (build shared identity)
    Peace = Reduce eta + Open theta + Resolve lam + Balance alpha + Thaw Theta
    """

    PHASES = [
        ("Ceasefire", {"eta": -0.20, "beta": +0.10}, "Weeks to months"),
        ("Dialogue", {"theta": +0.25}, "Months to years"),
        ("Truth & Reconciliation", {"lam": -0.15}, "Years to decades"),
        ("Power Sharing", {"alpha": 0.0}, "Years to decades"),
        ("Coexistence", {"theta": +0.15, "beta": +0.10}, "Generations"),
    ]

    def __init__(self, conflict):
        self.conflict = conflict
        self.log = []

    def run(self):
        self.log = []
        for phase, delta, duration in self.PHASES:
            self.conflict.escalate(delta)
            # Power sharing centers both alphas on 0.5
            if phase == "Power Sharing":
                self.conflict.theta_a["alpha"] = 0.5
                self.conflict.theta_b["alpha"] = 0.5
            self.log.append((phase, duration, dict(self.conflict.theta_a.p),
                             dict(self.conflict.theta_b.p)))
        done = self.conflict.is_war()
        return not done  # peace achieved when the war conditions no longer hold

    def summary(self, headers=("Phase", "Duration", "alpha_A", "theta_A",
                               "eta_A", "alpha_B", "theta_B", "eta_B")):
        print("%-24s %-16s %8s %7s %6s %8s %7s %6s"
              % headers)
        for phase, dur, pa, pb in self.log:
            print("%-24s %-16s %8s %7s %6s %8s %7s %6s" % (
                phase, dur,
                round(pa["alpha"], 2), round(pa["theta"], 2),
                round(pa["eta"], 2),
                round(pb["alpha"], 2), round(pb["theta"], 2),
                round(pb["eta"], 2)))


class UprootProcess:
    """Decoding / uprooting a lie.

      Uproot Lie = Expose Truth + Re-educate + Re-memorialize + Re-balance Power
      Process    : expose truth -> re-educate -> re-memorialize
                   -> re-balance power -> re-identify
    """

    STEPS = [
        "Expose the Truth",
        "Re-educate",
        "Re-memorialize",
        "Re-balance Power",
        "Re-identify",
    ]

    def __init__(self, seed):
        self.seed = seed

    def run(self):
        print("Uprooting lie of %s: %s" % (self.seed.conflict, self.seed.lie))
        for step in self.STEPS:
            print("  [%s]" % step)


# ============================================================
# DEMONSTRATION
# ============================================================

def _rwanda():
    print("=" * 72)
    print("RWANDA: GENOCIDE (1994) -> GACACA RESET")
    print("=" * 72)
    # Pre-genocide parameters: enforced memory lies, chaos, dominance, closed
    rw = ConflictSystem(
        "Rwanda (1994)",
        theta_a=None,   # Hutu Power government
        theta_b=None)
    rw.theta_a = IARParams(beta=0.8, gamma=0.9, lam=0.9, eta=0.65,
                           alpha=0.9, theta=0.1)
    rw.theta_b = IARParams(beta=0.6, gamma=0.9, lam=0.8, eta=0.65,
                           alpha=0.1, theta=0.1)
    rw.summary()
    print()
    # Gacaca: resolve memory, restore identity, reduce chaos
    gacaca = {"lam": -0.55, "eta": -0.35, "theta": +0.30}
    rw.escalate(gacaca)
    rw.theta_b["alpha"] = 0.4
    print("After Gacaca intervention (resolve lam, reduce eta, open theta):")
    rw.summary()
    return rw


def _egypt():
    print("=" * 72)
    print("ANCIENT EGYPT: THE LONGEST-LASTING CIVILIZATION")
    print("=" * 72)
    egypt = IARParams(beta=0.85, gamma=0.15, lam=4.5, eta=0.05,
                      alpha=0.5, theta=0.6)
    print("Egypt parameter set: %s" % egypt)
    print("Fate: %s" % fate_of(egypt))
    print("Longevity score (max lam + min eta + balanced alpha): %s"
          % longevity_score(egypt))
    for name, p in [
            ("Rome", IARParams(beta=0.8, gamma=0.5, lam=1.2, eta=0.5,
                               alpha=0.8, theta=0.7)),
            ("Greece", IARParams(beta=0.7, gamma=0.5, lam=2.0, eta=0.4,
                                 alpha=0.5, theta=0.8)),
            ("Mongol", IARParams(beta=0.9, gamma=0.9, lam=0.2, eta=0.8,
                                 alpha=1.0, theta=0.6))]:
        print("  %-8s fate=%-18s longevity=%s" % (name, fate_of(p),
                                                  longevity_score(p)))
    return egypt


def _master_key_demo():
    print("=" * 72)
    print("ISRAEL-PALESTINE: THE MASTER KEY AND THE FIVE PHASES")
    print("=" * 72)
    ip = ConflictSystem(
        "Israel-Palestine",
        theta_a=dict(beta=0.9, gamma=0.7, lam=0.95, eta=0.7,
                     alpha=0.75, theta=0.2),
        theta_b=dict(beta=0.9, gamma=0.7, lam=0.95, eta=0.7,
                     alpha=0.25, theta=0.2))
    ip.summary()
    print()
    ld = LieDetector(ip)
    ld.scan()
    print("Foundational lie present: %s" % ld.lie_present())
    print()
    peace = PeaceEngine(ip)
    achieved = peace.run()
    peace.summary()
    print("Peace achieved (war conditions cleared): %s" % achieved)


def main():
    _egypt()
    print()
    _rwanda()
    print()
    _master_key_demo()
    print()
    print("=" * 72)
    print("LIE-SEED TABLE (Law of the Lie-Seed)")
    print("=" * 72)
    seeds = [
        LieSeed("Rwanda", "Tutsis are inferior", 1930, 1994),
        LieSeed("Nazi Germany", "Jews are subhuman", 1880, 1941),
        LieSeed("Israel-Palestine", "The other wants to destroy us", 1948),
        LieSeed("Russia-Ukraine", "Ukraine is a Nazi state", 2014, 2022),
        LieSeed("Serbia-Kosovo", "Kosovo is the cradle of Serbian civilization",
                1389, 1999),
        LieSeed("USA (Race)", "White people are superior", 1619, 2020),
    ]
    for seed in seeds:
        seed.summary()
    print()
    UprootProcess(seeds[2]).run()


if __name__ == "__main__":
    main()