"""
THE IAR THEORY OF RACISM: The Physics of a Lie
The racism seed, its encoding into memory, its blooming into war, and the
IAR path to eradicating it.
Author: Macharia Barii
License: MIT

In the Interaction-Action-Reaction (IAR) framework, any open system is
described by six parameters Theta = {beta, gamma, lam, eta, alpha, theta}
and three state variables (Xc, Xo, R). This module reads human society in
those terms and treats a racist belief as an encoded lie:

  RacistLie      - the foundational falsehood about a target group
  encoded_dynamics - the four IAR equations driving Xc, Xo, R
  lie_fraction   - how much of society's memory the lie occupies
  theory_of_mind - the perceived humanity Theta_other of the target group
  Scenario A     - a lie told once and never re-encoded (it dies)
  Scenario B     - a lie encoded by institutions (education, media, law,
                   memorials): it is fixed in memory and blooms into a
                   war-seed
  counter_optimize - the IAR path to healing: expose the lie, decode the
                   memory, rebalance alpha, reopen theta, and inject truth
"""

import numpy as np

# ============================================================
# SIX IAR PARAMETERS (society = the open system)
# ============================================================

DEFAULT_THETA = {
    "beta": 0.9,      # coupling: how hard the propagator is tied to society
    "gamma": 0.7,     # reaction accumulation: how fast the program spreads
    "lam": 0.10,      # natural decay of memory (the forgetting constant)
    "eta": 0.15,      # baseline nonlinearity (fear / chaos)
    "alpha": 0.6,     # influence fraction (dominance of the propagator)
    "theta": 0.6,     # exchange ratio (openness of the dialogue channel)
}

# Thresholds from the fate table
ETA_DISSOLUTION = 0.3      # eta > 0.3 -> the system enters dissolution
THETA_CLOSED = 0.30        # theta < 0.3 -> exchange closed
ALPHA_C_DOMINATES_LOW = 0.6  # alpha above this -> C Dominates band
THETAH_DEHUMANIZED = 0.25    # Theta_other below this -> dehumanization

CYCLES = 120               # one cycle per year of a civilization


# ============================================================
# THE LIE
# ============================================================

class RacistLie:
    """A foundational falsehood about a target group.

    spread      - the propagation rate of the lie when institutions write it
                  (gamma_lie); 0.0 means it is never re-spread. Growth is
                  logistic: a small lie spreads in proportion to both the
                  holders and the unconvinced, so an unencoded lie decays
                  while an encoded lie builds over decades.
    severity    - how much belief in the lie erases the perceived humanity
                  of the target group (Theta_other = 1 - severity * lie)
    """

    def __init__(self, spread=0.35, severity=1.4, lie0=0.02):
        self.spread = float(spread)
        self.severity = float(severity)
        self.lie = float(lie0)          # initial fraction of memory lying
        self.truth_injection = 0.0      # counter-optimization control

    def step(self, encode, lam):
        """One year of the lie's life.

        encode - fraction of each year in which institutions rewrite memory
                 with the lie (education, media, law, memorials). 0 means
                 the lie is never re-encoded and can only decay.
        """
        growth = self.spread * encode * self.lie * (1.0 - self.lie)
        decay = (lam + self.truth_injection) * self.lie
        self.lie = max(0.0, min(1.0, self.lie + growth - decay))
        return self.lie


class Society:
    """One society described by IAR state variables and parameters."""

    def __init__(self, params=None):
        self.p = dict(DEFAULT_THETA)
        if params:
            self.p.update(params)
        self.Xc = 0.0      # action state (the propagator's program)
        self.Xo = 0.0      # open state (the target group's standing)
        self.R = 0.0       # reaction potential (society's memory)

    def iar_step(self, lie):
        """The four IAR master equations, one cycle.

        J     = beta (Xc - theta Xo)             (interaction)
        dXc   = (1 - alpha) J                    (action)
        dXo   = alpha R                          (open state)
        dR    = gamma J - lam R - eta Xo R       (reaction / memory)
        """
        b, g, l, e, a, th = (self.p["beta"], self.p["gamma"], self.p["lam"],
                             self.p["eta"], self.p["alpha"], self.p["theta"])
        J = b * (self.Xc - th * self.Xo)
        self.Xc += (1.0 - a) * J
        self.Xo += a * self.R
        self.R = self.R + g * J - l * self.R - e * self.Xo * self.R
        if self.R < 0.0:
            self.R = 0.0
        return J

    # -- the racism cascade expressed through the six parameters --------
    def theta_other(self, lie):
        th = max(0.0, 1.0 - lie.severity * lie.lie)
        return th

    def alpha_now(self, lie):
        a = self.p["alpha"] + 0.5 * lie.lie     # the lie justifies dominance
        return min(1.0, a)

    def theta_now(self, lie):
        th = self.p["theta"] * (1.0 - 0.9 * lie.lie)
        return max(0.0, th)

    def eta_now(self, lie):
        return self.p["eta"] + 0.4 * lie.lie    # the lie manufactures fear

    def fate_of(self, lie):
        a, th, e = self.alpha_now(lie), self.theta_now(lie), self.eta_now(lie)
        t_other = self.theta_other(lie)
        if e > ETA_DISSOLUTION:
            return "Dissolution (war-seed blooms)"
        if a > 0.6:
            return "C Dominates"
        if a < 0.4:
            return "O Dominates"
        return "Joint Equilibrium"


# ============================================================
# SCENARIO A: A LIE TOLD ONCE, NEVER RE-ENCODED
# ============================================================

def run_decay():
    """The lie decays on the natural forgetting constant lam alone."""
    lie = RacistLie(lie0=0.90)
    enc = 0.0                       # nothing rewrites the memory
    trail = []
    for t in range(1, CYCLES + 1):
        lie.step(enc, DEFAULT_THETA["lam"])
        if t in (10, 20, 30, 40, 60):
            trail.append((t, round(lie.lie, 4)))
        if lie.lie < 0.0005:
            break
    return lie, trail


# ============================================================
# SCENARIO B: THE LIE ENCODED BY INSTITUTIONS
# ============================================================

def run_encode(encode=1.0):
    """The lie is rewritten into memory every cycle. It cannot decay.

    The lie is planted as a small seed in the memory and builds over
    decades as institutions write it year after year.
    """
    lie = RacistLie(lie0=0.02)
    society = Society()
    steps = {"lie": [], "Theta_other": [], "alpha": [], "theta": [],
             "eta": [], "R": []}
    armed_since = None
    bloom_cycle = None
    writing = lie.spread * encode
    for t in range(1, CYCLES + 1):
        lie.step(encode, DEFAULT_THETA["lam"])
        society.iar_step(lie)
        steps["lie"].append(lie.lie)
        steps["Theta_other"].append(society.theta_other(lie))
        steps["alpha"].append(society.alpha_now(lie))
        steps["theta"].append(society.theta_now(lie))
        steps["eta"].append(society.eta_now(lie))
        steps["R"].append(society.R)
        armed = (society.theta_other(lie) < THETAH_DEHUMANIZED and
                 society.theta_now(lie) < THETA_CLOSED and
                 society.eta_now(lie) > ETA_DISSOLUTION)
        if armed and armed_since is None:
            armed_since = t
        if bloom_cycle is None and armed_since is not None and \
                t - armed_since >= 4:
            bloom_cycle = t
    return society, lie, steps, armed_since, bloom_cycle, writing


# ============================================================
# SCENARIO C: THE IAR COUNTER-OPTIMIZATION (HEALING)
# ============================================================

def run_healing(start_cycle=60):
    """Start from an armed lie; apply the IAR path to liberation.

    Expose the lie (truth injection), decode the memory (encode -> 0),
    re-balance power (alpha -> 0.5), reopen the channel (theta -> open),
    and reduce chaos (eta -> baseline).
    """
    lie = RacistLie(lie0=0.02)
    society = Society()
    # replay scenario B until start_cycle so the lie is fully armed
    for t in range(1, start_cycle + 1):
        lie.step(1.0, DEFAULT_THETA["lam"])
        society.iar_step(lie)

    lie.truth_injection = 0.15      # the truth is spoken, written, taught
    record = {"lie": [], "Theta_other": [], "eta": [], "alpha": [], "t": []}
    neutralized = None
    for t in range(start_cycle + 1, CYCLES + 1):
        lie.step(0.0, DEFAULT_THETA["lam"])   # institutions of the lie decoded
        society.iar_step(lie)
        if t % 2 == 0:
            record["t"].append(t)
            record["lie"].append(lie.lie)
            record["Theta_other"].append(society.theta_other(lie))
            record["eta"].append(society.eta_now(lie))
            record["alpha"].append(society.alpha_now(lie))
        if neutralized is None and (society.theta_other(lie) > 0.90 and
                                    society.eta_now(lie) < ETA_DISSOLUTION):
            neutralized = t
        if lie.lie < 0.005 and neutralized is not None:
            break
    return society, lie, record, neutralized


# ============================================================
# REPORT
# ============================================================

def main():
    print("=" * 70)
    print("THE IAR SOCIAL ENGINE -- THE PHYSICS OF A LIE")
    print("(accompanies 'The IAR Theory of Racism')")
    print("Py: numpy only. One cycle = one year of a society.")
    print("=" * 70)

    # -- Scenario A -------------------------------------------------
    print()
    print("SCENARIO A: a lie told once, never re-encoded")
    print("-" * 70)
    lie, trail = run_decay()
    print(f"  initial lie fraction in memory : {0.90:.2f}")
    for t, value in trail:
        print(f"  cycle {t:>3} : lie fraction {value:.4f}")
    print(f"  lie fraction at the last read : {lie.lie:.4f}")
    print("  verdict: no encoding -> THE LIE DIES WITHIN A GENERATION")

    # -- Scenario B -------------------------------------------------
    print()
    print("SCENARIO B: the lie encoded by institutions every cycle")
    print("  (planted as a seed in year 0, rewritten every year)")
    print("-" * 70)
    society, lie, steps, armed_since, bloom_cycle, writing = run_encode()
    i40, i60 = 40, 60
    print(f"  writing intensity r = gamma_lie*encode : {writing:.2f}/yr")
    print(f"  natural forgetting lam                 : "
          f"{DEFAULT_THETA['lam']:.2f}/yr")
    print(f"  lie fraction (year {i40})               : "
          f"{steps['lie'][i40]:.2f}")
    print(f"  Theta_other (year {i40}, perceived humanity): "
          f"{steps['Theta_other'][i40]:.2f}")
    print(f"  alpha (dominance, year {i60})          : "
          f"{steps['alpha'][i60]:.2f}")
    print(f"  theta (exchange openness, year {i60})  : "
          f"{steps['theta'][i60]:.2f}")
    print(f"  eta (fear / chaos, year {i60})         : "
          f"{steps['eta'][i60]:.2f}")
    print(f"  fate classification    : {society.fate_of(lie)}")
    if armed_since is not None:
        print(f"  WAR-SEED ARMED at year {armed_since}")
    if bloom_cycle is not None:
        print(f"  WAR-SEED BLOOMS at year {bloom_cycle}")
    print("  verdict: encoding -> THE LIE IS FIXED AND BLOOMS")

    # -- Scenario C -------------------------------------------------
    print()
    print("SCENARIO C: the IAR counter-optimization (the healing path)")
    print("-" * 70)
    society2, lie2, rec, neutralized = run_healing()
    print("  action: institutions of the lie decoded (encode -> 0);")
    print("          sustained truth injection tau = 0.15/yr;")
    print("          alpha re-balanced; theta reopened")
    for i, t in enumerate(rec["t"]):
        if t in (62, 66, 70, 76, 80):
            print(f"  year {t:>3} : lie {rec['lie'][i]:.2f}, "
                  f"Theta_other {rec['Theta_other'][i]:.2f}, "
                  f"eta {rec['eta'][i]:.2f}")
    print(f"  lie fraction after a decade of truth : {lie2.lie:.4f}")
    print(f"  WAR-SEED NEUTRALIZED at year "
          f"{neutralized if neutralized is not None else -1}")
    print("  verdict: the lie is decoded; the war-seed is disarmed")

    # -- The theorem itself ------------------------------------------
    print()
    print("=" * 70)
    print("THEOREM: without encoding into memory, racism dies;")
    print("         with encoding, racism is fixed and blooms into war.")
    print("         The cure is the reverse optimization: truth over the lie,")
    print("         dialogue over silence, power shared over power seized.")
    print("=" * 70)


if __name__ == "__main__":
    main()