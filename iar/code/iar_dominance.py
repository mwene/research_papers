"""
THE IAR THEORY OF HUMAN DOMINANCE: One Program, Many Flags
Racism, sexism, classism, colonialism, ableism, religious bigotry,
nationalism, hegemony, speciesism, totalitarianism -- one lie-writing
program run by every dominance system against a chosen target.
Author: Macharia Barii
License: MIT

This module runs the encoded-lie dynamics of the IAR social physics
against every member of the dominance spectrum and reads the universal
pattern: no matter the flag, the configuration is the same four-term move

    Racism-like vector:  Thoth -> 0  +  lambda weaponized
                        +  alpha -> 1  +  theta -> 0

The difference between the forms is only the severity of the dehumanization
(how far Thoth falls) and the strength of the institutional writing that
keeps the lie in memory. For every form the module prints:
    - the foundational lie,
    - the state of the society after forty years of writing,
    - the fate of the system,
    - the war-seed arming and blooming years,
and then runs the liberation pass (the reverse optimization) against every
form and shows that each returns to full theory of mind.
"""

import numpy as np

# ============================================================
# THE SHARED PHYSICS (same constants as the racism engine)
# ============================================================

LAM = 0.10          # natural forgetting, per year
SBASE = 0.15        # baseline nonlinearity (fear / chaos)
KETA = 0.4          # fear manufactured per unit of lie
CADOM = 0.5          # dominance gained per unit of lie
THETA_BASE = 0.6    # baseline exchange openness
THETA_CLOSED = 0.30
THETAH_DEHUMANIZED = 0.25
ETA_DISSOLUTION = 0.3
YEARS = 60
ARM_GRACE = 4       # years armed before the seed blooms

# The dominance spectrum: (name, foundational lie, severity, writing r)
FORMS = [
    ("Sexism",          "women are the weaker vessel",        0.7, 0.30),
    ("Ableism",         "they are broken, therefore lesser",  0.6, 0.26),
    ("Ageism",          "the old are useless",                0.6, 0.26),
    ("Nationalism",     "our nation is supreme by nature",    0.9, 0.28),
    ("Hegemony",        "our power needs no justification",   0.8, 0.30),
    ("Classism",        "the poor deserve their place",       1.3, 0.32),
    ("Racism",          "they are inferior by nature",        1.4, 0.35),
    ("Totalitarianism", "the state is the only person",       1.4, 0.35),
    ("Religious bigotry", "they are damned by the divine",    1.5, 0.33),
    ("Colonialism",     "we are civilizing them",             1.5, 0.34),
    ("Speciesism",      "other animals are property",         1.6, 0.36),
]


# ============================================================
# ONE DOMINANCE FORM
# ============================================================

class DominanceForm:
    """One member of the spectrum: a lie, its severity, its writers."""

    def __init__(self, name, lie, severity, writing):
        self.name = name
        self.lie_text = lie
        self.severity = float(severity)
        self.writing = float(writing)      # r = gamma_lie * encode

        self.L = 0.02                       # seed planted in year 0
        self.x_c = 0.0
        self.x_o = 0.0
        self.rp = 0.0                       # reaction potential R
        self.tau = 0.0                      # truth injection (liberation)

    # -- derived parameters ------------------------------------------
    def theta_other(self):
        return max(0.0, 1.0 - self.severity * self.L)

    def alpha(self):
        return min(1.0, 0.6 + CADOM * self.L)

    def theta(self):
        return max(0.0, THETA_BASE * (1.0 - 0.9 * self.L))

    def eta(self):
        return SBASE + KETA * self.L

    def fate(self):
        a, th, e = self.alpha(), self.theta(), self.eta()
        if e > ETA_DISSOLUTION and self.theta_other() < THETAH_DEHUMANIZED \
                and th < THETA_CLOSED:
            return "Dissolution (war-seed armed)"
        if e > ETA_DISSOLUTION:
            return "Dissolution pressure (dehumanization partial)"
        if a > 0.6:
            return "C Dominates"
        if a < 0.4:
            return "O Dominates"
        return "Joint Equilibrium"

    # -- dynamics ------------------------------------------------------
    def step(self, encode, healing=False):
        """One year. Writing rewrites the lie; truth erodes it."""
        growth = self.writing * self.L * (1.0 - self.L) if encode else 0.0
        decay = (LAM + self.tau) * self.L
        self.L = max(0.0, min(1.0, self.L + growth - decay))

        b, g, l, e, a, th = 0.9, 0.7, LAM, self.eta(), self.alpha(), \
            self.theta()
        J = b * (self.x_c - th * self.x_o)
        self.x_c += (1.0 - a) * J
        self.x_o += a * self.rp
        self.rp = max(0.0, self.rp + g * J - l * self.rp - e * self.x_o *
                      self.rp)

    def armed(self):
        return (self.theta_other() < THETAH_DEHUMANIZED and
                self.theta() < THETA_CLOSED and self.eta() > ETA_DISSOLUTION)


def run_encoded(form, years=YEARS):
    """Write the lie every year; record the trajectory."""
    state = {"t": [], "L": [], "thoth": [], "alpha": [], "theta": [],
             "eta": []}
    armed_since = None
    bloom = None
    for t in range(1, years + 1):
        form.step(encode=True)
        state["t"].append(t)
        state["L"].append(form.L)
        state["thoth"].append(form.theta_other())
        state["alpha"].append(form.alpha())
        state["theta"].append(form.theta())
        state["eta"].append(form.eta())
        if form.armed() and armed_since is None:
            armed_since = t
        if bloom is None and armed_since is not None and \
                t - armed_since >= ARM_GRACE:
            bloom = t
    return state, armed_since, bloom


def run_liberated(form):
    """From the armed year: decode the writing and inject truth."""
    form.tau = 0.15
    neutralized = None
    for t in range(1, YEARS + 1):
        form.step(encode=False, healing=True)
        if neutralized is None and form.theta_other() > 0.90 and \
                form.eta() < ETA_DISSOLUTION:
            neutralized = t
        if form.L < 0.005 and neutralized is not None:
            break
    return neutralized


# ============================================================
# REPORT
# ============================================================

def main():
    print("=" * 78)
    print("THE IAR SOCIAL ENGINE -- ONE PROGRAM, MANY FLAGS")
    print("(accompanies 'The IAR Theory of Human Dominance')")
    print("Py: numpy only. One cycle = one year of a society.")
    print("=" * 78)

    # --- the encoded state of every form after forty years -----------
    print()
    print("PART 1: THE DOMINANCE SPECTRUM, READ AT YEAR 40")
    print("  (each form runs the same encoded-lie dynamics)")
    print("-" * 78)
    print(f"{'form':<18}{'Thoth':>7}{'alpha':>7}{'theta':>7}"
          f"{'eta':>7}  {'fate':<35}  {'armed':>5} {'bloom':>5}")
    print(f"{'-'*18}{'-'*7}{'-'*7}{'-'*7}{'-'*7}  "
          f"{'-'*35}  {'-'*5} {'-'*5}")
    results = []
    for name, lie, severity, writing in FORMS:
        form = DominanceForm(name, lie, severity, writing)
        state, armed, bloom = run_encoded(form)
        i40 = 39
        row = (name, state["thoth"][i40], state["alpha"][i40],
               state["theta"][i40], state["eta"][i40], form.fate(),
               armed, bloom)
        results.append((form, state, armed, bloom, row))
        armed_s = str(armed) if armed is not None else "-"
        bloom_s = str(bloom) if bloom is not None else "-"
        print(f"{name:<18}{row[1]:>7.2f}{row[2]:>7.2f}{row[3]:>7.2f}"
              f"{row[4]:>7.2f}  {row[5]:<35}  {armed_s:>5} {bloom_s:>5}")

    # --- the universal pattern ----------------------------------------
    print()
    print("PART 2: THE UNIVERSAL PATTERN")
    print("-" * 78)
    print("  every form converges to the same configuration:")
    print("    theta_other -> 0 (or toward it), alpha -> 1,")
    print("    theta -> 0, eta -> 0.43      -- one four-term vector,")
    print("    (Thoth->0) + (lambda weaponized) + (alpha->1) + (theta->0)")
    print("  the only free parameters across the spectrum are the")
    print("  severity of the dehumanization and the strength of the")
    print("  institutional writing that upholds the lie.")

    # --- the liberation pass -----------------------------------------
    print()
    print("PART 3: THE LIBERATION PASS (the reverse optimization)")
    print("-" * 78)
    print(f"{'form':<20}{'Thoth before':>12}{'Thoth after':>12}"
          f"{'neutralized':>12}")
    print(f"{'-'*20}{'-'*12}{'-'*12}{'-'*12}")
    for form, state, armed, bloom, row in results:
        th_before = state["thoth"][39]
        yr = run_liberated(form)
        th_after = form.theta_other()
        yr_s = f"year {yr}" if yr is not None else "-"
        print(f"{form.name:<20}{th_before:>12.2f}{th_after:>12.2f}"
              f"{yr_s:>12}")

    print()
    print("=" * 78)
    print("THEOREM: under the same lie-writing law, every member of the")
    print("  dominance spectrum is the same vector with a different flag;")
    print("  and the same reverse optimization liberates every one of them.")
    print("  From sexism to speciesism: decode the lie, speak the truth,")
    print("  share the power, reopen the channel.")
    print("=" * 78)


if __name__ == "__main__":
    main()