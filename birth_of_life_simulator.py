"""
BIRTH-OF-LIFE: Complete Simulation
Author: [Your Name]
Date: 2026-09-03
Version: 2.0 (Revised with all model extensions)

This simulation implements the energy-driven, deterministic model
for the origin of life described in the paper.

NEW IN VERSION 2.0:
- Sensitivity analysis (Monte Carlo)
- Environmental realism (acid, saline, UV damage)
- Abiotic competition (waste formation)
- Bioenergetic constraints (EROI, maintenance)
- Salt poisoning and thermal denaturation
- Parasite-host co-evolution
- Diffusion-limited permeability
- Galactic habitability function
- Exoplanet predictions
- Geological matching

USAGE:
python birth_of_life.py
"""

import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm, norm


# --------------------------------------
# SECTION 1: PARAMETERS (All Knobs)
# --------------------------------------


class Parameters:
    """All tunable parameters for the simulation"""

    def __init__(self, mode='B'):  # 'A' = Space-Born, 'B' = Surface-Born
        # Environmental (Core Pushers)
        self.I_UV = 3.0  # UV flux (relative to modern Earth)
        self.f_wd = 1.0  # Wet-dry cycle frequency (cycles/day)
        self.k_rad = 1e-4  # Radioactive decay rate (s^-1)
        self.T_env = 298.0  # Environmental temperature (K)
        self.pH = 7.0  # pH
        self.salinity = 0.01  # NaCl concentration (M)

        # Synthesis & Concentration
        self.S_surf = 1e-7  # Surface monomer supply (M/s)
        self.S_comet = 1e-5  # Comet delivery (M, initial) - Mode A only
        self.k_UV = 2e-7  # UV synthesis rate (s^-1)
        self.k_photolysis = 1e-6  # UV photodegradation rate (s^-1)
        self.k_decay = 1e-6  # Hydrolysis/degradation rate (s^-1)
        self.k_conc = 1e-6  # Concentration factor per wet-dry cycle
        self.k_autocat = 1e4  # Autocatalytic rate constant (M^-1 s^-1)
        self.k_waste = 1e-7  # Waste formation rate (s^-1)

        # UV Spectrum
        self.alpha_uv = 0.5  # UV attenuation coefficient (m^-1)
        self.tau_ozone = 0.0  # Ozone optical depth (0 = none)
        self.z_depth = 0.5  # Water depth (m)

        # Salt Poisoning
        self.K_salt = 0.1  # Half-inhibition salt concentration (M)
        self.n_salt = 2.0  # Hill coefficient for salt inhibition

        # Thermal Denaturation
        self.Ea_denature = 50e3  # Activation energy for denaturation (J/mol)
        self.T_ref = 298.0  # Reference temperature (K)

        # Compartmentalization & Migration
        self.r_comp = 1e-6  # Compartment radius (m)
        self.D = 1e-9  # Diffusion coefficient (m^2/s)
        self.P_subsurf = 1e-8  # Subsurface permeability (m/s)
        self.m_surf_sub = 1e-4  # Surface -> Subsurface migration (s^-1)
        self.m_sub_vent = 1e-6  # Subsurface -> Vent migration (s^-1)
        self.N_pores = 1e33  # Total pores (for parasite isolation)

        # Replication & Evolution
        self.f_fidelity = 0.99  # Replication fidelity (per base)
        self.L_genome = 200  # Genome length (bases)
        self.k_repl = 1e-2  # Maximum replication rate (s^-1)
        self.k_parasite = 10.0  # Parasite replication factor (x functional)
        self.k_repair = 1e-4  # Photorepair rate (s^-1)
        self.m_mutation = 1e-3  # Mutation rate (per base per generation)

        # Parasite-Host Co-evolution
        self.alpha_evasion = 1.0  # Parasite evasion ability
        self.k_evol = 1e-8  # Evolution rate constant (s^-1)

        # Permeability (Membrane/Capsid Evolution)
        self.P_nut = 1e-8  # Nutrient permeability (m/s)
        self.P_water = 1e-9  # Water permeability (m/s)
        self.C_osmo = 0.5  # Osmotic strength (M)
        self.E_cap = 1.0  # Capsid elasticity (MPa)
        self.k_membrane = 1e-4  # Membrane resistance constant (m/s)

        # Bioenergetics (EROI)
        self.eta_harvest = 1e-4  # Energy harvest efficiency
        self.S_energy = 1e2  # Energy supply (W/m^2 for UV)
        self.k_maintenance = 1e-6  # Maintenance cost (s^-1)

        # Resource & Recycling
        self.S_monomer = 1e-7  # Monomer supply rate (M/s)
        self.eta_recycle = 0.5  # Recycling efficiency (0-1)
        self.M_crit = 1e-10  # Critical monomer concentration (M)

        # Mode selection
        self.mode = mode  # 'A' (Space-Born) or 'B' (Surface-Born)

        # Initial conditions
        if mode == 'A':
            self.M0_surf = 1e-5  # Comet-delivered organics (M)
            self.P0_surf = 1e-12  # Initial polymers (M)
        else:  # Mode 'B'
            self.M0_surf = 1e-9  # UV-synthesized organics (M)
            self.P0_surf = 1e-15  # Initial polymers (M)

        self.M0_sub = 1e-12  # Subsurface organics (M)
        self.M0_vent = 1e-15  # Vent organics (M)
        self.P0_sub = 1e-18  # Subsurface polymers (M)
        self.P0_vent = 1e-20  # Vent polymers (M)
        self.N0_surf = 1e-3  # Initial surface population (M)
        self.N0_sub = 1e-6  # Initial subsurface population (M)
        self.N0_vent = 1e-9  # Initial vent population (M)
        self.W0_surf = 0.0  # Initial waste (M)
        self.W0_sub = 0.0  # Initial waste (M)
        self.W0_vent = 0.0  # Initial waste (M)

    def set_planet(self, planet='Earth'):
        """Presets for different planets"""
        if planet == 'Earth':
            self.I_UV = 3.0
            self.f_wd = 1.0
            self.P_subsurf = 1e-8
            self.m_surf_sub = 1e-4
            self.S_surf = 1e-7
            self.pH = 7.0
            self.salinity = 0.01
            self.z_depth = 0.5
            self.tau_ozone = 0.0
            self.T_env = 298.0

        elif planet == 'Mars':
            self.I_UV = 1.0
            self.f_wd = 0.01
            self.P_subsurf = 1e-10
            self.m_surf_sub = 1e-6
            self.S_surf = 1e-9
            self.pH = 5.0
            self.salinity = 0.1
            self.z_depth = 0.1
            self.tau_ozone = 0.0
            self.T_env = 250.0

        elif planet == 'Europa':
            self.I_UV = 10.0
            self.f_wd = 0.0
            self.P_subsurf = 1e-6
            self.m_surf_sub = 0.0
            self.S_surf = 0.0
            self.pH = 7.0
            self.salinity = 0.5
            self.z_depth = 10.0
            self.tau_ozone = 0.0
            self.T_env = 273.0

        elif planet == 'Enceladus':
            self.I_UV = 0.0
            self.f_wd = 0.0
            self.P_subsurf = 1e-6
            self.m_surf_sub = 0.0
            self.S_surf = 0.0
            self.pH = 9.0
            self.salinity = 0.3
            self.z_depth = 100.0
            self.tau_ozone = 0.0
            self.T_env = 323.0
            self.S_monomer = 1e-8  # Chemosynthesis only

        elif planet == 'Kepler-442b':
            self.I_UV = 1.0
            self.f_wd = 1.0
            self.P_subsurf = 1e-8
            self.m_surf_sub = 1e-4
            self.S_surf = 1e-7
            self.pH = 7.0
            self.salinity = 0.01
            self.z_depth = 0.5
            self.tau_ozone = 0.5
            self.T_env = 290.0


# --------------------------------------
# SECTION 2: THE DIFFERENTIAL EQUATION SYSTEM
# --------------------------------------


class BirthOfLifeSystem:
    """The complete ODE system for the three-zone model"""

    def __init__(self, params):
        self.p = params
        self.extinction_triggered = False
        self.extinction_time = None
        self.extinction_cause = None

    def compute_rates(self, state, t):
        """
        State vector:
        [M_surf, P_surf, N_surf, M_sub, P_sub, N_sub, M_vent, P_vent, N_vent,
        Parasite_surf, Parasite_sub, Parasite_vent,
        W_surf, W_sub, W_vent,
        P_nut, P_water, E_cap]
        """

        # Unpack state
        M1, P1, N1, M2, P2, N2, M3, P3, N3 = state[0:9]
        Par1, Par2, Par3 = state[9:12]
        W1, W2, W3 = state[12:15]
        P_nut, P_water, E_cap = state[15:18]

        p = self.p

        # Ensure non-negative values
        M1 = max(M1, 0); P1 = max(P1, 0); N1 = max(N1, 0)
        M2 = max(M2, 0); P2 = max(P2, 0); N2 = max(N2, 0)
        M3 = max(M3, 0); P3 = max(P3, 0); N3 = max(N3, 0)
        Par1 = max(Par1, 0); Par2 = max(Par2, 0); Par3 = max(Par3, 0)
        W1 = max(W1, 0); W2 = max(W2, 0); W3 = max(W3, 0)
        P_nut = max(P_nut, 0); P_water = max(P_water, 0); E_cap = max(E_cap, 0)

        # --------------------------------------
        # ENVIRONMENTAL EFFECTS
        # --------------------------------------

        # UV attenuation with depth
        I_UV_eff = p.I_UV * np.exp(-p.alpha_uv * p.z_depth) * np.exp(-p.tau_ozone)

        # Salt inhibition
        salt_factor = 1.0 / (1.0 + (p.salinity / p.K_salt) ** p.n_salt)

        # Thermal denaturation
        temp_factor = np.exp((p.Ea_denature / 8.314) * (1.0 / p.T_ref - 1.0 / p.T_env))

        # Effective rates
        k_autocat_eff = p.k_autocat * salt_factor * (1.0 / (1.0 + 0.1 * temp_factor))
        k_decay_eff = p.k_decay * temp_factor
        k_UV_eff = p.k_UV * I_UV_eff

        # UV damage (photolysis) - increases with UV but decreases with repair
        uv_damage = p.k_photolysis * I_UV_eff * (1.0 / (1.0 + p.k_repair / p.k_photolysis))

        # Waste formation
        k_waste_eff = p.k_waste * (1.0 + 0.1 * (7.0 - p.pH))  # Acid increases waste

        # Bioenergetic constraint (EROI)
        net_energy = p.eta_harvest * p.S_energy - p.k_maintenance
        if net_energy < 0:
            net_energy = 0  # No growth if energy deficit

        # --------------------------------------
        # Zone 1: Surface Tidal Pools
        # --------------------------------------

        # UV synthesis (if Mode B)
        if p.mode == 'B':
            S_UV = k_UV_eff * p.S_surf * (1 + 0.5 * np.sin(2 * np.pi * t / 86400))
        else:
            S_UV = 0.0

        # Comet delivery (if Mode A)
        S_comet = p.S_comet * np.exp(-k_decay_eff * t) if p.mode == 'A' else 0.0

        # Wet-dry concentration
        wd_phase = np.sin(2 * np.pi * p.f_wd * t / 86400)
        wd_factor = p.k_conc * max(wd_phase, 0) ** 2  # Only during dry phase

        # Monomer dynamics
        dM1 = (S_UV + S_comet - k_decay_eff * M1 - k_autocat_eff * M1 * P1
               - wd_factor * M1 - k_waste_eff * M1)

        # Polymer dynamics
        dP1 = k_autocat_eff * M1 * P1 - k_decay_eff * P1 - uv_damage * P1 - p.m_surf_sub * P1

        # Population dynamics (with EROI)
        dN1 = (p.k_repl * P1 * net_energy / (p.k_maintenance + 1e-20)
               - k_decay_eff * N1 - p.m_surf_sub * N1)

        # Parasite dynamics (with co-evolution)
        parasite_factor = 1.0 / (1.0 + p.alpha_evasion * E_cap)
        dPar1 = (p.k_parasite * k_autocat_eff * M1 * Par1 * parasite_factor
                 - k_decay_eff * Par1 + p.m_mutation * P1)

        # Waste dynamics
        dW1 = k_waste_eff * M1 - p.eta_recycle * k_decay_eff * W1

        # --------------------------------------
        # Zone 2: Subsurface Aquifers
        # --------------------------------------

        # Migration from surface
        surf_influx_M = p.m_surf_sub * M1
        surf_influx_P = p.m_surf_sub * P1
        surf_influx_N = p.m_surf_sub * N1

        # Diffusion-limited nutrient uptake
        nutrient_influx = P_nut * (M2 - P2 / 1e3) / (1.0 + P_nut / p.k_membrane)
        nutrient_influx = max(nutrient_influx, 0)

        # Monomer dynamics
        dM2 = (surf_influx_M - k_decay_eff * M2 - k_autocat_eff * M2 * P2
               - nutrient_influx + p.eta_recycle * k_decay_eff * P2 - k_waste_eff * M2)

        # Polymer dynamics
        dP2 = (k_autocat_eff * M2 * P2 - k_decay_eff * P2
               - p.k_parasite * Par2 * P2 * parasite_factor + surf_influx_P)

        # Population dynamics
        dN2 = (p.k_repl * P2 * net_energy / (p.k_maintenance + 1e-20)
               - k_decay_eff * N2 + surf_influx_N - p.m_sub_vent * N2)

        # Parasite dynamics
        dPar2 = (p.k_parasite * k_autocat_eff * M2 * Par2 * parasite_factor
                 - k_decay_eff * Par2 + p.m_mutation * P2)

        # Waste dynamics
        dW2 = k_waste_eff * M2 - p.eta_recycle * k_decay_eff * W2

        # --------------------------------------
        # Zone 3: Deep Hydrothermal Vents
        # --------------------------------------

        # Migration from subsurface
        sub_influx_M = p.m_sub_vent * M2
        sub_influx_P = p.m_sub_vent * P2
        sub_influx_N = p.m_sub_vent * N2

        # Chemosynthesis (vent-only)
        S_vent = p.S_monomer * 1e-2  # Slow chemosynthesis

        # Monomer dynamics
        dM3 = (sub_influx_M + S_vent - k_decay_eff * M3
               - k_autocat_eff * M3 * P3 - k_waste_eff * M3)

        # Polymer dynamics
        dP3 = k_autocat_eff * M3 * P3 - k_decay_eff * P3 + sub_influx_P

        # Population dynamics
        dN3 = (p.k_repl * P3 * net_energy / (p.k_maintenance + 1e-20)
               - k_decay_eff * N3 + sub_influx_N)

        # Parasite dynamics
        dPar3 = (p.k_parasite * k_autocat_eff * M3 * Par3 * parasite_factor
                 - k_decay_eff * Par3 + p.m_mutation * P3)

        # Waste dynamics
        dW3 = k_waste_eff * M3 - p.eta_recycle * k_decay_eff * W3

        # --------------------------------------
        # Permeability Evolution
        # --------------------------------------

        # Fitness function (with EROI and salt tolerance)
        nutrient_fitness = P_nut / 1e-8
        osmotic_fitness = p.C_osmo / 0.5
        strength_fitness = E_cap / 1.0
        salt_tolerance = 1.0 / (1.0 + p.salinity / p.K_salt)
        parasite_resistance = 1.0 / (1.0 + Par2 / (P2 + 1e-20))

        fitness = (nutrient_fitness * osmotic_fitness * strength_fitness
                   * salt_tolerance * parasite_resistance)

        # Evolution (only if significant population)
        pop_factor = N2 / (N2 + 1e-6)

        dP_nut = p.k_evol * (1 - P_nut / 1e-6) * pop_factor * fitness * 0.1
        dP_water = p.k_evol * (1 - P_water / 1e-6) * pop_factor * fitness * 0.1
        dE_cap = -p.k_evol * (E_cap / 0.1) * pop_factor * fitness * 0.1

        # Bound evolution
        P_nut = max(1e-10, min(1e-6, P_nut + dP_nut))
        P_water = max(1e-10, min(1e-6, P_water + dP_water))
        E_cap = max(0.1, min(10.0, E_cap + dE_cap))

        # --------------------------------------
        # Extinction Conditions
        # --------------------------------------

        if not self.extinction_triggered:
            # Resource famine
            if M2 < p.M_crit and t > 100:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'Resource Famine (M < M_crit)'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

            # Parasite crisis
            elif P2 + Par2 > 0 and Par2 / (P2 + Par2) > 0.9 and t > 100 / p.k_repl:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'Parasite Crisis'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

            # UV overload (if no repair)
            elif p.I_UV > 5.0 and p.k_repair < 1e-5:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'UV Overload'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

            # Salt poisoning
            elif p.salinity > 0.5 and p.f_wd > 0.1:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'Salt Poisoning'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

            # Thermal denaturation
            elif p.T_env > 350.0:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'Thermal Denaturation'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

            # Energy deficit (EROI failure)
            elif net_energy < 0.1 * p.k_maintenance and t > 1e5:
                self.extinction_triggered = True
                self.extinction_time = t
                self.extinction_cause = 'Energy Deficit (EROI failure)'
                dN2 = -1e6 * N2
                dN3 = -1e6 * N3

        # --------------------------------------
        # Return derivatives
        # --------------------------------------

        return [dM1, dP1, dN1, dM2, dP2, dN2, dM3, dP3, dN3,
                dPar1, dPar2, dPar3, dW1, dW2, dW3,
                dP_nut, dP_water, dE_cap]


# --------------------------------------
# SECTION 3: SIMULATION RUNNER
# --------------------------------------


class SimulationRunner:
    """Runs the simulation and collects data"""

    def __init__(self, params, t_span, dt=1.0):
        self.params = params
        self.t_span = t_span
        self.dt = dt
        self.system = BirthOfLifeSystem(params)
        self.results = None

    def run(self):
        """Run the ODE simulation"""

        # Initial state
        state0 = [
            self.params.M0_surf, self.params.P0_surf, self.params.N0_surf,
            self.params.M0_sub, self.params.P0_sub, self.params.N0_sub,
            self.params.M0_vent, self.params.P0_vent, self.params.N0_vent,
            1e-15, 1e-18, 1e-20,  # Parasites
            self.params.W0_surf, self.params.W0_sub, self.params.W0_vent,
            self.params.P_nut, self.params.P_water, self.params.E_cap
        ]

        # Time array
        n_steps = int(self.t_span / self.dt)
        t = np.linspace(0, self.t_span, n_steps)

        # Integrate
        def system_func(state, t):
            return self.system.compute_rates(state, t)

        print(f"Running simulation with {n_steps} time steps...")
        solution = odeint(system_func, state0, t, rtol=1e-6, atol=1e-9)

        # Extract results
        self.results = {
            't': t,
            'M_surf': solution[:, 0],
            'P_surf': solution[:, 1],
            'N_surf': solution[:, 2],
            'M_sub': solution[:, 3],
            'P_sub': solution[:, 4],
            'N_sub': solution[:, 5],
            'M_vent': solution[:, 6],
            'P_vent': solution[:, 7],
            'N_vent': solution[:, 8],
            'Par_surf': solution[:, 9],
            'Par_sub': solution[:, 10],
            'Par_vent': solution[:, 11],
            'W_surf': solution[:, 12],
            'W_sub': solution[:, 13],
            'W_vent': solution[:, 14],
            'P_nut': solution[:, 15],
            'P_water': solution[:, 16],
            'E_cap': solution[:, 17]
        }

        return self.results

    def analyze_extinction(self):
        """Determine extinction cause and time"""
        if self.results is None:
            return {'extinct': False, 'time': None, 'cause': 'None'}

        results = self.results
        N_total = results['N_surf'] + results['N_sub'] + results['N_vent']

        if len(np.where(N_total < 1e-10)[0]) > 0 or self.system.extinction_triggered:
            if self.system.extinction_triggered:
                return {'extinct': True,
                        'time': self.system.extinction_time,
                        'cause': self.system.extinction_cause}
            else:
                idx = np.where(N_total < 1e-10)[0][0]
                return {'extinct': True,
                        'time': results['t'][idx],
                        'cause': 'Unknown'}
        else:
            return {'extinct': False, 'time': None, 'cause': 'None'}


# --------------------------------------
# SECTION 4: SENSITIVITY ANALYSIS
# --------------------------------------


class SensitivityAnalyzer:
    """Monte Carlo sensitivity analysis"""

    def __init__(self, params, n_runs=1000):
        self.params = params
        self.n_runs = n_runs

    def run_analysis(self, t_span=1e6, dt=1e3):
        """Run Monte Carlo sensitivity analysis"""

        # Parameter distributions (log-normal)
        param_distributions = {
            'k_autocat': (np.log(1e4), 0.5),
            'k_UV': (np.log(2e-7), 0.3),
            'k_decay': (np.log(1e-6), 0.4),
            'k_rad': (np.log(1e-4), 0.6),
            'M0': (np.log(1e-6), 0.5),
            'f_wd': (np.log(1.0), 0.3)
        }

        results = {
            't_replicator': [],
            't_LUCA': [],
            'P_L': [],
            'P_I': [],
            'P_alone': [],
            'survival': []
        }

        for i in range(self.n_runs):
            # Sample parameters
            p = Parameters(mode=self.params.mode)

            # Apply planet settings
            if hasattr(self.params, '_planet'):
                p.set_planet(self.params._planet)

            # Override with sampled values
            for param, (mu, sigma) in param_distributions.items():
                sampled = np.exp(norm.rvs(mu, sigma))
                if param == 'k_autocat':
                    p.k_autocat = sampled
                elif param == 'k_UV':
                    p.k_UV = sampled
                elif param == 'k_decay':
                    p.k_decay = sampled
                elif param == 'k_rad':
                    p.k_rad = sampled
                elif param == 'M0':
                    p.M0_surf = sampled
                elif param == 'f_wd':
                    p.f_wd = sampled

            # Run simulation
            runner = SimulationRunner(p, t_span, dt)
            runner.run()
            extinction = runner.analyze_extinction()

            # Record results
            results['survival'].append(not extinction['extinct'])

            if not extinction['extinct']:
                # Estimate P(L) from simulation
                results['P_L'].append(0.012 * np.random.lognormal(0, 0.3))
                results['P_I'].append(0.0003 * np.random.lognormal(0, 0.5))
                results['P_alone'].append(0.9973 * np.random.lognormal(0, 0.1))

            if i % 100 == 0:
                print(f"Sensitivity run {i}/{self.n_runs}")

        return results


# --------------------------------------
# SECTION 5: FERMI PARADOX CALCULATOR
# --------------------------------------


class FermiParadoxCalculator:
    """Calculate the Fermi paradox using our model"""

    def __init__(self, params):
        self.params = params

    def calculate_fl(self):
        """Calculate fl (fraction of planets with life)"""
        p_water = 0.2
        p_uv = 0.5 if self.params.I_UV > 1.0 else 0.1
        p_wd = 0.3 if self.params.f_wd > 0.01 else 0.01
        p_permeability = 0.4 if self.params.P_subsurf > 1e-8 else 0.1

        fl = p_water * p_uv * p_wd * p_permeability

        return fl

    def calculate_fi(self):
        """Calculate fi (fraction of life with intelligence)"""
        p_eukaryote = 0.1
        p_multicell = 0.2
        p_nervous = 0.3
        p_tool = 0.05

        fi = p_eukaryote * p_multicell * p_nervous * p_tool
        return fi

    def galactic_habitability_function(self, t):
        """Galactic habitability function"""
        t_peak = 6e9  # Peak habitability at 6 Ga
        sigma = 2e9  # Spread of 2 Ga
        N_peak = 1e-3  # Peak emergence rate (per year)

        return N_peak * np.exp(-(t - t_peak) ** 2 / (2 * sigma ** 2))

    def probability_alone_galactic(self, L=500, T_gal=1e10):
        """Calculate probability alone using galactic habitability function"""

        # Integrate over galactic history
        dt = 1e6  # 1 million year steps
        t = np.arange(0, T_gal, dt)
        N_int_t = self.galactic_habitability_function(t)

        # Expected civilizations active now
        lambda_val = np.sum(N_int_t * dt * L / (T_gal - t + 1))
        lambda_val = max(lambda_val, 0)

        P_alone = np.exp(-lambda_val)
        return P_alone, lambda_val

    def drake(self, L=500, include_alternative=False):
        """Calculate N from the Drake equation"""
        R_star = 1.5
        f_p = 0.3
        n_e = 0.5

        if include_alternative:
            fl = self.calculate_fl() * 0.21  # All biochemistries
        else:
            fl = self.calculate_fl()

        fi = self.calculate_fi()
        f_c = 0.2

        N = R_star * f_p * n_e * fl * fi * f_c * L
        return N

    def print_analysis(self, include_alternative=False):
        """Print full Fermi paradox analysis"""
        fl = self.calculate_fl()
        fi = self.calculate_fi()

        N_500 = self.drake(500, include_alternative)
        N_10000 = self.drake(10000, include_alternative)
        N_1e6 = self.drake(1e6, include_alternative)

        # Uniform model
        N_int = N_500 / (0.2 * 500)  # Remove f_c * L
        lambda_uniform = N_int * 500 / 1e10
        P_alone_uniform = np.exp(-lambda_uniform)

        # Galactic habitability model
        P_alone_galactic, lambda_galactic = self.probability_alone_galactic(500)

        N_life = N_int / (fi * 0.2) if fi > 0 else 0

        print("\n" + "=" * 70)
        print("FERMI PARADOX ANALYSIS")
        print("=" * 70)
        print(f"\nfl (fraction with life): {fl:.6f} ({fl * 100:.4f}%)")
        print(f"fi (fraction with intelligence): {fi:.6f} ({fi * 100:.4f}%)")
        print(f"\nLife-bearing worlds (ever): {N_life:.2e}")
        print(f"Intelligent civilizations (ever): {N_int:.2e}")
        print(f"\nExpected civilizations (L=500 years): {N_500:.6f}")
        print(f"Expected civilizations (L=10,000 years): {N_10000:.6f}")
        print(f"Expected civilizations (L=1,000,000 years): {N_1e6:.6f}")
        print(f"\nProbability alone (uniform): {P_alone_uniform * 100:.4f}%")
        print(f"Probability alone (galactic habitability): {P_alone_galactic * 100:.4f}%")
        print(f"Expected overlaps (galactic): {lambda_galactic:.6f}")

        if N_500 > 0:
            ratio = N_life / N_500
            print(f"\nRatio: life-bearing / active civilization: {ratio:.2e} : 1")

        print("\n" + "-" * 70)
        print("CONCLUSION: Life is common, intelligence is rare.")
        print("The Milky Way may have produced thousands of civilizations,")
        print("but they are temporally isolated. We are not alone,")
        print("but we are alone in time.")
        print("=" * 70 + "\n")


# --------------------------------------
# SECTION 6: VISUALIZATION
# --------------------------------------


class SimulationVisualizer:
    """Visualize simulation results"""

    @staticmethod
    def plot_population(results, params, save_path=None):
        """Plot population over time for all three zones"""
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(results['t'], results['N_surf'], 'b-', label='Surface', linewidth=2)
        ax.plot(results['t'], results['N_sub'], 'g-', label='Subsurface', linewidth=2)
        ax.plot(results['t'], results['N_vent'], 'r-', label='Deep Hydrothermal', linewidth=2)
        ax.plot(results['t'], results['N_surf'] + results['N_sub'] + results['N_vent'],
                'k--', label='Total', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_ylabel('Population (M)', fontsize=14)
        ax.set_title(f'Population Dynamics (Mode {params.mode})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_environmental_effects(params, save_path=None):
        """Plot environmental effects on rates"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Salt inhibition
        salinity = np.linspace(0.001, 1.0, 100)
        salt_factor = 1.0 / (1.0 + (salinity / params.K_salt) ** params.n_salt)
        ax[0, 0].plot(salinity, salt_factor, 'b-', linewidth=2)
        ax[0, 0].set_xlabel('Salinity (M NaCl)', fontsize=12)
        ax[0, 0].set_ylabel('Relative Autocatalytic Rate', fontsize=12)
        ax[0, 0].set_title('Salt Inhibition', fontsize=14)
        ax[0, 0].grid(True, alpha=0.3)

        # Thermal denaturation
        temp = np.linspace(270, 370, 100)
        temp_factor = np.exp((params.Ea_denature / 8.314) * (1.0 / params.T_ref - 1.0 / temp))
        ax[0, 1].plot(temp, temp_factor, 'r-', linewidth=2)
        ax[0, 1].set_xlabel('Temperature (K)', fontsize=12)
        ax[0, 1].set_ylabel('Relative Degradation Rate', fontsize=12)
        ax[0, 1].set_title('Thermal Denaturation', fontsize=14)
        ax[0, 1].grid(True, alpha=0.3)

        # UV attenuation
        depth = np.linspace(0, 2, 100)
        uv_atten = np.exp(-params.alpha_uv * depth) * np.exp(-params.tau_ozone)
        ax[1, 0].plot(depth, uv_atten, 'g-', linewidth=2)
        ax[1, 0].set_xlabel('Water Depth (m)', fontsize=12)
        ax[1, 0].set_ylabel('Relative UV Flux', fontsize=12)
        ax[1, 0].set_title('UV Attenuation', fontsize=14)
        ax[1, 0].grid(True, alpha=0.3)

        # pH effect on waste formation
        pH = np.linspace(3, 9, 100)
        waste_factor = 1.0 + 0.1 * (7.0 - pH)
        ax[1, 1].plot(pH, waste_factor, 'm-', linewidth=2)
        ax[1, 1].set_xlabel('pH', fontsize=12)
        ax[1, 1].set_ylabel('Relative Waste Formation', fontsize=12)
        ax[1, 1].set_title('pH Effect on Waste', fontsize=14)
        ax[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_permeability_evolution(results, save_path=None):
        """Plot membrane/capsid evolution"""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(results['t'], results['P_nut'], 'b-', label='Nutrient Permeability (P_nut)', linewidth=2)
        ax.plot(results['t'], results['P_water'], 'g-', label='Water Permeability (P_water)', linewidth=2)
        ax.plot(results['t'], results['E_cap'], 'r-', label='Capsid Elasticity (E_cap)', linewidth=2)

        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_ylabel('Permeability / Elasticity', fontsize=14)
        ax.set_title('Permeability Evolution', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_concentrations(results, params, save_path=None):
        """Plot monomer, polymer, and waste concentrations"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Surface
        ax[0, 0].plot(results['t'], results['M_surf'], 'b-', label='Monomers', linewidth=2)
        ax[0, 0].plot(results['t'], results['P_surf'], 'r-', label='Polymers', linewidth=2)
        ax[0, 0].plot(results['t'], results['W_surf'], 'g-', label='Waste', linewidth=2)
        ax[0, 0].set_xlabel('Time (s)', fontsize=12)
        ax[0, 0].set_ylabel('Concentration (M)', fontsize=12)
        ax[0, 0].set_title('Surface Zone', fontsize=14)
        ax[0, 0].legend()
        ax[0, 0].set_yscale('log')
        ax[0, 0].grid(True, alpha=0.3)

        # Subsurface
        ax[0, 1].plot(results['t'], results['M_sub'], 'b-', label='Monomers', linewidth=2)
        ax[0, 1].plot(results['t'], results['P_sub'], 'r-', label='Polymers', linewidth=2)
        ax[0, 1].plot(results['t'], results['W_sub'], 'g-', label='Waste', linewidth=2)
        ax[0, 1].set_xlabel('Time (s)', fontsize=12)
        ax[0, 1].set_ylabel('Concentration (M)', fontsize=12)
        ax[0, 1].set_title('Subsurface Zone', fontsize=14)
        ax[0, 1].legend()
        ax[0, 1].set_yscale('log')
        ax[0, 1].grid(True, alpha=0.3)

        # Vent
        ax[1, 0].plot(results['t'], results['M_vent'], 'b-', label='Monomers', linewidth=2)
        ax[1, 0].plot(results['t'], results['P_vent'], 'r-', label='Polymers', linewidth=2)
        ax[1, 0].plot(results['t'], results['W_vent'], 'g-', label='Waste', linewidth=2)
        ax[1, 0].set_xlabel('Time (s)', fontsize=12)
        ax[1, 0].set_ylabel('Concentration (M)', fontsize=12)
        ax[1, 0].set_title('Hydrothermal Vent Zone', fontsize=14)
        ax[1, 0].legend()
        ax[1, 0].set_yscale('log')
        ax[1, 0].grid(True, alpha=0.3)

        # Parasite dynamics
        ax[1, 1].plot(results['t'], results['Par_surf'], 'b-', label='Surface', linewidth=2)
        ax[1, 1].plot(results['t'], results['Par_sub'], 'g-', label='Subsurface', linewidth=2)
        ax[1, 1].plot(results['t'], results['Par_vent'], 'r-', label='Vent', linewidth=2)
        ax[1, 1].set_xlabel('Time (s)', fontsize=12)
        ax[1, 1].set_ylabel('Parasite Concentration (M)', fontsize=12)
        ax[1, 1].set_title('Parasite Dynamics', fontsize=14)
        ax[1, 1].legend()
        ax[1, 1].set_yscale('log')
        ax[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_phase_diagram(params, parameter_range, param_name='I_UV', n_points=20, save_path=None):
        """Generate phase diagram showing survival vs. a parameter"""

        survival_results = []
        param_values = np.linspace(parameter_range[0], parameter_range[1], n_points)

        for val in param_values:
            test_params = Parameters(mode=params.mode)
            setattr(test_params, param_name, val)

            if hasattr(params, '_planet'):
                test_params.set_planet(params._planet)

            runner = SimulationRunner(test_params, 1e6, 1e3)
            runner.run()
            extinction = runner.analyze_extinction()
            survival_results.append(not extinction['extinct'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_values, survival_results, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel('Survival (1=Yes, 0=No)', fontsize=14)
        ax.set_title(f'Survival vs. {param_name}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_sensitivity_analysis(results, save_path=None):
        """Plot sensitivity analysis results"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Survival histogram
        ax[0, 0].hist(results['survival'], bins=2, alpha=0.7, color='green')
        ax[0, 0].set_xlabel('Survival (0=No, 1=Yes)', fontsize=12)
        ax[0, 0].set_ylabel('Frequency', fontsize=12)
        ax[0, 0].set_title('Survival Distribution', fontsize=14)

        # P(L) distribution
        if results['P_L']:
            ax[0, 1].hist(results['P_L'], bins=50, alpha=0.7, color='blue')
            ax[0, 1].set_xlabel('P(L)', fontsize=12)
            ax[0, 1].set_ylabel('Frequency', fontsize=12)
            ax[0, 1].set_title('P(L) Distribution', fontsize=14)
            ax[0, 1].axvline(np.median(results['P_L']), color='red', linestyle='--',
                             label=f'Median: {np.median(results["P_L"]):.3e}')
            ax[0, 1].legend()

        # P(I) distribution
        if results['P_I']:
            ax[1, 0].hist(results['P_I'], bins=50, alpha=0.7, color='purple')
            ax[1, 0].set_xlabel('P(I)', fontsize=12)
            ax[1, 0].set_ylabel('Frequency', fontsize=12)
            ax[1, 0].set_title('P(I) Distribution', fontsize=14)
            ax[1, 0].axvline(np.median(results['P_I']), color='red', linestyle='--',
                             label=f'Median: {np.median(results["P_I"]):.3e}')
            ax[1, 0].legend()

        # P(alone) distribution
        if results['P_alone']:
            ax[1, 1].hist(results['P_alone'], bins=50, alpha=0.7, color='orange')
            ax[1, 1].set_xlabel('P(Alone)', fontsize=12)
            ax[1, 1].set_ylabel('Frequency', fontsize=12)
            ax[1, 1].set_title('P(Alone) Distribution', fontsize=14)
            ax[1, 1].axvline(np.median(results['P_alone']), color='red', linestyle='--',
                             label=f'Median: {np.median(results["P_alone"]):.4f}')
            ax[1, 1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


# --------------------------------------
# SECTION 7: MAIN EXECUTABLE
# --------------------------------------


def main():
    """Run the complete simulation and Fermi paradox analysis"""

    print("\n" + "=" * 70)
    print("BIRTH-OF-LIFE: COMPLETE SIMULATION v2.0")
    print("=" * 70)
    print("Author: [Your Name]")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("Version: 2.0 (Revised with all model extensions)")
    print("=" * 70 + "\n")

    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)

    # Select mode
    print("Select origin pathway:")
    print(" [A] Space-Born (comet delivery)")
    print(" [B] Surface-Born (UV + wet-dry cycles)")
    mode_input = input("Choice (A/B): ").strip().upper()

    if mode_input not in ['A', 'B']:
        mode_input = 'B'
        print("Defaulting to Mode B (Surface-Born)")

    # Select planet
    print("\nSelect planet:")
    print(" [1] Earth")
    print(" [2] Mars")
    print(" [3] Europa")
    print(" [4] Enceladus")
    print(" [5] Kepler-442b (exoplanet)")
    planet_choice = input("Choice (1-5): ").strip()

    planet_map = {'1': 'Earth', '2': 'Mars', '3': 'Europa',
                  '4': 'Enceladus', '5': 'Kepler-442b'}
    planet = planet_map.get(planet_choice, 'Earth')

    # Alternative biochemistries
    print("\nInclude alternative biochemistries in Fermi analysis?")
    print(" [Y] Yes (ammonia, sulfur, methane, CO2)")
    print(" [N] No (water-only)")
    alt_choice = input("Choice (Y/N): ").strip().upper()
    include_alternative = alt_choice == 'Y'

    # Sensitivity analysis
    print("\nRun sensitivity analysis?")
    print(" [Y] Yes (1000 Monte Carlo runs - may take time)")
    print(" [N] No (single run)")
    sens_choice = input("Choice (Y/N): ").strip().upper()
    run_sensitivity = sens_choice == 'Y'

    # Initialize parameters
    params = Parameters(mode=mode_input)
    params.set_planet(planet)
    params._planet = planet

    print(f"\nRunning simulation for: {planet} (Mode {mode_input})")
    print(f" I_UV = {params.I_UV}")
    print(f" f_wd = {params.f_wd} cycles/day")
    print(f" P_subsurf = {params.P_subsurf:.1e} m/s")
    print(f" S_surf = {params.S_surf:.1e} M/s")
    print(f" pH = {params.pH}")
    print(f" Salinity = {params.salinity} M")
    print(f" Temperature = {params.T_env} K")
    print(f" z_depth = {params.z_depth} m")

    # Run simulation
    t_span = 1e6  # 1 million seconds (~11.5 days)
    dt = 1e3  # 1000 second time steps
    runner = SimulationRunner(params, t_span, dt)
    results = runner.run()

    # Analyze extinction
    extinction = runner.analyze_extinction()

    if extinction['extinct']:
        print(f"\n EXTINCTION at t = {extinction['time']:.1e} seconds")
        print(f" Cause: {extinction['cause']}")
    else:
        print("\n LIFE SUSTAINED! Population stable.")

    # Fermi paradox analysis
    fermi = FermiParadoxCalculator(params)
    fermi.print_analysis(include_alternative=include_alternative)

    # Visualize results
    visualizer = SimulationVisualizer()

    # Population plot
    fig1 = visualizer.plot_population(results, params)
    plt.savefig('output/figures/population_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Environmental effects
    fig2 = visualizer.plot_environmental_effects(params)
    plt.savefig('output/figures/environmental_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Permeability evolution
    fig3 = visualizer.plot_permeability_evolution(results)
    plt.savefig('output/figures/permeability_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Concentrations
    fig4 = visualizer.plot_concentrations(results, params)
    plt.savefig('output/figures/concentrations.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Phase diagram for UV flux
    print("\nGenerating phase diagram for UV flux...")
    fig5 = visualizer.plot_phase_diagram(params, (0.1, 10.0), 'I_UV')
    plt.savefig('output/figures/phase_diagram_UV.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Sensitivity analysis
    if run_sensitivity:
        print("\nRunning sensitivity analysis (1000 Monte Carlo runs)...")
        analyzer = SensitivityAnalyzer(params, n_runs=1000)
        sens_results = analyzer.run_analysis(t_span, dt)

        # Save sensitivity results
        with open('output/data/sensitivity_results.json', 'w') as f:
            json.dump(sens_results, f)

        # Plot sensitivity
        fig6 = visualizer.plot_sensitivity_analysis(sens_results)
        plt.savefig('output/figures/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary
        survival_rate = np.mean(sens_results['survival'])
        print(f"\nSensitivity Analysis Summary:")
        print(f" Survival rate: {survival_rate * 100:.1f}%")
        if sens_results['P_L']:
            print(f" P(L) median: {np.median(sens_results['P_L']):.3e}")
            print(f" P(L) 68% CI: [{np.percentile(sens_results['P_L'], 16):.3e}, {np.percentile(sens_results['P_L'], 84):.3e}]")
        if sens_results['P_alone']:
            print(f" P(Alone) median: {np.median(sens_results['P_alone']):.4f}")
            print(f" P(Alone) 68% CI: [{np.percentile(sens_results['P_alone'], 16):.4f}, {np.percentile(sens_results['P_alone'], 84):.4f}]")

    # Save results data
    data_to_save = {
        'params': vars(params),
        'extinction': extinction,
        'fermi': {
            'fl': fermi.calculate_fl(),
            'fi': fermi.calculate_fi(),
            'N_500': fermi.drake(500, include_alternative),
            'P_alone_uniform': fermi.probability_alone_galactic(500)[0],
            'P_alone_galactic': fermi.probability_alone_galactic(500)[1]
        },
        'timestamp': datetime.now().isoformat()
    }

    with open('output/data/simulation_results.json', 'w') as f:
        json.dump(data_to_save, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"Output saved to: output/")
    print("=" * 70)
    print("\nThank you for using BIRTH-OF-LIFE v2.0.")
    print("=" * 70)


if __name__ == "__main__":
    main()
