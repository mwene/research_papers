"""
EVOLUTION-OF-LIFE: Complete Simulation
Author: [Your Name]
Date: 2026-09-03
Version: 1.0

This simulation implements the predictive adaptation framework
for the evolution of life, as described in the paper.

USAGE:
python evolution_of_life.py

DEPENDENCIES:
numpy, scipy, matplotlib
"""

import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.signal import periodogram


# ----------------------------------------------------------------------
# SECTION 1: PARAMETERS
# ----------------------------------------------------------------------


class Parameters:
    """All tunable parameters for the simulation"""

    def __init__(self):
        # Environmental
        self.env_regime = 'periodic'  # stable, periodic, chaotic, trending
        self.env_complexity = 0.5  # Number of environmental variables
        self.env_noise = 0.1  # Noise amplitude

        # Predictive Adaptation
        self.alpha = 0.1  # Prediction error penalty
        self.beta = 0.1  # Homeostatic deviation penalty
        self.gamma = 0.01  # Energy efficiency reward

        # Evolution Rates
        self.k_learn = 1e-4  # Learning rate
        self.k_mutate = 1e-3  # Mutation rate
        self.k_sense = 1e-5  # Sensing evolution rate
        self.k_mem = 1e-5  # Memory evolution rate
        self.k_horizon = 1e-4  # Horizon evolution rate
        self.k_homeo = 1e-4  # Homeostatic regulation rate

        # Energy
        self.energy_harvest = 1.0  # Energy harvest rate (J/s)
        self.energy_cost_per_bit = 1e-6  # Energy per bit of information
        self.max_energy = 100.0  # Maximum energy available

        # Initial Conditions
        self.M0 = np.zeros(5)  # Initial model parameters
        self.S0 = 0.1  # Initial sensing acuity
        self.C0 = 0.1  # Initial memory capacity
        self.T0 = 100.0  # Initial prediction horizon (s)
        self.H0 = 0.5  # Initial homeostatic state

        # Simulation
        self.dt = 1.0  # Time step (s)
        self.t_span = 1e6  # Total simulation time (s)

    def set_regime(self, regime):
        """Set environmental regime"""
        self.env_regime = regime

        if regime == 'stable':
            self.env_noise = 0.01
            self.env_complexity = 1
        elif regime == 'periodic':
            self.env_noise = 0.1
            self.env_complexity = 3
        elif regime == 'chaotic':
            self.env_noise = 0.5
            self.env_complexity = 5
        elif regime == 'trending':
            self.env_noise = 0.05
            self.env_complexity = 4


# ----------------------------------------------------------------------
# SECTION 2: ENVIRONMENT
# ----------------------------------------------------------------------


class Environment:
    """Environmental dynamics"""

    def __init__(self, params):
        self.params = params
        self.t = 0

    def get_state(self, t):
        """Return environmental state at time t"""
        n_vars = int(self.params.env_complexity)

        # Base states
        if self.params.env_regime == 'stable':
            states = np.ones(n_vars) * 0.5
            noise = np.random.randn(n_vars) * 0.01

        elif self.params.env_regime == 'periodic':
            # Daily cycles
            daily = np.sin(2 * np.pi * t / 86400)
            seasonal = np.sin(2 * np.pi * t / 3.15e7)

            states = np.array([
                0.5 + 0.3 * daily + 0.2 * seasonal,  # Temperature
                0.5 + 0.2 * daily,  # pH
                0.5 + 0.1 * daily + 0.1 * seasonal,  # Salinity
                0.5 + 0.4 * daily,  # UV
                0.5 + 0.3 * seasonal  # Nutrients
            ])[:n_vars]

            noise = np.random.randn(n_vars) * 0.1

        elif self.params.env_regime == 'chaotic':
            # Ornstein-Uhlenbeck process
            states = 0.5 + 0.5 * np.random.randn(n_vars)
            noise = np.random.randn(n_vars) * 0.5

        elif self.params.env_regime == 'trending':
            # Gradual change over time
            trend = t / 1e9  # Over billions of seconds
            states = 0.5 + 0.5 * trend * np.ones(n_vars)
            noise = np.random.randn(n_vars) * 0.05

        else:
            states = np.ones(n_vars) * 0.5
            noise = np.random.randn(n_vars) * 0.01

        # Ensure bounded
        states = np.clip(states + noise, 0, 1)

        return states


# ----------------------------------------------------------------------
# SECTION 3: PREDICTIVE ADAPTATION SYSTEM
# ----------------------------------------------------------------------


class PredictiveAdaptationSystem:
    """Implements the predictive adaptation dynamics"""

    def __init__(self, params):
        self.params = params
        self.env = Environment(params)
        self.t = 0
        self.history = []

    def compute_rates(self, state, t):
        """
        State vector:
        [M0, M1, M2, M3, M4, Sensing, Memory, Horizon, Homeostatic]
        """

        # Unpack state
        M = state[0:5]
        S = state[5]
        C = state[6]
        T_pred = state[7]
        H = state[8]

        p = self.params
        self.t = t

        # Get environmental state
        E = self.env.get_state(t)
        E_future = self.env.get_state(t + T_pred)

        # Generate prediction
        omega1 = 2 * np.pi / 86400  # Daily
        omega2 = 2 * np.pi / 3.15e7  # Seasonal
        omega3 = 2 * np.pi / 1e6  # Random

        P = (M[0] + M[1] * t +
             M[2] * np.sin(omega1 * t) +
             M[3] * np.sin(omega2 * t) +
             M[4] * np.cos(omega3 * t))

        # Prediction error (scalar)
        epsilon = abs(E_future.mean() - P)

        # Homeostatic deviation
        H_dev = abs(H - 0.5)

        # Energy cost
        E_cost = (S * 0.1 + C * 0.01 + T_pred * 1e-6 +
                  np.sum(np.abs(M)) * 0.01)

        # Energy harvest
        E_harvest = p.energy_harvest * (1.0 - epsilon / (1.0 + epsilon))

        # Energy efficiency
        if E_cost > 0:
            energy_efficiency = E_harvest / E_cost
        else:
            energy_efficiency = 0

        # Information bottleneck
        I_model = np.sum(np.abs(M))
        I_bottleneck = E_harvest / p.energy_cost_per_bit

        if I_model > I_bottleneck:
            info_penalty = (I_model - I_bottleneck) / I_bottleneck
        else:
            info_penalty = 0

        # Fitness
        fitness = (-p.alpha * epsilon ** 2 -
                   p.beta * H_dev ** 2 +
                   p.gamma * energy_efficiency -
                   info_penalty * 0.1)

        # Store history
        self.history.append({
            't': t,
            'E': E.mean(),
            'P': P,
            'epsilon': epsilon,
            'fitness': fitness,
            'S': S,
            'C': C,
            'T_pred': T_pred,
            'H': H
        })
        if len(self.history) > 10000:
            self.history.pop(0)

        # Derivative of fitness with respect to prediction error
        dF_deps = -2 * p.alpha * epsilon

        # Derivative of epsilon with respect to model parameters
        deps_dM = np.array([
            1.0 / (1.0 + epsilon),  # M0
            t / (1.0 + epsilon),  # M1
            np.sin(omega1 * t) / (1.0 + epsilon),  # M2
            np.sin(omega2 * t) / (1.0 + epsilon),  # M3
            np.cos(omega3 * t) / (1.0 + epsilon)  # M4
        ])

        # Model evolution
        dM = p.k_learn * dF_deps * deps_dM + p.k_mutate * np.random.randn(5) * 0.01

        # Sensing evolution
        dS = p.k_sense * (-2 * p.alpha * epsilon * (-1.0 / (1.0 + epsilon) ** 2)) - 0.001 * S

        # Memory evolution
        dC = p.k_mem * (-2 * p.alpha * epsilon * (-1.0 / (1.0 + epsilon) ** 2)) - 0.001 * C

        # Horizon evolution
        dT_pred = p.k_horizon * (-2 * p.alpha * epsilon * (-1.0 / (1.0 + epsilon) ** 2)) - 0.01 * T_pred

        # Homeostatic regulation
        dH = -p.k_homeo * (H - 0.5) + 0.1 * (1.0 - H_dev)

        return np.concatenate([dM, [dS, dC, dT_pred, dH]])


# ----------------------------------------------------------------------
# SECTION 4: SIMULATION RUNNER
# ----------------------------------------------------------------------


class SimulationRunner:
    """Runs the predictive adaptation simulation"""

    def __init__(self, params):
        self.params = params
        self.system = PredictiveAdaptationSystem(params)
        self.results = None

    def run(self):
        """Run the simulation"""

        # Initial state
        state0 = np.concatenate([
            self.params.M0,
            [self.params.S0, self.params.C0, self.params.T0, self.params.H0]
        ])

        # Time array
        t = np.arange(0, self.params.t_span, self.params.dt)

        # Integrate
        def system_func(state, t):
            return self.system.compute_rates(state, t)

        print(f"Running predictive adaptation simulation with {len(t)} time steps...")
        solution = odeint(system_func, state0, t, rtol=1e-6, atol=1e-9)

        # Extract results
        self.results = {
            't': t,
            'M': solution[:, 0:5],
            'S': solution[:, 5],
            'C': solution[:, 6],
            'T_pred': solution[:, 7],
            'H': solution[:, 8],
            'history': self.system.history
        }

        return self.results


# ----------------------------------------------------------------------
# SECTION 5: VISUALIZATION
# ----------------------------------------------------------------------


class Visualizer:
    """Visualize simulation results"""

    @staticmethod
    def plot_prediction(results, save_path=None):
        """Plot prediction vs. actual environment"""
        history = results['history']
        if not history:
            print("No history available")
            return None

        # Extract data
        t = [h['t'] for h in history]
        E_actual = [h['E'] for h in history]
        P_pred = [h['P'] for h in history]
        eps = [h['epsilon'] for h in history]

        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

        # Environment
        ax[0].plot(t, E_actual, 'b-', label='Actual', linewidth=2)
        ax[0].plot(t, P_pred, 'r--', label='Predicted', linewidth=2)
        ax[0].set_xlabel('Time (s)', fontsize=12)
        ax[0].set_ylabel('Environment', fontsize=12)
        ax[0].set_title('Prediction vs. Actual', fontsize=14)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Prediction error
        ax[1].plot(t, eps, 'g-', linewidth=2)
        ax[1].set_xlabel('Time (s)', fontsize=12)
        ax[1].set_ylabel('Prediction Error', fontsize=12)
        ax[1].set_title('Prediction Error Over Time', fontsize=14)
        ax[1].grid(True, alpha=0.3)
        ax[1].set_yscale('log')

        # Fitness
        fitness = [h['fitness'] for h in history]
        ax[2].plot(t, fitness, 'm-', linewidth=2)
        ax[2].set_xlabel('Time (s)', fontsize=12)
        ax[2].set_ylabel('Fitness', fontsize=12)
        ax[2].set_title('Fitness Over Time', fontsize=14)
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_evolution(results, save_path=None):
        """Plot evolution of predictive variables"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Sensing
        ax[0, 0].plot(results['t'], results['S'], 'b-', linewidth=2)
        ax[0, 0].set_xlabel('Time (s)', fontsize=12)
        ax[0, 0].set_ylabel('Sensing Acuity', fontsize=12)
        ax[0, 0].set_title('Sensing Evolution', fontsize=14)
        ax[0, 0].grid(True, alpha=0.3)
        ax[0, 0].set_ylim(0, 1.1)

        # Memory
        ax[0, 1].plot(results['t'], results['C'], 'r-', linewidth=2)
        ax[0, 1].set_xlabel('Time (s)', fontsize=12)
        ax[0, 1].set_ylabel('Memory Capacity', fontsize=12)
        ax[0, 1].set_title('Memory Evolution', fontsize=14)
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].set_ylim(0, 1.1)

        # Prediction horizon
        ax[1, 0].plot(results['t'], results['T_pred'], 'g-', linewidth=2)
        ax[1, 0].set_xlabel('Time (s)', fontsize=12)
        ax[1, 0].set_ylabel('Prediction Horizon (s)', fontsize=12)
        ax[1, 0].set_title('Prediction Horizon Evolution', fontsize=14)
        ax[1, 0].grid(True, alpha=0.3)
        ax[1, 0].set_yscale('log')

        # Homeostatic state
        ax[1, 1].plot(results['t'], results['H'], 'm-', linewidth=2)
        ax[1, 1].set_xlabel('Time (s)', fontsize=12)
        ax[1, 1].set_ylabel('Homeostatic State', fontsize=12)
        ax[1, 1].set_title('Homeostatic Regulation', fontsize=14)
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].set_ylim(0, 1.1)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_phase_space(results, save_path=None):
        """Plot phase space of predictive adaptation"""
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(results['S'], results['C'],
                             c=results['t'], cmap='viridis',
                             s=5, alpha=0.5)

        ax.set_xlabel('Sensing Acuity', fontsize=14)
        ax.set_ylabel('Memory Capacity', fontsize=14)
        ax.set_title('Phase Space: Sensing vs. Memory', fontsize=16)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (s)', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


# ----------------------------------------------------------------------
# SECTION 6: ANALYSIS
# ----------------------------------------------------------------------


class Analyzer:
    """Analyze simulation results"""

    @staticmethod
    def calculate_intelligence(results, threshold=0.5):
        """Calculate intelligence metric"""
        # Intelligence is measured as predictive capacity
        history = results['history']
        if not history:
            return None

        final_eps = history[-1]['epsilon']
        final_S = results['S'][-1]
        final_C = results['C'][-1]
        final_T = results['T_pred'][-1]

        # Intelligence score (higher is better)
        intelligence = (1.0 / (1.0 + final_eps)) * (0.5 + 0.5 * final_S) * (0.5 + 0.5 * final_C)

        # Check if intelligence threshold is reached
        reached = intelligence > threshold

        return {
            'intelligence': intelligence,
            'threshold_reached': reached,
            'final_epsilon': final_eps,
            'final_sensing': final_S,
            'final_memory': final_C,
            'final_horizon': final_T
        }


# ----------------------------------------------------------------------
# SECTION 7: MAIN EXECUTABLE
# ----------------------------------------------------------------------


def main():
    """Run the complete simulation"""

    print("\n" + "=" * 70)
    print("EVOLUTION-OF-LIFE: PREDICTIVE ADAPTATION SIMULATION")
    print("=" * 70)
    print("Author: [Your Name]")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70 + "\n")

    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)

    # Select environmental regime
    print("Select environmental regime:")
    print(" [1] Stable (deep ocean)")
    print(" [2] Periodic (tidal pools)")
    print(" [3] Chaotic (early Earth)")
    print(" [4] Trending (climate change)")
    regime_choice = input("Choice (1-4): ").strip()

    regime_map = {'1': 'stable', '2': 'periodic',
                  '3': 'chaotic', '4': 'trending'}
    regime = regime_map.get(regime_choice, 'periodic')

    print(f"\nRunning predictive adaptation simulation...")
    print(f" Regime: {regime}")

    # Initialize parameters
    params = Parameters()
    params.set_regime(regime)

    # Run simulation
    runner = SimulationRunner(params)
    results = runner.run()

    # Analyze
    analyzer = Analyzer()
    intelligence = analyzer.calculate_intelligence(results)

    if intelligence:
        print(f"\nIntelligence Analysis:")
        print(f" Intelligence score: {intelligence['intelligence']:.4f}")
        print(f" Threshold reached: {intelligence['threshold_reached']}")
        print(f" Final prediction error: {intelligence['final_epsilon']:.4f}")
        print(f" Final sensing acuity: {intelligence['final_sensing']:.4f}")
        print(f" Final memory capacity: {intelligence['final_memory']:.4f}")

        # Visualize
        visualizer = Visualizer()

        fig1 = visualizer.plot_prediction(results)
        if fig1:
            plt.savefig('output/figures/prediction_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        fig2 = visualizer.plot_evolution(results)
        plt.savefig('output/figures/evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

        fig3 = visualizer.plot_phase_space(results)
        plt.savefig('output/figures/phase_space.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print(f"Output saved to: output/")
        print("=" * 70)
        print("\nThank you for using EVOLUTION-OF-LIFE.")
        print("=" * 70)


if __name__ == "__main__":
    main()
