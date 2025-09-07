#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 2: Mathematics
Example 3: State Vectors and Quantum State Representations

This script explores quantum state vectors, Bloch sphere representations,
and different ways to describe quantum states mathematically.

Author: Quantum Computing 101 Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from pathlib import Path

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
import warnings

warnings.filterwarnings("ignore")


class QuantumStateAnalyzer:
    """Comprehensive quantum state vector analysis and visualization."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.states_analyzed = []

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[StateAnalysis] {message}")

    def create_qubit_states(self):
        """Create and analyze various single-qubit states."""
        print("\n=== Single-Qubit State Vectors ===")

        # Computational basis states
        state_0 = Statevector([1, 0])  # |0⟩
        state_1 = Statevector([0, 1])  # |1⟩

        # Superposition states
        state_plus = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])  # |+⟩
        state_minus = Statevector([1 / np.sqrt(2), -1 / np.sqrt(2)])  # |-⟩

        # General parameterized state
        theta = np.pi / 3
        phi = np.pi / 4
        state_general = Statevector(
            [np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)]
        )

        states = {
            "|0⟩": state_0,
            "|1⟩": state_1,
            "|+⟩": state_plus,
            "|-⟩": state_minus,
            "General": state_general,
        }

        # Analyze each state
        for name, state in states.items():
            print(f"\nState {name}:")
            self._analyze_single_state(state)
            self.states_analyzed.append((name, state))

        return states

    def _analyze_single_state(self, state):
        """Analyze properties of a single quantum state."""
        # Get state vector
        vector = state.data

        print(f"  State Vector: {vector}")
        print(f"  Amplitudes: α = {vector[0]:.4f}, β = {vector[1]:.4f}")

        # Probabilities
        prob_0 = abs(vector[0]) ** 2
        prob_1 = abs(vector[1]) ** 2
        print(f"  Probabilities: P(|0⟩) = {prob_0:.4f}, P(|1⟩) = {prob_1:.4f}")

        # Normalization check
        norm = np.linalg.norm(vector)
        print(f"  Normalization: ||ψ|| = {norm:.6f}")

        # Bloch sphere coordinates
        bloch_coords = self._state_to_bloch(vector)
        print(
            f"  Bloch coordinates: (x={bloch_coords[0]:.4f}, y={bloch_coords[1]:.4f}, z={bloch_coords[2]:.4f})"
        )

        if self.verbose:
            # Phase information
            if abs(vector[1]) > 1e-10:  # Avoid division by zero
                relative_phase = (
                    np.angle(vector[1] / vector[0])
                    if abs(vector[0]) > 1e-10
                    else np.angle(vector[1])
                )
                print(
                    f"  Relative Phase: {relative_phase:.4f} rad ({np.degrees(relative_phase):.2f}°)"
                )

    def _state_to_bloch(self, state_vector):
        """Convert state vector to Bloch sphere coordinates."""
        α, β = state_vector[0], state_vector[1]

        # Calculate Bloch coordinates
        x = 2 * np.real(α * np.conj(β))
        y = 2 * np.imag(α * np.conj(β))
        z = abs(α) ** 2 - abs(β) ** 2

        return np.array([x, y, z])

    def create_multiqubit_states(self):
        """Create and analyze multi-qubit state vectors."""
        print("\n=== Multi-Qubit State Vectors ===")

        # Two-qubit product states
        state_00 = Statevector([1, 0, 0, 0])  # |00⟩
        state_01 = Statevector([0, 1, 0, 0])  # |01⟩
        state_10 = Statevector([0, 0, 1, 0])  # |10⟩
        state_11 = Statevector([0, 0, 0, 1])  # |11⟩

        # Bell states (maximally entangled)
        bell_phi_plus = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])  # |Φ+⟩
        bell_phi_minus = Statevector([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)])  # |Φ-⟩
        bell_psi_plus = Statevector([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])  # |Ψ+⟩
        bell_psi_minus = Statevector([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])  # |Ψ-⟩

        # Partially entangled state
        partial_entangled = Statevector([0.6, 0.8j, 0, 0])
        partial_entangled = partial_entangled / partial_entangled.norm()

        states_2q = {
            "|00⟩": state_00,
            "|01⟩": state_01,
            "|10⟩": state_10,
            "|11⟩": state_11,
            "|Φ+⟩": bell_phi_plus,
            "|Φ-⟩": bell_phi_minus,
            "|Ψ+⟩": bell_psi_plus,
            "|Ψ-⟩": bell_psi_minus,
            "Partial": partial_entangled,
        }

        # Analyze each state
        for name, state in states_2q.items():
            print(f"\nTwo-qubit state {name}:")
            self._analyze_two_qubit_state(state)

        return states_2q

    def _analyze_two_qubit_state(self, state):
        """Analyze properties of a two-qubit quantum state."""
        vector = state.data

        print(f"  State Vector: {vector}")

        # Computational basis probabilities
        probs = [abs(amp) ** 2 for amp in vector]
        basis_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

        print("  Measurement Probabilities:")
        for i, (basis, prob) in enumerate(zip(basis_states, probs)):
            print(f"    P({basis}) = {prob:.4f}")

        # Entanglement analysis using Schmidt decomposition
        entanglement = self._calculate_entanglement(vector)
        print(f"  Entanglement (von Neumann entropy): {entanglement:.4f}")

        if entanglement > 0.1:
            print("    → This state is entangled!")
        else:
            print("    → This state is separable (product state)")

    def _calculate_entanglement(self, state_vector):
        """Calculate entanglement using von Neumann entropy of reduced state."""
        # Reshape to 2x2 matrix for 2-qubit system
        state_matrix = state_vector.reshape(2, 2)

        # Calculate reduced density matrix for first qubit
        rho_reduced = np.dot(state_matrix, state_matrix.conj().T)

        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(rho_reduced)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros

        # Calculate von Neumann entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))

        return entropy

    def demonstrate_state_representations(self):
        """Show different mathematical representations of quantum states."""
        print("\n=== State Representation Methods ===")

        # Create a general single-qubit state
        theta = 2 * np.pi / 3
        phi = np.pi / 2
        state = Statevector([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])

        print(f"Example state with θ = {theta:.4f}, φ = {phi:.4f}")

        # 1. State vector representation
        print("\n1. State Vector |ψ⟩:")
        vector = state.data
        print(f"   |ψ⟩ = {vector[0]:.4f}|0⟩ + {vector[1]:.4f}|1⟩")

        # 2. Density matrix representation
        print("\n2. Density Matrix ρ = |ψ⟩⟨ψ|:")
        density_matrix = DensityMatrix(state)
        rho = density_matrix.data
        print(f"   ρ = {rho}")

        # 3. Bloch sphere representation
        print("\n3. Bloch Sphere Representation:")
        bloch_coords = self._state_to_bloch(vector)
        print(
            f"   r⃗ = ({bloch_coords[0]:.4f}, {bloch_coords[1]:.4f}, {bloch_coords[2]:.4f})"
        )

        # 4. Spherical coordinates
        print("\n4. Spherical Coordinates:")
        r = np.linalg.norm(bloch_coords)
        theta_bloch = np.arccos(bloch_coords[2] / r) if r > 1e-10 else 0
        phi_bloch = np.arctan2(bloch_coords[1], bloch_coords[0])
        print(f"   r = {r:.4f}, θ = {theta_bloch:.4f} rad, φ = {phi_bloch:.4f} rad")

        return state, density_matrix, bloch_coords

    def visualize_bloch_sphere(self, states_dict):
        """Create 3D visualization of states on the Bloch sphere."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color="lightblue")

        # Draw coordinate axes
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], "k-", alpha=0.3)

        # Plot states
        colors = ["red", "blue", "green", "orange", "purple"]
        for i, (name, state) in enumerate(states_dict.items()):
            if len(state.data) == 2:  # Single-qubit states only
                bloch_coords = self._state_to_bloch(state.data)
                x, y, z = bloch_coords

                ax.scatter(x, y, z, color=colors[i % len(colors)], s=100, label=name)
                ax.plot(
                    [0, x], [0, y], [0, z], color=colors[i % len(colors)], alpha=0.7
                )

        # Labels and formatting
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Quantum States on the Bloch Sphere")
        ax.legend()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)

        plt.tight_layout()
        plt.show()

    def visualize_state_evolution(self):
        """Visualize quantum state evolution through gates."""
        print("\n=== State Evolution Visualization ===")

        # Create circuit with state evolution
        qc = QuantumCircuit(1)

        # Start with |0⟩ and apply various gates
        initial_state = Statevector([1, 0])
        states_evolution = [("Initial |0⟩", initial_state)]

        # Apply X gate
        qc.x(0)
        state_after_x = initial_state.evolve(qc)
        states_evolution.append(("After X", state_after_x))

        # Apply Hadamard
        qc.h(0)
        state_after_h = initial_state.evolve(qc)
        states_evolution.append(("After X,H", state_after_h))

        # Apply phase gate
        qc.s(0)
        state_after_s = initial_state.evolve(qc)
        states_evolution.append(("After X,H,S", state_after_s))

        # Analyze evolution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, (name, state) in enumerate(states_evolution):
            if i < 4:  # Only plot first 4 states
                vector = state.data

                # Amplitude plot
                ax = axes[i]
                x_pos = [0, 1]
                amplitudes_real = [np.real(vector[0]), np.real(vector[1])]
                amplitudes_imag = [np.imag(vector[0]), np.imag(vector[1])]

                width = 0.35
                ax.bar(
                    [x - width / 2 for x in x_pos],
                    amplitudes_real,
                    width,
                    label="Real",
                    alpha=0.7,
                    color="blue",
                )
                ax.bar(
                    [x + width / 2 for x in x_pos],
                    amplitudes_imag,
                    width,
                    label="Imaginary",
                    alpha=0.7,
                    color="red",
                )

                ax.set_xlabel("Basis State")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"{name}\n{vector}")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(["|0⟩", "|1⟩"])
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return states_evolution

    def generate_summary_report(self):
        """Generate a comprehensive summary of the analysis."""
        print("\n" + "=" * 60)
        print("QUANTUM STATE VECTORS - ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 States Analyzed: {len(self.states_analyzed)}")

        print("\n🔬 Key Concepts Demonstrated:")
        print("  • State vector representation in computational basis")
        print("  • Amplitude and probability relationships")
        print("  • Normalization requirements")
        print("  • Bloch sphere coordinates")
        print("  • Multi-qubit state vectors")
        print("  • Entanglement quantification")
        print("  • Different state representations")
        print("  • State evolution through quantum gates")

        print("\n📚 Mathematical Insights:")
        print("  • |ψ⟩ = α|0⟩ + β|1⟩ with |α|² + |β|² = 1")
        print("  • Bloch vector: (2Re(αβ*), 2Im(αβ*), |α|² - |β|²)")
        print("  • Entanglement measured by von Neumann entropy")
        print("  • Density matrices provide complete state description")

        print("\n🎯 Learning Outcomes:")
        print("  ✓ Understanding quantum state vector mathematics")
        print("  ✓ Visualization of states on Bloch sphere")
        print("  ✓ Analysis of entanglement in multi-qubit systems")
        print("  ✓ Multiple representations of quantum states")

        print("\n🚀 Next Steps:")
        print("  → Explore inner products and orthogonality")
        print("  → Study tensor products in detail")
        print("  → Investigate quantum state tomography")
        print("  → Advanced entanglement measures")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Quantum State Vectors and Representations Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_state_vectors_representations.py
  python 03_state_vectors_representations.py --verbose
  python 03_state_vectors_representations.py --show-evolution
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed calculations",
    )
    parser.add_argument(
        "--show-bloch", action="store_true", help="Display Bloch sphere visualization"
    )
    parser.add_argument(
        "--show-evolution",
        action="store_true",
        help="Display state evolution visualization",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis without visualizations",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 2: Mathematics")
    print("Example 3: State Vectors and Quantum State Representations")
    print("=" * 65)

    # Initialize analyzer
    analyzer = QuantumStateAnalyzer(verbose=args.verbose)

    try:
        # Analyze single-qubit states
        single_qubit_states = analyzer.create_qubit_states()

        # Analyze multi-qubit states
        multi_qubit_states = analyzer.create_multiqubit_states()

        # Demonstrate different representations
        state, density_matrix, bloch_coords = (
            analyzer.demonstrate_state_representations()
        )

        # Visualizations (optional)
        if not args.analysis_only:
            if args.show_bloch:
                analyzer.visualize_bloch_sphere(single_qubit_states)

            if args.show_evolution:
                evolution_states = analyzer.visualize_state_evolution()

        # Generate summary
        analyzer.generate_summary_report()

        print(f"\n✅ Analysis completed successfully!")
        print(
            f"📊 Analyzed {len(single_qubit_states)} single-qubit and {len(multi_qubit_states)} two-qubit states"
        )

        if args.verbose:
            print(f"\n🔍 Detailed mathematical analysis enabled")
            print(f"📈 Use --show-bloch and --show-evolution for visualizations")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
