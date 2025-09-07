#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 2: Mathematics
Example 4: Inner Products and Orthogonality

This script explores inner products between quantum states, orthogonal bases,
and the mathematical foundations of quantum measurement.

Author: Quantum Computing 101 Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
from qiskit_aer import AerSimulator
import warnings

warnings.filterwarnings("ignore")


class InnerProductAnalyzer:
    """Comprehensive analysis of inner products and orthogonality in quantum mechanics."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.computed_products = []

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[InnerProduct] {message}")

    def basic_inner_products(self):
        """Demonstrate basic inner product calculations."""
        print("\n=== Basic Inner Product Calculations ===")

        # Define fundamental states
        state_0 = np.array([1, 0], dtype=complex)
        state_1 = np.array([0, 1], dtype=complex)
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        state_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

        states = {"|0⟩": state_0, "|1⟩": state_1, "|+⟩": state_plus, "|-⟩": state_minus}

        print("Inner products between fundamental states:")
        print("⟨ψ|φ⟩ = ψ*ᵀ · φ")
        print()

        # Calculate all pairwise inner products
        state_names = list(states.keys())
        for i, name1 in enumerate(state_names):
            for j, name2 in enumerate(state_names):
                psi = states[name1]
                phi = states[name2]

                # Calculate inner product ⟨ψ|φ⟩
                inner_product = np.vdot(psi, phi)
                self.computed_products.append((name1, name2, inner_product))

                print(f"⟨{name1}|{name2}⟩ = {inner_product:.4f}")

                if self.verbose:
                    # Show detailed calculation
                    print(f"  Calculation: {np.conj(psi)} · {phi}")
                    print(
                        f"  = {np.conj(psi[0]):.4f} × {phi[0]:.4f} + {np.conj(psi[1]):.4f} × {phi[1]:.4f}"
                    )
                    print(f"  = {inner_product:.4f}")

        return states

    def orthogonality_analysis(self):
        """Analyze orthogonal relationships between quantum states."""
        print("\n=== Orthogonality Analysis ===")

        # Computational basis - orthogonal
        print("1. Computational Basis {|0⟩, |1⟩}:")
        state_0 = np.array([1, 0], dtype=complex)
        state_1 = np.array([0, 1], dtype=complex)

        inner_01 = np.vdot(state_0, state_1)
        print(f"   ⟨0|1⟩ = {inner_01:.6f}")
        print(f"   → {'Orthogonal' if abs(inner_01) < 1e-10 else 'Not orthogonal'}")

        # Hadamard basis - orthogonal
        print("\n2. Hadamard Basis {|+⟩, |-⟩}:")
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        state_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

        inner_pm = np.vdot(state_plus, state_minus)
        print(f"   ⟨+|-⟩ = {inner_pm:.6f}")
        print(f"   → {'Orthogonal' if abs(inner_pm) < 1e-10 else 'Not orthogonal'}")

        # Circular basis - orthogonal
        print("\n3. Circular Basis {|R⟩, |L⟩}:")
        state_R = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # Right circular
        state_L = np.array([1, -1j], dtype=complex) / np.sqrt(2)  # Left circular

        inner_RL = np.vdot(state_R, state_L)
        print(f"   ⟨R|L⟩ = {inner_RL:.6f}")
        print(f"   → {'Orthogonal' if abs(inner_RL) < 1e-10 else 'Not orthogonal'}")

        # Non-orthogonal example
        print("\n4. Non-orthogonal States:")
        state_A = np.array([1, 0], dtype=complex)
        state_B = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)], dtype=complex)

        inner_AB = np.vdot(state_A, state_B)
        overlap = abs(inner_AB) ** 2
        print(f"   |ψ₁⟩ = |0⟩")
        print(f"   |ψ₂⟩ = cos(π/6)|0⟩ + sin(π/6)|1⟩")
        print(f"   ⟨ψ₁|ψ₂⟩ = {inner_AB:.4f}")
        print(f"   |⟨ψ₁|ψ₂⟩|² = {overlap:.4f} (overlap)")

        return {
            "computational": (state_0, state_1),
            "hadamard": (state_plus, state_minus),
            "circular": (state_R, state_L),
        }

    def gram_schmidt_orthogonalization(self):
        """Demonstrate Gram-Schmidt orthogonalization process."""
        print("\n=== Gram-Schmidt Orthogonalization ===")

        # Start with linearly independent but non-orthogonal vectors
        v1 = np.array([1, 0], dtype=complex)
        v2 = np.array([1, 1], dtype=complex)

        print("Starting vectors:")
        print(f"v₁ = {v1}")
        print(f"v₂ = {v2}")

        # Check if they're orthogonal
        initial_inner = np.vdot(v1, v2)
        print(f"⟨v₁|v₂⟩ = {initial_inner:.4f}")

        # Gram-Schmidt process
        print("\nGram-Schmidt orthogonalization:")

        # Step 1: Normalize first vector
        u1 = v1 / np.linalg.norm(v1)
        print(f"u₁ = v₁/||v₁|| = {u1}")

        # Step 2: Remove projection of v2 onto u1
        projection = np.vdot(u1, v2) * u1
        u2_unnormalized = v2 - projection
        u2 = u2_unnormalized / np.linalg.norm(u2_unnormalized)

        print(f"proj_u₁(v₂) = ⟨u₁|v₂⟩u₁ = {projection}")
        print(f"u₂ = (v₂ - proj_u₁(v₂))/||v₂ - proj_u₁(v₂)|| = {u2}")

        # Verify orthogonality
        final_inner = np.vdot(u1, u2)
        print(f"\nVerification: ⟨u₁|u₂⟩ = {final_inner:.10f}")
        print(f"→ {'Orthogonal!' if abs(final_inner) < 1e-10 else 'Not orthogonal'}")

        # Verify normalization
        norm1 = np.linalg.norm(u1)
        norm2 = np.linalg.norm(u2)
        print(f"||u₁|| = {norm1:.6f}, ||u₂|| = {norm2:.6f}")

        return u1, u2

    def measurement_probability_analysis(self):
        """Analyze measurement probabilities using inner products."""
        print("\n=== Measurement Probability Analysis ===")

        # Create a general quantum state
        theta = np.pi / 3
        phi = np.pi / 4
        psi = np.array(
            [np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)], dtype=complex
        )

        print(f"Quantum state: |ψ⟩ = {psi[0]:.4f}|0⟩ + {psi[1]:.4f}|1⟩")

        # Measurement in computational basis
        print("\n1. Measurement in Computational Basis:")
        prob_0 = abs(np.vdot(np.array([1, 0]), psi)) ** 2
        prob_1 = abs(np.vdot(np.array([0, 1]), psi)) ** 2

        print(f"   P(|0⟩) = |⟨0|ψ⟩|² = {prob_0:.4f}")
        print(f"   P(|1⟩) = |⟨1|ψ⟩|² = {prob_1:.4f}")
        print(f"   Sum = {prob_0 + prob_1:.6f}")

        # Measurement in Hadamard basis
        print("\n2. Measurement in Hadamard Basis:")
        state_plus = np.array([1, 1]) / np.sqrt(2)
        state_minus = np.array([1, -1]) / np.sqrt(2)

        prob_plus = abs(np.vdot(state_plus, psi)) ** 2
        prob_minus = abs(np.vdot(state_minus, psi)) ** 2

        print(f"   P(|+⟩) = |⟨+|ψ⟩|² = {prob_plus:.4f}")
        print(f"   P(|-⟩) = |⟨-|ψ⟩|² = {prob_minus:.4f}")
        print(f"   Sum = {prob_plus + prob_minus:.6f}")

        # Measurement in arbitrary direction
        print("\n3. Measurement in Arbitrary Direction:")
        # Define measurement direction
        alpha = np.pi / 6
        beta = np.pi / 8

        measure_state = np.array(
            [np.cos(alpha / 2), np.sin(alpha / 2) * np.exp(1j * beta)]
        )

        prob_measure = abs(np.vdot(measure_state, psi)) ** 2
        prob_orthogonal = 1 - prob_measure

        print(
            f"   Measurement state: |m⟩ = {measure_state[0]:.4f}|0⟩ + {measure_state[1]:.4f}|1⟩"
        )
        print(f"   P(|m⟩) = |⟨m|ψ⟩|² = {prob_measure:.4f}")
        print(f"   P(orthogonal) = {prob_orthogonal:.4f}")

        return psi, {
            "computational": (prob_0, prob_1),
            "hadamard": (prob_plus, prob_minus),
            "arbitrary": (prob_measure, prob_orthogonal),
        }

    def visualize_inner_products(self, states_dict):
        """Visualize inner product relationships."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Inner product matrix visualization
        state_names = list(states_dict.keys())
        n_states = len(state_names)
        inner_matrix = np.zeros((n_states, n_states), dtype=complex)

        for i, name1 in enumerate(state_names):
            for j, name2 in enumerate(state_names):
                inner_matrix[i, j] = np.vdot(states_dict[name1], states_dict[name2])

        # Plot magnitude of inner products
        im1 = ax1.imshow(np.abs(inner_matrix), cmap="viridis", vmin=0, vmax=1)
        ax1.set_title("|⟨ψᵢ|ψⱼ⟩| Matrix")
        ax1.set_xticks(range(n_states))
        ax1.set_yticks(range(n_states))
        ax1.set_xticklabels(state_names)
        ax1.set_yticklabels(state_names)

        # Add text annotations
        for i in range(n_states):
            for j in range(n_states):
                text = ax1.text(
                    j,
                    i,
                    f"{np.abs(inner_matrix[i, j]):.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        plt.colorbar(im1, ax=ax1)

        # 2. Phase visualization
        im2 = ax2.imshow(np.angle(inner_matrix), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax2.set_title("Phase of ⟨ψᵢ|ψⱼ⟩")
        ax2.set_xticks(range(n_states))
        ax2.set_yticks(range(n_states))
        ax2.set_xticklabels(state_names)
        ax2.set_yticklabels(state_names)
        plt.colorbar(im2, ax=ax2)

        # 3. State overlap visualization
        angles = np.linspace(0, 2 * np.pi, 100)

        # Plot unit circle
        ax3.plot(np.cos(angles), np.sin(angles), "k--", alpha=0.3, label="Unit circle")

        # Plot states as vectors
        colors = ["red", "blue", "green", "orange"]
        for i, (name, state) in enumerate(states_dict.items()):
            # Project to 2D for visualization
            x = np.real(state[0])
            y = np.real(state[1])

            ax3.arrow(
                0,
                0,
                x,
                y,
                head_width=0.05,
                head_length=0.05,
                fc=colors[i],
                ec=colors[i],
                label=name,
            )
            ax3.text(x * 1.1, y * 1.1, name, fontsize=10, ha="center")

        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_xlabel("Real part")
        ax3.set_ylabel("Imaginary part")
        ax3.set_title("State Vectors (Real Parts)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_aspect("equal")

        # 4. Orthogonality test results
        orthogonal_pairs = []
        non_orthogonal_pairs = []

        for i, name1 in enumerate(state_names):
            for j, name2 in enumerate(state_names):
                if i < j:  # Avoid duplicates
                    inner_prod = inner_matrix[i, j]
                    if abs(inner_prod) < 1e-10:
                        orthogonal_pairs.append((name1, name2))
                    else:
                        non_orthogonal_pairs.append((name1, name2, abs(inner_prod)))

        # Bar plot of overlaps
        pair_names = []
        overlaps = []

        for pair in non_orthogonal_pairs:
            pair_names.append(f"{pair[0]},{pair[1]}")
            overlaps.append(pair[2])

        if overlaps:
            bars = ax4.bar(range(len(overlaps)), overlaps, alpha=0.7)
            ax4.set_xlabel("State Pairs")
            ax4.set_ylabel("|⟨ψᵢ|ψⱼ⟩|")
            ax4.set_title("Non-orthogonal State Overlaps")
            ax4.set_xticks(range(len(pair_names)))
            ax4.set_xticklabels(pair_names, rotation=45)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "All states are orthogonal!",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )
            ax4.set_title("Orthogonality Analysis")

        plt.tight_layout()
        plt.show()

    def advanced_inner_product_properties(self):
        """Explore advanced properties of inner products."""
        print("\n=== Advanced Inner Product Properties ===")

        # Generate random quantum states
        np.random.seed(42)  # For reproducibility
        psi = random_statevector(2).data
        phi = random_statevector(2).data
        chi = random_statevector(2).data

        print("Random quantum states:")
        print(f"|ψ⟩ = {psi}")
        print(f"|φ⟩ = {phi}")
        print(f"|χ⟩ = {chi}")

        # 1. Conjugate symmetry: ⟨ψ|φ⟩ = ⟨φ|ψ⟩*
        print("\n1. Conjugate Symmetry:")
        inner_psi_phi = np.vdot(psi, phi)
        inner_phi_psi = np.vdot(phi, psi)

        print(f"   ⟨ψ|φ⟩ = {inner_psi_phi:.6f}")
        print(f"   ⟨φ|ψ⟩ = {inner_phi_psi:.6f}")
        print(f"   ⟨φ|ψ⟩* = {np.conj(inner_phi_psi):.6f}")
        print(
            f"   → Conjugate symmetry: {'✓' if abs(inner_psi_phi - np.conj(inner_phi_psi)) < 1e-10 else '✗'}"
        )

        # 2. Linearity in second argument
        print("\n2. Linearity in Second Argument:")
        alpha, beta = 0.6 + 0.8j, 0.3 - 0.4j

        # ⟨ψ|αφ + βχ⟩ = α⟨ψ|φ⟩ + β⟨ψ|χ⟩
        linear_combination = alpha * phi + beta * chi

        left_side = np.vdot(psi, linear_combination)
        right_side = alpha * np.vdot(psi, phi) + beta * np.vdot(psi, chi)

        print(f"   α = {alpha:.3f}, β = {beta:.3f}")
        print(f"   ⟨ψ|αφ + βχ⟩ = {left_side:.6f}")
        print(f"   α⟨ψ|φ⟩ + β⟨ψ|χ⟩ = {right_side:.6f}")
        print(f"   → Linearity: {'✓' if abs(left_side - right_side) < 1e-10 else '✗'}")

        # 3. Positive definiteness: ⟨ψ|ψ⟩ ≥ 0
        print("\n3. Positive Definiteness:")
        norm_squared_psi = np.vdot(psi, psi)
        norm_squared_phi = np.vdot(phi, phi)

        print(f"   ⟨ψ|ψ⟩ = {norm_squared_psi:.6f}")
        print(f"   ⟨φ|φ⟩ = {norm_squared_phi:.6f}")
        print(
            f"   → Positive definiteness: {'✓' if np.real(norm_squared_psi) >= 0 and np.real(norm_squared_phi) >= 0 else '✗'}"
        )

        # 4. Cauchy-Schwarz inequality
        print("\n4. Cauchy-Schwarz Inequality:")
        left_cs = abs(np.vdot(psi, phi)) ** 2
        right_cs = np.vdot(psi, psi) * np.vdot(phi, phi)

        print(f"   |⟨ψ|φ⟩|² = {left_cs:.6f}")
        print(f"   ⟨ψ|ψ⟩⟨φ|φ⟩ = {np.real(right_cs):.6f}")
        print(
            f"   → Cauchy-Schwarz: {'✓' if left_cs <= np.real(right_cs) + 1e-10 else '✗'}"
        )

        return psi, phi, chi

    def generate_summary_report(self):
        """Generate comprehensive summary of inner product analysis."""
        print("\n" + "=" * 60)
        print("INNER PRODUCTS AND ORTHOGONALITY - ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 Inner Products Computed: {len(self.computed_products)}")

        print("\n🔬 Key Concepts Demonstrated:")
        print("  • Inner product calculation: ⟨ψ|φ⟩ = ψ*ᵀ · φ")
        print("  • Orthogonality condition: ⟨ψ|φ⟩ = 0")
        print("  • Measurement probabilities: P = |⟨measurement|state⟩|²")
        print("  • Gram-Schmidt orthogonalization")
        print("  • Inner product properties (linearity, conjugate symmetry)")

        print("\n📚 Mathematical Foundations:")
        print("  • Computational basis: {|0⟩, |1⟩} - orthogonal")
        print("  • Hadamard basis: {|+⟩, |-⟩} - orthogonal")
        print("  • Circular basis: {|R⟩, |L⟩} - orthogonal")
        print("  • Born rule: P(outcome) = |⟨outcome|ψ⟩|²")
        print("  • Completeness: Σᵢ |⟨eᵢ|ψ⟩|² = 1")

        print("\n🎯 Learning Outcomes:")
        print("  ✓ Understanding inner product mathematics")
        print("  ✓ Recognizing orthogonal quantum states")
        print("  ✓ Computing measurement probabilities")
        print("  ✓ Applying Gram-Schmidt orthogonalization")
        print("  ✓ Verifying inner product properties")

        print("\n🚀 Next Steps:")
        print("  → Explore tensor products and multi-qubit systems")
        print("  → Study quantum state spaces and Hilbert spaces")
        print("  → Investigate Schmidt decomposition")
        print("  → Advanced measurement theory")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Inner Products and Orthogonality in Quantum Mechanics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_inner_products_orthogonality.py
  python 04_inner_products_orthogonality.py --verbose
  python 04_inner_products_orthogonality.py --show-visualization
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed calculations",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display inner product visualizations",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis without visualizations",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 2: Mathematics")
    print("Example 4: Inner Products and Orthogonality")
    print("=" * 50)

    # Initialize analyzer
    analyzer = InnerProductAnalyzer(verbose=args.verbose)

    try:
        # Basic inner product calculations
        states = analyzer.basic_inner_products()

        # Orthogonality analysis
        orthogonal_bases = analyzer.orthogonality_analysis()

        # Gram-Schmidt demonstration
        u1, u2 = analyzer.gram_schmidt_orthogonalization()

        # Measurement probability analysis
        state, probabilities = analyzer.measurement_probability_analysis()

        # Advanced properties
        psi, phi, chi = analyzer.advanced_inner_product_properties()

        # Visualization (optional)
        if not args.analysis_only and args.show_visualization:
            analyzer.visualize_inner_products(states)

        # Generate summary
        analyzer.generate_summary_report()

        print(f"\n✅ Inner product analysis completed successfully!")
        print(f"📊 Computed {len(analyzer.computed_products)} inner products")

        if args.verbose:
            print(f"\n🔍 Detailed mathematical calculations enabled")
            print(f"📈 Use --show-visualization for plots")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
