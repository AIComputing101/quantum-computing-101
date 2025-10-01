#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 4
Quantum Entanglement

This example explores quantum entanglement - Einstein's "spooky action at a
distance" - one of the most fascinating and important phenomena in quantum
mechanics that makes quantum computers possible.

🎯 BEGINNER-FRIENDLY LEARNING OBJECTIVES:
- Understand what quantum entanglement really means (it's NOT faster-than-light communication)
- Learn to create Bell states - the simplest entangled quantum states
- See how measuring one qubit instantly affects its entangled partner
- Explore the difference between correlation and causation in quantum mechanics
- Understand why entanglement is essential for quantum computing

💡 KEY CONCEPTS YOU'LL LEARN:
- Entanglement: A quantum connection where qubits share a single quantum state
- Bell States: The four maximally entangled two-qubit states
- Quantum Correlation: Statistical relationships that are stronger than classical physics allows
- Non-locality: Quantum effects that seem to happen instantly across distances
- Measurement outcomes: How measuring entangled qubits gives correlated results

🚀 WHY THIS MATTERS:
Entanglement is the "quantum magic" that gives quantum computers their power!
Without entanglement, a quantum computer would be no better than a classical one.

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator


def explain_entanglement_concept():
    """Explain entanglement in beginner-friendly terms."""
    print("🪄 WELCOME TO QUANTUM ENTANGLEMENT!")
    print("Einstein called it 'spooky action at a distance'")
    print("=" * 50)
    print()

    print("🤔 What is Quantum Entanglement?")
    print("Imagine two magic coins that are forever connected:")
    print("- When one lands heads, the other INSTANTLY lands tails")
    print("- This happens no matter how far apart they are")
    print("- It's not that they 'communicated' - they share a single quantum state")
    print()

    print("📊 Classical vs Quantum Correlations:")
    print()
    print("CLASSICAL CORRELATION:")
    print("- Two coins in a box, one heads, one tails")
    print("- When you look at one, you know the other")
    print("- But they were always determined - you just didn't know")
    print()

    print("QUANTUM ENTANGLEMENT:")
    print("- Two qubits that share a single quantum state")
    print("- Neither has a definite state until measured")
    print("- Measuring one instantly determines the other")
    print("- This correlation is stronger than classical physics allows!")
    print()

    print("🎯 Key Insights:")
    print("1. 🚫 NO COMMUNICATION: Information doesn't travel between qubits")
    print("2. 📏 MEASUREMENT MATTERS: Results are correlated, but random")
    print("3. 🔗 SHARED STATE: Entangled qubits can't be described separately")
    print("4. ⚡ QUANTUM POWER: This gives quantum computers their advantage")
    print()

    print("🔮 What You'll See:")
    print("- How to create entangled states (Bell states)")
    print("- Perfect correlations in measurement outcomes")
    print("- Why this violates classical intuition")
    print("- How entanglement enables quantum algorithms")
    print()


def create_bell_states():
    """Create the four Bell states (maximally entangled states)."""
    print("=== BELL STATES (MAXIMALLY ENTANGLED STATES) ===")
    print()

    bell_states = {}

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    qc_phi_plus = QuantumCircuit(2)
    qc_phi_plus.h(0)
    qc_phi_plus.cx(0, 1)
    bell_states["|Φ+⟩ = (|00⟩ + |11⟩)/√2"] = qc_phi_plus

    # Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2
    qc_phi_minus = QuantumCircuit(2)
    qc_phi_minus.h(0)
    qc_phi_minus.z(0)
    qc_phi_minus.cx(0, 1)
    bell_states["|Φ-⟩ = (|00⟩ - |11⟩)/√2"] = qc_phi_minus

    # Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    qc_psi_plus = QuantumCircuit(2)
    qc_psi_plus.h(0)
    qc_psi_plus.cx(0, 1)
    qc_psi_plus.x(1)
    bell_states["|Ψ+⟩ = (|01⟩ + |10⟩)/√2"] = qc_psi_plus

    # Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    qc_psi_minus = QuantumCircuit(2)
    qc_psi_minus.h(0)
    qc_psi_minus.z(0)
    qc_psi_minus.cx(0, 1)
    qc_psi_minus.x(1)
    bell_states["|Ψ-⟩ = (|01⟩ - |10⟩)/√2"] = qc_psi_minus

    # Analyze each Bell state
    for label, circuit in bell_states.items():
        state = Statevector.from_instruction(circuit)
        print(f"{label}:")
        print(f"  Statevector: {state}")
        print(
            f"  Amplitudes: |00⟩:{state[0]:.3f}, |01⟩:{state[1]:.3f}, |10⟩:{state[2]:.3f}, |11⟩:{state[3]:.3f}"
        )
        print()

    return bell_states


def demonstrate_entanglement_creation():
    """Show step-by-step entanglement creation."""
    print("=== STEP-BY-STEP ENTANGLEMENT CREATION ===")
    print()

    steps = []

    # Step 0: Initial state |00⟩
    qc0 = QuantumCircuit(2)
    steps.append(("Initial |00⟩", qc0))

    # Step 1: Apply Hadamard to first qubit
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    steps.append(("After H gate: (|00⟩ + |10⟩)/√2", qc1))

    # Step 2: Apply CNOT gate
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    steps.append(("After CNOT: (|00⟩ + |11⟩)/√2", qc2))

    for i, (description, circuit) in enumerate(steps):
        state = Statevector.from_instruction(circuit)
        print(f"Step {i}: {description}")
        print(f"  State: {state}")

        # Check if entangled (for 2-qubit states)
        if circuit.num_qubits == 2:
            # Compute reduced density matrices
            rho_A = partial_trace(state, [1])  # Trace out qubit 1
            rho_B = partial_trace(state, [0])  # Trace out qubit 0

            # Check purity (pure states have purity = 1, mixed states < 1)
            purity_A = np.trace(rho_A.data @ rho_A.data).real
            purity_B = np.trace(rho_B.data @ rho_B.data).real

            entangled = purity_A < 0.99 or purity_B < 0.99
            print(f"  Entangled: {entangled}")
            print(f"  Purity A: {purity_A:.3f}, Purity B: {purity_B:.3f}")

        print()

    print("Key insight: CNOT gate creates entanglement between qubits!")
    print()

    return steps


def demonstrate_quantum_correlations():
    """Demonstrate the strange correlations in entangled states."""
    print("=== QUANTUM CORRELATIONS ===")
    print()

    # Create Bell state |Φ+⟩
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)

    simulator = AerSimulator()
    shots = 1000

    print(f"Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"Measuring both qubits ({shots} shots):")
    print()

    # Measure both qubits in Z-basis
    qc_measure = qc.copy()
    # Create circuit with classical bits for measurement
    if not hasattr(qc_measure, "clbits") or not qc_measure.clbits:
        # Recreate circuit with classical register
        temp_circuit = QuantumCircuit(qc_measure.num_qubits, qc_measure.num_qubits)
        temp_circuit = temp_circuit.compose(qc_measure)
        qc_measure = temp_circuit
    qc_measure.measure_all()

    job = simulator.run(transpile(qc_measure, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts()

    print("Z-basis measurement (computational basis):")
    for outcome, count in counts.items():
        percentage = (count / shots) * 100
        print(f"  |{outcome}⟩: {count} times ({percentage:.1f}%)")
    print()

    # Calculate correlation
    corr_same = counts.get("00", 0) + counts.get("11", 0)
    corr_diff = counts.get("01", 0) + counts.get("10", 0)

    print(f"Correlation analysis:")
    print(f"  Same outcomes (00 or 11): {corr_same} ({100*corr_same/shots:.1f}%)")
    print(f"  Different outcomes (01 or 10): {corr_diff} ({100*corr_diff/shots:.1f}%)")
    print(f"  Perfect correlation: {corr_same / shots:.3f}")
    print()

    return counts


def demonstrate_measurement_basis_effects():
    """Show how measurement basis affects entangled states."""
    print("=== MEASUREMENT BASIS EFFECTS ON ENTANGLEMENT ===")
    print()

    # Create Bell state
    base_circuit = QuantumCircuit(2)
    base_circuit.h(0)
    base_circuit.cx(0, 1)

    measurement_setups = {}

    # Z-Z measurement (computational basis)
    qc_zz = base_circuit.copy()
    qc_zz.measure_all()
    measurement_setups["Z-Z (computational)"] = qc_zz

    # X-X measurement
    qc_xx = base_circuit.copy()
    qc_xx.h(0)  # Rotate to X-basis
    qc_xx.h(1)  # Rotate to X-basis
    qc_xx.measure_all()
    measurement_setups["X-X (Hadamard basis)"] = qc_xx

    # Z-X measurement (mixed bases)
    qc_zx = base_circuit.copy()
    qc_zx.h(1)  # Rotate second qubit to X-basis
    qc_zx.measure_all()
    measurement_setups["Z-X (mixed bases)"] = qc_zx

    simulator = AerSimulator()
    shots = 1000

    results = {}
    correlations = {}

    for setup_name, circuit in measurement_setups.items():
        job = simulator.run(transpile(circuit, simulator), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results[setup_name] = counts

        # Calculate correlation coefficient
        same_parity = counts.get("00", 0) + counts.get("11", 0)
        diff_parity = counts.get("01", 0) + counts.get("10", 0)
        correlation = (same_parity - diff_parity) / shots
        correlations[setup_name] = correlation

        print(f"{setup_name}:")
        for outcome, count in counts.items():
            percentage = (count / shots) * 100
            print(f"  |{outcome}⟩: {count} times ({percentage:.1f}%)")
        print(f"  Correlation coefficient: {correlation:.3f}")
        print()

    # Visualize results
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3))
    if len(results) == 1:
        axes = [axes]

    for i, (setup_name, counts) in enumerate(results.items()):
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i].bar(list(counts.keys()), list(counts.values()))
            axes[i].set_xlabel("Measurement Outcome")
            axes[i].set_ylabel("Counts")
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i].set_title(
            f"{setup_name}\nCorr: {correlations[setup_name]:.3f}", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(
        "module1_04_entanglement_measurements.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Key insights:")
    print("• Entangled states show correlations in any measurement basis")
    print("• Z-Z and X-X measurements show perfect correlation (+1)")
    print("• Mixed basis measurements can show partial correlations")
    print()

    return results, correlations


def demonstrate_separable_vs_entangled():
    """Compare separable and entangled states."""
    print("=== SEPARABLE vs ENTANGLED STATES ===")
    print()

    states = {}

    # Separable state: |0⟩ ⊗ |+⟩
    qc_sep = QuantumCircuit(2)
    qc_sep.h(1)  # Second qubit in |+⟩ state
    states["Separable: |0⟩ ⊗ |+⟩"] = qc_sep

    # Entangled state: Bell state
    qc_ent = QuantumCircuit(2)
    qc_ent.h(0)
    qc_ent.cx(0, 1)
    states["Entangled: (|00⟩ + |11⟩)/√2"] = qc_ent

    for label, circuit in states.items():
        print(f"{label}:")
        state = Statevector.from_instruction(circuit)
        print(f"  Full state: {state}")

        # Analyze individual qubits
        rho_A = partial_trace(state, [1])  # First qubit
        rho_B = partial_trace(state, [0])  # Second qubit

        # Check if qubits are in pure states
        purity_A = np.trace(rho_A.data @ rho_A.data).real
        purity_B = np.trace(rho_B.data @ rho_B.data).real

        print(f"  First qubit purity: {purity_A:.3f}")
        print(f"  Second qubit purity: {purity_B:.3f}")
        print(f"  Individual qubits pure: {purity_A > 0.99 and purity_B > 0.99}")
        print()

    print("Key differences:")
    print("• Separable: Individual qubits have definite states (pure)")
    print("• Entangled: Individual qubits are in mixed states")
    print("• Entanglement creates quantum correlations impossible classically")
    print()


def explore_entanglement_applications():
    """Explore practical applications of entanglement."""
    print("=== ENTANGLEMENT APPLICATIONS ===")
    print()

    print("1. Quantum Communication:")
    print("   • Quantum key distribution (QKD)")
    print("   • Quantum teleportation")
    print("   • Superdense coding")
    print()

    print("2. Quantum Computing:")
    print("   • Quantum algorithms (Shor's, Grover's)")
    print("   • Quantum error correction")
    print("   • Quantum machine learning")
    print()

    print("3. Quantum Sensing:")
    print("   • Enhanced measurement precision")
    print("   • Quantum metrology")
    print("   • Atomic clocks")
    print()

    print("4. Fundamental Physics:")
    print("   • Testing Bell inequalities")
    print("   • Exploring quantum foundations")
    print("   • Quantum gravity research")
    print()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Quantum Entanglement Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of measurement shots (default: 1000)",
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 1, Example 4")
    print("Quantum Entanglement")
    print("=" * 50)
    print()

    try:
        # Create Bell states
        bell_states = create_bell_states()

        # Demonstrate entanglement creation
        creation_steps = demonstrate_entanglement_creation()

        # Demonstrate quantum correlations
        correlation_counts = demonstrate_quantum_correlations()

        # Show measurement basis effects
        basis_results, correlations = demonstrate_measurement_basis_effects()

        # Compare separable vs entangled
        demonstrate_separable_vs_entangled()

        # Explore applications
        explore_entanglement_applications()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module1_04_entanglement_measurements.png - Measurement correlations")
        print()
        print("🎯 Key takeaways:")
        print("• Entanglement creates non-local correlations between qubits")
        print("• Measurement of one qubit instantly affects the other")
        print("• Bell states are maximally entangled two-qubit states")
        print("• Entanglement is a key resource for quantum technologies")
        print("• 'Spooky action at a distance' - Einstein's concern!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
