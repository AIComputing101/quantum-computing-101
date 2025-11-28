#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 7
No-Cloning Theorem

This example demonstrates the no-cloning theorem, one of the most fundamental
principles of quantum mechanics that explains why quantum information cannot
be copied perfectly. This is a key concept that makes quantum computing
fundamentally different from classical computing.

Learning objectives:
- Understand why qubits cannot be copied like classical bits
- See the mathematical proof of the no-cloning theorem
- Explore the implications for quantum computing and communication
- Learn why quantum teleportation is necessary instead of copying

Based on concepts from "Quantum Computing in Action" Chapter 6

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector, random_statevector

# Handle different Qiskit versions for fidelity
try:
    from qiskit.quantum_info import fidelity
except ImportError:
    # For older Qiskit versions, use state_fidelity
    try:
        from qiskit.quantum_info import state_fidelity as fidelity
    except ImportError:
        # Fallback implementation
        def fidelity(state1, state2):
            return abs(np.vdot(state1.data, state2.data)) ** 2


from qiskit_aer import AerSimulator
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from quantum_helpers import create_bell_state, run_circuit_with_shots


def demonstrate_classical_copying():
    """
    Show how classical bits can be copied perfectly.
    
    Mathematical Foundation - Classical Copying:
    -------------------------------------------
    Classical information can be copied arbitrarily many times
    without any loss or change to the original.
    
    Classical COPY Operation:
    ------------------------
    Input: |original⟩, |blank⟩
    Output: |original⟩, |original⟩
    
    For classical bits (b ∈ {0,1}):
    COPY(b, 0) = (b, b)
    
    Examples:
    COPY(0, 0) = (0, 0) ✓
    COPY(1, 0) = (1, 1) ✓
    
    Properties:
    ----------
    1. Perfect fidelity: Copy is identical to original
    2. Non-destructive: Original unchanged
    3. Unlimited copies: Can repeat infinitely
    4. Independent: Reading doesn't affect the bit
    
    Why This Works Classically:
    --------------------------
    Classical bits have definite values (0 or 1).
    Reading a bit tells us its value without changing it.
    We can then set another bit to the same value.
    
    Mathematical representation:
    If original = b, then after COPY:
    - First bit = b (unchanged)
    - Second bit = b (copied value)
    - Correlation: 100% (perfect copy)
    
    This is IMPOSSIBLE for quantum states! (No-Cloning Theorem)
    
    Returns:
        tuple: (original_bit, copied_bit)
    """
    print("=== CLASSICAL BIT COPYING ===")
    print()

    print("Classical bits can be copied perfectly:")
    original_bit = 1
    print(f"Original bit: {original_bit}")

    # Classical copying is trivial - just read and set
    # Reading a classical bit doesn't change it (unlike quantum measurement!)
    copied_bit = original_bit
    print(f"Copied bit:   {copied_bit}")
    print(f"Are they identical? {original_bit == copied_bit}")
    print()

    print("Classical copying process:")
    print("1. Read the original bit (this doesn't change it)")
    print("2. Set the copy bit to the same value")
    print("3. Now you have two identical bits")
    print("4. This works for any classical information")
    print()
    
    print("Mathematical view:")
    print(f"  Before: bit₁ = {original_bit}, bit₂ = 0")
    print(f"  After:  bit₁ = {original_bit}, bit₂ = {copied_bit}")
    print("  Both bits are independent and identical!")
    print()

    return original_bit, copied_bit


def attempt_quantum_copying_naive():
    """Attempt to copy a qubit using a naive classical approach."""
    print("=== NAIVE QUANTUM COPYING ATTEMPT ===")
    print()

    print("Let's try to copy a qubit the classical way...")
    print()

    # Create a qubit in superposition
    qc = QuantumCircuit(1)
    qc.h(0)  # Put in superposition: |+⟩ = (|0⟩ + |1⟩)/√2

    print("Original qubit state: |+⟩ = (|0⟩ + |1⟩)/√2")
    print("This is a superposition of |0⟩ and |1⟩")
    print()

    # Get the state vector
    state = Statevector.from_instruction(qc)
    print(f"State vector: {state.data}")
    print()

    # The problem: if we measure to "read" the qubit...
    print("Problem with naive copying:")
    print("1. To read the qubit classically, we must measure it")
    print("2. Measurement collapses superposition to |0⟩ or |1⟩")
    print("3. We lose the original superposition information!")
    print("4. We can only copy the collapsed state, not the original")
    print()

    # Demonstrate measurement destroying superposition
    from qiskit.circuit import ClassicalRegister
    qc_measure = QuantumCircuit(1, 1)
    qc_measure.h(0)  # Recreate the superposition
    qc_measure.measure(0, 0)

    print("After measurement, we get either:")
    print("- |0⟩ with 50% probability, OR")
    print("- |1⟩ with 50% probability")
    print("The superposition is gone forever!")
    print()

    return state


def demonstrate_no_cloning_theorem():
    """Demonstrate the mathematical impossibility of quantum cloning."""
    print("=== NO-CLONING THEOREM PROOF ===")
    print()

    print("Mathematical proof that quantum cloning is impossible:")
    print()

    print("Suppose there exists a perfect quantum cloning machine:")
    print("U|ψ⟩|0⟩ = |ψ⟩|ψ⟩  (copy any state |ψ⟩ to blank qubit |0⟩)")
    print()

    print("This would mean:")
    print("U|0⟩|0⟩ = |0⟩|0⟩  (copying |0⟩)")
    print("U|1⟩|0⟩ = |1⟩|1⟩  (copying |1⟩)")
    print()

    print("But what about superposition |+⟩ = (|0⟩ + |1⟩)/√2?")
    print()

    print("Linearity of quantum mechanics requires:")
    print("U|+⟩|0⟩ = U[(|0⟩ + |1⟩)/√2]|0⟩")
    print("        = [U|0⟩|0⟩ + U|1⟩|0⟩]/√2")
    print("        = [|0⟩|0⟩ + |1⟩|1⟩]/√2")
    print()

    print("But perfect copying would require:")
    print("U|+⟩|0⟩ = |+⟩|+⟩ = [(|0⟩ + |1⟩)/√2][(|0⟩ + |1⟩)/√2]")
    print("        = [|0⟩|0⟩ + |0⟩|1⟩ + |1⟩|0⟩ + |1⟩|1⟩]/2")
    print()

    print("These are different! Contradiction proves no-cloning theorem.")
    print()


def demonstrate_approximate_cloning():
    """Show that approximate cloning is possible but imperfect."""
    print("=== APPROXIMATE QUANTUM CLONING ===")
    print()

    print("While perfect cloning is impossible, approximate cloning exists...")
    print()

    # Create two different quantum states
    qc1 = QuantumCircuit(1)
    qc1.ry(np.pi / 3, 0)  # Some arbitrary state
    state1 = Statevector.from_instruction(qc1)

    qc2 = QuantumCircuit(1)
    qc2.ry(np.pi / 4, 0)  # Different arbitrary state
    state2 = Statevector.from_instruction(qc2)

    print(f"Original state 1: {state1.data}")
    print(f"Original state 2: {state2.data}")
    print()

    # Simulate approximate cloning by adding noise
    def approximate_clone(state, fidelity_target=0.83):
        """Simulate approximate cloning with limited fidelity."""
        # This is a simplified simulation - real approximate cloning
        # requires more complex quantum circuits
        noise = np.random.normal(0, 0.1, len(state.data))
        noisy_data = state.data + noise
        # Renormalize
        noisy_data = noisy_data / np.linalg.norm(noisy_data)
        return Statevector(noisy_data)

    clone1 = approximate_clone(state1)
    clone2 = approximate_clone(state2)

    # Calculate fidelities
    fid1 = fidelity(state1, clone1)
    fid2 = fidelity(state2, clone2)

    print(f"Cloning fidelity for state 1: {fid1:.3f}")
    print(f"Cloning fidelity for state 2: {fid2:.3f}")
    print()

    print("Approximate cloning limitations:")
    print("- Cannot achieve perfect fidelity (1.0) for all states")
    print("- Trade-off between fidelity and universality")
    print("- Fundamental quantum mechanical limits")
    print()

    return fid1, fid2


def demonstrate_implications():
    """Explore the implications of the no-cloning theorem."""
    print("=== IMPLICATIONS OF NO-CLONING ===")
    print()

    print("🔐 QUANTUM CRYPTOGRAPHY:")
    print("- Eavesdropping can be detected (measurement disturbs qubits)")
    print("- Quantum key distribution relies on this property")
    print("- No perfect copying means no undetectable interception")
    print()

    print("📡 QUANTUM COMMUNICATION:")
    print("- Quantum teleportation needed instead of copying")
    print("- Original qubit is destroyed during teleportation")
    print("- Classical communication required alongside quantum")
    print()

    print("💾 QUANTUM COMPUTING:")
    print("- Cannot easily backup quantum states")
    print("- Error correction must work differently than classical")
    print("- Quantum algorithms must be designed carefully")
    print()

    print("🔬 QUANTUM PHYSICS:")
    print("- Fundamental difference from classical information")
    print("- Conservation of quantum information")
    print("- Measurement as an irreversible process")
    print()


def compare_with_error_correction():
    """Compare no-cloning with quantum error correction."""
    print("=== NO-CLONING VS QUANTUM ERROR CORRECTION ===")
    print()

    print('🤔 "But wait, don\'t quantum computers use error correction?"')
    print()

    print("Quantum error correction is NOT cloning because:")
    print()

    print("1. REDUNDANT ENCODING:")
    print("   - Original quantum state is encoded across multiple qubits")
    print("   - No copying happens - just spreading information")
    print("   - Like storing a message in multiple parts")
    print()

    print("2. ERROR SYNDROME DETECTION:")
    print("   - Measure only the error syndrome, not the state itself")
    print("   - The quantum information remains untouched")
    print("   - Like checking a checksum without reading the data")
    print()

    print("3. QUANTUM ERROR CORRECTION EXAMPLE:")
    print("   - Encode 1 logical qubit into 3 physical qubits")
    print("   - Original: |ψ⟩ = α|0⟩ + β|1⟩")
    print("   - Encoded: |ψ_L⟩ = α|000⟩ + β|111⟩")
    print("   - No cloning occurred - just entanglement!")
    print()


def visualize_no_cloning():
    """Create visualizations showing why cloning fails."""
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Classical vs Quantum copying
    ax = axes[0, 0]

    # Classical copying success
    classical_data = [1, 1, 1, 1]  # Perfect copies
    ax.bar(
        ["Original", "Copy 1", "Copy 2", "Copy 3"],
        classical_data,
        color="green",
        alpha=0.7,
    )
    ax.set_title("Classical Copying: Perfect Success")
    ax.set_ylabel("Information Fidelity")
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Perfect Copying")
    ax.legend()

    # 2. Quantum copying attempts
    ax = axes[0, 1]

    # Approximate quantum copying
    quantum_data = [1.0, 0.83, 0.81, 0.79]  # Decreasing fidelity
    ax.bar(
        ["Original", "Clone 1", "Clone 2", "Clone 3"],
        quantum_data,
        color="orange",
        alpha=0.7,
    )
    ax.set_title("Quantum Cloning: Fundamental Limits")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Perfect Cloning (Impossible)")
    ax.axhline(y=0.83, color="blue", linestyle="--", label="Universal Cloning Limit")
    ax.legend()

    # 3. Measurement collapse
    ax = axes[1, 0]

    # Superposition before measurement
    angles = np.linspace(0, 2 * np.pi, 100)
    prob_0 = 0.5 * np.ones_like(angles)  # Constant probability
    prob_1 = 0.5 * np.ones_like(angles)

    ax.fill_between(angles, 0, prob_0, alpha=0.5, label="P(|0⟩) = 0.5")
    ax.fill_between(angles, prob_0, prob_0 + prob_1, alpha=0.5, label="P(|1⟩) = 0.5")
    ax.set_title("Superposition Before Measurement")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Probability")
    ax.legend()

    # 4. After measurement collapse
    ax = axes[1, 1]

    # Two possible outcomes after measurement
    measurement_results = ["Outcome 1\n|0⟩", "Outcome 2\n|1⟩"]
    probabilities = [0.5, 0.5]
    colors = ["blue", "red"]

    bars = ax.bar(measurement_results, probabilities, color=colors, alpha=0.7)
    ax.set_title("After Measurement: Information Lost")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    # Add text
    ax.text(
        0.5,
        0.8,
        "Superposition\nDestroyed!",
        ha="center",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    plt.suptitle(
        "No-Cloning Theorem: Why Quantum States Cannot Be Copied",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="No-Cloning Theorem Demonstration")
    parser.add_argument(
        "--skip-visualization", action="store_true", help="Skip the visualization plots"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed explanations"
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 1: Fundamental Concepts")
    print("Example 7: No-Cloning Theorem")
    print("=" * 50)

    try:
        print("\n🎯 Welcome to the No-Cloning Theorem!")
        print("This is one of the most important principles in quantum mechanics.")
        print(
            "It explains why quantum computing is fundamentally different from classical computing."
        )
        print()

        # Classical copying demonstration
        classical_original, classical_copy = demonstrate_classical_copying()

        # Naive quantum copying attempt
        quantum_state = attempt_quantum_copying_naive()

        # Mathematical proof
        demonstrate_no_cloning_theorem()

        # Approximate cloning
        print("🔬 Testing approximate cloning...")
        fid1, fid2 = demonstrate_approximate_cloning()

        # Implications
        demonstrate_implications()

        # Error correction comparison
        compare_with_error_correction()

        # Key takeaways
        print("🎓 KEY TAKEAWAYS:")
        print("=" * 40)
        print("1. ❌ Quantum states CANNOT be cloned perfectly")
        print("2. 📏 This is a fundamental law of quantum mechanics")
        print("3. 🔐 Enables secure quantum communication")
        print("4. 📡 Makes quantum teleportation necessary")
        print("5. 💾 Quantum error correction works differently")
        print("6. 🎯 Approximate cloning has fundamental limits")
        print()

        if not args.skip_visualization:
            visualize_no_cloning()

        if args.verbose:
            print("\n🔬 DEEPER DIVE:")
            print("The no-cloning theorem, proven by Wootters and Zurek in 1982,")
            print("states that it is impossible to create an independent and")
            print("identical copy of an arbitrary unknown quantum state.")
            print()
            print("This has profound implications for:")
            print("- Quantum cryptography (BB84 protocol)")
            print("- Quantum computing (error correction strategies)")
            print("- Quantum communication (teleportation vs copying)")
            print("- Fundamental physics (information conservation)")
            print()

        print("✅ No-cloning theorem demonstration completed!")
        print()
        print("💡 Next Steps:")
        print("- Try the quantum teleportation example (06_quantum_teleportation.py)")
        print("- Explore quantum cryptography applications")
        print("- Learn about quantum error correction in Module 5")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
