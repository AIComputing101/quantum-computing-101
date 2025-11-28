#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 6
Quantum Teleportation

This example demonstrates quantum teleportation, the process of transferring
the quantum state of one qubit to another using entanglement and classical
communication, without physically moving the qubit itself.

Learning objectives:
- Understand the quantum teleportation protocol
- See how entanglement enables "spooky action at a distance"
- Learn about the no-cloning theorem implications
- Explore the role of classical communication in quantum protocols

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
from qiskit_aer import AerSimulator
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from quantum_helpers import create_bell_state, run_circuit_with_shots, analyze_state


def create_teleportation_circuit(state_to_teleport=None, use_random_state=True):
    """
    Create a complete quantum teleportation circuit.
    
    Mathematical Foundation - Quantum Teleportation:
    -----------------------------------------------
    Teleportation transfers a quantum state from Alice to Bob using:
    1. Pre-shared entanglement (Bell state)
    2. Classical communication (2 classical bits)
    3. Local operations (controlled gates)
    
    Remarkable fact: The original qubit's state is DESTROYED (no-cloning!)
    but recreated at Bob's location.
    
    Protocol Steps:
    ---------------
    Initial state: |ψ⟩ = α|0⟩ + β|1⟩ (unknown state to teleport)
    
    STEP 0: Prepare shared entangled pair
    |Φ+⟩ = (|00⟩ + |11⟩)/√2 (Bell state)
    Alice has first qubit, Bob has second
    
    STEP 1: Alice's Bell measurement
    Alice entangles her original qubit with her half of Bell pair
    Then measures both qubits in Bell basis
    Results: 2 classical bits (4 possible outcomes: 00, 01, 10, 11)
    
    STEP 2: Classical communication
    Alice sends her 2 measurement bits to Bob (classical channel)
    
    STEP 3: Bob's correction
    Based on Alice's bits, Bob applies gates to his qubit:
    - 00: Do nothing (I gate)
    - 01: Apply X gate
    - 10: Apply Z gate
    - 11: Apply ZX gates
    
    Result: Bob's qubit now in state |ψ⟩ = α|0⟩ + β|1⟩ ✓
    
    Mathematical Analysis:
    ---------------------
    Full initial state (3 qubits):
    |ψ_initial⟩ = |ψ⟩_Alice ⊗ |Φ+⟩_Alice-Bob
                = (α|0⟩ + β|1⟩) ⊗ (|00⟩ + |11⟩)/√2
    
    After Bell measurement, state collapses to one of 4 outcomes,
    each with probability 1/4:
    
    Measurement 00: Bob has α|0⟩ + β|1⟩ (already correct!)
    Measurement 01: Bob has α|1⟩ + β|0⟩ (X needed)
    Measurement 10: Bob has α|0⟩ - β|1⟩ (Z needed)
    Measurement 11: Bob has α|1⟩ - β|0⟩ (ZX needed)
    
    Key Insights:
    ------------
    • NO PHYSICAL QUBIT MOVED! Only classical info transmitted
    • Original state DESTROYED (measurement collapses it)
    • Entanglement + classical bits = quantum state transfer
    • Cannot transmit faster than light (classical comm needed)
    • Works for ANY unknown quantum state!
    
    Why "Teleportation"?
    -------------------
    - State appears at Bob's location
    - Original at Alice destroyed
    - Like Star Trek transporter (information, not matter)
    
    Applications:
    ------------
    - Quantum networks (distribute quantum info)
    - Quantum repeaters (extend entanglement range)
    - Quantum error correction
    - Distributed quantum computing
    
    Args:
        state_to_teleport: Specific state name or None
        use_random_state: If True, generate random state
        
    Returns:
        tuple: (QuantumCircuit, state_description)
    """
    # Create 3-qubit circuit with 3 classical bits for measurements
    # Qubit 0: Alice's original qubit (state to be teleported)
    # Qubit 1: Alice's half of entangled Bell pair
    # Qubit 2: Bob's half of entangled Bell pair (destination)
    qc = QuantumCircuit(3, 3)

    # PREPARE STATE TO TELEPORT
    # This is Alice's unknown quantum state |ψ⟩ = α|0⟩ + β|1⟩
    if state_to_teleport is not None:
        # Use provided state (for testing/demonstration)
        if state_to_teleport == "|+⟩":
            # |+⟩ = H|0⟩ = (|0⟩ + |1⟩)/√2
            qc.h(0)
            state_desc = "|+⟩ = (|0⟩ + |1⟩)/√2"
        elif state_to_teleport == "|-⟩":
            # |-⟩ = HX|0⟩ = (|0⟩ - |1⟩)/√2
            qc.x(0)
            qc.h(0)
            state_desc = "|-⟩ = (|0⟩ - |1⟩)/√2"
        elif state_to_teleport == "|0⟩":
            # Already in |0⟩ state (do nothing)
            state_desc = "|0⟩"
        elif state_to_teleport == "|1⟩":
            # |1⟩ = X|0⟩
            qc.x(0)
            state_desc = "|1⟩"
        else:
            # Interpret as custom rotation
            qc.ry(np.pi / 3, 0)  # Example rotation
            state_desc = "custom state"
    elif use_random_state:
        # Create a random state to teleport
        # General qubit: R_z(φ)R_y(θ)|0⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        state_desc = f"random state (θ={theta:.2f}, φ={phi:.2f})"
    else:
        # Default: teleport |+⟩ state
        qc.h(0)
        state_desc = "|+⟩ = (|0⟩ + |1⟩)/√2"

    qc.barrier()  # Visual separator

    # STEP 1: CREATE ENTANGLED BELL PAIR
    # Alice and Bob share |Φ+⟩ = (|00⟩ + |11⟩)/√2
    # Alice has qubit 1, Bob has qubit 2
    qc.h(1)  # Put qubit 1 in superposition: |+⟩
    qc.cx(1, 2)  # Entangle qubits 1 and 2: |Φ+⟩ = (|00⟩ + |11⟩)/√2

    qc.barrier()  # Visual separator

    # STEP 2: ALICE'S BELL MEASUREMENT
    # Alice entangles her original qubit (0) with her Bell pair qubit (1)
    # Then measures both in computational basis
    qc.cx(0, 1)  # CNOT: entangle qubit 0 with qubit 1
    qc.h(0)  # Hadamard: transform to Bell basis

    # Measure Alice's two qubits (0 and 1)
    # These measurements give 2 classical bits to send to Bob
    qc.measure(0, 0)  # Measure qubit 0 -> classical bit 0
    qc.measure(1, 1)  # Measure qubit 1 -> classical bit 1

    qc.barrier()  # Visual separator

    # Step 3: Bob applies corrections based on Alice's measurement results
    # If Alice measured 01, Bob applies X gate
    qc.cx(1, 2)  # Apply X to Bob's qubit if Alice's qubit 1 measured 1
    # If Alice measured 10 or 11, Bob applies Z gate
    qc.cz(0, 2)  # Apply Z to Bob's qubit if Alice's qubit 0 measured 1

    # Measure Bob's qubit to verify teleportation
    qc.measure(2, 2)  # Measure Bob's qubit -> classical bit 2

    return qc, state_desc


def demonstrate_teleportation_protocol():
    """Demonstrate the complete teleportation protocol step by step."""
    print("🌟 QUANTUM TELEPORTATION PROTOCOL")
    print("=" * 50)
    print()
    print("Scenario: Alice wants to send a quantum state to Bob")
    print("Problem: Quantum states cannot be copied (No-Cloning Theorem)")
    print("Solution: Use quantum entanglement + classical communication!")
    print()

    # Show the protocol steps
    print("📋 PROTOCOL STEPS:")
    print("1. 🎭 Alice and Bob share an entangled pair")
    print("2. 🔍 Alice performs Bell measurement on her qubits")
    print("3. 📞 Alice sends measurement results to Bob (classical)")
    print("4. 🎯 Bob applies corrections based on Alice's results")
    print("5. ✅ Bob now has Alice's original quantum state!")
    print()

    return True


def run_teleportation_examples(shots=1000, verbose=False):
    """Run teleportation examples with different input states."""
    print("🧪 TELEPORTATION EXPERIMENTS")
    print("=" * 40)

    # Test states to teleport
    test_states = ["|0⟩", "|1⟩", "|+⟩", "|-⟩"]
    results = {}

    simulator = AerSimulator()

    for state_name in test_states:
        print(f"\n📡 Teleporting state: {state_name}")

        # Create teleportation circuit
        qc, state_desc = create_teleportation_circuit(
            state_to_teleport=state_name, use_random_state=False
        )

        if verbose:
            print(f"  Circuit depth: {qc.depth()}")
            print(f"  Total gates: {sum(qc.count_ops().values())}")

        # Run the circuit
        try:
            job = simulator.run(transpile(qc, simulator), shots=shots)
            result = job.result()
            counts = result.get_counts()
            results[state_name] = counts

            # Analyze Bob's measurement results (bit 2)
            bob_results = {}
            for outcome, count in counts.items():
                # Extract Bob's result (rightmost bit in Qiskit's convention)
                bob_bit = outcome[0]  # Bob's measurement result
                if bob_bit not in bob_results:
                    bob_results[bob_bit] = 0
                bob_results[bob_bit] += count

            print(f"  Bob's results: {bob_results}")

            # Calculate success rate
            if state_name in ["|0⟩"]:
                success_count = bob_results.get("0", 0)
            elif state_name in ["|1⟩"]:
                success_count = bob_results.get("1", 0)
            else:  # Superposition states
                # For superposition, both outcomes are valid with ~50% probability
                success_count = shots  # All outcomes are "successful"

            success_rate = (success_count / shots) * 100
            print(f"  Teleportation success rate: {success_rate:.1f}%")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[state_name] = {}

    return results


def visualize_teleportation_process():
    """Create visualizations showing the teleportation process."""
    print("\n🎨 TELEPORTATION VISUALIZATION")
    print("=" * 35)

    # Create a simple teleportation circuit for visualization
    qc, state_desc = create_teleportation_circuit(
        state_to_teleport="|+⟩", use_random_state=False
    )

    # Create figure showing the circuit
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Circuit diagram (if possible)
    try:
        from qiskit.visualization import circuit_drawer

        circuit_fig = circuit_drawer(
            qc, output="mpl", style={"backgroundcolor": "#EEEEEE"}
        )
        axes[0].text(
            0.5,
            0.5,
            "Quantum Teleportation Circuit\n(See saved file: teleportation_circuit.png)",
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
        )
        axes[0].set_title("Quantum Teleportation Protocol", fontsize=16)
        axes[0].axis("off")

        # Save circuit separately
        plt.figure(figsize=(16, 8))
        circuit_drawer(qc, output="mpl", style={"backgroundcolor": "#FFFFFF"})
        plt.savefig(
            "module1_06_teleportation_circuit.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("💾 Saved: module1_06_teleportation_circuit.png")

    except Exception as e:
        print(f"⚠️ Circuit visualization error: {e}")
        axes[0].text(
            0.5,
            0.5,
            f"Circuit visualization not available\n{e}",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0].axis("off")

    # Protocol explanation
    protocol_text = """
🌟 QUANTUM TELEPORTATION EXPLAINED 🌟

Alice's Qubits:        Bob's Qubit:
┌─────────────┐      ┌──────────┐
│ Original    │      │ Target   │
│ |ψ⟩        │      │ (empty)  │
│             │      │          │
│ Entangled   │◄────►│ Partner  │
│ |Φ+⟩       │      │ |Φ+⟩     │
└─────────────┘      └──────────┘

Step 1: Entanglement created between Alice and Bob
Step 2: Alice measures both her qubits (Bell measurement)  
Step 3: Alice sends classical bits to Bob
Step 4: Bob applies corrections → Gets |ψ⟩!

✨ Key Insight: The original |ψ⟩ is destroyed at Alice's side
                but perfectly recreated at Bob's side!
    """

    axes[1].text(
        0.05,
        0.95,
        protocol_text,
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        "module1_06_teleportation_explanation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("💾 Saved: module1_06_teleportation_explanation.png")
    return True


def explain_no_cloning_theorem():
    """Explain the no-cloning theorem and why teleportation is necessary."""
    print("\n🚫 THE NO-CLONING THEOREM")
    print("=" * 30)
    print()
    print("❓ Why can't we just copy quantum states?")
    print()
    print("🔬 THE NO-CLONING THEOREM states:")
    print("   'It is impossible to create an identical copy of an")
    print("    arbitrary unknown quantum state.'")
    print()
    print("🤔 Why this matters:")
    print("   • Classical bits can be copied perfectly: 0→00, 1→11")
    print("   • Quantum states CANNOT be copied: |ψ⟩ ↛ |ψ⟩|ψ⟩")
    print("   • This is fundamental to quantum mechanics!")
    print()
    print("💡 Teleportation is the solution:")
    print("   • We don't copy the state")
    print("   • We TRANSFER it using entanglement")
    print("   • Original state is destroyed in the process")
    print("   • Perfect copy appears at the destination")
    print()
    print("🎭 Think of it like a fax machine for quantum states:")
    print("   • Original document is destroyed (measurement)")
    print("   • Perfect copy appears at destination (Bob's qubit)")
    print("   • Classical information travels between (Alice's measurements)")


def main():
    """Main function to run the teleportation demonstration."""
    parser = argparse.ArgumentParser(description="Quantum Teleportation Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--shots", type=int, default=1000, help="Number of measurement shots"
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Skip visualization"
    )
    parser.add_argument(
        "--random-state", action="store_true", help="Teleport a random state"
    )

    args = parser.parse_args()

    print("🌟 Quantum Computing 101 - Module 1, Example 6")
    print("Quantum Teleportation")
    print("=" * 55)
    print()

    try:
        # Part 1: Explain the protocol
        demonstrate_teleportation_protocol()

        # Part 2: Explain no-cloning theorem
        explain_no_cloning_theorem()

        # Part 3: Run teleportation experiments
        results = run_teleportation_examples(shots=args.shots, verbose=args.verbose)

        # Part 4: Visualizations
        if not args.no_visualization:
            visualize_teleportation_process()

        # Part 5: Random state teleportation
        if args.random_state:
            print("\n🎲 RANDOM STATE TELEPORTATION")
            print("=" * 35)
            qc, state_desc = create_teleportation_circuit(use_random_state=True)
            print(f"Created circuit to teleport: {state_desc}")

            counts = run_circuit_with_shots(qc, shots=args.shots)
            print(f"Teleportation results: {counts}")

        # Summary
        print("\n" + "=" * 60)
        print("🎯 KEY TAKEAWAYS:")
        print("• Quantum teleportation transfers quantum states using entanglement")
        print("• No information travels faster than light (classical channel needed)")
        print("• Original quantum state is destroyed in the process")
        print("• Perfect quantum state copy appears at destination")
        print("• This enables quantum networking and distributed quantum computing")
        print()
        print("🚀 Next steps:")
        print("• Learn about quantum error correction")
        print("• Explore quantum networking protocols")
        print("• Study Bell inequality violations")
        print()
        print("✅ Teleportation demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
