#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3, Example 6
Quantum Debugging Guide for Beginners

This example provides a comprehensive guide to debugging quantum programs,
including common errors, debugging strategies, and tools for understanding
what's happening in quantum circuits.

Learning objectives:
- Learn common quantum programming errors and how to fix them
- Understand debugging strategies specific to quantum computing
- Explore tools for visualizing and analyzing quantum circuits
- Practice troubleshooting quantum programs step-by-step

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector, DensityMatrix

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
import warnings
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from quantum_helpers import run_circuit_with_shots


def common_quantum_errors_demo():
    """Demonstrate the most common errors beginners make."""
    print("=== COMMON QUANTUM PROGRAMMING ERRORS ===")
    print()

    print("🚫 ERROR 1: FORGOTTEN MEASUREMENTS")
    print("Problem: Quantum circuit without measurement gives no classical output")
    print()

    # Bad example - no measurement
    print("❌ BAD CODE:")
    qc_bad = QuantumCircuit(2)
    qc_bad.h(0)
    qc_bad.cx(0, 1)
    print("qc = QuantumCircuit(2)")
    print("qc.h(0)")
    print("qc.cx(0, 1)")
    print("# Missing: qc.measure_all() or explicit measurements")
    print()

    # Good example - with measurement
    print("✅ FIXED CODE:")
    qc_good = QuantumCircuit(2, 2)
    qc_good.h(0)
    qc_good.cx(0, 1)
    qc_good.measure_all()
    print("qc = QuantumCircuit(2, 2)  # Include classical registers!")
    print("qc.h(0)")
    print("qc.cx(0, 1)")
    print("qc.measure_all()  # Essential for getting results!")
    print()

    print("🚫 ERROR 2: WRONG QUBIT INDEXING")
    print("Problem: Qubits are 0-indexed, not 1-indexed")
    print()

    print("❌ BAD CODE (for 3-qubit circuit):")
    print("qc.h(3)  # Error! Valid indices are 0, 1, 2")
    print()
    print("✅ FIXED CODE:")
    print("qc.h(2)  # Correct! Last qubit in 3-qubit system")
    print()

    print("🚫 ERROR 3: CLASSICAL REGISTER MISMATCH")
    print("Problem: Number of classical bits doesn't match measurements")
    print()

    print("❌ BAD CODE:")
    print("qc = QuantumCircuit(3, 2)  # 3 qubits, 2 classical bits")
    print("qc.measure_all()  # Tries to measure 3 qubits into 2 bits!")
    print()
    print("✅ FIXED CODE:")
    print("qc = QuantumCircuit(3, 3)  # Matching numbers")
    print("# OR measure only some qubits:")
    print("qc.measure(0, 0)  # Measure qubit 0 into bit 0")
    print()

    return qc_good


def measurement_timing_issues():
    """Demonstrate issues with measurement timing."""
    print("=== MEASUREMENT TIMING ISSUES ===")
    print()

    print("🚫 ERROR 4: OPERATIONS AFTER MEASUREMENT")
    print("Problem: Cannot perform quantum gates after measuring a qubit")
    print()

    print("❌ BAD CODE:")
    print("qc.h(0)         # Create superposition")
    print("qc.measure(0, 0)  # Measure qubit 0")
    print("qc.x(0)         # ERROR: Cannot gate measured qubit!")
    print()

    print("✅ UNDERSTANDING THE PROBLEM:")
    qc_demo = QuantumCircuit(1, 1)
    qc_demo.h(0)
    print("Before measurement: |+⟩ = (|0⟩ + |1⟩)/√2")

    # Show state before measurement
    state_before = Statevector.from_instruction(qc_demo)
    print(f"State vector: {state_before.data}")

    qc_demo.measure(0, 0)
    print("After measurement: |0⟩ OR |1⟩ (collapsed!)")
    print("The superposition is gone - no more quantum operations possible!")
    print()

    print("🚫 ERROR 5: MID-CIRCUIT MEASUREMENTS WITHOUT CONDITIONALS")
    print("Problem: Measuring in the middle without using the result")
    print()

    print("❌ CONFUSING CODE:")
    qc_confusing = QuantumCircuit(2, 2)
    qc_confusing.h(0)
    qc_confusing.measure(0, 0)  # Mid-circuit measurement
    qc_confusing.cx(0, 1)  # What state is qubit 0 in now?
    qc_confusing.measure(1, 1)
    print("qc.h(0)")
    print("qc.measure(0, 0)  # Collapses superposition")
    print("qc.cx(0, 1)       # Qubit 0 is now |0⟩ or |1⟩, not |+⟩!")
    print()

    print("✅ CLEARER APPROACH:")
    print("Either use conditional operations:")
    print("qc.h(0)")
    print("qc.measure(0, 0)")
    print("with qc.if_test((creg, 0), 1):")  # Modern Qiskit syntax
    print("    qc.x(1)  # Only if measurement was 1")
    print("OR avoid mid-circuit measurements until you understand them!")
    print()


def state_visualization_debugging():
    """Show how to debug using state visualization."""
    print("=== STATE VISUALIZATION FOR DEBUGGING ===")
    print()

    print("🔍 DEBUGGING STRATEGY: VISUALIZE THE STATE")
    print("Instead of guessing what your circuit does, look at the quantum state!")
    print()

    # Create a potentially buggy circuit
    print("Example: Debugging a 'Bell state' circuit")
    print()

    print("INTENDED CIRCUIT (Bell state):")
    qc_intended = QuantumCircuit(2)
    qc_intended.h(0)
    qc_intended.cx(0, 1)

    print("qc.h(0)    # Superposition")
    print("qc.cx(0, 1)  # Entanglement")
    print("Expected: (|00⟩ + |11⟩)/√2")
    print()

    # Show intended state
    intended_state = Statevector.from_instruction(qc_intended)
    print(f"Intended state: {intended_state.data}")
    print()

    # Create a buggy version
    print("BUGGY CIRCUIT (common mistake):")
    qc_buggy = QuantumCircuit(2)
    qc_buggy.h(0)
    qc_buggy.h(1)  # BUG: H on both qubits instead of CNOT
    qc_buggy.cx(0, 1)

    print("qc.h(0)    # Superposition on qubit 0")
    print("qc.h(1)    # BUG: Superposition on qubit 1 too!")
    print("qc.cx(0, 1)  # CNOT")
    print()

    # Show buggy state
    buggy_state = Statevector.from_instruction(qc_buggy)
    print(f"Actual state: {buggy_state.data}")
    print()

    # Compare states
    fidelity_score = fidelity(intended_state, buggy_state)
    print(f"State fidelity: {fidelity_score:.4f}")
    print("(1.0 = identical, 0.0 = orthogonal)")
    print()

    print("🔍 DEBUGGING STEPS:")
    print("1. Check state after each gate")
    print("2. Compare with expected theoretical state")
    print("3. Use fidelity to quantify differences")
    print("4. Visualize on Bloch sphere if single qubit")
    print()


def circuit_construction_debugging():
    """Debug issues with circuit construction."""
    print("=== CIRCUIT CONSTRUCTION DEBUGGING ===")
    print()

    print("🔧 DEBUGGING TOOL: CIRCUIT INSPECTION")
    print("Always inspect your circuit before running it!")
    print()

    # Example circuit with potential issues
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    print("Example circuit:")
    print(qc.draw())
    print()

    print("🔍 INSPECTION CHECKLIST:")
    print(f"✅ Number of qubits: {qc.num_qubits}")
    print(f"✅ Number of classical bits: {qc.num_clbits}")
    print(f"✅ Circuit depth: {qc.depth()}")
    print(f"✅ Gate count: {sum(qc.count_ops().values())}")
    print(f"✅ Gate types: {list(qc.count_ops().keys())}")
    print()

    print("🔧 COMMON CONSTRUCTION ERRORS:")
    print()

    print("ERROR: Wrong gate order")
    print("❌ qc.cx(0, 1); qc.h(0)  # H after CNOT changes meaning!")
    print("✅ qc.h(0); qc.cx(0, 1)  # H before CNOT for Bell state")
    print()

    print("ERROR: Swapped control/target")
    print("❌ qc.cx(1, 0)  # Qubit 1 controls qubit 0")
    print("✅ qc.cx(0, 1)  # Qubit 0 controls qubit 1")
    print()

    print("ERROR: Wrong parameter values")
    print("❌ qc.ry(90, 0)    # Angle in degrees - wrong!")
    print("✅ qc.ry(np.pi/2, 0)  # Angle in radians - correct!")
    print()


def measurement_analysis_debugging():
    """Debug measurement and results analysis."""
    print("=== MEASUREMENT ANALYSIS DEBUGGING ===")
    print()

    print("🎲 DEBUGGING MEASUREMENT RESULTS")
    print("Quantum results are probabilistic - analyze them statistically!")
    print()

    # Create a simple circuit for demonstration
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # 50/50 superposition
    qc.cx(0, 1)  # Bell state
    qc.measure_all()

    print("Test circuit: Bell state")
    print("Expected: 50% |00⟩, 50% |11⟩, 0% |01⟩ or |10⟩")
    print()

    # Run with different shot counts to show debugging
    simulator = AerSimulator()
    shot_counts = [10, 100, 1000, 10000]

    print("🔍 SHOT COUNT DEBUGGING:")
    for shots in shot_counts:
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=shots)
        counts = job.result().get_counts()

        print(f"\nShots: {shots}")
        for outcome, count in sorted(counts.items()):
            percentage = 100 * count / shots
            print(f"  {outcome}: {count:4d} ({percentage:5.1f}%)")

        # Check for unexpected outcomes
        unexpected = set(counts.keys()) - {"00", "11"}
        if unexpected:
            print(f"  ⚠️  Unexpected outcomes: {unexpected}")

    print()
    print("🔧 MEASUREMENT DEBUGGING TIPS:")
    print("1. Use enough shots (≥1000) for reliable statistics")
    print("2. Check for unexpected measurement outcomes")
    print("3. Compare experimental vs theoretical probabilities")
    print("4. Look for systematic deviations (not just noise)")
    print()


def simulator_vs_hardware_debugging():
    """Debug differences between simulator and real hardware."""
    print("=== SIMULATOR vs HARDWARE DEBUGGING ===")
    print()

    print("🖥️  SIMULATOR BEHAVIOR:")
    print("- Perfect gates (no errors)")
    print("- Infinite coherence time")
    print("- Perfect measurements")
    print("- All-to-all connectivity")
    print("- No noise or decoherence")
    print()

    print("🔧 REAL HARDWARE BEHAVIOR:")
    print("- Gate errors (~0.1-1%)")
    print("- Limited coherence time (~μs)")
    print("- Measurement errors (~1-5%)")
    print("- Limited connectivity")
    print("- Noise and decoherence")
    print()

    # Simulate the difference
    print("🧪 SIMULATING HARDWARE EFFECTS:")

    # Perfect circuit
    qc_perfect = QuantumCircuit(2, 2)
    qc_perfect.h(0)
    qc_perfect.cx(0, 1)
    qc_perfect.measure_all()

    # Add some noise to simulate hardware
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # Create noise model
    noise_model = NoiseModel()
    gate_error = 0.01  # 1% error rate
    measurement_error = 0.02  # 2% error rate

    # Add errors to gates
    error_1q = depolarizing_error(gate_error, 1)
    error_2q = depolarizing_error(gate_error, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    print("Comparing perfect vs noisy simulation...")

    # Run perfect simulation
    simulator = AerSimulator()
    transpiled_perfect = transpile(qc_perfect, simulator)
    job_perfect = simulator.run(transpiled_perfect, shots=1000)
    counts_perfect = job_perfect.result().get_counts()

    # Run noisy simulation
    transpiled_noisy = transpile(qc_perfect, simulator)
    job_noisy = simulator.run(transpiled_noisy, shots=1000, noise_model=noise_model)
    counts_noisy = job_noisy.result().get_counts()

    print("\nPerfect simulator results:")
    for outcome, count in sorted(counts_perfect.items()):
        print(f"  {outcome}: {count:4d} ({100*count/1000:5.1f}%)")

    print("\nNoisy simulator results:")
    for outcome, count in sorted(counts_noisy.items()):
        print(f"  {outcome}: {count:4d} ({100*count/1000:5.1f}%)")

    print()
    print("🔧 HARDWARE DEBUGGING TIPS:")
    print("1. Test on simulator first (faster, easier to debug)")
    print("2. Use noise models to approximate hardware")
    print("3. Expect different results on real hardware")
    print("4. Error rates increase with circuit depth")
    print("5. Calibration data helps explain hardware behavior")


def debugging_checklist():
    """Provide a comprehensive debugging checklist."""
    print("=== QUANTUM DEBUGGING CHECKLIST ===")
    print()

    checklist = {
        "🏗️  Circuit Construction": [
            "□ Correct number of qubits and classical bits",
            "□ Gates applied in correct order",
            "□ Control/target qubits correct for 2-qubit gates",
            "□ Parameter values in correct units (radians, not degrees)",
            "□ No gates applied after measuring the same qubit",
        ],
        "📏 Measurements": [
            "□ All qubits have corresponding classical bits",
            "□ measure_all() or explicit measurements included",
            "□ Mid-circuit measurements used properly",
            "□ Conditional operations implemented correctly",
        ],
        "🧪 State Analysis": [
            "□ Statevector matches theoretical expectations",
            "□ Probability amplitudes are normalized",
            "□ Complex phases are correct",
            "□ Entanglement structure is as intended",
        ],
        "🎲 Results Analysis": [
            "□ Sufficient number of shots (≥1000)",
            "□ Results match theoretical probabilities",
            "□ No unexpected measurement outcomes",
            "□ Statistical fluctuations within expected range",
        ],
        "🔧 Hardware Considerations": [
            "□ Circuit transpiled for target hardware",
            "□ Connectivity constraints satisfied",
            "□ Circuit depth reasonable for hardware coherence",
            "□ Error rates and noise effects considered",
        ],
    }

    for category, items in checklist.items():
        print(f"{category}:")
        for item in items:
            print(f"  {item}")
        print()

    print("💡 DEBUGGING WORKFLOW:")
    print("1. 🔍 Inspect circuit visually (draw())")
    print("2. 🧮 Check state theoretically (Statevector)")
    print("3. 🎲 Run with many shots (≥1000)")
    print("4. 📊 Analyze measurement statistics")
    print("5. 🔧 Test on noisy simulator")
    print("6. 🏃‍♂️ Try on real hardware")
    print()


def interactive_debugging_demo():
    """Interactive debugging demonstration."""
    print("=== INTERACTIVE DEBUGGING DEMO ===")
    print()

    print("🐛 DEBUG THIS CIRCUIT:")
    print("A student wants to create the state |10⟩ (qubit 0 in |1⟩, qubit 1 in |0⟩)")
    print()

    # Student's buggy attempt
    qc_student = QuantumCircuit(2, 2)
    qc_student.h(0)  # BUG: Creates superposition instead of |1⟩
    qc_student.measure_all()

    print("Student's code:")
    print("qc = QuantumCircuit(2, 2)")
    print("qc.h(0)  # Student thinks this creates |1⟩")
    print("qc.measure_all()")
    print()

    print("🔍 DEBUGGING ANALYSIS:")

    # Analyze the state
    qc_statevector = qc_student.copy()
    qc_statevector.remove_final_measurements(
        inplace=False
    )  # Remove measurements for statevector

    state = Statevector.from_instruction(qc_statevector)
    print(f"Actual state: {state.data}")
    print("Expected |10⟩: [0, 0, 1, 0] (in computational basis)")
    print()

    # Run simulation
    simulator = AerSimulator()
    transpiled = transpile(qc_student, simulator)
    job = simulator.run(transpiled, shots=1000)
    counts = job.result().get_counts()

    print("Measurement results:")
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count} times ({100*count/1000:.1f}%)")

    print()
    print("🐛 PROBLEM IDENTIFIED:")
    print("- H gate creates superposition (|0⟩ + |1⟩)/√2, not |1⟩")
    print("- Student gets 50% |00⟩ and 50% |10⟩ instead of 100% |10⟩")
    print()

    print("✅ CORRECT SOLUTION:")
    qc_fixed = QuantumCircuit(2, 2)
    qc_fixed.x(0)  # X gate creates |1⟩ from |0⟩
    qc_fixed.measure_all()

    print("qc = QuantumCircuit(2, 2)")
    print("qc.x(0)  # X gate flips |0⟩ to |1⟩")
    print("qc.measure_all()")
    print()

    # Verify the fix
    transpiled_fixed = transpile(qc_fixed, simulator)
    job_fixed = simulator.run(transpiled_fixed, shots=1000)
    counts_fixed = job_fixed.result().get_counts()

    print("Fixed circuit results:")
    for outcome, count in sorted(counts_fixed.items()):
        print(f"  {outcome}: {count} times ({100*count/1000:.1f}%)")

    print("✅ Success! 100% |10⟩ state as intended.")


def main():
    parser = argparse.ArgumentParser(description="Quantum Debugging Guide")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive debugging demos"
    )
    parser.add_argument(
        "--skip-noise", action="store_true", help="Skip noise model demonstrations"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed explanations"
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 3: Programming")
    print("Example 6: Quantum Debugging Guide for Beginners")
    print("=" * 55)

    try:
        print("\n🐛 Welcome to Quantum Debugging!")
        print("Learn to troubleshoot quantum programs like a pro.")
        print()

        # Common errors
        good_circuit = common_quantum_errors_demo()

        # Measurement timing
        measurement_timing_issues()

        # State visualization
        state_visualization_debugging()

        # Circuit construction
        circuit_construction_debugging()

        # Measurement analysis
        measurement_analysis_debugging()

        # Simulator vs hardware
        if not args.skip_noise:
            simulator_vs_hardware_debugging()

        # Debugging checklist
        debugging_checklist()

        # Interactive demo
        if args.interactive:
            interactive_debugging_demo()

        print("🎓 DEBUGGING MASTERY TIPS:")
        print("=" * 40)
        print("1. 🔍 Always visualize your circuit first")
        print("2. 🧮 Check states match your expectations")
        print("3. 🎲 Use enough shots for reliable statistics")
        print("4. 🔧 Test on simulators before hardware")
        print("5. 📚 Understand quantum mechanics fundamentals")
        print("6. 🤝 Ask for help in quantum computing communities")
        print()

        print("✅ Quantum debugging guide completed!")
        print()
        print("💡 Next Steps:")
        print("- Practice debugging with real quantum algorithms")
        print("- Try debugging on actual quantum hardware")
        print("- Learn advanced debugging tools (StrangeFX, Quantum Inspector)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("This is a perfect example of debugging in action!")
        print("Check that you have installed: pip install qiskit qiskit-aer matplotlib")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
