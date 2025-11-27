#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3: Advanced Programming
Example 3: Quantum Circuit Patterns

This script demonstrates common quantum circuit patterns, design principles,
and reusable quantum programming constructs.

Author: Quantum Computing 101 Course
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import (
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, GroverOperator
import warnings

warnings.filterwarnings("ignore")


class QuantumCircuitPatterns:
    """Demonstrate common quantum circuit patterns and design principles."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.patterns_catalog = {}

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[CircuitPatterns] {message}")

    def basic_circuit_patterns(self):
        """Demonstrate fundamental quantum circuit patterns."""
        print("\n=== Basic Circuit Patterns ===")

        patterns = {}

        # 1. State Preparation Pattern
        print("\n1. State Preparation Pattern:")
        state_prep = QuantumCircuit(2)

        # Arbitrary single-qubit state preparation
        theta, phi, lam = np.pi / 3, np.pi / 4, np.pi / 6
        state_prep.u(theta, phi, lam, 0)

        # Bell state preparation
        state_prep.h(1)
        state_prep.cx(1, 0)

        patterns["state_preparation"] = state_prep

        print(f"   Circuit depth: {state_prep.depth()}")
        print(f"   Gate count: {len(state_prep.data)}")

        if self.verbose:
            print("   Pattern: Arbitrary state preparation + entanglement")
            print(f"   {state_prep.draw()}")

        # 2. Unitary Evolution Pattern
        print("\n2. Unitary Evolution Pattern:")
        evolution = QuantumCircuit(3)

        # Time evolution simulation pattern
        t = Parameter("t")
        for i in range(3):
            evolution.rx(2 * t, i)  # Single-qubit evolution

        # Interaction terms
        evolution.cx(0, 1)
        evolution.rz(t, 1)
        evolution.cx(0, 1)

        evolution.cx(1, 2)
        evolution.rz(t, 2)
        evolution.cx(1, 2)

        patterns["unitary_evolution"] = evolution

        print(f"   Parameterized gates: {len(evolution.parameters)}")
        print(f"   Circuit depth: {evolution.depth()}")

        if self.verbose:
            print("   Pattern: Parameterized evolution with interactions")
            print(f"   {evolution.draw()}")

        # 3. Measurement Pattern
        print("\n3. Measurement Pattern:")
        measurement = QuantumCircuit(2, 2)

        # Prepare state
        measurement.h(0)
        measurement.cx(0, 1)

        # Measurement in different bases
        measurement.barrier()

        # Computational basis measurement
        measurement.measure(0, 0)

        # X-basis measurement on qubit 1
        measurement.h(1)
        measurement.measure(1, 1)

        patterns["measurement"] = measurement

        print(f"   Classical bits: {measurement.num_clbits}")
        print(f"   Measurement operations: 2")

        if self.verbose:
            print("   Pattern: Multi-basis measurement")
            print(f"   {measurement.draw()}")

        self.patterns_catalog.update(patterns)
        return patterns

    def advanced_circuit_patterns(self):
        """Demonstrate advanced quantum circuit patterns."""
        print("\n=== Advanced Circuit Patterns ===")

        patterns = {}

        # 1. Variational Ansatz Pattern
        print("\n1. Variational Ansatz Pattern:")

        n_qubits = 4
        n_layers = 3

        # Create parameter vector
        params = ParameterVector("θ", n_qubits * n_layers * 2)

        ansatz = QuantumCircuit(n_qubits)
        param_idx = 0

        for layer in range(n_layers):
            # Single-qubit rotations
            for qubit in range(n_qubits):
                ansatz.ry(params[param_idx], qubit)
                param_idx += 1
                ansatz.rz(params[param_idx], qubit)
                param_idx += 1

            # Entangling gates
            for qubit in range(n_qubits - 1):
                ansatz.cx(qubit, qubit + 1)

            # Optional: Add barrier for visualization
            ansatz.barrier()

        patterns["variational_ansatz"] = ansatz

        print(f"   Qubits: {n_qubits}")
        print(f"   Layers: {n_layers}")
        print(f"   Parameters: {len(ansatz.parameters)}")
        print(f"   Circuit depth: {ansatz.depth()}")

        if self.verbose:
            print("   Pattern: Layered variational ansatz")
            print(f"   Parameters per layer: {n_qubits * 2}")

        # 2. Quantum Fourier Transform Pattern
        print("\n2. Quantum Fourier Transform Pattern:")

        qft_qubits = 4
        qft_circuit = QFT(qft_qubits, do_swaps=False)

        # Create a circuit with QFT pattern
        qft_example = QuantumCircuit(qft_qubits)
        qft_example.compose(qft_circuit, inplace=True)

        patterns["qft"] = qft_example

        print(f"   Qubits: {qft_qubits}")
        print(f"   Gate count: {len(qft_example.data)}")
        print(f"   Circuit depth: {qft_example.depth()}")

        if self.verbose:
            print("   Pattern: Hierarchical controlled rotations")
            print("   Applications: Period finding, phase estimation")

        # 3. Error Correction Pattern
        print("\n3. Error Correction Pattern (3-qubit repetition code):")

        error_correction = QuantumCircuit(
            7, 3
        )  # 3 data + 4 ancilla qubits, 3 syndrome bits

        # Encode logical qubit (data qubit 0) into 3 physical qubits
        error_correction.cx(0, 1)
        error_correction.cx(0, 2)

        # Syndrome measurement
        error_correction.barrier()

        # Measure parity of qubits 0,1
        error_correction.cx(0, 3)
        error_correction.cx(1, 3)
        error_correction.measure(3, 0)

        # Measure parity of qubits 1,2
        error_correction.cx(1, 4)
        error_correction.cx(2, 4)
        error_correction.measure(4, 1)

        # Measure parity of qubits 0,2
        error_correction.cx(0, 5)
        error_correction.cx(2, 5)
        error_correction.measure(5, 2)

        patterns["error_correction"] = error_correction

        print(f"   Data qubits: 3")
        print(f"   Ancilla qubits: 4")
        print(f"   Syndrome bits: 3")

        if self.verbose:
            print("   Pattern: Syndrome extraction + error correction")
            print("   Detects single-qubit X errors")

        # 4. Conditional Quantum Operations Pattern
        print("\n4. Conditional Operations Pattern:")

        conditional = QuantumCircuit(3, 3)

        # Initial state preparation
        conditional.h(0)
        conditional.measure(0, 0)

        # Conditional operations based on measurement
        conditional.barrier()

        # If measurement result is 1, apply operations
        with conditional.if_test((0, 1)):
            conditional.x(1)
            conditional.h(2)

        # Measure final state
        conditional.measure([1, 2], [1, 2])

        patterns["conditional"] = conditional

        print(f"   Conditional blocks: 1")
        print(f"   Operations inside condition: 2")

        if self.verbose:
            print("   Pattern: Classical feedback control")
            print("   Enables adaptive quantum algorithms")

        self.patterns_catalog.update(patterns)
        return patterns

    def reusable_quantum_modules(self):
        """Create reusable quantum circuit modules."""
        print("\n=== Reusable Quantum Modules ===")

        modules = {}

        # 1. Quantum Adder Module
        print("\n1. Quantum Adder Module:")

        def quantum_adder(a_qubits, b_qubits, carry_qubit):
            """Create a quantum adder circuit (simplified)."""
            n_bits = len(a_qubits)
            qc = QuantumCircuit(a_qubits + b_qubits + [carry_qubit])

            # Simplified adder implementation (XOR for addition without carry propagation)
            for i in range(n_bits):
                # XOR operation: a XOR b
                qc.cx(a_qubits[i], b_qubits[i])
                
            # Compute final carry (AND of highest bits)
            if n_bits > 0:
                qc.ccx(a_qubits[-1], b_qubits[-1], carry_qubit)

            return qc

        # Example usage
        a_reg = QuantumRegister(3, "a")
        b_reg = QuantumRegister(3, "b")
        carry_reg = QuantumRegister(1, "carry")

        adder_circuit = QuantumCircuit(a_reg, b_reg, carry_reg)
        adder_module = quantum_adder(list(a_reg), list(b_reg), carry_reg[0])
        adder_circuit.compose(adder_module, inplace=True)

        modules["quantum_adder"] = adder_circuit

        print(f"   Input bits: {len(a_reg)}")
        print(f"   Total qubits: {adder_circuit.num_qubits}")
        print(f"   Gate count: {len(adder_circuit.data)}")

        # 2. Quantum Comparator Module
        print("\n2. Quantum Comparator Module:")

        def quantum_comparator(a_qubits, b_qubits, result_qubit):
            """Create a quantum comparator (a > b)."""
            n_bits = len(a_qubits)
            qc = QuantumCircuit(a_qubits + b_qubits + [result_qubit])

            # Compare from most significant bit
            for i in range(n_bits - 1, -1, -1):
                # If a_i = 1 and b_i = 0, then a > b
                qc.x(b_qubits[i])
                qc.ccx(a_qubits[i], b_qubits[i], result_qubit)
                qc.x(b_qubits[i])

                # If we already determined a > b, don't change result
                # (This is simplified; full implementation needs more logic)

            return qc

        comp_a = QuantumRegister(3, "comp_a")
        comp_b = QuantumRegister(3, "comp_b")
        comp_result = QuantumRegister(1, "result")

        comparator_circuit = QuantumCircuit(comp_a, comp_b, comp_result)
        comp_module = quantum_comparator(list(comp_a), list(comp_b), comp_result[0])
        comparator_circuit.compose(comp_module, inplace=True)

        modules["quantum_comparator"] = comparator_circuit

        print(f"   Input bits: {len(comp_a)}")
        print(f"   Output bits: 1")
        print(f"   Gate count: {len(comparator_circuit.data)}")

        # 3. Quantum Random Number Generator Module
        print("\n3. Quantum Random Number Generator:")

        def qrng_module(n_bits):
            """Create a quantum random number generator."""
            qc = QuantumCircuit(n_bits, n_bits)

            # Apply Hadamard to all qubits
            for i in range(n_bits):
                qc.h(i)

            # Measure all qubits
            qc.measure_all()

            return qc

        qrng_bits = 4
        qrng_circuit = qrng_module(qrng_bits)

        modules["qrng"] = qrng_circuit

        print(f"   Random bits: {qrng_bits}")
        print(f"   Entropy: {qrng_bits} bits")
        print(f"   Gate count: {len(qrng_circuit.data)}")

        # 4. Quantum State Teleportation Module
        print("\n4. Quantum Teleportation Module:")

        def teleportation_module():
            """Create a quantum teleportation circuit."""
            qc = QuantumCircuit(3, 2)

            # Prepare Bell pair between qubits 1 and 2
            qc.h(1)
            qc.cx(1, 2)

            # Bell measurement on qubits 0 and 1
            qc.cx(0, 1)
            qc.h(0)
            qc.measure([0, 1], [0, 1])

            # Conditional operations on qubit 2
            qc.barrier()
            with qc.if_test((1, 1)):
                qc.x(2)
            with qc.if_test((0, 1)):
                qc.z(2)

            return qc

        teleport_circuit = teleportation_module()

        modules["teleportation"] = teleport_circuit

        print(f"   Qubits: 3 (sender, ancilla, receiver)")
        print(f"   Classical bits: 2")
        print(f"   Fidelity: 100% (ideal)")

        self.patterns_catalog.update(modules)
        return modules

    def circuit_optimization_patterns(self):
        """Demonstrate circuit optimization patterns."""
        print("\n=== Circuit Optimization Patterns ===")

        optimizations = {}

        # 1. Gate Decomposition Pattern
        print("\n1. Gate Decomposition Optimization:")

        # Original circuit with complex gates
        original = QuantumCircuit(2)
        original.u(np.pi / 4, np.pi / 3, np.pi / 6, 0)
        original.cu(np.pi / 5, np.pi / 7, np.pi / 11, np.pi / 13, 0, 1)

        # Decomposed version
        decomposed = QuantumCircuit(2)

        # Decompose U gate into elementary gates
        theta, phi, lam = np.pi / 4, np.pi / 3, np.pi / 6
        decomposed.rz(phi, 0)
        decomposed.ry(theta, 0)
        decomposed.rz(lam, 0)

        # Decompose controlled-U
        # (Simplified decomposition)
        decomposed.ry(np.pi / 10, 1)
        decomposed.cx(0, 1)
        decomposed.ry(-np.pi / 10, 1)
        decomposed.cx(0, 1)
        decomposed.rz(np.pi / 13, 0)

        optimizations["original"] = original
        optimizations["decomposed"] = decomposed

        print(f"   Original gates: {len(original.data)}")
        print(f"   Decomposed gates: {len(decomposed.data)}")
        print(
            f"   Basis gates only: {'Yes' if all(gate.name in ['rx', 'ry', 'rz', 'cx'] for gate in decomposed.data) else 'No'}"
        )

        # 2. Circuit Depth Optimization
        print("\n2. Circuit Depth Optimization:")

        # Create a circuit with unnecessary depth
        deep_circuit = QuantumCircuit(4)
        for i in range(4):
            deep_circuit.h(i)
            deep_circuit.barrier()
        for i in range(3):
            deep_circuit.cx(i, i + 1)
            deep_circuit.barrier()

        # Optimized version with parallel operations
        shallow_circuit = QuantumCircuit(4)
        # All Hadamards can be applied in parallel
        for i in range(4):
            shallow_circuit.h(i)
        shallow_circuit.barrier()

        # CNOTs can be partially parallelized
        shallow_circuit.cx(0, 1)
        shallow_circuit.cx(2, 3)
        shallow_circuit.barrier()
        shallow_circuit.cx(1, 2)

        optimizations["deep_circuit"] = deep_circuit
        optimizations["shallow_circuit"] = shallow_circuit

        print(f"   Original depth: {deep_circuit.depth()}")
        print(f"   Optimized depth: {shallow_circuit.depth()}")
        print(
            f"   Depth reduction: {(deep_circuit.depth() - shallow_circuit.depth())/deep_circuit.depth()*100:.1f}%"
        )

        # 3. Gate Count Optimization
        print("\n3. Gate Count Optimization:")

        # Circuit with redundant operations
        redundant = QuantumCircuit(2)
        redundant.x(0)
        redundant.x(0)  # Double X = identity
        redundant.h(1)
        redundant.z(1)
        redundant.h(1)  # HZH = X
        redundant.cx(0, 1)
        redundant.cx(0, 1)  # Double CNOT = identity

        # Optimized version
        optimized = QuantumCircuit(2)
        optimized.x(1)  # Result of HZH sequence

        optimizations["redundant"] = redundant
        optimizations["optimized"] = optimized

        print(f"   Original gates: {len(redundant.data)}")
        print(f"   Optimized gates: {len(optimized.data)}")
        print(
            f"   Gate reduction: {(len(redundant.data) - len(optimized.data))/len(redundant.data)*100:.1f}%"
        )

        # Verify equivalence
        original_state = Statevector.from_instruction(redundant)
        optimized_state = Statevector.from_instruction(optimized)
        fidelity = abs(np.vdot(original_state.data, optimized_state.data)) ** 2

        print(f"   State fidelity: {fidelity:.10f}")
        print(f"   Equivalent: {'Yes' if fidelity > 0.9999 else 'No'}")

        self.patterns_catalog.update(optimizations)
        return optimizations

    def analyze_circuit_complexity(self, circuits_dict):
        """Analyze complexity metrics for different circuit patterns."""
        print("\n=== Circuit Complexity Analysis ===")

        complexity_metrics = {}

        for name, circuit in circuits_dict.items():
            if hasattr(circuit, "depth"):
                metrics = {
                    "num_qubits": circuit.num_qubits,
                    "num_clbits": circuit.num_clbits,
                    "depth": circuit.depth(),
                    "size": circuit.size(),
                    "gate_count": len(circuit.data),
                    "cx_count": circuit.count_ops().get("cx", 0),
                    "single_qubit_gates": sum(
                        1 for gate in circuit.data if len(gate.qubits) == 1
                    ),
                    "two_qubit_gates": sum(
                        1 for gate in circuit.data if len(gate.qubits) == 2
                    ),
                    "parameters": (
                        len(circuit.parameters) if hasattr(circuit, "parameters") else 0
                    ),
                }

                complexity_metrics[name] = metrics

                print(f"\n{name}:")
                print(f"   Qubits: {metrics['num_qubits']}")
                print(f"   Depth: {metrics['depth']}")
                print(f"   Total gates: {metrics['gate_count']}")
                print(f"   Two-qubit gates: {metrics['two_qubit_gates']}")
                print(f"   Parameters: {metrics['parameters']}")

                if self.verbose:
                    print(f"   Classical bits: {metrics['num_clbits']}")
                    print(f"   CNOT gates: {metrics['cx_count']}")
                    print(f"   Single-qubit gates: {metrics['single_qubit_gates']}")

        return complexity_metrics

    def visualize_circuit_patterns(self, circuits_dict):
        """Visualize circuit patterns and their properties."""
        # Filter out circuits that are too large for visualization
        visualizable_circuits = {
            name: circuit
            for name, circuit in circuits_dict.items()
            if hasattr(circuit, "depth")
            and circuit.num_qubits <= 4
            and circuit.depth() <= 20
        }

        if not visualizable_circuits:
            print("\nNo circuits suitable for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. Circuit depth comparison
        names = list(visualizable_circuits.keys())
        depths = [circuit.depth() for circuit in visualizable_circuits.values()]

        axes[0].bar(range(len(names)), depths, alpha=0.7, color="skyblue")
        axes[0].set_xlabel("Circuit Pattern")
        axes[0].set_ylabel("Circuit Depth")
        axes[0].set_title("Circuit Depth Comparison")
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=45, ha="right")
        axes[0].grid(True, alpha=0.3)

        # 2. Gate count comparison
        gate_counts = [circuit.size() for circuit in visualizable_circuits.values()]

        axes[1].bar(range(len(names)), gate_counts, alpha=0.7, color="lightcoral")
        axes[1].set_xlabel("Circuit Pattern")
        axes[1].set_ylabel("Total Gates")
        axes[1].set_title("Gate Count Comparison")
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right")
        axes[1].grid(True, alpha=0.3)

        # 3. Two-qubit gate ratio
        two_qubit_ratios = []
        for circuit in visualizable_circuits.values():
            total_gates = len(circuit.data)
            two_qubit_gates = sum(1 for gate in circuit.data if len(gate.qubits) == 2)
            ratio = two_qubit_gates / total_gates if total_gates > 0 else 0
            two_qubit_ratios.append(ratio)

        axes[2].bar(range(len(names)), two_qubit_ratios, alpha=0.7, color="lightgreen")
        axes[2].set_xlabel("Circuit Pattern")
        axes[2].set_ylabel("Two-Qubit Gate Ratio")
        axes[2].set_title("Two-Qubit Gate Density")
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, rotation=45, ha="right")
        axes[2].grid(True, alpha=0.3)

        # 4. Qubit utilization
        qubit_counts = [
            circuit.num_qubits for circuit in visualizable_circuits.values()
        ]

        axes[3].bar(range(len(names)), qubit_counts, alpha=0.7, color="orange")
        axes[3].set_xlabel("Circuit Pattern")
        axes[3].set_ylabel("Number of Qubits")
        axes[3].set_title("Qubit Utilization")
        axes[3].set_xticks(range(len(names)))
        axes[3].set_xticklabels(names, rotation=45, ha="right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()

    def generate_summary_report(self):
        """Generate comprehensive summary of circuit patterns analysis."""
        print("\n" + "=" * 60)
        print("QUANTUM CIRCUIT PATTERNS - ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 Patterns Demonstrated: {len(self.patterns_catalog)}")

        print("\n🔬 Pattern Categories:")
        print("  • Basic patterns: State preparation, evolution, measurement")
        print("  • Advanced patterns: Variational ansatz, QFT, error correction")
        print("  • Reusable modules: Adder, comparator, QRNG, teleportation")
        print("  • Optimization patterns: Decomposition, depth, gate count")

        print("\n📚 Design Principles:")
        print("  • Modularity: Reusable circuit components")
        print("  • Parameterization: Flexible circuit families")
        print("  • Optimization: Depth and gate count reduction")
        print("  • Abstraction: High-level quantum operations")

        print("\n🎯 Key Insights:")
        print("  • Circuit patterns enable systematic quantum programming")
        print("  • Modular design improves maintainability")
        print("  • Optimization reduces hardware requirements")
        print("  • Parameterization enables variational algorithms")

        print("\n🚀 Applications:")
        print("  → Quantum algorithm development")
        print("  → Variational quantum algorithms")
        print("  → Quantum error correction")
        print("  → Hardware-efficient circuit design")

        print("\n💡 Best Practices:")
        print("  → Use standard patterns as building blocks")
        print("  → Optimize for target hardware constraints")
        print("  → Validate circuit equivalence after optimization")
        print("  → Document reusable modules clearly")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Patterns and Design Principles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_quantum_circuit_patterns.py
  python 03_quantum_circuit_patterns.py --verbose
  python 03_quantum_circuit_patterns.py --show-visualization
  python 03_quantum_circuit_patterns.py --pattern-type basic
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed information",
    )
    parser.add_argument(
        "--pattern-type",
        choices=["basic", "advanced", "modules", "optimization", "all"],
        default="all",
        help="Focus on specific pattern type",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display circuit pattern visualizations",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis without circuit drawings",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 3: Advanced Programming")
    print("Example 3: Quantum Circuit Patterns")
    print("=" * 45)

    # Initialize pattern analyzer
    pattern_analyzer = QuantumCircuitPatterns(verbose=args.verbose)

    try:
        all_circuits = {}

        # Run selected pattern types
        if args.pattern_type in ["basic", "all"]:
            basic_patterns = pattern_analyzer.basic_circuit_patterns()
            all_circuits.update(basic_patterns)

        if args.pattern_type in ["advanced", "all"]:
            advanced_patterns = pattern_analyzer.advanced_circuit_patterns()
            all_circuits.update(advanced_patterns)

        if args.pattern_type in ["modules", "all"]:
            modules = pattern_analyzer.reusable_quantum_modules()
            all_circuits.update(modules)

        if args.pattern_type in ["optimization", "all"]:
            optimizations = pattern_analyzer.circuit_optimization_patterns()
            all_circuits.update(optimizations)

        # Analyze circuit complexity
        complexity_metrics = pattern_analyzer.analyze_circuit_complexity(all_circuits)

        # Visualization (optional)
        if args.show_visualization and not args.analysis_only:
            pattern_analyzer.visualize_circuit_patterns(all_circuits)

        # Generate summary
        pattern_analyzer.generate_summary_report()

        print(f"\n✅ Circuit pattern analysis completed successfully!")
        print(f"📊 Analyzed {len(all_circuits)} circuit patterns")

        if args.verbose:
            print(f"🔍 Detailed analysis enabled")
            print(f"📈 Use --show-visualization for charts")

    except Exception as e:
        print(f"\n❌ Error during pattern analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
