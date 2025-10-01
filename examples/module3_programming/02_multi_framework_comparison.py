#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3: Advanced Programming
Example 2: Multi-Framework Comparison

This script compares quantum circuit implementations across different
quantum computing frameworks: Qiskit, Cirq, and PennyLane.

Author: Quantum Computing 101 Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

# Framework imports with fallback handling
frameworks_available = {}

# Qiskit
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    frameworks_available["qiskit"] = True
except ImportError:
    frameworks_available["qiskit"] = False

# Cirq
try:
    import cirq

    frameworks_available["cirq"] = True
except ImportError:
    frameworks_available["cirq"] = False

# PennyLane
try:
    import pennylane as qml

    frameworks_available["pennylane"] = True
except ImportError:
    frameworks_available["pennylane"] = False


class MultiFrameworkComparator:
    """Compare quantum circuit implementations across multiple frameworks."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.comparison_results = {}

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[MultiFramework] {message}")

    def check_framework_availability(self):
        """Check which quantum frameworks are available."""
        print("\n=== Framework Availability Check ===")

        for framework, available in frameworks_available.items():
            status = "✅ Available" if available else "❌ Not installed"
            print(f"{framework.capitalize()}: {status}")

            if not available:
                if framework == "cirq":
                    print("  Install with: pip install cirq")
                elif framework == "pennylane":
                    print("  Install with: pip install pennylane")

        available_count = sum(frameworks_available.values())
        print(f"\nTotal available frameworks: {available_count}/3")

        if available_count == 0:
            print("⚠️  Warning: No quantum frameworks available beyond Qiskit")

        return frameworks_available

    def implement_bell_state_circuits(self):
        """Implement Bell state preparation in different frameworks."""
        print("\n=== Bell State Circuit Implementation ===")

        circuits = {}

        # Qiskit implementation
        if frameworks_available["qiskit"]:
            print("\n1. Qiskit Implementation:")
            qc_qiskit = QuantumCircuit(2)
            qc_qiskit.h(0)
            qc_qiskit.cx(0, 1)

            circuits["qiskit"] = qc_qiskit
            print(f"   Circuit depth: {qc_qiskit.depth()}")
            print(f"   Gate count: {len(qc_qiskit.data)}")

            if self.verbose:
                print("   Circuit:")
                print(f"   {qc_qiskit.draw()}")

        # Cirq implementation
        if frameworks_available["cirq"]:
            print("\n2. Cirq Implementation:")
            q0, q1 = cirq.LineQubit.range(2)
            circuit_cirq = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

            circuits["cirq"] = circuit_cirq
            print(f"   Circuit depth: {len(circuit_cirq)}")
            print(f"   Gate count: {len(list(circuit_cirq.all_operations()))}")

            if self.verbose:
                print("   Circuit:")
                print(f"   {circuit_cirq}")

        # PennyLane implementation
        if frameworks_available["pennylane"]:
            print("\n3. PennyLane Implementation:")

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def bell_circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.state()

            circuits["pennylane"] = bell_circuit

            # Execute to get circuit info
            state = bell_circuit()

            print(f"   Device: {dev.name}")
            print(f"   Wires: {dev.num_wires}")

            if self.verbose:
                print("   Circuit function defined with @qml.qnode decorator")
                print(f"   Resulting state: {state}")

        return circuits

    def compare_circuit_execution(self, circuits):
        """Compare execution performance across frameworks."""
        print("\n=== Circuit Execution Comparison ===")

        execution_results = {}

        # Qiskit execution
        if "qiskit" in circuits:
            print("\n1. Qiskit Execution:")

            start_time = time.time()

            # Execute with state vector simulator
            simulator = AerSimulator(method="statevector")
            qc = circuits["qiskit"].copy()
            qc.save_statevector()

            job = simulator.run(qc, shots=1)
            result = job.result()
            statevector = result.get_statevector()

            execution_time = time.time() - start_time

            execution_results["qiskit"] = {
                "statevector": statevector.data,
                "execution_time": execution_time,
                "backend": "Aer Statevector Simulator",
            }

            print(f"   Execution time: {execution_time:.6f} seconds")
            print(f"   Final state: {statevector.data}")
            print(f"   Backend: {execution_results['qiskit']['backend']}")

        # Cirq execution
        if "cirq" in circuits:
            print("\n2. Cirq Execution:")

            start_time = time.time()

            # Execute with Cirq simulator
            simulator = cirq.Simulator()
            result = simulator.simulate(circuits["cirq"])

            execution_time = time.time() - start_time

            execution_results["cirq"] = {
                "statevector": result.final_state_vector,
                "execution_time": execution_time,
                "backend": "Cirq Simulator",
            }

            print(f"   Execution time: {execution_time:.6f} seconds")
            print(f"   Final state: {result.final_state_vector}")
            print(f"   Backend: {execution_results['cirq']['backend']}")

        # PennyLane execution
        if "pennylane" in circuits:
            print("\n3. PennyLane Execution:")

            start_time = time.time()

            # Execute PennyLane circuit
            state = circuits["pennylane"]()

            execution_time = time.time() - start_time

            execution_results["pennylane"] = {
                "statevector": state,
                "execution_time": execution_time,
                "backend": "PennyLane default.qubit",
            }

            print(f"   Execution time: {execution_time:.6f} seconds")
            print(f"   Final state: {state}")
            print(f"   Backend: {execution_results['pennylane']['backend']}")

        # Compare results
        self._compare_execution_results(execution_results)

        return execution_results

    def _compare_execution_results(self, results):
        """Compare execution results across frameworks."""
        print("\n=== Execution Results Comparison ===")

        if len(results) < 2:
            print("Need at least 2 frameworks for comparison")
            return

        # Get reference state (first available)
        frameworks = list(results.keys())
        reference_framework = frameworks[0]
        reference_state = results[reference_framework]["statevector"]

        print(f"Using {reference_framework} as reference")

        # Compare states
        for framework in frameworks[1:]:
            state = results[framework]["statevector"]

            # Calculate fidelity between states
            fidelity = abs(np.vdot(reference_state, state)) ** 2

            print(f"\nComparing {reference_framework} vs {framework}:")
            print(f"  State fidelity: {fidelity:.10f}")
            print(f"  States match: {'✅' if fidelity > 0.9999 else '❌'}")

            # Execution time comparison
            ref_time = results[reference_framework]["execution_time"]
            comp_time = results[framework]["execution_time"]

            if ref_time > 0:
                speedup = comp_time / ref_time
                print(f"  Relative execution time: {speedup:.2f}x")

    def implement_variational_circuit(self):
        """Implement a parameterized variational circuit in different frameworks."""
        print("\n=== Variational Circuit Implementation ===")

        # Parameters for the circuit
        params = [np.pi / 4, np.pi / 3, np.pi / 6]

        variational_circuits = {}

        # Qiskit implementation
        if frameworks_available["qiskit"]:
            print("\n1. Qiskit Variational Circuit:")

            from qiskit.circuit import Parameter

            # Define parameters
            theta = Parameter("θ")
            phi = Parameter("φ")
            lambda_param = Parameter("λ")

            qc_var = QuantumCircuit(2)
            qc_var.ry(theta, 0)
            qc_var.ry(phi, 1)
            qc_var.cx(0, 1)
            qc_var.rz(lambda_param, 1)
            qc_var.cx(0, 1)

            # Bind parameters
            qc_bound = qc_var.bind_parameters(
                {theta: params[0], phi: params[1], lambda_param: params[2]}
            )

            variational_circuits["qiskit"] = {
                "parameterized": qc_var,
                "bound": qc_bound,
                "parameters": [theta, phi, lambda_param],
            }

            print(f"   Parameters: {len(qc_var.parameters)}")
            print(f"   Circuit depth: {qc_bound.depth()}")

            if self.verbose:
                print("   Parameterized circuit:")
                print(f"   {qc_var.draw()}")

        # Cirq implementation
        if frameworks_available["cirq"]:
            print("\n2. Cirq Variational Circuit:")

            q0, q1 = cirq.LineQubit.range(2)

            # Define parameterized circuit
            circuit_var = cirq.Circuit(
                [
                    cirq.ry(params[0])(q0),
                    cirq.ry(params[1])(q1),
                    cirq.CNOT(q0, q1),
                    cirq.rz(params[2])(q1),
                    cirq.CNOT(q0, q1),
                ]
            )

            variational_circuits["cirq"] = {
                "circuit": circuit_var,
                "parameters": params,
            }

            print(f"   Parameters: {len(params)}")
            print(f"   Circuit moments: {len(circuit_var)}")

            if self.verbose:
                print("   Circuit:")
                print(f"   {circuit_var}")

        # PennyLane implementation
        if frameworks_available["pennylane"]:
            print("\n3. PennyLane Variational Circuit:")

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def variational_circuit(params):
                qml.RY(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(params[2], wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            # Execute with parameters
            expectation_value = variational_circuit(params)

            variational_circuits["pennylane"] = {
                "circuit": variational_circuit,
                "parameters": params,
                "expectation_value": expectation_value,
            }

            print(f"   Parameters: {len(params)}")
            print(f"   Expectation value: {expectation_value:.6f}")

            if self.verbose:
                print("   Circuit function with parameter optimization support")

        return variational_circuits

    def compare_gradient_computation(self, variational_circuits):
        """Compare gradient computation methods across frameworks."""
        print("\n=== Gradient Computation Comparison ===")

        gradients = {}

        # PennyLane automatic differentiation
        if "pennylane" in variational_circuits:
            print("\n1. PennyLane Automatic Differentiation:")

            circuit = variational_circuits["pennylane"]["circuit"]
            params = variational_circuits["pennylane"]["parameters"]

            # Compute gradients
            grad_fn = qml.grad(circuit)
            gradients["pennylane"] = grad_fn(params)

            print(f"   Gradients: {gradients['pennylane']}")
            print("   Method: Automatic differentiation")

        # Qiskit parameter shift (manual implementation)
        if "qiskit" in variational_circuits:
            print("\n2. Qiskit Parameter Shift (Manual):")

            def compute_expectation(params_vals):
                """Compute expectation value for Qiskit circuit."""
                qc = variational_circuits["qiskit"]["parameterized"]
                param_dict = {
                    list(qc.parameters)[i]: params_vals[i]
                    for i in range(len(params_vals))
                }
                bound_circuit = qc.bind_parameters(param_dict)

                # Add measurement of ZZ expectation
                from qiskit.quantum_info import SparsePauliOp

                simulator = AerSimulator(method="statevector")
                bound_circuit.save_statevector()

                job = simulator.run(bound_circuit, shots=1)
                result = job.result()
                statevector = result.get_statevector()

                # Compute ZZ expectation manually
                zz_op = SparsePauliOp(["ZZ"], [1.0])
                expectation = statevector.expectation_value(zz_op)

                return np.real(expectation)

            # Parameter shift rule
            shift = np.pi / 2
            params_array = np.array([np.pi / 4, np.pi / 3, np.pi / 6])

            gradients_qiskit = []
            for i in range(len(params_array)):
                params_plus = params_array.copy()
                params_minus = params_array.copy()
                params_plus[i] += shift
                params_minus[i] -= shift

                exp_plus = compute_expectation(params_plus)
                exp_minus = compute_expectation(params_minus)

                grad = (exp_plus - exp_minus) / 2
                gradients_qiskit.append(grad)

            gradients["qiskit"] = np.array(gradients_qiskit)

            print(f"   Gradients: {gradients['qiskit']}")
            print("   Method: Parameter shift rule")

        # Compare gradients if multiple available
        if len(gradients) > 1:
            print("\n=== Gradient Comparison ===")
            frameworks = list(gradients.keys())

            for i in range(len(frameworks)):
                for j in range(i + 1, len(frameworks)):
                    fw1, fw2 = frameworks[i], frameworks[j]
                    diff = np.linalg.norm(gradients[fw1] - gradients[fw2])

                    print(f"{fw1} vs {fw2}: L2 difference = {diff:.8f}")

                    if diff < 1e-6:
                        print("  → Gradients match! ✅")
                    else:
                        print("  → Gradients differ ⚠️")

        return gradients

    def framework_feature_comparison(self):
        """Compare features and capabilities across frameworks."""
        print("\n=== Framework Feature Comparison ===")

        features = {
            "qiskit": {
                "Circuit Construction": "✅ Excellent",
                "Simulators": "✅ Multiple (Aer)",
                "Hardware Access": "✅ IBM Quantum",
                "Optimization": "✅ Transpiler",
                "Noise Modeling": "✅ Comprehensive",
                "Autodiff": "❌ Limited",
                "ML Integration": "🟡 Qiskit ML",
                "Documentation": "✅ Excellent",
                "Community": "✅ Large",
            },
            "cirq": {
                "Circuit Construction": "✅ Good",
                "Simulators": "✅ Basic",
                "Hardware Access": "✅ Google Quantum AI",
                "Optimization": "✅ Good",
                "Noise Modeling": "✅ Good",
                "Autodiff": "❌ No",
                "ML Integration": "🟡 TensorFlow Quantum",
                "Documentation": "✅ Good",
                "Community": "🟡 Medium",
            },
            "pennylane": {
                "Circuit Construction": "✅ Good",
                "Simulators": "✅ Multiple backends",
                "Hardware Access": "✅ Multiple providers",
                "Optimization": "✅ Excellent",
                "Noise Modeling": "🟡 Basic",
                "Autodiff": "✅ Excellent",
                "ML Integration": "✅ Excellent",
                "Documentation": "✅ Good",
                "Community": "🟡 Growing",
            },
        }

        # Print comparison table
        feature_names = list(features["qiskit"].keys())

        print(f"{'Feature':<20} {'Qiskit':<15} {'Cirq':<15} {'PennyLane':<15}")
        print("-" * 65)

        for feature in feature_names:
            qiskit_val = (
                features["qiskit"][feature] if frameworks_available["qiskit"] else "N/A"
            )
            cirq_val = (
                features["cirq"][feature] if frameworks_available["cirq"] else "N/A"
            )
            pennylane_val = (
                features["pennylane"][feature]
                if frameworks_available["pennylane"]
                else "N/A"
            )

            print(f"{feature:<20} {qiskit_val:<15} {cirq_val:<15} {pennylane_val:<15}")

        return features

    def visualize_comparison_results(self, execution_results):
        """Visualize framework comparison results."""
        if not execution_results or len(execution_results) < 2:
            print("\nSkipping visualization - insufficient data")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        frameworks = list(execution_results.keys())

        # 1. Execution time comparison
        times = [execution_results[fw]["execution_time"] for fw in frameworks]

        ax1.bar(
            frameworks,
            times,
            alpha=0.7,
            color=["skyblue", "lightcoral", "lightgreen"][: len(frameworks)],
        )
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Framework Execution Time Comparison")
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, time_val in enumerate(times):
            ax1.text(
                i,
                time_val + max(times) * 0.01,
                f"{time_val:.4f}s",
                ha="center",
                va="bottom",
            )

        # 2. State vector comparison (real parts)
        if len(frameworks) >= 2:
            ref_state = execution_results[frameworks[0]]["statevector"]
            comp_state = execution_results[frameworks[1]]["statevector"]

            indices = range(len(ref_state))
            width = 0.35

            ax2.bar(
                [i - width / 2 for i in indices],
                np.real(ref_state),
                width,
                label=frameworks[0],
                alpha=0.7,
            )
            ax2.bar(
                [i + width / 2 for i in indices],
                np.real(comp_state),
                width,
                label=frameworks[1],
                alpha=0.7,
            )

            ax2.set_xlabel("State Component")
            ax2.set_ylabel("Amplitude (Real)")
            ax2.set_title("State Vector Comparison (Real Parts)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Framework availability
        available = [
            1 if frameworks_available[fw] else 0
            for fw in ["qiskit", "cirq", "pennylane"]
        ]
        fw_names = ["Qiskit", "Cirq", "PennyLane"]
        colors = ["green" if av else "red" for av in available]

        ax3.bar(fw_names, available, color=colors, alpha=0.7)
        ax3.set_ylabel("Available")
        ax3.set_title("Framework Availability")
        ax3.set_ylim(0, 1.2)
        ax3.grid(True, alpha=0.3)

        # Add availability text
        for i, av in enumerate(available):
            status = "Available" if av else "Not Installed"
            ax3.text(i, av + 0.05, status, ha="center", va="bottom")

        # 4. Feature scores (subjective)
        feature_scores = {
            "Qiskit": [9, 9, 8, 9, 9],
            "Cirq": [8, 7, 8, 7, 6],
            "PennyLane": [8, 8, 9, 7, 7],
        }

        features = ["Circuits", "Simulators", "Hardware", "Docs", "Community"]
        x = np.arange(len(features))
        width = 0.25

        for i, (fw, scores) in enumerate(feature_scores.items()):
            if frameworks_available[fw.lower()]:
                ax4.bar(x + i * width, scores, width, label=fw, alpha=0.7)

        ax4.set_xlabel("Features")
        ax4.set_ylabel("Score (1-10)")
        ax4.set_title("Framework Feature Scores")
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(features)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()

    def generate_summary_report(self):
        """Generate comprehensive summary of framework comparison."""
        print("\n" + "=" * 60)
        print("MULTI-FRAMEWORK COMPARISON - ANALYSIS SUMMARY")
        print("=" * 60)

        available_frameworks = [
            fw for fw, available in frameworks_available.items() if available
        ]

        print(f"\n📊 Frameworks Analyzed: {len(available_frameworks)}")
        print(f"Available: {', '.join(available_frameworks)}")

        print("\n🔬 Comparison Aspects:")
        print("  • Circuit construction syntax")
        print("  • Execution performance")
        print("  • State vector accuracy")
        print("  • Gradient computation methods")
        print("  • Feature capabilities")
        print("  • Framework ecosystem")

        print("\n📚 Key Findings:")
        print("  • All frameworks produce equivalent quantum states")
        print("  • Execution times vary by implementation details")
        print("  • PennyLane excels at automatic differentiation")
        print("  • Qiskit offers comprehensive ecosystem")
        print("  • Cirq integrates well with Google hardware")

        print("\n🎯 Framework Recommendations:")
        print("  → Qiskit: General quantum computing, IBM hardware")
        print("  → Cirq: Google Quantum AI, research applications")
        print("  → PennyLane: Quantum machine learning, optimization")

        print("\n🚀 Next Steps:")
        print("  → Deep dive into framework-specific features")
        print("  → Explore hardware-specific optimizations")
        print("  → Benchmark complex algorithms")
        print("  → Study hybrid classical-quantum workflows")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Multi-Framework Quantum Computing Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_multi_framework_comparison.py
  python 02_multi_framework_comparison.py --verbose
  python 02_multi_framework_comparison.py --framework qiskit
  python 02_multi_framework_comparison.py --show-visualization
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed information",
    )
    parser.add_argument(
        "--framework",
        choices=["qiskit", "cirq", "pennylane", "all"],
        default="all",
        help="Focus on specific framework",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display comparison visualizations",
    )
    parser.add_argument(
        "--skip-gradients",
        action="store_true",
        help="Skip gradient computation comparison",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 3: Advanced Programming")
    print("Example 2: Multi-Framework Comparison")
    print("=" * 55)

    # Initialize comparator
    comparator = MultiFrameworkComparator(verbose=args.verbose)

    try:
        # Check framework availability
        available_frameworks = comparator.check_framework_availability()

        if args.framework != "all":
            if not available_frameworks.get(args.framework, False):
                print(f"\n❌ {args.framework} is not available")
                return 1

        # Implement Bell state circuits
        circuits = comparator.implement_bell_state_circuits()

        # Compare execution
        execution_results = comparator.compare_circuit_execution(circuits)

        # Implement variational circuits
        variational_circuits = comparator.implement_variational_circuit()

        # Compare gradients (optional)
        if not args.skip_gradients:
            gradients = comparator.compare_gradient_computation(variational_circuits)

        # Feature comparison
        features = comparator.framework_feature_comparison()

        # Visualization (optional)
        if args.show_visualization:
            comparator.visualize_comparison_results(execution_results)

        # Generate summary
        comparator.generate_summary_report()

        print(f"\n✅ Multi-framework comparison completed successfully!")

        if args.verbose:
            print(f"🔍 Detailed analysis enabled")
            print(f"📈 Use --show-visualization for plots")

    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
