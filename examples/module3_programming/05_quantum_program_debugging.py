#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3: Advanced Programming
Example 5: Quantum Program Debugging

This script provides comprehensive debugging tools and techniques for quantum programs,
including state analysis, circuit validation, and performance optimization.

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
import time
import warnings

warnings.filterwarnings("ignore")

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    process_fidelity,
    state_fidelity,
)
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Note: Fake providers have been updated in newer Qiskit versions
try:
    from qiskit.providers.fake_provider import FakeManila
except ImportError:
    # Use AerSimulator for newer Qiskit versions
    FakeManila = lambda: AerSimulator()
    print("ℹ️  Using AerSimulator instead of deprecated FakeManila provider")
import logging


class QuantumDebugger:
    """Comprehensive quantum program debugging toolkit."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.debug_log = []
        self.circuit_snapshots = []
        self.performance_metrics = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def log_debug(self, message, level="INFO"):
        """Add entry to debug log."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {level}: {message}"
        self.debug_log.append(log_entry)

        if self.verbose:
            print(log_entry)

    def validate_circuit(self, circuit):
        """Perform comprehensive circuit validation."""
        self.log_debug("Starting circuit validation")

        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "metrics": {},
        }

        try:
            # Basic structural checks
            validation_results["metrics"]["num_qubits"] = circuit.num_qubits
            validation_results["metrics"]["num_clbits"] = circuit.num_clbits
            validation_results["metrics"]["depth"] = circuit.depth()
            validation_results["metrics"]["size"] = circuit.size()

            # Check for common issues

            # 1. Empty circuit
            if circuit.size() == 0:
                validation_results["warnings"].append("Circuit is empty")

            # 2. Unused qubits
            used_qubits = set()
            for instruction in circuit.data:
                for qubit in instruction.qubits:
                    used_qubits.add(circuit.find_bit(qubit).index)

            unused_qubits = set(range(circuit.num_qubits)) - used_qubits
            if unused_qubits:
                validation_results["warnings"].append(f"Unused qubits: {unused_qubits}")

            # 3. Measurements without classical bits
            has_measurements = any(
                instr.operation.name == "measure" for instr in circuit.data
            )
            if has_measurements and circuit.num_clbits == 0:
                validation_results["errors"].append(
                    "Circuit has measurements but no classical bits"
                )
                validation_results["valid"] = False

            # 4. Very deep circuits (potential performance issue)
            if circuit.depth() > 100:
                validation_results["warnings"].append(
                    f"Very deep circuit (depth={circuit.depth()})"
                )

            # 5. High gate count
            if circuit.size() > 1000:
                validation_results["warnings"].append(
                    f"High gate count ({circuit.size()})"
                )

            # 6. Check for duplicate gates on same qubits
            self._check_gate_redundancy(circuit, validation_results)

            # 7. Validate parameter binding
            if circuit.parameters:
                validation_results["warnings"].append(
                    f"Circuit has {len(circuit.parameters)} unbound parameters"
                )

            self.log_debug(
                f"Circuit validation completed. Valid: {validation_results['valid']}"
            )

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            self.log_debug(f"Circuit validation failed: {str(e)}", level="ERROR")

        return validation_results

    def _check_gate_redundancy(self, circuit, validation_results):
        """Check for redundant gate sequences."""
        redundancies = []

        # Check for double X gates (cancel out)
        gate_history = {}
        for i, instruction in enumerate(circuit.data):
            gate_name = instruction.operation.name
            qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)

            if gate_name == "x" and len(qubits) == 1:
                qubit = qubits[0]
                if qubit in gate_history and gate_history[qubit][-1][0] == "x":
                    redundancies.append(
                        f"Double X gate on qubit {qubit} at positions {gate_history[qubit][-1][1]}, {i}"
                    )

                if qubit not in gate_history:
                    gate_history[qubit] = []
                gate_history[qubit].append((gate_name, i))

        if redundancies:
            validation_results["warnings"].extend(redundancies)

    def trace_execution(self, circuit, steps=None):
        """Trace quantum circuit execution step by step."""
        self.log_debug("Starting execution trace")

        if steps is None:
            steps = len(circuit.data)

        trace_results = {"steps": [], "final_state": None, "intermediate_states": []}

        try:
            # Create step-by-step circuits
            current_circuit = QuantumCircuit(circuit.num_qubits)

            for step, instruction in enumerate(circuit.data[:steps]):
                # Add current instruction
                current_circuit.append(
                    instruction.operation, instruction.qubits, instruction.clbits
                )

                # Get state after this step (if no measurements)
                if instruction.operation.name != "measure":
                    try:
                        state = Statevector.from_instruction(current_circuit)

                        step_info = {
                            "step": step,
                            "instruction": instruction.operation.name,
                            "qubits": [
                                circuit.find_bit(q).index for q in instruction.qubits
                            ],
                            "state_vector": state.data.copy(),
                            "probabilities": np.abs(state.data) ** 2,
                            "entanglement": self._calculate_entanglement(
                                state.data, circuit.num_qubits
                            ),
                        }

                        trace_results["steps"].append(step_info)
                        trace_results["intermediate_states"].append(state.data.copy())

                        self.log_debug(
                            f"Step {step}: {instruction.operation.name} on qubits {step_info['qubits']}"
                        )

                    except Exception as e:
                        self.log_debug(
                            f"Could not compute state at step {step}: {str(e)}",
                            level="WARNING",
                        )
                        break
                else:
                    step_info = {
                        "step": step,
                        "instruction": instruction.operation.name,
                        "qubits": [
                            circuit.find_bit(q).index for q in instruction.qubits
                        ],
                        "note": "Measurement - state vector not available",
                    }
                    trace_results["steps"].append(step_info)
                    break

            # Final state
            if trace_results["intermediate_states"]:
                trace_results["final_state"] = trace_results["intermediate_states"][-1]

            self.log_debug(
                f"Execution trace completed for {len(trace_results['steps'])} steps"
            )

        except Exception as e:
            self.log_debug(f"Execution trace failed: {str(e)}", level="ERROR")

        return trace_results

    def _calculate_entanglement(self, state_vector, num_qubits):
        """Calculate entanglement measure for the state."""
        if num_qubits == 1:
            return 0.0

        # For 2-qubit systems, calculate concurrence
        if num_qubits == 2:
            # Reshape state vector to 2x2 matrix
            state_matrix = state_vector.reshape(2, 2)

            # Calculate reduced density matrix
            rho_A = np.dot(state_matrix, state_matrix.conj().T)

            # Calculate von Neumann entropy
            eigenvals = np.linalg.eigvals(rho_A)
            eigenvals = eigenvals[eigenvals > 1e-12]

            if len(eigenvals) > 1:
                entropy = -np.sum(eigenvals * np.log2(eigenvals))
                return entropy

        return 0.0  # Placeholder for higher dimensions

    def analyze_state_evolution(self, trace_results):
        """Analyze how quantum state evolves through the circuit."""
        self.log_debug("Analyzing state evolution")

        analysis = {
            "probability_evolution": [],
            "entanglement_evolution": [],
            "state_distances": [],
            "gate_effects": [],
        }

        if not trace_results["steps"]:
            return analysis

        # Track probability distributions
        for step_info in trace_results["steps"]:
            if "probabilities" in step_info:
                analysis["probability_evolution"].append(step_info["probabilities"])
                analysis["entanglement_evolution"].append(step_info["entanglement"])

        # Calculate state distances between consecutive steps
        for i in range(1, len(trace_results["intermediate_states"])):
            prev_state = trace_results["intermediate_states"][i - 1]
            curr_state = trace_results["intermediate_states"][i]

            # State fidelity
            fidelity = abs(np.vdot(prev_state, curr_state)) ** 2
            distance = 1 - fidelity

            analysis["state_distances"].append(
                {
                    "step": i,
                    "fidelity": fidelity,
                    "distance": distance,
                    "gate": trace_results["steps"][i]["instruction"],
                }
            )

        # Analyze gate effects
        gate_effects = {}
        for step_info in trace_results["steps"]:
            if "instruction" in step_info:
                gate_name = step_info["instruction"]
                if gate_name not in gate_effects:
                    gate_effects[gate_name] = {"count": 0, "avg_entanglement_change": 0}
                gate_effects[gate_name]["count"] += 1

        analysis["gate_effects"] = gate_effects

        self.log_debug(f"State evolution analysis completed")
        return analysis

    def performance_profiling(self, circuit, backend=None):
        """Profile circuit performance on different backends."""
        self.log_debug("Starting performance profiling")

        if backend is None:
            backend = AerSimulator()

        profiling_results = {
            "transpilation_time": 0,
            "execution_time": 0,
            "original_circuit": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "cx_count": circuit.count_ops().get("cx", 0),
            },
            "transpiled_circuit": {},
            "backend_info": str(backend),
        }

        try:
            # Transpilation profiling
            start_time = time.time()
            transpiled_circuit = transpile(circuit, backend)
            profiling_results["transpilation_time"] = time.time() - start_time

            profiling_results["transpiled_circuit"] = {
                "depth": transpiled_circuit.depth(),
                "size": transpiled_circuit.size(),
                "cx_count": transpiled_circuit.count_ops().get("cx", 0),
            }

            # Execution profiling (if circuit has measurements)
            if circuit.num_clbits > 0:
                start_time = time.time()
                job = backend.run(transpiled_circuit, shots=100)
                result = job.result()
                profiling_results["execution_time"] = time.time() - start_time
                profiling_results["measurement_counts"] = result.get_counts()

            self.log_debug(f"Performance profiling completed")

        except Exception as e:
            self.log_debug(f"Performance profiling failed: {str(e)}", level="ERROR")

        return profiling_results

    def error_detection(self, circuit, expected_state=None):
        """Detect potential errors in quantum circuit."""
        self.log_debug("Starting error detection")

        errors = {"logical_errors": [], "performance_issues": [], "quantum_errors": []}

        # Validate circuit first
        validation = self.validate_circuit(circuit)
        if not validation["valid"]:
            errors["logical_errors"].extend(validation["errors"])

        # Check for performance issues
        if circuit.depth() > 50:
            errors["performance_issues"].append(
                f"Deep circuit may suffer from decoherence (depth={circuit.depth()})"
            )

        if circuit.count_ops().get("cx", 0) > circuit.num_qubits * 5:
            errors["performance_issues"].append(
                "High number of two-qubit gates may reduce fidelity"
            )

        # Check against expected behavior
        if expected_state is not None:
            try:
                actual_state = Statevector.from_instruction(circuit)
                fidelity = state_fidelity(actual_state, expected_state)

                if fidelity < 0.99:
                    errors["quantum_errors"].append(
                        f"State fidelity below threshold: {fidelity:.4f}"
                    )

            except Exception as e:
                errors["quantum_errors"].append(f"Could not verify state: {str(e)}")

        self.log_debug(
            f"Error detection completed. Found {sum(len(v) for v in errors.values())} issues"
        )
        return errors

    def generate_debug_report(
        self, circuit, trace_results=None, performance_results=None
    ):
        """Generate comprehensive debug report."""
        print("\n" + "=" * 60)
        print("QUANTUM PROGRAM DEBUG REPORT")
        print("=" * 60)

        # Circuit overview
        print(f"\n📊 Circuit Overview:")
        print(f"   Qubits: {circuit.num_qubits}")
        print(f"   Classical bits: {circuit.num_clbits}")
        print(f"   Depth: {circuit.depth()}")
        print(f"   Gate count: {circuit.size()}")
        print(f"   Parameters: {len(circuit.parameters)}")

        # Validation results
        validation = self.validate_circuit(circuit)
        print(f"\n🔍 Validation Results:")
        print(f"   Valid: {'✅' if validation['valid'] else '❌'}")

        if validation["warnings"]:
            print(f"   Warnings ({len(validation['warnings'])}):")
            for warning in validation["warnings"]:
                print(f"     ⚠️  {warning}")

        if validation["errors"]:
            print(f"   Errors ({len(validation['errors'])}):")
            for error in validation["errors"]:
                print(f"     ❌ {error}")

        # Execution trace summary
        if trace_results:
            print(f"\n🚀 Execution Trace Summary:")
            print(f"   Steps traced: {len(trace_results['steps'])}")

            if trace_results["steps"]:
                final_step = trace_results["steps"][-1]
                if "entanglement" in final_step:
                    print(f"   Final entanglement: {final_step['entanglement']:.4f}")

        # Performance summary
        if performance_results:
            print(f"\n⚡ Performance Summary:")
            print(
                f"   Transpilation time: {performance_results['transpilation_time']:.4f}s"
            )
            print(f"   Execution time: {performance_results['execution_time']:.4f}s")

            orig = performance_results["original_circuit"]
            trans = performance_results["transpiled_circuit"]

            if trans:
                print(f"   Depth change: {orig['depth']} → {trans['depth']}")
                print(f"   Gate change: {orig['size']} → {trans['size']}")

        # Debug log summary
        print(f"\n📝 Debug Log ({len(self.debug_log)} entries):")
        for entry in self.debug_log[-5:]:  # Show last 5 entries
            print(f"   {entry}")

        if len(self.debug_log) > 5:
            print(f"   ... and {len(self.debug_log) - 5} more entries")

    def visualize_debugging_results(self, trace_results, performance_results=None):
        """Visualize debugging results."""
        if not trace_results["steps"]:
            print("No trace results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Probability evolution
        prob_evolution = []
        steps = []
        for step_info in trace_results["steps"]:
            if "probabilities" in step_info:
                prob_evolution.append(step_info["probabilities"])
                steps.append(step_info["step"])

        if prob_evolution:
            prob_array = np.array(prob_evolution)

            for i in range(min(4, prob_array.shape[1])):  # Show first 4 basis states
                axes[0, 0].plot(
                    steps,
                    prob_array[:, i],
                    label=f"|{i:0{int(np.log2(prob_array.shape[1]))}b}⟩",
                    marker="o",
                )

            axes[0, 0].set_xlabel("Circuit Step")
            axes[0, 0].set_ylabel("Probability")
            axes[0, 0].set_title("State Probability Evolution")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Entanglement evolution
        entanglement_values = []
        for step_info in trace_results["steps"]:
            if "entanglement" in step_info:
                entanglement_values.append(step_info["entanglement"])

        if entanglement_values:
            axes[0, 1].plot(
                steps[: len(entanglement_values)], entanglement_values, "r-o"
            )
            axes[0, 1].set_xlabel("Circuit Step")
            axes[0, 1].set_ylabel("Entanglement")
            axes[0, 1].set_title("Entanglement Evolution")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Gate usage analysis
        gate_counts = {}
        for step_info in trace_results["steps"]:
            if "instruction" in step_info:
                gate = step_info["instruction"]
                gate_counts[gate] = gate_counts.get(gate, 0) + 1

        if gate_counts:
            gates = list(gate_counts.keys())
            counts = list(gate_counts.values())

            axes[1, 0].bar(gates, counts, alpha=0.7, color="skyblue")
            axes[1, 0].set_xlabel("Gate Type")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Gate Usage Distribution")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Performance metrics
        if performance_results:
            metrics = ["Depth", "Size", "CX Count"]
            original = [
                performance_results["original_circuit"]["depth"],
                performance_results["original_circuit"]["size"],
                performance_results["original_circuit"]["cx_count"],
            ]
            transpiled = [
                performance_results["transpiled_circuit"].get("depth", 0),
                performance_results["transpiled_circuit"].get("size", 0),
                performance_results["transpiled_circuit"].get("cx_count", 0),
            ]

            x = np.arange(len(metrics))
            width = 0.35

            axes[1, 1].bar(x - width / 2, original, width, label="Original", alpha=0.7)
            axes[1, 1].bar(
                x + width / 2, transpiled, width, label="Transpiled", alpha=0.7
            )

            axes[1, 1].set_xlabel("Metric")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Original vs Transpiled Circuit")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Quantum Program Debugging Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_quantum_program_debugging.py --example bell-state
  python 05_quantum_program_debugging.py --example grover --verbose
  python 05_quantum_program_debugging.py --trace-steps 5
        """,
    )

    parser.add_argument(
        "--example",
        choices=["bell-state", "grover", "qft", "custom"],
        default="bell-state",
        help="Example circuit to debug",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=None,
        help="Number of steps to trace (default: all)",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Show debugging visualizations",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debugging output"
    )
    parser.add_argument(
        "--profile-backend", action="store_true", help="Include backend profiling"
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 3: Advanced Programming")
    print("Example 5: Quantum Program Debugging")
    print("=" * 43)

    # Initialize debugger
    debugger = QuantumDebugger(verbose=args.verbose)

    try:
        # Create example circuit
        if args.example == "bell-state":
            circuit = QuantumCircuit(2, 2)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure_all()

        elif args.example == "grover":
            circuit = QuantumCircuit(3, 3)
            # Superposition
            for i in range(3):
                circuit.h(i)
            # Oracle (mark |101⟩)
            circuit.x(0)
            circuit.ccx(0, 1, 2)
            circuit.x(0)
            # Diffuser
            for i in range(3):
                circuit.h(i)
            for i in range(3):
                circuit.x(i)
            circuit.ccx(0, 1, 2)
            for i in range(3):
                circuit.x(i)
            for i in range(3):
                circuit.h(i)
            circuit.measure_all()

        elif args.example == "qft":
            circuit = QuantumCircuit(3, 3)
            # Simple QFT implementation
            circuit.h(0)
            circuit.cp(np.pi / 2, 0, 1)
            circuit.cp(np.pi / 4, 0, 2)
            circuit.h(1)
            circuit.cp(np.pi / 2, 1, 2)
            circuit.h(2)
            circuit.swap(0, 2)
            circuit.measure_all()

        else:  # custom
            circuit = QuantumCircuit(2, 2)
            circuit.h(0)
            circuit.cx(0, 1)
            # Intentional redundancy for debugging demo
            circuit.x(0)
            circuit.x(0)
            circuit.measure_all()

        print(f"\nDebugging {args.example} circuit...")
        print(
            f"Circuit: {circuit.num_qubits} qubits, {circuit.depth()} depth, {circuit.size()} gates"
        )

        # Trace execution
        trace_results = debugger.trace_execution(circuit, steps=args.trace_steps)

        # Analyze state evolution
        evolution_analysis = debugger.analyze_state_evolution(trace_results)

        # Performance profiling
        performance_results = None
        if args.profile_backend:
            performance_results = debugger.performance_profiling(circuit)

        # Error detection
        errors = debugger.error_detection(circuit)

        # Generate debug report
        debugger.generate_debug_report(circuit, trace_results, performance_results)

        # Show errors if any
        if any(errors.values()):
            print(f"\n🐛 Detected Issues:")
            for category, issue_list in errors.items():
                if issue_list:
                    print(f"   {category.replace('_', ' ').title()}:")
                    for issue in issue_list:
                        print(f"     • {issue}")

        # Visualization
        if args.show_visualization:
            debugger.visualize_debugging_results(trace_results, performance_results)

        print(f"\n✅ Quantum program debugging completed!")
        print(f"📝 Generated {len(debugger.debug_log)} debug log entries")

    except Exception as e:
        print(f"\n❌ Error during debugging: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
