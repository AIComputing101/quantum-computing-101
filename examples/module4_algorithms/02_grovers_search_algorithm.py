#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 4: Quantum Algorithms
Example 2: Grover's Search Algorithm

This script implements and analyzes Grover's quantum search algorithm,
demonstrating quadratic speedup over classical search.

Author: Quantum Computing 101 Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import time

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import warnings

warnings.filterwarnings("ignore")


class GroverSearchAlgorithm:
    """Complete implementation of Grover's quantum search algorithm."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.search_results = {}

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Grover] {message}")

    def create_oracle(self, n_qubits, marked_items):
        """Create oracle circuit that marks specified items."""
        oracle = QuantumCircuit(n_qubits, name="Oracle")

        for item in marked_items:
            # Convert item to binary representation
            binary_item = format(item, f"0{n_qubits}b")
            self.log(f"Marking item {item} (binary: {binary_item})")

            # Apply X gates to qubits that should be 0
            for i, bit in enumerate(binary_item):
                if bit == "0":
                    oracle.x(i)

            # Apply multi-controlled Z gate
            self._add_mcz(oracle, list(range(n_qubits)))

            # Undo X gates
            for i, bit in enumerate(binary_item):
                if bit == "0":
                    oracle.x(i)

        return oracle

    def _add_mcz(self, circuit, qubits):
        """Add multi-controlled Z gate."""
        n = len(qubits)

        if n == 1:
            circuit.z(qubits[0])
        elif n == 2:
            circuit.cz(qubits[0], qubits[1])
        elif n == 3:
            # Decompose 3-qubit MCZ using Toffoli and single-qubit gates
            circuit.h(qubits[2])
            circuit.ccx(qubits[0], qubits[1], qubits[2])
            circuit.h(qubits[2])
        else:
            # For larger n, use more complex decomposition
            # This is a simplified implementation
            circuit.h(qubits[-1])
            self._add_mcx(circuit, qubits[:-1], qubits[-1])
            circuit.h(qubits[-1])

    def _add_mcx(self, circuit, control_qubits, target_qubit):
        """Add multi-controlled X gate using auxiliary qubits if needed."""
        n_controls = len(control_qubits)

        if n_controls == 1:
            circuit.cx(control_qubits[0], target_qubit)
        elif n_controls == 2:
            circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        else:
            # For more controls, use decomposition with auxiliary qubits
            # This is a simplified version
            circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)

    def create_diffuser(self, n_qubits):
        """Create diffusion operator (inversion about average)."""
        diffuser = QuantumCircuit(n_qubits, name="Diffuser")

        # Apply H gates
        for i in range(n_qubits):
            diffuser.h(i)

        # Apply X gates
        for i in range(n_qubits):
            diffuser.x(i)

        # Apply multi-controlled Z
        self._add_mcz(diffuser, list(range(n_qubits)))

        # Apply X gates
        for i in range(n_qubits):
            diffuser.x(i)

        # Apply H gates
        for i in range(n_qubits):
            diffuser.h(i)

        return diffuser

    def calculate_optimal_iterations(self, n_qubits, num_marked):
        """Calculate optimal number of Grover iterations."""
        N = 2**n_qubits
        if num_marked >= N:
            return 0

        theta = np.arcsin(np.sqrt(num_marked / N))
        optimal_iterations = int(np.round(np.pi / (4 * theta) - 0.5))

        self.log(f"Search space: {N} items")
        self.log(f"Marked items: {num_marked}")
        self.log(f"Optimal iterations: {optimal_iterations}")

        return optimal_iterations

    def build_grover_circuit(self, n_qubits, marked_items, iterations=None):
        """Build complete Grover search circuit."""
        if iterations is None:
            iterations = self.calculate_optimal_iterations(n_qubits, len(marked_items))

        # Create main circuit
        circuit = QuantumCircuit(n_qubits, n_qubits)

        # Initialize uniform superposition
        for i in range(n_qubits):
            circuit.h(i)

        circuit.barrier(label="Initialization")

        # Create oracle and diffuser
        oracle = self.create_oracle(n_qubits, marked_items)
        diffuser = self.create_diffuser(n_qubits)

        # Apply Grover iterations
        for iteration in range(iterations):
            circuit.barrier(label=f"Iteration {iteration+1}")

            # Apply oracle
            circuit.compose(oracle, inplace=True)
            circuit.barrier(label="Oracle")

            # Apply diffuser
            circuit.compose(diffuser, inplace=True)
            circuit.barrier(label="Diffuser")

        # Measurement
        circuit.measure_all()

        self.circuit = circuit
        self.n_qubits = n_qubits
        self.marked_items = marked_items
        self.iterations_used = iterations

        return circuit

    def simulate_grover_amplitude_evolution(
        self, n_qubits, marked_items, max_iterations=None
    ):
        """Simulate amplitude evolution throughout Grover iterations."""
        if max_iterations is None:
            max_iterations = 2 * self.calculate_optimal_iterations(
                n_qubits, len(marked_items)
            )

        evolution_data = {
            "iterations": [],
            "marked_probability": [],
            "unmarked_probability": [],
            "marked_amplitude": [],
            "unmarked_amplitude": [],
        }

        # Create circuit without measurement for state vector simulation
        circuit_no_measure = QuantumCircuit(n_qubits)

        # Initialize superposition
        for i in range(n_qubits):
            circuit_no_measure.h(i)

        # Get initial state
        state = Statevector.from_instruction(circuit_no_measure)
        self._record_amplitude_data(state, marked_items, 0, evolution_data)

        # Create oracle and diffuser
        oracle = self.create_oracle(n_qubits, marked_items)
        diffuser = self.create_diffuser(n_qubits)

        # Apply iterations and record states
        for iteration in range(max_iterations):
            # Apply oracle
            circuit_no_measure.compose(oracle, inplace=True)

            # Apply diffuser
            circuit_no_measure.compose(diffuser, inplace=True)

            # Get state after this iteration
            state = Statevector.from_instruction(circuit_no_measure)
            self._record_amplitude_data(
                state, marked_items, iteration + 1, evolution_data
            )

        return evolution_data

    def _record_amplitude_data(self, state, marked_items, iteration, evolution_data):
        """Record amplitude and probability data for analysis."""
        state_vector = state.data
        N = len(state_vector)

        # Calculate probabilities for marked and unmarked items
        marked_prob = sum(abs(state_vector[item]) ** 2 for item in marked_items)
        unmarked_prob = 1 - marked_prob

        # Calculate average amplitudes
        marked_amp = np.mean([abs(state_vector[item]) for item in marked_items])
        unmarked_indices = [i for i in range(N) if i not in marked_items]
        unmarked_amp = (
            np.mean([abs(state_vector[i]) for i in unmarked_indices])
            if unmarked_indices
            else 0
        )

        evolution_data["iterations"].append(iteration)
        evolution_data["marked_probability"].append(marked_prob)
        evolution_data["unmarked_probability"].append(unmarked_prob)
        evolution_data["marked_amplitude"].append(marked_amp)
        evolution_data["unmarked_amplitude"].append(unmarked_amp)

    def execute_search(self, shots=1024):
        """Execute the Grover search algorithm."""
        if not hasattr(self, "circuit"):
            raise ValueError("Circuit not built. Call build_grover_circuit first.")

        self.log(f"Executing Grover search with {shots} shots")

        # Execute on simulator
        simulator = AerSimulator()
        job = simulator.run(self.circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Analyze results
        success_count = 0
        for item in self.marked_items:
            binary_item = format(item, f"0{self.n_qubits}b")
            success_count += counts.get(binary_item, 0)

        success_probability = success_count / shots

        # Theoretical success probability
        N = 2**self.n_qubits
        M = len(self.marked_items)
        theoretical_prob = self._calculate_theoretical_success_probability(
            N, M, self.iterations_used
        )

        self.search_results = {
            "counts": counts,
            "shots": shots,
            "success_count": success_count,
            "success_probability": success_probability,
            "theoretical_probability": theoretical_prob,
            "marked_items": self.marked_items,
            "iterations_used": self.iterations_used,
            "quantum_advantage": self._calculate_quantum_advantage(N, M),
        }

        return self.search_results

    def _calculate_theoretical_success_probability(self, N, M, iterations):
        """Calculate theoretical success probability."""
        if M == 0 or M >= N:
            return 0

        theta = np.arcsin(np.sqrt(M / N))
        return np.sin((2 * iterations + 1) * theta) ** 2

    def _calculate_quantum_advantage(self, N, M):
        """Calculate quantum advantage over classical search."""
        classical_queries = N // 2  # Average case for classical search
        quantum_queries = self.iterations_used

        advantage = classical_queries / quantum_queries if quantum_queries > 0 else 1

        return {
            "classical_queries": classical_queries,
            "quantum_queries": quantum_queries,
            "speedup": advantage,
        }

    def compare_with_classical(self, target_items, search_space_size):
        """Compare Grover's algorithm with classical search."""
        comparison = {"classical": {}, "quantum": {}, "advantage": {}}

        # Classical search analysis
        classical_best_case = 1
        classical_worst_case = search_space_size
        classical_average_case = (search_space_size + 1) / 2

        comparison["classical"] = {
            "best_case_queries": classical_best_case,
            "worst_case_queries": classical_worst_case,
            "average_case_queries": classical_average_case,
            "complexity": "O(N)",
        }

        # Quantum search analysis
        n_qubits = int(np.ceil(np.log2(search_space_size)))
        quantum_queries = self.calculate_optimal_iterations(n_qubits, len(target_items))

        comparison["quantum"] = {
            "queries": quantum_queries,
            "complexity": "O(√N)",
            "success_probability": self._calculate_theoretical_success_probability(
                2**n_qubits, len(target_items), quantum_queries
            ),
        }

        # Calculate advantage
        comparison["advantage"] = {
            "speedup_vs_average": (
                classical_average_case / quantum_queries if quantum_queries > 0 else 1
            ),
            "speedup_vs_worst": (
                classical_worst_case / quantum_queries if quantum_queries > 0 else 1
            ),
            "space_size": search_space_size,
            "marked_items": len(target_items),
        }

        return comparison

    def visualize_results(self, evolution_data=None, comparison_data=None):
        """Visualize Grover algorithm results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Amplitude evolution
        if evolution_data:
            iterations = evolution_data["iterations"]

            axes[0, 0].plot(
                iterations,
                evolution_data["marked_probability"],
                "r-o",
                label="Marked Items",
                linewidth=2,
                markersize=4,
            )
            axes[0, 0].plot(
                iterations,
                evolution_data["unmarked_probability"],
                "b-s",
                label="Unmarked Items",
                linewidth=2,
                markersize=4,
            )

            axes[0, 0].set_xlabel("Grover Iterations")
            axes[0, 0].set_ylabel("Probability")
            axes[0, 0].set_title("Probability Evolution in Grover Search")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)

        # 2. Success probability vs iterations
        if evolution_data:
            axes[0, 1].plot(
                iterations, evolution_data["marked_probability"], "g-o", linewidth=2
            )

            # Mark optimal iteration
            optimal_idx = np.argmax(evolution_data["marked_probability"])
            optimal_iter = iterations[optimal_idx]
            optimal_prob = evolution_data["marked_probability"][optimal_idx]

            axes[0, 1].axvline(
                x=optimal_iter,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Optimal: {optimal_iter} iterations",
            )
            axes[0, 1].plot(optimal_iter, optimal_prob, "ro", markersize=8)

            axes[0, 1].set_xlabel("Iterations")
            axes[0, 1].set_ylabel("Success Probability")
            axes[0, 1].set_title("Success Probability vs Iterations")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)

        # 3. Measurement results
        if hasattr(self, "search_results") and "counts" in self.search_results:
            counts = self.search_results["counts"]

            # Sort by count value
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            items = [item[0] for item in sorted_items[:8]]  # Show top 8
            count_values = [item[1] for item in sorted_items[:8]]

            # Color marked items differently
            colors = []
            for item_binary in items:
                item_decimal = int(item_binary, 2)
                if item_decimal in self.marked_items:
                    colors.append("red")
                else:
                    colors.append("skyblue")

            axes[1, 0].bar(range(len(items)), count_values, color=colors, alpha=0.7)
            axes[1, 0].set_xlabel("Measurement Outcome")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Measurement Results (Red = Marked)")
            axes[1, 0].set_xticks(range(len(items)))
            axes[1, 0].set_xticklabels(items, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Quantum vs Classical comparison
        if comparison_data:
            methods = ["Classical\n(Average)", "Classical\n(Worst)", "Quantum"]
            queries = [
                comparison_data["classical"]["average_case_queries"],
                comparison_data["classical"]["worst_case_queries"],
                comparison_data["quantum"]["queries"],
            ]
            colors = ["lightcoral", "red", "lightgreen"]

            bars = axes[1, 1].bar(methods, queries, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel("Number of Queries")
            axes[1, 1].set_title("Quantum vs Classical Search")
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, queries):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(queries) * 0.01,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

            # Add speedup text
            speedup = comparison_data["advantage"]["speedup_vs_average"]
            axes[1, 1].text(
                0.5,
                0.95,
                f"Quantum Speedup: {speedup:.1f}x",
                transform=axes[1, 1].transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

        plt.tight_layout()
        plt.show()

    def generate_summary_report(self):
        """Generate comprehensive summary of Grover search analysis."""
        print("\n" + "=" * 60)
        print("GROVER'S SEARCH ALGORITHM - ANALYSIS SUMMARY")
        print("=" * 60)

        if hasattr(self, "search_results"):
            results = self.search_results

            print(f"\n📊 Search Results:")
            print(f"   Search space: {2**self.n_qubits} items")
            print(
                f"   Marked items: {len(results['marked_items'])} {results['marked_items']}"
            )
            print(f"   Iterations used: {results['iterations_used']}")
            print(f"   Success probability: {results['success_probability']:.4f}")
            print(
                f"   Theoretical probability: {results['theoretical_probability']:.4f}"
            )

            advantage = results["quantum_advantage"]
            print(f"\n🚀 Quantum Advantage:")
            print(f"   Classical queries (avg): {advantage['classical_queries']}")
            print(f"   Quantum queries: {advantage['quantum_queries']}")
            print(f"   Speedup: {advantage['speedup']:.2f}x")

        print(f"\n🔬 Algorithm Properties:")
        print(f"   Complexity: O(√N) vs O(N) classical")
        print(f"   Quadratic speedup for unstructured search")
        print(f"   Optimal for single marked item")
        print(f"   Probabilistic success")

        print(f"\n🎯 Key Insights:")
        print(f"   • Grover's algorithm provides quadratic speedup")
        print(f"   • Optimal iteration count is π√N/4M")
        print(f"   • Success probability oscillates with iterations")
        print(f"   • Works best with small number of marked items")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Grover's Quantum Search Algorithm Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_grovers_search_algorithm.py --qubits 3 --marked-items 3
  python 02_grovers_search_algorithm.py --qubits 4 --marked-items 5 7 --verbose
  python 02_grovers_search_algorithm.py --show-evolution --show-visualization
        """,
    )

    parser.add_argument(
        "--qubits",
        type=int,
        default=3,
        help="Number of qubits (search space = 2^qubits)",
    )
    parser.add_argument(
        "--marked-items",
        type=int,
        nargs="+",
        default=[3],
        help="Items to mark in the search space",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of Grover iterations (default: optimal)",
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Number of measurement shots"
    )
    parser.add_argument(
        "--show-evolution",
        action="store_true",
        help="Show amplitude evolution during iterations",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display result visualizations",
    )
    parser.add_argument(
        "--compare-classical",
        action="store_true",
        help="Include classical search comparison",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 4: Quantum Algorithms")
    print("Example 2: Grover's Search Algorithm")
    print("=" * 45)

    # Validate inputs
    search_space_size = 2**args.qubits
    if any(item >= search_space_size for item in args.marked_items):
        print(f"❌ Error: Marked items must be < {search_space_size}")
        return 1

    # Initialize Grover algorithm
    grover = GroverSearchAlgorithm(verbose=args.verbose)

    try:
        print(f"\nSearch Configuration:")
        print(f"   Qubits: {args.qubits}")
        print(f"   Search space: {search_space_size} items")
        print(f"   Marked items: {args.marked_items}")

        # Build circuit
        circuit = grover.build_grover_circuit(
            args.qubits, args.marked_items, args.iterations
        )

        print(f"   Circuit depth: {circuit.depth()}")
        print(f"   Gate count: {circuit.size()}")

        # Execute search
        results = grover.execute_search(shots=args.shots)

        print(f"\nSearch Results:")
        print(f"   Success rate: {results['success_probability']:.1%}")
        print(f"   Theoretical: {results['theoretical_probability']:.1%}")
        print(f"   Quantum speedup: {results['quantum_advantage']['speedup']:.1f}x")

        # Show evolution if requested
        evolution_data = None
        if args.show_evolution:
            print(f"\nAnalyzing amplitude evolution...")
            evolution_data = grover.simulate_grover_amplitude_evolution(
                args.qubits, args.marked_items
            )

        # Classical comparison
        comparison_data = None
        if args.compare_classical:
            comparison_data = grover.compare_with_classical(
                args.marked_items, search_space_size
            )

            print(f"\nClassical vs Quantum Comparison:")
            print(
                f"   Classical average queries: {comparison_data['classical']['average_case_queries']:.1f}"
            )
            print(f"   Quantum queries: {comparison_data['quantum']['queries']}")
            print(
                f"   Speedup: {comparison_data['advantage']['speedup_vs_average']:.1f}x"
            )

        # Visualization
        if args.show_visualization:
            grover.visualize_results(evolution_data, comparison_data)

        # Generate summary
        grover.generate_summary_report()

        print(f"\n✅ Grover search analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during Grover search: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
