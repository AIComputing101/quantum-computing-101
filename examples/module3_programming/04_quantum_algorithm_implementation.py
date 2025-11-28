#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3: Advanced Programming
Example 4: Quantum Algorithm Implementation

This script provides a comprehensive framework for implementing and analyzing
quantum algorithms with proper structure and documentation.

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
from abc import ABC, abstractmethod

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import warnings

warnings.filterwarnings("ignore")


class QuantumAlgorithm(ABC):
    """Abstract base class for quantum algorithm implementations."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.circuit = None
        self.results = {}

    @abstractmethod
    def build_circuit(self, *args, **kwargs):
        """Build the quantum circuit for the algorithm."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the algorithm and return results."""
        pass

    def analyze_complexity(self):
        """Analyze the computational complexity of the algorithm."""
        if self.circuit is None:
            return None

        return {
            "qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "gates": self.circuit.size(),
            "cx_gates": self.circuit.count_ops().get("cx", 0),
        }


class DeutschJozsaAlgorithm(QuantumAlgorithm):
    """Implementation of the Deutsch-Jozsa algorithm."""

    def __init__(self):
        super().__init__(
            "Deutsch-Jozsa Algorithm",
            "Determines if a function is constant or balanced with a single query",
        )

    def build_circuit(self, n_qubits, oracle_type="balanced"):
        """Build Deutsch-Jozsa circuit with specified oracle."""
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits + 1, n_qubits)

        # Initialize ancilla qubit in |1⟩
        self.circuit.x(n_qubits)

        # Apply Hadamard to all qubits
        for i in range(n_qubits + 1):
            self.circuit.h(i)

        # Apply oracle
        self._add_oracle(oracle_type)

        # Apply Hadamard to input qubits
        for i in range(n_qubits):
            self.circuit.h(i)

        # Measure input qubits
        self.circuit.measure(range(n_qubits), range(n_qubits))

        return self.circuit

    def _add_oracle(self, oracle_type):
        """
        Add Deutsch-Jozsa oracle based on function type.
        
        Mathematical Concept - Quantum Oracle:
        -------------------------------------
        An oracle is a "black box" that computes f(x) and encodes
        the result into the phase of a quantum state.
        
        Oracle Operation:
        ----------------
        U_f: |x⟩|y⟩ → |x⟩|y ⊕ f(x)⟩
        
        where:
        - |x⟩: n-qubit input register
        - |y⟩: 1-qubit ancilla in state |−⟩ = (|0⟩−|1⟩)/√2
        - f(x): Boolean function (0 or 1)
        - ⊕: XOR operation
        
        Phase Kickback:
        --------------
        When y = |−⟩, the oracle induces a phase:
        
        U_f|x⟩|−⟩ = |x⟩(|0⊕f(x)⟩−|1⊕f(x)⟩)/√2
                  = (−1)^f(x)|x⟩|−⟩
        
        Result: f(x) encoded in the PHASE (not the amplitude)!
        
        Oracle Types:
        ------------
        1. Constant: f(x) = 0 or f(x) = 1 for all x
        2. Balanced: f(x) = 0 for half of inputs, 1 for the other half
        
        Deutsch-Jozsa Promise: f is guaranteed to be one of these two types
        
        Implementation Examples:
        -----------------------
        - Constant_0: f(x) = 0 → Do nothing (identity)
        - Constant_1: f(x) = 1 → Apply X to ancilla
        - Balanced: f(x) = x_0 → CNOT from qubit 0 to ancilla
        - Balanced_parity: f(x) = x_0 ⊕ ... ⊕ x_{n-1} → CNOTs from all qubits
        
        Why This Works:
        - Constant function → All phases same → Interference constructive at |0...0⟩
        - Balanced function → Phases differ → Interference destructive at |0...0⟩
        """
        n = self.n_qubits

        if oracle_type == "constant_0":
            # f(x) = 0 for all x (identity operation)
            # Math: U_f|x⟩|y⟩ = |x⟩|y⟩ (no change)
            pass
        elif oracle_type == "constant_1":
            # f(x) = 1 for all x
            # Math: U_f|x⟩|y⟩ = |x⟩|y⊕1⟩ → flip ancilla
            self.circuit.x(n)
        elif oracle_type == "balanced":
            # f(x) = x_0 (output equals first input bit)
            # Math: CNOT implements |x⟩|y⟩ → |x⟩|y⊕x_0⟩
            self.circuit.cx(0, n)
        elif oracle_type == "balanced_parity":
            # f(x) = x_0 ⊕ x_1 ⊕ ... ⊕ x_{n-1} (parity of all bits)
            # Math: Each CNOT adds one bit to the parity
            for i in range(n):
                self.circuit.cx(i, n)

    def execute(self, shots=1024, oracle_type="balanced"):
        """Execute the algorithm."""
        self.build_circuit(
            self.n_qubits if hasattr(self, "n_qubits") else 3, oracle_type
        )

        simulator = AerSimulator()
        job = simulator.run(self.circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Analyze results
        zero_string = "0" * self.n_qubits
        is_constant = zero_string in counts and counts[zero_string] == shots

        self.results = {
            "counts": counts,
            "is_constant": is_constant,
            "oracle_type": oracle_type,
            "theoretical_constant": oracle_type.startswith("constant"),
        }

        return self.results


class GroverAlgorithm(QuantumAlgorithm):
    """Implementation of Grover's search algorithm."""

    def __init__(self):
        super().__init__(
            "Grover's Search Algorithm",
            "Searches an unsorted database quadratically faster than classical",
        )

    def build_circuit(self, n_qubits, marked_items):
        """Build Grover's algorithm circuit."""
        self.n_qubits = n_qubits
        self.marked_items = marked_items

        # Calculate optimal number of iterations
        N = 2**n_qubits
        self.iterations = int(np.pi / 4 * np.sqrt(N))

        self.circuit = QuantumCircuit(n_qubits, n_qubits)

        # Initialize superposition
        for i in range(n_qubits):
            self.circuit.h(i)

        # Grover iterations
        for _ in range(self.iterations):
            # Oracle
            self._add_oracle(marked_items)

            # Diffuser
            self._add_diffuser()

        # Measure
        self.circuit.measure_all()

        return self.circuit

    def _add_oracle(self, marked_items):
        """
        Add Grover oracle that marks specified items.
        
        Mathematical Concept - Grover Oracle:
        ------------------------------------
        The oracle flips the phase of the marked state(s).
        
        Oracle Operation:
        ----------------
        O|x⟩ = {  −|x⟩  if x is marked (target)
              {   |x⟩  otherwise
        
        Mathematically:
        O = I − 2|w⟩⟨w|
        
        where |w⟩ is the marked state.
        
        Phase Flip Implementation:
        -------------------------
        To flip phase of state |w⟩ = |w_n...w_1w_0⟩:
        
        1. Apply X gates to qubits where w_i = 0
           (This makes target state |1...1⟩)
        
        2. Apply multi-controlled Z gate
           (Flips phase of |1...1⟩ state)
           Math: Z|1⟩ = −|1⟩
        
        3. Undo X gates (restore original basis)
        
        Example for |101⟩:
        - Apply X to qubit 1: |101⟩ → |111⟩
        - Apply CCZ: −|111⟩
        - Undo X: −|101⟩ (phase flipped!)
        
        Multi-Controlled Z Gate:
        -----------------------
        For n qubits, need (n-1)-controlled Z:
        - Decompose using H-CCX-H pattern
        - CCX = Toffoli gate (2-controlled X)
        
        Why Phase Flip?
        - Allows interference to amplify marked state
        - Combined with diffusion → amplitude amplification
        """
        for item in marked_items:
            # Convert item to binary and apply multi-controlled Z
            binary_item = format(item, f"0{self.n_qubits}b")

            # Step 1: Flip qubits where marked state has 0
            # Transform |w⟩ to |1...1⟩
            for i, bit in enumerate(binary_item):
                if bit == "0":
                    self.circuit.x(i)

            # Step 2: Multi-controlled Z gate
            # Flips phase of |1...1⟩ state
            if self.n_qubits == 1:
                self.circuit.z(0)
            elif self.n_qubits == 2:
                self.circuit.cz(0, 1)
            else:
                # Use H-CCX-H for multi-controlled Z
                # Math: HZH = X, so HXH = Z
                self.circuit.h(self.n_qubits - 1)
                self._multi_controlled_x(
                    list(range(self.n_qubits - 1)), self.n_qubits - 1
                )
                self.circuit.h(self.n_qubits - 1)

            # Step 3: Flip qubits back
            # Restore original basis
            for i, bit in enumerate(binary_item):
                if bit == "0":
                    self.circuit.x(i)

    def _add_diffuser(self):
        """
        Add diffusion operator (inversion about average).
        
        Mathematical Concept - Grover Diffusion Operator:
        -------------------------------------------------
        The diffusion operator inverts all amplitudes about their average.
        This is the key to amplitude amplification!
        
        Diffusion Operator D:
        --------------------
        D = 2|s⟩⟨s| − I
        
        where |s⟩ = H^⊗n|0⟩ = (1/√N)Σ|x⟩ is the uniform superposition
        
        Effect on Amplitude:
        -------------------
        For state ψ = Σ α_x|x⟩:
        
        α_x → 2⟨α⟩ − α_x
        
        where ⟨α⟩ = (1/N)Σα_x is the average amplitude
        
        Geometric Interpretation:
        ------------------------
        - Reflects amplitudes about their mean
        - States below average → become above average
        - States above average → become below average
        - After oracle marks target, diffusion amplifies it!
        
        Implementation: D = H^⊗n (2|0⟩⟨0| − I) H^⊗n
        
        Step-by-Step Construction:
        -------------------------
        1. H^⊗n: Transform to computational basis
           |s⟩ → |0⟩
        
        2. Apply (2|0⟩⟨0| − I):
           - Flip all qubits: |0⟩ → |1...1⟩
           - Multi-controlled Z: Flip phase of |1...1⟩
           - Flip qubits back
           This implements: −(I − 2|0⟩⟨0|) = 2|0⟩⟨0| − I
        
        3. H^⊗n: Transform back to superposition
           |0⟩ → |s⟩
        
        Combined with Oracle (Grover Operator):
        ---------------------------------------
        G = D·O (diffusion after oracle)
        
        Effect: Each iteration rotates state vector toward marked state
        Amplitude of marked state increases by ~2/√N each iteration
        
        After k iterations:
        α_marked ≈ sin((2k+1)θ) where θ = arcsin(1/√N)
        
        Optimal iterations: k ≈ (π/4)√N
        """
        # Step 1: Transform from superposition to computational basis
        # H^⊗n: |s⟩ → |0⟩
        for i in range(self.n_qubits):
            self.circuit.h(i)

        # Step 2a: Flip all qubits
        # Transform |0⟩ → |1...1⟩
        for i in range(self.n_qubits):
            self.circuit.x(i)

        # Step 2b: Multi-controlled Z
        # Flips phase of |1...1⟩ state
        # This implements conditional phase flip about |0⟩
        if self.n_qubits == 1:
            self.circuit.z(0)
        elif self.n_qubits == 2:
            self.circuit.cz(0, 1)
        else:
            self.circuit.h(self.n_qubits - 1)
            self._multi_controlled_x(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            self.circuit.h(self.n_qubits - 1)

        # Step 2c: Flip qubits back
        # Transform |1...1⟩ → |0⟩
        for i in range(self.n_qubits):
            self.circuit.x(i)

        # Step 3: Transform back to superposition
        # H^⊗n: |0⟩ → |s⟩
        # Now have implemented: 2|s⟩⟨s| − I
        for i in range(self.n_qubits):
            self.circuit.h(i)

    def _multi_controlled_x(self, control_qubits, target_qubit):
        """Implement multi-controlled X gate."""
        if len(control_qubits) == 1:
            self.circuit.cx(control_qubits[0], target_qubit)
        elif len(control_qubits) == 2:
            self.circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        # For more controls, would need decomposition

    def execute(self, shots=1024, marked_items=None):
        """Execute Grover's algorithm."""
        if marked_items is None:
            marked_items = [3]  # Default: mark item 3

        self.build_circuit(
            self.n_qubits if hasattr(self, "n_qubits") else 3, marked_items
        )

        simulator = AerSimulator()
        job = simulator.run(self.circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Analyze success probability
        success_counts = 0
        for item in marked_items:
            binary_item = format(item, f"0{self.n_qubits}b")
            success_counts += counts.get(binary_item, 0)

        success_probability = success_counts / shots

        self.results = {
            "counts": counts,
            "marked_items": marked_items,
            "success_probability": success_probability,
            "iterations_used": self.iterations,
            "theoretical_probability": 1.0,  # Ideal case
        }

        return self.results


class QuantumPhaseEstimation(QuantumAlgorithm):
    """Implementation of Quantum Phase Estimation algorithm."""

    def __init__(self):
        super().__init__(
            "Quantum Phase Estimation",
            "Estimates the phase of an eigenvalue of a unitary operator",
        )

    def build_circuit(self, n_counting_qubits, unitary_power=1):
        """Build quantum phase estimation circuit."""
        self.n_counting = n_counting_qubits
        self.n_total = n_counting_qubits + 1  # +1 for eigenstate qubit

        self.circuit = QuantumCircuit(self.n_total, n_counting_qubits)

        # Prepare eigenstate |1⟩ on the last qubit
        self.circuit.x(self.n_total - 1)

        # Initialize counting qubits in superposition
        for i in range(n_counting_qubits):
            self.circuit.h(i)

        # Controlled unitary operations
        for i in range(n_counting_qubits):
            # Apply controlled-U^(2^i)
            repetitions = unitary_power * (2**i)
            for _ in range(repetitions):
                self._add_controlled_unitary(i, self.n_total - 1)

        # Inverse QFT on counting qubits
        self._add_inverse_qft(list(range(n_counting_qubits)))

        # Measure counting qubits
        self.circuit.measure(range(n_counting_qubits), range(n_counting_qubits))

        return self.circuit

    def _add_controlled_unitary(self, control_qubit, target_qubit):
        """Add controlled unitary operation (using T gate as example)."""
        # Example: controlled-T gate (phase = π/4)
        self.circuit.cp(np.pi / 4, control_qubit, target_qubit)

    def _add_inverse_qft(self, qubits):
        """Add inverse quantum Fourier transform."""
        n = len(qubits)

        for i in range(n // 2):
            self.circuit.swap(qubits[i], qubits[n - 1 - i])

        for i in range(n):
            for j in range(i):
                self.circuit.cp(-np.pi / 2 ** (i - j), qubits[j], qubits[i])
            self.circuit.h(qubits[i])

    def execute(self, shots=1024):
        """Execute quantum phase estimation."""
        n_counting = 4  # Default precision
        self.build_circuit(n_counting)

        simulator = AerSimulator()
        job = simulator.run(self.circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Estimate phase from measurement results
        estimated_phases = {}
        for bitstring, count in counts.items():
            # Convert binary to decimal and normalize
            decimal_value = int(bitstring, 2)
            estimated_phase = decimal_value / (2**n_counting)
            estimated_phases[estimated_phase] = count

        # Find most probable phase
        best_phase = max(estimated_phases.keys(), key=lambda x: estimated_phases[x])

        self.results = {
            "counts": counts,
            "estimated_phases": estimated_phases,
            "best_phase_estimate": best_phase,
            "theoretical_phase": 1 / 8,  # π/4 normalized to [0,1)
            "precision_qubits": n_counting,
        }

        return self.results


class AlgorithmBenchmark:
    """Benchmark and compare quantum algorithms."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.algorithms = {}
        self.benchmark_results = {}

    def register_algorithm(self, algorithm):
        """Register an algorithm for benchmarking."""
        self.algorithms[algorithm.name] = algorithm

    def benchmark_algorithm(self, algorithm_name, **kwargs):
        """Benchmark a specific algorithm."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not registered")

        algorithm = self.algorithms[algorithm_name]

        print(f"\n=== Benchmarking {algorithm_name} ===")
        print(f"Description: {algorithm.description}")

        # Execute algorithm
        start_time = time.time()
        results = algorithm.execute(**kwargs)
        execution_time = time.time() - start_time

        # Analyze complexity
        complexity = algorithm.analyze_complexity()

        benchmark_result = {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "complexity": complexity,
            "results": results,
        }

        self.benchmark_results[algorithm_name] = benchmark_result

        # Print results
        print(f"Execution time: {execution_time:.4f} seconds")
        if complexity:
            print(f"Circuit qubits: {complexity['qubits']}")
            print(f"Circuit depth: {complexity['depth']}")
            print(f"Total gates: {complexity['gates']}")

        return benchmark_result

    def compare_algorithms(self):
        """Compare benchmarked algorithms."""
        if len(self.benchmark_results) < 2:
            print("Need at least 2 algorithms for comparison")
            return

        print("\n=== Algorithm Comparison ===")

        # Create comparison table
        print(
            f"{'Algorithm':<25} {'Qubits':<8} {'Depth':<8} {'Gates':<8} {'Time(s)':<10}"
        )
        print("-" * 70)

        for name, result in self.benchmark_results.items():
            complexity = result["complexity"]
            if complexity:
                print(
                    f"{name:<25} {complexity['qubits']:<8} {complexity['depth']:<8} "
                    f"{complexity['gates']:<8} {result['execution_time']:<10.4f}"
                )

    def visualize_benchmarks(self):
        """Visualize benchmark results."""
        if not self.benchmark_results:
            print("No benchmark results to visualize")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        algorithms = list(self.benchmark_results.keys())

        # 1. Execution time comparison
        times = [result["execution_time"] for result in self.benchmark_results.values()]

        ax1.bar(algorithms, times, alpha=0.7, color="skyblue")
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Algorithm Execution Time")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Circuit complexity
        qubits = []
        depths = []
        gates = []

        for result in self.benchmark_results.values():
            complexity = result["complexity"]
            if complexity:
                qubits.append(complexity["qubits"])
                depths.append(complexity["depth"])
                gates.append(complexity["gates"])

        x = np.arange(len(algorithms))
        width = 0.25

        ax2.bar(x - width, qubits, width, label="Qubits", alpha=0.7)
        ax2.bar(x, depths, width, label="Depth", alpha=0.7)
        ax2.bar(x + width, gates, width, label="Gates", alpha=0.7)

        ax2.set_xlabel("Algorithm")
        ax2.set_ylabel("Count")
        ax2.set_title("Circuit Complexity Metrics")
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Success rates (where applicable)
        success_rates = []
        for result in self.benchmark_results.values():
            alg_results = result["results"]
            if "success_probability" in alg_results:
                success_rates.append(alg_results["success_probability"])
            elif "is_constant" in alg_results:
                # For Deutsch-Jozsa, success is correct classification
                success_rates.append(
                    1.0
                    if alg_results["is_constant"] == alg_results["theoretical_constant"]
                    else 0.0
                )
            else:
                success_rates.append(0.5)  # Default

        ax3.bar(algorithms, success_rates, alpha=0.7, color="lightgreen")
        ax3.set_ylabel("Success Rate")
        ax3.set_title("Algorithm Success Rates")
        ax3.tick_params(axis="x", rotation=45)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)

        # 4. Gate efficiency (gates per qubit)
        efficiencies = []
        for result in self.benchmark_results.values():
            complexity = result["complexity"]
            if complexity and complexity["qubits"] > 0:
                efficiency = complexity["gates"] / complexity["qubits"]
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)

        ax4.bar(algorithms, efficiencies, alpha=0.7, color="orange")
        ax4.set_ylabel("Gates per Qubit")
        ax4.set_title("Gate Efficiency")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Quantum Algorithm Implementation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_quantum_algorithm_implementation.py --algorithm deutsch-jozsa
  python 04_quantum_algorithm_implementation.py --algorithm grover --qubits 3
  python 04_quantum_algorithm_implementation.py --benchmark-all
        """,
    )

    parser.add_argument(
        "--algorithm",
        choices=["deutsch-jozsa", "grover", "phase-estimation", "all"],
        default="all",
        help="Algorithm to run",
    )
    parser.add_argument(
        "--qubits", type=int, default=3, help="Number of qubits for algorithms"
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Number of measurement shots"
    )
    parser.add_argument(
        "--benchmark-all", action="store_true", help="Benchmark all algorithms"
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Show benchmark visualizations",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 3: Advanced Programming")
    print("Example 4: Quantum Algorithm Implementation")
    print("=" * 52)

    # Initialize benchmark
    benchmark = AlgorithmBenchmark(verbose=args.verbose)

    # Register algorithms
    dj_algorithm = DeutschJozsaAlgorithm()
    dj_algorithm.n_qubits = args.qubits

    grover_algorithm = GroverAlgorithm()
    grover_algorithm.n_qubits = args.qubits

    qpe_algorithm = QuantumPhaseEstimation()

    benchmark.register_algorithm(dj_algorithm)
    benchmark.register_algorithm(grover_algorithm)
    benchmark.register_algorithm(qpe_algorithm)

    try:
        if args.benchmark_all or args.algorithm == "all":
            # Benchmark all algorithms
            benchmark.benchmark_algorithm(
                "Deutsch-Jozsa Algorithm", shots=args.shots, oracle_type="balanced"
            )
            benchmark.benchmark_algorithm(
                "Grover's Search Algorithm", shots=args.shots, marked_items=[3]
            )
            benchmark.benchmark_algorithm("Quantum Phase Estimation", shots=args.shots)

            benchmark.compare_algorithms()
        else:
            # Run specific algorithm
            if args.algorithm == "deutsch-jozsa":
                benchmark.benchmark_algorithm(
                    "Deutsch-Jozsa Algorithm", shots=args.shots, oracle_type="balanced"
                )
            elif args.algorithm == "grover":
                benchmark.benchmark_algorithm(
                    "Grover's Search Algorithm", shots=args.shots, marked_items=[3]
                )
            elif args.algorithm == "phase-estimation":
                benchmark.benchmark_algorithm(
                    "Quantum Phase Estimation", shots=args.shots
                )

        if args.show_visualization:
            benchmark.visualize_benchmarks()

        print(f"\n✅ Algorithm implementation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during algorithm execution: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
