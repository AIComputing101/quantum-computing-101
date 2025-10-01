#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 4: Quantum Algorithms
Example 4: Shor's Algorithm Demo

Demonstration of Shor's algorithm for integer factorization (simplified version).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import math
import warnings

warnings.filterwarnings("ignore")


class ShorsAlgorithmDemo:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def gcd(self, a, b):
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

    def modular_exponentiation(self, base, exp, mod):
        """Modular exponentiation: base^exp mod mod."""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result

    def find_period_classical(self, a, N):
        """Classical period finding for verification."""
        period = 1
        current = a % N
        while current != 1:
            current = (current * a) % N
            period += 1
            if period > N:  # Safety check
                return None
        return period

    def build_quantum_period_circuit(self, a, N, n_counting_qubits):
        """Build quantum circuit for period finding."""
        # Number of qubits needed for N
        n_work_qubits = int(np.ceil(np.log2(N)))
        total_qubits = n_counting_qubits + n_work_qubits

        circuit = QuantumCircuit(total_qubits, n_counting_qubits)

        # Initialize counting register in superposition
        for i in range(n_counting_qubits):
            circuit.h(i)

        # Initialize work register to |1⟩
        circuit.x(n_counting_qubits)

        # Controlled modular exponentiation
        for i in range(n_counting_qubits):
            power = 2**i
            # Simplified implementation - in practice would use
            # controlled modular multiplication circuits
            for _ in range(power % 4):  # Simplified period 4 case
                circuit.cx(i, n_counting_qubits)

        # Apply inverse QFT to counting register
        self.apply_inverse_qft(circuit, range(n_counting_qubits))

        # Measure counting register
        circuit.measure(range(n_counting_qubits), range(n_counting_qubits))

        return circuit

    def apply_inverse_qft(self, circuit, qubits):
        """Apply inverse QFT to specified qubits."""
        n = len(qubits)

        # Swap qubits
        for i in range(n // 2):
            circuit.swap(qubits[i], qubits[n - 1 - i])

        # Apply inverse QFT gates
        for i in range(n):
            for j in range(i):
                angle = -np.pi / (2 ** (i - j))
                circuit.cp(angle, qubits[j], qubits[i])
            circuit.h(qubits[i])

    def demonstrate_factorization(self, N, a=None):
        """Demonstrate Shor's factorization algorithm."""
        if self.verbose:
            print(f"Factoring N = {N}")

        # Choose random a if not provided
        if a is None:
            a = np.random.randint(2, N)
            while self.gcd(a, N) != 1:
                a = np.random.randint(2, N)

        print(f"Chosen a = {a}")

        # Check if gcd(a, N) != 1 (lucky guess)
        g = self.gcd(a, N)
        if g != 1:
            print(f"Lucky! gcd({a}, {N}) = {g}")
            return {"factors": [g, N // g], "method": "gcd"}

        # Find period classically for this demo
        period = self.find_period_classical(a, N)
        if period is None:
            return {"error": "Period finding failed"}

        print(f"Period of {a}^x mod {N} is {period}")

        # Check if period is even
        if period % 2 != 0:
            print("Period is odd, trying different a")
            return {"error": "Odd period"}

        # Check if a^(r/2) ≡ -1 (mod N)
        half_power = self.modular_exponentiation(a, period // 2, N)
        if half_power == N - 1:
            print(f"a^(r/2) ≡ -1 (mod N), trying different a")
            return {"error": "Trivial case"}

        # Find factors
        factor1 = self.gcd(half_power - 1, N)
        factor2 = self.gcd(half_power + 1, N)

        if factor1 > 1 and factor1 < N:
            factors = [factor1, N // factor1]
            print(f"Factors found: {factors}")
            return {"factors": factors, "period": period, "a": a}
        elif factor2 > 1 and factor2 < N:
            factors = [factor2, N // factor2]
            print(f"Factors found: {factors}")
            return {"factors": factors, "period": period, "a": a}
        else:
            return {"error": "Factorization failed"}

    def quantum_period_finding_demo(self, a, N):
        """Demonstrate quantum period finding (simplified)."""
        n_counting = 4  # Small example

        circuit = self.build_quantum_period_circuit(a, N, n_counting)

        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Analyze results to find period
        measured_values = []
        for outcome, count in counts.items():
            value = int(outcome, 2)
            measured_values.extend([value] * count)

        # Find most common non-zero measurement
        non_zero_counts = {k: v for k, v in counts.items() if int(k, 2) != 0}
        if non_zero_counts:
            most_common = max(non_zero_counts, key=non_zero_counts.get)
            measured_s = int(most_common, 2)

            # Extract period using continued fractions (simplified)
            if measured_s > 0:
                fraction = measured_s / (2**n_counting)
                # Simple fraction approximation
                for r in range(1, 16):
                    if abs(fraction - round(fraction * r) / r) < 0.1:
                        return {
                            "period_estimate": r,
                            "circuit": circuit,
                            "counts": counts,
                        }

        return {"period_estimate": None, "circuit": circuit, "counts": counts}

    def visualize_results(self, factorization_result, period_result=None):
        """Visualize factorization results."""
        fig, axes = plt.subplots(1, 2 if period_result else 1, figsize=(15, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Factorization summary
        ax1 = axes[0]
        if "factors" in factorization_result:
            factors = factorization_result["factors"]
            ax1.bar(["Factor 1", "Factor 2"], factors, alpha=0.7, color=["blue", "red"])
            ax1.set_title(f"Factors of {np.prod(factors)}")
            ax1.set_ylabel("Value")
            ax1.grid(True, alpha=0.3)

            # Add verification text
            verification = (
                f"Verification: {factors[0]} × {factors[1]} = {np.prod(factors)}"
            )
            ax1.text(
                0.5,
                max(factors) * 0.8,
                verification,
                transform=ax1.transData,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax1.text(
                0.5,
                0.5,
                f"Error: {factorization_result.get('error', 'Unknown')}",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
            )
            ax1.set_title("Factorization Result")

        # Period finding results
        if period_result and len(axes) > 1:
            ax2 = axes[1]
            counts = period_result["counts"]

            outcomes = list(counts.keys())
            values = list(counts.values())

            ax2.bar(outcomes, values, alpha=0.7, color="green")
            ax2.set_title("Quantum Period Finding Results")
            ax2.set_xlabel("Measured State")
            ax2.set_ylabel("Count")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Shor's Algorithm Demonstration")
    parser.add_argument("--number", type=int, default=15, help="Number to factor")
    parser.add_argument("--base", type=int, help="Base for period finding")
    parser.add_argument(
        "--quantum-demo", action="store_true", help="Run quantum period finding demo"
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 4: Quantum Algorithms")
    print("Example 4: Shor's Algorithm Demo")
    print("=" * 40)

    shor_demo = ShorsAlgorithmDemo(verbose=args.verbose)

    try:
        # Main factorization demonstration
        print(f"\nFactoring {args.number}...")
        result = shor_demo.demonstrate_factorization(args.number, args.base)

        if "factors" in result:
            factors = result["factors"]
            print(
                f"✅ Successfully factored {args.number} = {factors[0]} × {factors[1]}"
            )
        else:
            print(f"❌ Factorization failed: {result.get('error')}")

        # Quantum period finding demo
        period_result = None
        if args.quantum_demo and "factors" in result:
            print(f"\nDemonstrating quantum period finding...")
            a = result.get("a", 2)
            period_result = shor_demo.quantum_period_finding_demo(a, args.number)

            if period_result["period_estimate"]:
                print(f"Quantum estimated period: {period_result['period_estimate']}")
                print(f"Classical period: {result.get('period', 'Unknown')}")
            else:
                print("Quantum period estimation inconclusive")

        # Show educational notes
        print(f"\n📚 Educational Notes:")
        print(f"• Shor's algorithm provides exponential speedup for factorization")
        print(f"• Quantum period finding is the key subroutine")
        print(f"• This demo uses classical period finding for verification")
        print(
            f"• Real implementation requires sophisticated modular arithmetic circuits"
        )

        if args.show_visualization:
            shor_demo.visualize_results(result, period_result)

        print(f"\n✅ Shor's algorithm demo completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
