#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 5
First Quantum Algorithm - Quantum Random Number Generator

This example implements a complete quantum algorithm: a true quantum
random number generator that demonstrates quantum superposition and measurement.

Learning objectives:
- Build a complete quantum algorithm from start to finish
- Compare quantum vs classical randomness
- Understand the practical value of quantum randomness
- Create scalable quantum circuits

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit_aer import AerSimulator
from scipy import stats


class QuantumRandomNumberGenerator:
    """A true quantum random number generator using quantum superposition."""

    def __init__(self, backend=None):
        """Initialize the quantum RNG.

        Args:
            backend: Quantum backend to use (defaults to simulator)
        """
        self.backend = backend or AerSimulator()
        self.history = []

    def generate_single_bit(self, shots=1):
        """
        Generate a single random bit using quantum superposition.
        
        Mathematical Foundation - Quantum Random Number Generation:
        -----------------------------------------------------------
        This is a TRUE quantum random number generator, not pseudo-random!
        
        Algorithm:
        ----------
        1. Initialize: |ψ₀⟩ = |0⟩ (ground state)
        
        2. Apply Hadamard gate:
           H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩ (equal superposition)
           
           State after H: |ψ₁⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
        
        3. Measure in computational basis:
           - Probability of |0⟩: P(0) = |(1/√2)|² = 1/2 = 50%
           - Probability of |1⟩: P(1) = |(1/√2)|² = 1/2 = 50%
        
        Mathematical Details:
        --------------------
        Hadamard gate matrix:
        H = (1/√2)[[1,  1],
                   [1, -1]]
        
        Action on |0⟩ = [1, 0]ᵀ:
        H|0⟩ = (1/√2)[[1,  1],    [1]   = (1/√2)[1]   = [1/√2]
                      [1, -1]]  ×  [0]            [1]     [1/√2]
        
        This creates perfect 50-50 superposition!
        
        Born Rule (Measurement):
        -----------------------
        For state |ψ⟩ = α|0⟩ + β|1⟩:
        - P(measuring 0) = |α|² = |1/√2|² = 1/2
        - P(measuring 1) = |β|² = |1/√2|² = 1/2
        
        Why This Is Truly Random:
        -------------------------
        • Classical RNG: Deterministic algorithms (pseudo-random)
          - Given seed → predictable sequence
          - Computable by anyone with the algorithm
        
        • Quantum RNG: Fundamental quantum randomness
          - Measurement outcome is intrinsically random
          - No hidden variables (Bell's theorem)
          - Impossible to predict (even in principle!)
        
        This randomness comes from quantum mechanics itself,
        not from our lack of knowledge!
        
        Applications:
        ------------
        - Cryptography (truly unpredictable keys)
        - Monte Carlo simulations
        - Gambling (provably fair)
        - Scientific experiments requiring random sampling
        
        Args:
            shots (int): Number of measurements (repetitions)
            
        Returns:
            int or list: Single bit (0 or 1) if shots=1, 
                        list of bits if shots > 1
        """
        # Create quantum circuit with 1 qubit and 1 classical bit
        # Qubit holds quantum superposition
        # Classical bit stores measurement result
        qc = QuantumCircuit(1, 1)

        # Apply Hadamard gate to create equal superposition
        # Mathematical operation: |0⟩ → (|0⟩ + |1⟩)/√2
        # This creates the source of quantum randomness!
        qc.h(0)

        # Measure the qubit in computational basis
        # Collapse: (|0⟩ + |1⟩)/√2 → |0⟩ with 50% or |1⟩ with 50%
        # This measurement is fundamentally random (quantum mechanics)
        qc.measure(0, 0)

        # Execute the circuit on quantum backend (simulator or real hardware)
        # Each shot is an independent quantum experiment
        job = self.backend.run(transpile(qc, self.backend), shots=shots)
        result = job.result()
        counts = result.get_counts()

        if shots == 1:
            # Return single bit: either 0 or 1 (50% chance each)
            return int(list(counts.keys())[0])
        else:
            # Return list of bits from multiple measurements
            # Statistical distribution should approach 50-50 as shots → ∞
            bits = []
            for outcome, count in counts.items():
                bits.extend([int(outcome)] * count)
            np.random.shuffle(bits)  # Shuffle to remove any ordering artifacts
            return bits

    def generate_random_integer(self, n_bits=8):
        """Generate a random integer using n_bits quantum bits.

        Args:
            n_bits: Number of bits (default: 8 for 0-255 range)

        Returns:
            int: Random integer
        """
        # Create quantum circuit with n_bits qubits
        qc = QuantumCircuit(n_bits, n_bits)

        # Put all qubits in superposition
        for i in range(n_bits):
            qc.h(i)

        # Measure all qubits explicitly
        for i in range(n_bits):
            qc.measure(i, i)

        # Execute the circuit
        job = self.backend.run(transpile(qc, self.backend), shots=1)
        result = job.result()
        counts = result.get_counts()

        # Convert binary string to integer
        # Remove any spaces that might be in the binary string
        binary_string = list(counts.keys())[0].replace(' ', '')
        integer_value = int(binary_string, 2)

        self.history.append(integer_value)
        return integer_value

    def generate_random_float(self, n_bits=16):
        """Generate a random float between 0 and 1.

        Args:
            n_bits: Number of bits for precision (default: 16)

        Returns:
            float: Random float between 0 and 1
        """
        max_value = 2**n_bits - 1
        random_int = self.generate_random_integer(n_bits)
        return random_int / max_value


def demonstrate_basic_qrng():
    """Demonstrate basic quantum random number generation."""
    print("=== BASIC QUANTUM RANDOM NUMBER GENERATION ===")
    print()

    qrng = QuantumRandomNumberGenerator()

    # Generate single bits
    print("Generating single random bits:")
    for i in range(10):
        bit = qrng.generate_single_bit()
        print(f"  Bit {i+1}: {bit}")
    print()

    # Generate random integers
    print("Generating random integers (8-bit, 0-255):")
    for i in range(5):
        number = qrng.generate_random_integer(8)
        print(f"  Number {i+1}: {number}")
    print()

    # Generate random floats
    print("Generating random floats (0.0-1.0):")
    for i in range(5):
        number = qrng.generate_random_float(16)
        print(f"  Float {i+1}: {number:.6f}")
    print()


def analyze_randomness_quality():
    """Analyze the quality of quantum randomness."""
    print("=== RANDOMNESS QUALITY ANALYSIS ===")
    print()

    qrng = QuantumRandomNumberGenerator()

    # Generate large sample of bits
    print("Generating 1000 random bits for analysis...")
    bits = qrng.generate_single_bit(shots=1000)

    # Basic statistics
    zeros = bits.count(0)
    ones = bits.count(1)

    print(f"Bit distribution:")
    print(f"  Zeros: {zeros} ({100*zeros/len(bits):.1f}%)")
    print(f"  Ones: {ones} ({100*ones/len(bits):.1f}%)")
    print(f"  Expected: ~50% each for true randomness")
    print()

    # Chi-square test for uniformity
    observed = [zeros, ones]
    expected = [len(bits) / 2, len(bits) / 2]
    chi2_stat, p_value = stats.chisquare(observed, expected)

    print(f"Chi-square test for uniformity:")
    print(f"  Chi-square statistic: {chi2_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Random (p > 0.05): {p_value > 0.05}")
    print()

    # Runs test for independence
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1

    expected_runs = (2 * zeros * ones) / len(bits) + 1

    print(f"Runs test for independence:")
    print(f"  Observed runs: {runs}")
    print(f"  Expected runs: {expected_runs:.1f}")
    print(f"  Ratio: {runs/expected_runs:.3f}")
    print(f"  Good randomness: ratio should be close to 1.0")
    print()

    return bits


def compare_quantum_classical_randomness():
    """Compare quantum and classical random number generation."""
    print("=== QUANTUM vs CLASSICAL RANDOMNESS ===")
    print()

    # Generate quantum random numbers
    qrng = QuantumRandomNumberGenerator()
    quantum_numbers = [qrng.generate_random_integer(8) for _ in range(100)]

    # Generate classical pseudorandom numbers
    np.random.seed(42)  # Fixed seed for reproducibility
    classical_numbers = [np.random.randint(0, 256) for _ in range(100)]

    # Reset seed for different classical sequence
    np.random.seed(123)
    classical_numbers_2 = [np.random.randint(0, 256) for _ in range(100)]

    print("First 10 numbers from each source:")
    print(f"Quantum:     {quantum_numbers[:10]}")
    print(f"Classical 1: {classical_numbers[:10]}")
    print(f"Classical 2: {classical_numbers_2[:10]}")
    print()

    # Statistical comparison
    def analyze_sequence(numbers, name):
        mean = np.mean(numbers)
        std = np.std(numbers)
        min_val = np.min(numbers)
        max_val = np.max(numbers)
        unique = len(set(numbers))

        print(f"{name} statistics:")
        print(f"  Mean: {mean:.2f} (expected: ~127.5)")
        print(f"  Std dev: {std:.2f} (expected: ~73.8)")
        print(f"  Range: {min_val}-{max_val} (expected: 0-255)")
        print(f"  Unique values: {unique}/100")
        print()

    analyze_sequence(quantum_numbers, "Quantum")
    analyze_sequence(classical_numbers, "Classical 1")
    analyze_sequence(classical_numbers_2, "Classical 2")

    # Visualize distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(quantum_numbers, bins=20, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_title("Quantum Random Numbers")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(classical_numbers, bins=20, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_title("Classical Random Numbers (seed=42)")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(
        classical_numbers_2, bins=20, alpha=0.7, color="green", edgecolor="black"
    )
    axes[2].set_title("Classical Random Numbers (seed=123)")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("module1_05_randomness_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Key differences:")
    print("• Quantum: True randomness from quantum measurement")
    print("• Classical: Pseudorandom (deterministic algorithm)")
    print("• Quantum: Unpredictable even with perfect knowledge")
    print("• Classical: Reproducible with same seed")
    print()

    return quantum_numbers, classical_numbers


def demonstrate_scalable_qrng():
    """Demonstrate scalable quantum random number generation."""
    print("=== SCALABLE QUANTUM RNG ===")
    print()

    qrng = QuantumRandomNumberGenerator()

    # Test different bit widths
    bit_widths = [1, 4, 8, 16]

    print("Random numbers with different bit widths:")
    for bits in bit_widths:
        max_value = 2**bits - 1
        number = qrng.generate_random_integer(bits)
        print(f"  {bits:2d} bits: {number:5d} (range: 0-{max_value})")
    print()

    # Demonstrate circuit scaling
    print("Circuit complexity scaling:")
    for bits in bit_widths:
        qc = QuantumCircuit(bits, bits)
        for i in range(bits):
            qc.h(i)
        qc.measure_all()

        print(f"  {bits:2d} bits: {qc.num_qubits} qubits, depth {qc.depth()}")
    print()


def build_practical_qrng_application():
    """Build a practical application using quantum RNG."""
    print("=== PRACTICAL APPLICATION: QUANTUM DICE ===")
    print()

    class QuantumDice:
        """A quantum dice that uses true quantum randomness."""

        def __init__(self, sides=6):
            self.sides = sides
            self.qrng = QuantumRandomNumberGenerator()

        def roll(self):
            """Roll the quantum dice."""
            # Need enough bits to represent all possible outcomes
            bits_needed = int(np.ceil(np.log2(self.sides)))

            while True:
                # Generate random number
                random_number = self.qrng.generate_random_integer(bits_needed)

                # Map to dice range (rejection sampling for uniform distribution)
                if random_number < self.sides:
                    return random_number + 1  # Dice are 1-indexed

    # Demonstrate quantum dice
    dice = QuantumDice(6)

    print("Rolling quantum dice 20 times:")
    rolls = []
    for i in range(20):
        roll = dice.roll()
        rolls.append(roll)
        print(f"  Roll {i+1:2d}: {roll}")
    print()

    # Analyze distribution
    print("Distribution analysis:")
    for value in range(1, 7):
        count = rolls.count(value)
        percentage = (count / len(rolls)) * 100
        print(f"  {value}: {count:2d} times ({percentage:5.1f}%)")
    print()

    # Visualize rolls
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of rolls
    ax1.hist(rolls, bins=range(1, 8), alpha=0.7, color="purple", edgecolor="black")
    ax1.set_title("Quantum Dice Rolls")
    ax1.set_xlabel("Dice Value")
    ax1.set_ylabel("Frequency")
    ax1.set_xticks(range(1, 7))

    # Time series of rolls
    ax2.plot(range(1, len(rolls) + 1), rolls, "o-", color="purple", markersize=6)
    ax2.set_title("Quantum Dice Roll Sequence")
    ax2.set_xlabel("Roll Number")
    ax2.set_ylabel("Dice Value")
    ax2.set_ylim(0.5, 6.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module1_05_quantum_dice.png", dpi=300, bbox_inches="tight")
    plt.close()

    return rolls


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Quantum Random Number Generator Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of shots for randomness analysis (default: 1000)",
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 1, Example 5")
    print("First Quantum Algorithm: Quantum Random Number Generator")
    print("=" * 60)
    print()

    try:
        # Basic QRNG demonstration
        demonstrate_basic_qrng()

        # Analyze randomness quality
        random_bits = analyze_randomness_quality()

        # Compare quantum vs classical
        quantum_nums, classical_nums = compare_quantum_classical_randomness()

        # Demonstrate scalability
        demonstrate_scalable_qrng()

        # Build practical application
        dice_rolls = build_practical_qrng_application()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print(
            "• module1_05_randomness_comparison.png - Quantum vs classical comparison"
        )
        print("• module1_05_quantum_dice.png - Quantum dice application")
        print()
        print("🎯 Key takeaways:")
        print("• Quantum RNG provides true randomness (not pseudorandom)")
        print("• Quantum superposition enables perfect bit randomness")
        print("• Quantum randomness is valuable for cryptography and simulation")
        print("• This is your first complete quantum algorithm!")
        print()
        print("🎉 Congratulations! You've completed Module 1!")
        print("Next: Module 2 - Mathematical Foundations")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy scipy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
