#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 4, Example 1
Deutsch-Jozsa Algorithm

This example implements the complete Deutsch-Jozsa algorithm, demonstrating
quantum advantage for determining if a function is constant or balanced.

Learning objectives:
- Implement the Deutsch-Jozsa algorithm from scratch
- Understand oracle construction for different functions
- Demonstrate exponential quantum speedup
- Compare with classical brute-force approach

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit_aer import AerSimulator
import random
import time


class DeutschJozsaOracle:
    """Oracle implementation for Deutsch-Jozsa algorithm.
    
    An oracle is a 'black box' quantum operation that computes a classical
    function f(x) and stores the result in a quantum circuit. The key insight
    from "Quantum Computing in Action" is that oracles transform the problem
    of function evaluation into quantum circuit operations.
    
    Oracle Design Principles:
    1. Reversibility: All quantum operations must be reversible
    2. Unitary: Oracle operations preserve quantum superposition
    3. Function embedding: f(x) is embedded as |x⟩|y⟩ → |x⟩|y ⊕ f(x)⟩
    """
    
    def __init__(self, n_qubits, function_type='random'):
        """Initialize the oracle.
        
        Args:
            n_qubits: Number of input qubits (determines function domain size)
            function_type: 'constant_0', 'constant_1', 'balanced', or 'random'
        """
        self.n_qubits = n_qubits
        self.function_type = function_type
        self.truth_table = self._generate_function()
        
        print(f"🔮 Oracle created: {function_type} function with {n_qubits} input qubits")
        print(f"   Function domain: {2**n_qubits} possible inputs")
        if len(self.truth_table) <= 16:  # Show truth table for small functions
            print(f"   Truth table: {self.truth_table}")
        else:
            print(f"   Truth table too large to display ({len(self.truth_table)} entries)")
        print(f"   Expected result: {'CONSTANT' if self._is_constant() else 'BALANCED'}")
    
    def _generate_function(self):
        """Generate the function based on type."""
        n_inputs = 2 ** self.n_qubits
        
        if self.function_type == 'constant_0':
            return [0] * n_inputs
        elif self.function_type == 'constant_1':
            return [1] * n_inputs
        elif self.function_type == 'balanced':
            # Create balanced function (half 0s, half 1s)
            truth_table = [0] * (n_inputs // 2) + [1] * (n_inputs // 2)
            random.shuffle(truth_table)
            return truth_table
        elif self.function_type == 'random':
            # Randomly choose constant or balanced
            if random.choice([True, False]):
                return self._generate_function() if self.function_type != 'constant_0' else [0] * n_inputs
            else:
                temp_type = self.function_type
                self.function_type = 'balanced'
                result = self._generate_function()
                self.function_type = temp_type
                return result
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def is_constant(self):
        """Check if the function is constant."""
        return len(set(self.truth_table)) == 1
    
    def create_oracle_circuit(self):
        """Create the quantum oracle circuit."""
        # n input qubits + 1 output qubit
        qc = QuantumCircuit(self.n_qubits + 1, name='Oracle')
        
        # Implement oracle based on truth table
        for i, output in enumerate(self.truth_table):
            if output == 1:
                # Create controlled X gate for this input pattern
                # Convert i to binary representation
                binary_str = format(i, f'0{self.n_qubits}b')
                
                # Apply X gates to qubits that should be 0 in the pattern
                for qubit, bit in enumerate(binary_str):
                    if bit == '0':
                        qc.x(qubit)
                
                # Apply multi-controlled X gate
                if self.n_qubits == 1:
                    qc.cx(0, self.n_qubits)
                elif self.n_qubits == 2:
                    qc.ccx(0, 1, self.n_qubits)
                else:
                    # For more qubits, use multi-controlled X
                    control_qubits = list(range(self.n_qubits))
                    qc.mcx(control_qubits, self.n_qubits)
                
                # Undo X gates
                for qubit, bit in enumerate(binary_str):
                    if bit == '0':
                        qc.x(qubit)
        
        return qc


def implement_deutsch_jozsa_algorithm(oracle):
    """Implement the complete Deutsch-Jozsa algorithm."""
    print(f"=== DEUTSCH-JOZSA ALGORITHM ({oracle.n_qubits} qubits) ===")
    print()
    
    n = oracle.n_qubits
    
    # Create quantum circuit
    # n input qubits + 1 output qubit + n classical bits
    qc = QuantumCircuit(n + 1, n)
    
    # Step 1: Initialize output qubit in |1⟩ state
    qc.x(n)
    
    # Step 2: Apply Hadamard gates to all qubits
    for i in range(n + 1):
        qc.h(i)
    
    # Step 3: Apply oracle
    oracle_circuit = oracle.create_oracle_circuit()
    qc = qc.compose(oracle_circuit)
    
    # Step 4: Apply Hadamard gates to input qubits
    for i in range(n):
        qc.h(i)
    
    # Step 5: Measure input qubits
    qc.measure(range(n), range(n))
    
    print("Complete Deutsch-Jozsa circuit:")
    print(qc.draw())
    print()
    
    # Execute the algorithm
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    print("Measurement results:")
    for outcome, count in sorted(counts.items()):
        percentage = (count / 1024) * 100
        print(f"  {outcome}: {count} times ({percentage:.1f}%)")
    
    # Determine if function is constant or balanced
    zero_string = '0' * n
    if zero_string in counts and counts[zero_string] > 512:  # Threshold for quantum measurement
        algorithm_result = "CONSTANT"
    else:
        algorithm_result = "BALANCED"
    
    actual_result = "CONSTANT" if oracle.is_constant() else "BALANCED"
    correct = algorithm_result == actual_result
    
    print()
    print(f"Algorithm prediction: {algorithm_result}")
    print(f"Actual function type: {actual_result}")
    print(f"Correct prediction: {correct}")
    print()
    
    return qc, counts, correct


def classical_solution(oracle):
    """Implement classical solution for comparison."""
    print("=== CLASSICAL SOLUTION ===")
    print()
    
    truth_table = oracle.truth_table
    n_inputs = len(truth_table)
    
    # Classical algorithm needs to check at most 2^(n-1) + 1 inputs
    max_checks = (n_inputs // 2) + 1
    
    print(f"Function truth table: {truth_table}")
    print(f"Total possible inputs: {n_inputs}")
    print(f"Maximum checks needed: {max_checks}")
    print()
    
    # Simulate classical checking
    checks_performed = 0
    first_output = truth_table[0]
    
    for i, output in enumerate(truth_table):
        checks_performed += 1
        
        if output != first_output:
            # Found different output, function is balanced
            classical_result = "BALANCED"
            break
        
        if checks_performed >= max_checks:
            # Checked enough, function is constant
            classical_result = "CONSTANT"
            break
    
    actual_result = "CONSTANT" if oracle.is_constant() else "BALANCED"
    correct = classical_result == actual_result
    
    print(f"Classical algorithm result: {classical_result}")
    print(f"Function evaluations performed: {checks_performed}")
    print(f"Actual function type: {actual_result}")
    print(f"Correct prediction: {correct}")
    print()
    
    return classical_result, checks_performed


def demonstrate_quantum_advantage():
    """Demonstrate quantum advantage of Deutsch-Jozsa algorithm."""
    print("=== QUANTUM ADVANTAGE DEMONSTRATION ===")
    print()
    
    qubit_counts = [1, 2, 3, 4, 5]
    quantum_queries = [1] * len(qubit_counts)  # Always 1 for quantum
    classical_queries = []
    
    for n in qubit_counts:
        # Classical worst case: 2^(n-1) + 1 queries
        classical_worst = (2 ** (n-1)) + 1
        classical_queries.append(classical_worst)
    
    print("Quantum vs Classical Query Complexity:")
    print("Qubits | Quantum Queries | Classical Queries (worst case)")
    print("-------|-----------------|-----------------------------")
    for n, q_queries, c_queries in zip(qubit_counts, quantum_queries, classical_queries):
        print(f"  {n:2d}   |       {q_queries:2d}        |            {c_queries:4d}")
    print()
    
    # Visualize the advantage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear vs exponential plot
    ax1.plot(qubit_counts, quantum_queries, 'b-o', label='Quantum', linewidth=3, markersize=8)
    ax1.plot(qubit_counts, classical_queries, 'r-s', label='Classical', linewidth=3, markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Number of Function Queries')
    ax1.set_title('Query Complexity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup factor
    speedup = [c / q for c, q in zip(classical_queries, quantum_queries)]
    ax2.bar(qubit_counts, speedup, alpha=0.7, color='green')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Quantum Speedup (Classical/Quantum)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for x, y in zip(qubit_counts, speedup):
        ax2.text(x, y + y*0.1, f'{y:.0f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('module4_01_quantum_advantage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return speedup


def test_different_functions():
    """Test the algorithm on different types of functions."""
    print("=== TESTING DIFFERENT FUNCTION TYPES ===")
    print()
    
    n_qubits = 3
    function_types = ['constant_0', 'constant_1', 'balanced']
    
    results = {}
    
    for func_type in function_types:
        print(f"Testing {func_type} function:")
        oracle = DeutschJozsaOracle(n_qubits, func_type)
        
        # Run quantum algorithm
        qc, counts, quantum_correct = implement_deutsch_jozsa_algorithm(oracle)
        
        # Run classical algorithm
        classical_result, classical_queries = classical_solution(oracle)
        
        results[func_type] = {
            'oracle': oracle,
            'quantum_correct': quantum_correct,
            'classical_queries': classical_queries,
            'counts': counts
        }
        
        print("-" * 50)
        print()
    
    # Visualize results
    fig, axes = plt.subplots(1, len(function_types), figsize=(5*len(function_types), 4))
    if len(function_types) == 1:
        axes = [axes]
    
    for i, func_type in enumerate(function_types):
        counts = results[func_type]['counts']
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i].bar(list(counts.keys()), list(counts.values()))
            axes[i].set_xlabel('Measurement Outcome')
            axes[i].set_ylabel('Counts')
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i].set_title(f'{func_type.replace("_", " ").title()} Function')
    
    plt.tight_layout()
    plt.savefig('module4_01_function_types.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def analyze_success_probability():
    """Analyze success probability with noise and finite shots."""
    print("=== SUCCESS PROBABILITY ANALYSIS ===")
    print()
    
    n_qubits = 2
    shot_counts = [10, 50, 100, 500, 1000, 5000]
    success_rates = {'constant': [], 'balanced': []}
    
    for shots in shot_counts:
        print(f"Testing with {shots} shots:")
        
        for func_type in ['constant_0', 'balanced']:
            successes = 0
            trials = 20  # Run multiple trials for statistics
            
            for trial in range(trials):
                oracle = DeutschJozsaOracle(n_qubits, func_type)
                
                # Create and run circuit
                n = oracle.n_qubits
                qc = QuantumCircuit(n + 1, n)
                
                # Deutsch-Jozsa algorithm
                qc.x(n)
                for i in range(n + 1):
                    qc.h(i)
                
                oracle_circuit = oracle.create_oracle_circuit()
                qc = qc.compose(oracle_circuit)
                
                for i in range(n):
                    qc.h(i)
                qc.measure(range(n), range(n))
                
                # Execute
                simulator = AerSimulator()
                job = simulator.run(transpile(qc, simulator), shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                # Check result
                zero_string = '0' * n
                if zero_string in counts:
                    zero_prob = counts[zero_string] / shots
                else:
                    zero_prob = 0
                
                if func_type == 'constant_0':
                    # For constant function, should measure all zeros
                    if zero_prob > 0.7:  # Threshold accounting for noise
                        successes += 1
                else:
                    # For balanced function, should not measure all zeros
                    if zero_prob < 0.3:
                        successes += 1
            
            success_rate = successes / trials
            category = 'constant' if 'constant' in func_type else 'balanced'
            success_rates[category].append(success_rate)
            
            print(f"  {func_type}: {success_rate:.2f} success rate")
        
        print()
    
    # Plot success rates
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(shot_counts, success_rates['constant'], 'b-o', label='Constant Function', linewidth=2, markersize=8)
    ax.plot(shot_counts, success_rates['balanced'], 'r-s', label='Balanced Function', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Success Rate')
    ax.set_title('Algorithm Success Rate vs Number of Shots')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('module4_01_success_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return success_rates


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='Deutsch-Jozsa Algorithm Demo')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--qubits', type=int, default=3,
                       help='Number of input qubits (default: 3)')
    parser.add_argument('--function-type', choices=['constant_0', 'constant_1', 'balanced', 'random'],
                       default='random', help='Type of function to test (default: random)')
    args = parser.parse_args()
    
    print("🚀 Quantum Computing 101 - Module 4, Example 1")
    print("Deutsch-Jozsa Algorithm")
    print("=" * 50)
    print()
    
    try:
        # Create oracle and run main algorithm
        oracle = DeutschJozsaOracle(args.qubits, args.function_type)
        qc, counts, quantum_correct = implement_deutsch_jozsa_algorithm(oracle)
        
        # Run classical comparison
        classical_result, classical_queries = classical_solution(oracle)
        
        # Demonstrate quantum advantage
        speedup_factors = demonstrate_quantum_advantage()
        
        # Test different function types
        function_results = test_different_functions()
        
        # Analyze success probability
        success_rates = analyze_success_probability()
        
        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module4_01_quantum_advantage.png - Quantum speedup analysis")
        print("• module4_01_function_types.png - Different function type results")
        print("• module4_01_success_probability.png - Success rate analysis")
        print()
        print("🎯 Key takeaways:")
        print("• Deutsch-Jozsa provides exponential quantum speedup")
        print("• Quantum algorithm always uses exactly 1 function query")
        print("• Classical algorithm needs up to 2^(n-1) + 1 queries")
        print("• Algorithm demonstrates quantum parallelism")
        print("• Perfect example of quantum advantage for specific problems")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
