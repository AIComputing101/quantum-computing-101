#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction  
Example 4: Quantum Error Correction Protocols

Implementation of quantum error correction protocols and fault-tolerant operations.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, random_statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings
warnings.filterwarnings('ignore')

class QuantumErrorCorrectionProtocols:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def three_qubit_bit_flip_code(self):
        """Implementation of 3-qubit bit flip code."""
        # Encoding circuit
        encoding = QuantumCircuit(3, name='3-qubit_encode')
        encoding.cx(0, 1)
        encoding.cx(0, 2)
        
        # Decoding circuit
        decoding = QuantumCircuit(3, name='3-qubit_decode')
        decoding.cx(0, 1)
        decoding.cx(0, 2)
        
        # Syndrome measurement circuit
        syndrome = QuantumCircuit(5, 2, name='syndrome_measure')
        # Ancilla qubits for syndrome measurement
        syndrome.cx(0, 3)  # Check qubit 0 vs 1
        syndrome.cx(1, 3)
        syndrome.cx(1, 4)  # Check qubit 1 vs 2
        syndrome.cx(2, 4)
        syndrome.measure([3, 4], [0, 1])
        
        return {
            'encoding': encoding,
            'decoding': decoding,
            'syndrome': syndrome,
            'code_distance': 3,
            'correctable_errors': 1
        }
    
    def three_qubit_phase_flip_code(self):
        """Implementation of 3-qubit phase flip code."""
        # Encoding circuit (in X basis)
        encoding = QuantumCircuit(3, name='phase_flip_encode')
        encoding.h([0, 1, 2])  # Change to X basis
        encoding.cx(0, 1)
        encoding.cx(0, 2)
        encoding.h([0, 1, 2])  # Back to Z basis
        
        # Decoding circuit
        decoding = QuantumCircuit(3, name='phase_flip_decode')
        decoding.h([0, 1, 2])
        decoding.cx(0, 1)
        decoding.cx(0, 2)
        decoding.h([0, 1, 2])
        
        return {
            'encoding': encoding,
            'decoding': decoding,
            'code_distance': 3,
            'correctable_errors': 1
        }
    
    def shor_nine_qubit_code(self):
        """Implementation of Shor's 9-qubit code."""
        # Encoding: combines bit flip and phase flip codes
        encoding = QuantumCircuit(9, name='shor_9_encode')
        
        # First layer: bit flip encoding for each logical qubit
        for i in range(3):
            base = i * 3
            encoding.cx(base, base + 1)
            encoding.cx(base, base + 2)
        
        # Second layer: phase flip encoding
        encoding.h([0, 3, 6])
        encoding.cx(0, 3)
        encoding.cx(0, 6)
        encoding.h([0, 3, 6])
        
        # Repeat for other qubits in each block
        for offset in [1, 2]:
            encoding.h([offset, offset + 3, offset + 6])
            encoding.cx(offset, offset + 3)
            encoding.cx(offset, offset + 6)
            encoding.h([offset, offset + 3, offset + 6])
        
        return {
            'encoding': encoding,
            'logical_qubits': 1,
            'physical_qubits': 9,
            'code_distance': 3,
            'correctable_errors': 1
        }
    
    def fault_tolerant_cnot(self, code_type='steane'):
        """Implement fault-tolerant CNOT gate."""
        if code_type == 'steane':
            # Steane code transversal CNOT
            ft_cnot = QuantumCircuit(14, name='FT_CNOT_Steane')
            for i in range(7):
                ft_cnot.cx(i, i + 7)  # Transversal CNOT
        
        elif code_type == 'shor':
            # Shor code CNOT (requires additional operations)
            ft_cnot = QuantumCircuit(18, name='FT_CNOT_Shor')
            for i in range(9):
                ft_cnot.cx(i, i + 9)
        
        else:
            # Simple 3-qubit code
            ft_cnot = QuantumCircuit(6, name='FT_CNOT_3qubit')
            for i in range(3):
                ft_cnot.cx(i, i + 3)
        
        return ft_cnot
    
    def error_correction_cycle(self, code_qubits, syndrome_qubits, error_type='x'):
        """Complete error correction cycle."""
        total_qubits = len(code_qubits) + len(syndrome_qubits)
        
        # Initialize circuit
        circuit = QuantumCircuit(total_qubits, len(syndrome_qubits))
        
        # Syndrome extraction
        if error_type == 'x':
            # X-type stabilizers
            for i, anc in enumerate(syndrome_qubits):
                for j, code in enumerate(code_qubits):
                    if self.get_stabilizer_matrix(len(code_qubits))[i][j]:
                        circuit.cx(code, anc)
        
        elif error_type == 'z':
            # Z-type stabilizers
            for i, anc in enumerate(syndrome_qubits):
                for j, code in enumerate(code_qubits):
                    if self.get_stabilizer_matrix(len(code_qubits))[i][j]:
                        circuit.cz(code, anc)
        
        # Measure syndrome
        circuit.measure(syndrome_qubits, range(len(syndrome_qubits)))
        
        return circuit
    
    def get_stabilizer_matrix(self, n_qubits):
        """Get stabilizer matrix for given code."""
        if n_qubits == 3:
            # 3-qubit bit flip code stabilizers
            return np.array([
                [1, 1, 0],
                [0, 1, 1]
            ])
        elif n_qubits == 7:
            # Steane code stabilizers (simplified)
            return np.array([
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1]
            ])
        else:
            # Default identity-like matrix
            return np.eye(min(n_qubits-1, 3), n_qubits)
    
    def simulate_error_correction(self, code_type='3-qubit', error_rate=0.01, n_cycles=5):
        """Simulate error correction over multiple cycles."""
        results = {
            'code_type': code_type,
            'error_rate': error_rate,
            'cycles': n_cycles,
            'cycle_results': []
        }
        
        # Get code parameters
        if code_type == '3-qubit':
            code_info = self.three_qubit_bit_flip_code()
            n_code_qubits = 3
            n_syndrome_qubits = 2
        elif code_type == 'steane':
            n_code_qubits = 7
            n_syndrome_qubits = 3
        else:
            n_code_qubits = 9
            n_syndrome_qubits = 8
        
        # Simulate each cycle
        for cycle in range(n_cycles):
            cycle_result = self.simulate_single_cycle(
                n_code_qubits, n_syndrome_qubits, error_rate
            )
            cycle_result['cycle'] = cycle
            results['cycle_results'].append(cycle_result)
        
        # Calculate statistics
        results['success_rate'] = np.mean([r['correction_success'] for r in results['cycle_results']])
        results['average_syndrome_weight'] = np.mean([r['syndrome_weight'] for r in results['cycle_results']])
        
        return results
    
    def simulate_single_cycle(self, n_code_qubits, n_syndrome_qubits, error_rate):
        """Simulate a single error correction cycle."""
        # Create circuit
        total_qubits = n_code_qubits + n_syndrome_qubits
        circuit = QuantumCircuit(total_qubits, n_syndrome_qubits)
        
        # Initialize logical state (|0⟩_L)
        # For simplicity, start with all qubits in |0⟩
        
        # Add random errors
        n_errors = 0
        for i in range(n_code_qubits):
            if np.random.random() < error_rate:
                circuit.x(i)  # Bit flip error
                n_errors += 1
        
        # Error correction cycle
        code_qubits = list(range(n_code_qubits))
        syndrome_qubits = list(range(n_code_qubits, total_qubits))
        
        error_correction_circuit = self.error_correction_cycle(
            code_qubits, syndrome_qubits, 'x'
        )
        circuit.compose(error_correction_circuit, inplace=True)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Analyze syndrome
        syndrome = list(counts.keys())[0]
        syndrome_weight = syndrome.count('1')
        
        # Determine if correction was successful (simplified)
        correction_success = (n_errors <= 1)  # Can correct single errors
        
        return {
            'n_errors': n_errors,
            'syndrome': syndrome,
            'syndrome_weight': syndrome_weight,
            'correction_success': correction_success
        }
    
    def threshold_analysis(self, code_type='3-qubit', error_rates=None):
        """Analyze error threshold for given code."""
        if error_rates is None:
            error_rates = np.logspace(-4, -1, 10)  # 10^-4 to 10^-1
        
        results = {
            'error_rates': error_rates,
            'logical_error_rates': [],
            'code_type': code_type
        }
        
        for p in error_rates:
            # Simulate many cycles at this error rate
            simulation = self.simulate_error_correction(
                code_type, error_rate=p, n_cycles=100
            )
            
            # Calculate logical error rate
            logical_error_rate = 1 - simulation['success_rate']
            results['logical_error_rates'].append(logical_error_rate)
        
        # Find threshold (where logical error rate equals physical error rate)
        physical_rates = error_rates
        logical_rates = results['logical_error_rates']
        
        # Find crossing point
        threshold_idx = None
        for i in range(len(physical_rates)-1):
            if (logical_rates[i] <= physical_rates[i] and 
                logical_rates[i+1] > physical_rates[i+1]):
                threshold_idx = i
                break
        
        if threshold_idx is not None:
            results['threshold'] = physical_rates[threshold_idx]
        else:
            results['threshold'] = None
        
        return results
    
    def visualize_protocols(self, simulation_results, threshold_results=None):
        """Visualize error correction protocol results."""
        fig = plt.figure(figsize=(16, 12))
        
        # Cycle-by-cycle results
        ax1 = plt.subplot(2, 3, 1)
        cycles = [r['cycle'] for r in simulation_results['cycle_results']]
        errors = [r['n_errors'] for r in simulation_results['cycle_results']]
        successes = [1 if r['correction_success'] else 0 for r in simulation_results['cycle_results']]
        
        ax1.scatter(cycles, errors, c=successes, cmap='RdYlGn', alpha=0.7, s=50)
        ax1.set_title('Error Correction Performance')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Number of Errors')
        ax1.grid(True, alpha=0.3)
        
        # Success rate
        ax2 = plt.subplot(2, 3, 2)
        success_rate = simulation_results['success_rate']
        ax2.pie([success_rate, 1-success_rate], 
               labels=['Success', 'Failure'],
               autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral'],
               startangle=90)
        ax2.set_title(f'Overall Success Rate\n({simulation_results["code_type"]} code)')
        
        # Syndrome weight distribution
        ax3 = plt.subplot(2, 3, 3)
        syndrome_weights = [r['syndrome_weight'] for r in simulation_results['cycle_results']]
        ax3.hist(syndrome_weights, bins=range(max(syndrome_weights)+2), 
                alpha=0.7, color='blue', edgecolor='black')
        ax3.set_title('Syndrome Weight Distribution')
        ax3.set_xlabel('Syndrome Weight')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Error rate vs cycle
        ax4 = plt.subplot(2, 3, 4)
        error_counts = [r['n_errors'] for r in simulation_results['cycle_results']]
        ax4.plot(cycles, error_counts, 'b-o', alpha=0.7, markersize=4)
        ax4.set_title('Errors per Cycle')
        ax4.set_xlabel('Cycle')
        ax4.set_ylabel('Number of Errors')
        ax4.grid(True, alpha=0.3)
        
        # Threshold plot
        if threshold_results:
            ax5 = plt.subplot(2, 3, 5)
            physical_rates = threshold_results['error_rates']
            logical_rates = threshold_results['logical_error_rates']
            
            ax5.loglog(physical_rates, logical_rates, 'b-o', label='Logical error rate')
            ax5.loglog(physical_rates, physical_rates, 'r--', label='Physical error rate')
            
            if threshold_results['threshold']:
                ax5.axvline(x=threshold_results['threshold'], color='green', 
                           linestyle=':', label=f'Threshold ≈ {threshold_results["threshold"]:.2e}')
            
            ax5.set_title('Error Threshold Analysis')
            ax5.set_xlabel('Physical Error Rate')
            ax5.set_ylabel('Logical Error Rate')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Code comparison
        ax6 = plt.subplot(2, 3, 6)
        codes = ['3-qubit\nBit Flip', '3-qubit\nPhase Flip', 'Shor\n9-qubit', 'Steane\n7-qubit']
        distances = [3, 3, 3, 3]
        rates = [1/3, 1/3, 1/9, 1/7]
        
        x = np.arange(len(codes))
        width = 0.35
        
        ax6_twin = ax6.twinx()
        bars1 = ax6.bar(x - width/2, distances, width, label='Distance', alpha=0.7, color='blue')
        bars2 = ax6_twin.bar(x + width/2, rates, width, label='Rate', alpha=0.7, color='red')
        
        ax6.set_xlabel('Code Type')
        ax6.set_ylabel('Distance', color='blue')
        ax6_twin.set_ylabel('Encoding Rate', color='red')
        ax6.set_title('Code Comparison')
        ax6.set_xticks(x)
        ax6.set_xticklabels(codes, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Quantum Error Correction Protocols")
    parser.add_argument('--code', choices=['3-qubit', 'steane', 'shor'], default='3-qubit')
    parser.add_argument('--cycles', type=int, default=20, help='Number of correction cycles')
    parser.add_argument('--error-rate', type=float, default=0.01, help='Physical error rate')
    parser.add_argument('--threshold-analysis', action='store_true')
    parser.add_argument('--show-visualization', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("Quantum Computing 101 - Module 5: Error Correction")
    print("Example 4: Quantum Error Correction Protocols")
    print("=" * 47)
    
    protocols = QuantumErrorCorrectionProtocols(verbose=args.verbose)
    
    try:
        # Get code information
        if args.code == '3-qubit':
            code_info = protocols.three_qubit_bit_flip_code()
            print(f"\n📋 3-Qubit Bit Flip Code:")
            print(f"   Distance: {code_info['code_distance']}")
            print(f"   Correctable errors: {code_info['correctable_errors']}")
        
        elif args.code == 'steane':
            print(f"\n📋 Steane 7-Qubit Code:")
            print(f"   Parameters: [7, 1, 3]")
            print(f"   Encodes 1 logical qubit in 7 physical qubits")
            print(f"   Can correct any single-qubit error")
        
        elif args.code == 'shor':
            code_info = protocols.shor_nine_qubit_code()
            print(f"\n📋 Shor 9-Qubit Code:")
            print(f"   Logical qubits: {code_info['logical_qubits']}")
            print(f"   Physical qubits: {code_info['physical_qubits']}")
            print(f"   Distance: {code_info['code_distance']}")
        
        # Simulate error correction
        print(f"\n🔄 Simulating {args.cycles} correction cycles...")
        simulation = protocols.simulate_error_correction(
            args.code, args.error_rate, args.cycles
        )
        
        print(f"\n📊 Simulation Results:")
        print(f"   Success rate: {simulation['success_rate']:.1%}")
        print(f"   Average syndrome weight: {simulation['average_syndrome_weight']:.2f}")
        print(f"   Error rate: {args.error_rate}")
        
        # Analyze specific cycles
        failed_cycles = [r for r in simulation['cycle_results'] if not r['correction_success']]
        if failed_cycles:
            print(f"   Failed corrections: {len(failed_cycles)}")
            avg_errors_failed = np.mean([r['n_errors'] for r in failed_cycles])
            print(f"   Average errors in failed cycles: {avg_errors_failed:.1f}")
        
        # Threshold analysis
        threshold_results = None
        if args.threshold_analysis:
            print(f"\n🎯 Analyzing error threshold...")
            threshold_results = protocols.threshold_analysis(args.code)
            
            if threshold_results['threshold']:
                print(f"   Estimated threshold: {threshold_results['threshold']:.2e}")
            else:
                print(f"   No clear threshold found in tested range")
        
        # Display fault-tolerant operations
        print(f"\n🛠️  Fault-Tolerant Operations:")
        ft_cnot = protocols.fault_tolerant_cnot(args.code)
        print(f"   FT CNOT qubits: {ft_cnot.num_qubits}")
        print(f"   FT CNOT depth: {ft_cnot.depth()}")
        
        if args.show_visualization:
            protocols.visualize_protocols(simulation, threshold_results)
        
        print(f"\n📚 Key Insights:")
        print(f"   • Error correction enables fault-tolerant quantum computing")
        print(f"   • Threshold theorem: below threshold, logical error rate decreases")
        print(f"   • Trade-off between code distance and encoding efficiency")
        print(f"   • Syndrome measurement reveals error information without destroying data")
        
        print(f"\n✅ Error correction protocol analysis completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
