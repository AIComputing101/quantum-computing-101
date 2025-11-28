#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction
Example 7: Google Willow Surface Code Scaling

Analysis of Google's December 2024 breakthrough: first demonstration
of below-threshold quantum error correction with exponential error suppression.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")


class WillowSurfaceCodeAnalysis:
    """
    Google Willow Surface Code Analysis
    
    Demonstrates the threshold theorem and exponential error suppression
    achieved in Google's Willow chip (December 2024).
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
        # Google Willow specifications (approximate)
        self.willow_specs = {
            'physical_qubits': 105,
            'physical_error_rate': 0.001,  # ~0.1% per gate
            'T1_time': 100e-6,  # 100 microseconds
            'T2_time': 50e-6,   # 50 microseconds
            'gate_fidelity': 0.999,  # 99.9%
            'gate_time': 50e-9,  # 50 nanoseconds
            'threshold': 0.01    # ~1% threshold for surface codes
        }
    
    def threshold_theorem_analysis(self):
        """
        Analyze the threshold theorem
        
        Key equation: p_logical(d) = (p_physical / p_threshold)^((d+1)/2)
        
        Below threshold: λ = p_physical / p_threshold < 1
        → Logical errors decrease exponentially with code distance
        """
        print(f"\n🎯 Threshold Theorem Analysis")
        print(f"=" * 45)
        
        p_phys = self.willow_specs['physical_error_rate']
        p_th = self.willow_specs['threshold']
        lambda_factor = p_phys / p_th
        
        print(f"\n📊 Willow Chip Parameters:")
        print(f"   Physical error rate (p_phys): {p_phys*100:.2f}%")
        print(f"   Threshold (p_th): {p_th*100:.1f}%")
        print(f"   λ = p_phys / p_th: {lambda_factor:.3f}")
        
        if lambda_factor < 1:
            print(f"   ✅ BELOW THRESHOLD (λ < 1)")
            print(f"   → Scaling up improves performance!")
        else:
            print(f"   ❌ ABOVE THRESHOLD (λ > 1)")
            print(f"   → Scaling up makes things worse")
        
        return lambda_factor
    
    def calculate_logical_error_rates(self, distances):
        """
        Calculate logical error rates for different code distances
        
        Args:
            distances: List of surface code distances
            
        Returns:
            Dictionary with error rates and qubit counts
        """
        p_phys = self.willow_specs['physical_error_rate']
        p_th = self.willow_specs['threshold']
        lambda_factor = p_phys / p_th
        
        results = {
            'distances': [],
            'physical_qubits': [],
            'logical_error_rates': [],
            'effective_coherence': [],
            'lambda_factors': []
        }
        
        for d in distances:
            # Surface code requires approximately d^2 + (d-1)^2 qubits
            n_qubits = d**2 + (d-1)**2
            
            # Logical error rate (simplified model)
            p_logical = lambda_factor**((d+1)/2)
            
            # Effective coherence time improvement
            coherence_improvement = 1 / p_logical
            
            results['distances'].append(d)
            results['physical_qubits'].append(n_qubits)
            results['logical_error_rates'].append(p_logical)
            results['effective_coherence'].append(coherence_improvement)
            results['lambda_factors'].append(lambda_factor**((d+1)/2))
        
        return results
    
    def demonstrate_willow_results(self):
        """
        Demonstrate Google Willow's experimental results
        
        Key achievement: First proof that logical errors decrease
        exponentially as code distance increases
        """
        print(f"\n🌟 Google Willow Experimental Results (Dec 2024)")
        print(f"=" * 50)
        
        # Test distances used by Google
        distances = [3, 5, 7, 9, 11]
        results = self.calculate_logical_error_rates(distances)
        
        print(f"\n{'Distance':<10} {'Qubits':<10} {'Logical Error':<15} {'Coherence×':<12}")
        print(f"-" * 50)
        
        for i, d in enumerate(results['distances']):
            print(f"{d:<10} {results['physical_qubits'][i]:<10} "
                  f"{results['logical_error_rates'][i]:.3e}        "
                  f"{results['effective_coherence'][i]:>10.1f}×")
        
        # Calculate error suppression ratios
        print(f"\n📉 Error Suppression Ratios:")
        for i in range(1, len(distances)):
            ratio = results['logical_error_rates'][i-1] / results['logical_error_rates'][i]
            print(f"   d={distances[i-1]} → d={distances[i]}: {ratio:.2f}× reduction")
        
        # Key metrics
        d3_error = results['logical_error_rates'][0]
        d7_error = results['logical_error_rates'][2]
        total_improvement = d3_error / d7_error
        
        print(f"\n🎯 Key Achievement:")
        print(f"   Distance-3: {d3_error:.3e} logical error rate")
        print(f"   Distance-7: {d7_error:.3e} logical error rate")
        print(f"   Total improvement: {total_improvement:.1f}×")
        print(f"   ✨ First experimental proof of exponential suppression!")
        
        return results
    
    def compare_above_below_threshold(self):
        """
        Compare performance above vs below threshold
        
        Shows why Willow's achievement is revolutionary
        """
        print(f"\n⚖️  Above vs Below Threshold Comparison")
        print(f"=" * 45)
        
        distances = [3, 5, 7, 9, 11]
        
        # Below threshold (Willow-quality hardware)
        p_below = 0.001
        p_th = 0.01
        lambda_below = p_below / p_th  # = 0.1 < 1
        
        # Above threshold (poor hardware)
        p_above = 0.02
        lambda_above = p_above / p_th  # = 2.0 > 1
        
        print(f"\n📊 Below Threshold (Good Hardware, λ=0.1):")
        print(f"{'Distance':<10} {'Qubits':<10} {'Logical Error':<15}")
        print(f"-" * 35)
        for d in distances:
            n_qubits = d**2 + (d-1)**2
            p_logical = lambda_below**((d+1)/2)
            print(f"{d:<10} {n_qubits:<10} {p_logical:.3e}")
        
        print(f"\n⚠️  Above Threshold (Poor Hardware, λ=2.0):")
        print(f"{'Distance':<10} {'Qubits':<10} {'Logical Error':<15}")
        print(f"-" * 35)
        for d in distances:
            n_qubits = d**2 + (d-1)**2
            p_logical = lambda_above**((d+1)/2)
            print(f"{d:<10} {n_qubits:<10} {p_logical:.3e}")
        
        print(f"\n🎯 Key Insight:")
        print(f"   Below threshold: Errors DECREASE with more qubits ✅")
        print(f"   Above threshold: Errors INCREASE with more qubits ❌")
        print(f"   Willow proved we're in the \"good regime\"!")
    
    def visualize_willow_results(self, results, save_path='willow_analysis.png'):
        """Visualize Google Willow's achievements"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Logical Error Rate vs Distance
        ax1 = axes[0, 0]
        ax1.semilogy(results['distances'], results['logical_error_rates'], 
                    'o-', linewidth=2, markersize=8, color='blue', label='Willow (λ=0.1)')
        
        # Add comparison: above threshold
        p_above = 0.02
        p_th = 0.01
        lambda_above = p_above / p_th
        above_threshold_errors = [lambda_above**((d+1)/2) for d in results['distances']]
        ax1.semilogy(results['distances'], above_threshold_errors,
                    'x--', linewidth=2, markersize=8, color='red', 
                    label='Poor Hardware (λ=2.0)', alpha=0.7)
        
        ax1.set_xlabel('Code Distance (d)', fontsize=12)
        ax1.set_ylabel('Logical Error Rate', fontsize=12)
        ax1.set_title('Exponential Error Suppression\n(Google Willow, Dec 2024)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Physical Qubits vs Distance
        ax2 = axes[0, 1]
        ax2.plot(results['distances'], results['physical_qubits'], 
                'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Code Distance (d)', fontsize=12)
        ax2.set_ylabel('Physical Qubits Required', fontsize=12)
        ax2.set_title('Qubit Overhead vs Protection Level', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Add annotation
        for i, d in enumerate(results['distances']):
            ax2.annotate(f'd={d}\n{results["physical_qubits"][i]} qubits',
                        (d, results['physical_qubits'][i]),
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=8)
        
        # Plot 3: Coherence Time Improvement
        ax3 = axes[1, 0]
        ax3.semilogy(results['distances'], results['effective_coherence'],
                    'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Code Distance (d)', fontsize=12)
        ax3.set_ylabel('Effective Coherence Time (×)', fontsize=12)
        ax3.set_title('Coherence Time Extension via QEC', fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Key Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        🌟 Google Willow Key Achievements
        {'=' * 40}
        
        📊 Hardware Specifications:
        • Physical qubits: {self.willow_specs['physical_qubits']}
        • Physical error: {self.willow_specs['physical_error_rate']*100:.2f}%
        • Gate fidelity: {self.willow_specs['gate_fidelity']*100:.2f}%
        • T1 time: {self.willow_specs['T1_time']*1e6:.0f} μs
        
        🎯 Breakthrough Results:
        • First below-threshold demonstration ✅
        • Exponential error suppression verified ✅
        • d=7: {results['logical_error_rates'][2]:.3e} logical error
        • Coherence: {results['effective_coherence'][2]:.0f}× improvement
        
        💡 Significance:
        Proves that scaling quantum computers
        actually works as theory predicts!
        
        This is the turning point from NISQ
        to fault-tolerant quantum computing.
        
        Reference: Google Quantum AI (Dec 2024)
        "Quantum Error Correction Below the
        Surface Code Threshold"
        """
        
        ax4.text(0.05, 0.5, summary_text, fontsize=9,
                family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Google Willow Surface Code Analysis (Dec 2024)"
    )
    parser.add_argument("--max-distance", type=int, default=13,
                       help="Maximum surface code distance to analyze")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  Quantum Computing 101 - Module 5: Error Correction   ║")
    print("║  Example 7: Google Willow Surface Code (Dec 2024)     ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    try:
        # Initialize analyzer
        analyzer = WillowSurfaceCodeAnalysis(verbose=args.verbose)
        
        # Threshold theorem analysis
        lambda_factor = analyzer.threshold_theorem_analysis()
        
        # Demonstrate Willow results
        results = analyzer.demonstrate_willow_results()
        
        # Compare above/below threshold
        analyzer.compare_above_below_threshold()
        
        # Visualize if requested
        if args.visualize:
            analyzer.visualize_willow_results(results)
        
        print(f"\n🎓 Educational Takeaways:")
        print(f"   1. Threshold theorem experimentally verified ✅")
        print(f"   2. Below-threshold operation is achievable ✅")
        print(f"   3. Scaling quantum computers actually works ✅")
        print(f"   4. Path to fault-tolerant QC is clear ✅")
        
        print(f"\n🚀 What This Means for Quantum Computing:")
        print(f"   • We can now build larger, more reliable systems")
        print(f"   • Logical qubits are becoming practical")
        print(f"   • Transition from NISQ to fault-tolerant era")
        print(f"   • Real quantum advantage is within reach")
        
        print(f"\n✅ Google Willow analysis completed successfully!")
        return 0
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

