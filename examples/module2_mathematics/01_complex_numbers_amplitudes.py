#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 2, Example 1
Complex Numbers and Quantum Amplitudes

This example explores complex numbers in the context of quantum computing,
showing how they represent quantum amplitudes and phases.

Learning objectives:
- Master complex number arithmetic in quantum contexts
- Visualize quantum amplitudes on the complex plane
- Understand the relationship between amplitudes and probabilities
- Explore phase relationships in quantum states

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import cmath


def demonstrate_complex_basics():
    """Demonstrate basic complex number operations."""
    print("=== COMPLEX NUMBERS BASICS ===")
    print()
    
    # Define some complex numbers
    z1 = 3 + 4j
    z2 = 1 - 2j
    z3 = 2 * cmath.exp(1j * np.pi / 4)  # Polar form
    
    print("Basic complex numbers:")
    print(f"z1 = {z1} = {z1.real} + {z1.imag}i")
    print(f"z2 = {z2} = {z2.real} + {z2.imag}i")
    print(f"z3 = {z3:.3f} = 2*e^(iπ/4) (polar form)")
    print()
    
    # Complex operations
    print("Complex arithmetic:")
    print(f"z1 + z2 = {z1 + z2}")
    print(f"z1 * z2 = {z1 * z2}")
    print(f"z1 / z2 = {z1 / z2:.3f}")
    print(f"z1* (conjugate) = {z1.conjugate()}")
    print()
    
    # Magnitude and phase
    print("Magnitude and phase:")
    for i, z in enumerate([z1, z2, z3], 1):
        magnitude = abs(z)
        phase = cmath.phase(z)
        print(f"z{i}: |z| = {magnitude:.3f}, arg(z) = {phase:.3f} rad = {np.degrees(phase):.1f}°")
    print()
    
    return [z1, z2, z3]


def visualize_complex_plane(complex_numbers, labels=None):
    """Visualize complex numbers on the complex plane."""
    print("=== COMPLEX PLANE VISUALIZATION ===")
    print()
    
    if labels is None:
        labels = [f"z{i}" for i in range(1, len(complex_numbers) + 1)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Complex plane with vectors
    for z, label in zip(complex_numbers, labels):
        ax1.arrow(0, 0, z.real, z.imag, head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', alpha=0.7, length_includes_head=True)
        ax1.plot(z.real, z.imag, 'ro', markersize=8)
        ax1.annotate(f'{label} = {z:.2f}', (z.real, z.imag), 
                    xytext=(10, 10), textcoords='offset points')
    
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Complex Numbers as Vectors')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_aspect('equal')
    
    # Plot 2: Polar representation
    angles = [cmath.phase(z) for z in complex_numbers]
    magnitudes = [abs(z) for z in complex_numbers]
    
    ax2 = plt.subplot(122, projection='polar')
    colors = plt.cm.Set3(np.linspace(0, 1, len(complex_numbers)))
    
    for angle, mag, label, color in zip(angles, magnitudes, labels, colors):
        ax2.arrow(0, 0, angle, mag, head_width=0.1, head_length=0.1, 
                 fc=color, ec=color, alpha=0.7, length_includes_head=True)
        ax2.plot(angle, mag, 'o', color=color, markersize=8)
        ax2.annotate(label, (angle, mag), xytext=(10, 10), 
                    textcoords='offset points')
    
    ax2.set_title('Polar Representation\n(Magnitude and Phase)')
    
    plt.tight_layout()
    plt.savefig('module2_01_complex_plane.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_quantum_amplitudes():
    """Show how complex numbers represent quantum amplitudes."""
    print("=== QUANTUM AMPLITUDES ===")
    print()
    
    # Create quantum states with different amplitudes
    states = {}
    
    # |0⟩ state - real amplitude
    alpha_0 = 1 + 0j
    beta_0 = 0 + 0j
    states['|0⟩'] = [alpha_0, beta_0]
    
    # |1⟩ state - real amplitude
    alpha_1 = 0 + 0j
    beta_1 = 1 + 0j
    states['|1⟩'] = [alpha_1, beta_1]
    
    # |+⟩ state - real amplitudes
    alpha_plus = 1/np.sqrt(2) + 0j
    beta_plus = 1/np.sqrt(2) + 0j
    states['|+⟩'] = [alpha_plus, beta_plus]
    
    # |-⟩ state - negative real amplitude
    alpha_minus = 1/np.sqrt(2) + 0j
    beta_minus = -1/np.sqrt(2) + 0j
    states['|-⟩'] = [alpha_minus, beta_minus]
    
    # |i+⟩ state - complex amplitude
    alpha_i = 1/np.sqrt(2) + 0j
    beta_i = 0 + 1j/np.sqrt(2)
    states['|i+⟩'] = [alpha_i, beta_i]
    
    # Analyze each state
    for label, [alpha, beta] in states.items():
        print(f"State {label}:")
        print(f"  α = {alpha:.3f} (amplitude of |0⟩)")
        print(f"  β = {beta:.3f} (amplitude of |1⟩)")
        print(f"  |α|² = {abs(alpha)**2:.3f} (probability of |0⟩)")
        print(f"  |β|² = {abs(beta)**2:.3f} (probability of |1⟩)")
        print(f"  |α|² + |β|² = {abs(alpha)**2 + abs(beta)**2:.3f} (normalization)")
        
        if alpha != 0:
            print(f"  Phase of α: {np.degrees(cmath.phase(alpha)):.1f}°")
        if beta != 0:
            print(f"  Phase of β: {np.degrees(cmath.phase(beta)):.1f}°")
        print()
    
    return states


def visualize_quantum_amplitudes(states):
    """Visualize quantum amplitudes and their properties."""
    print("=== AMPLITUDE VISUALIZATION ===")
    print()
    
    n_states = len(states)
    fig, axes = plt.subplots(2, n_states, figsize=(4*n_states, 8))
    if n_states == 1:
        axes = axes.reshape(2, 1)
    
    # Plot amplitudes on complex plane
    for i, (label, [alpha, beta]) in enumerate(states.items()):
        # Amplitude visualization
        ax = axes[0, i]
        ax.arrow(0, 0, alpha.real, alpha.imag, head_width=0.02, head_length=0.02,
                fc='red', ec='red', alpha=0.7, length_includes_head=True, label='α (|0⟩)')
        ax.arrow(0, 0, beta.real, beta.imag, head_width=0.02, head_length=0.02,
                fc='blue', ec='blue', alpha=0.7, length_includes_head=True, label='β (|1⟩)')
        
        ax.plot(alpha.real, alpha.imag, 'ro', markersize=8)
        ax.plot(beta.real, beta.imag, 'bo', markersize=8)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Amplitudes: {label}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Probability bar chart
        ax2 = axes[1, i]
        probabilities = [abs(alpha)**2, abs(beta)**2]
        bars = ax2.bar(['|0⟩', '|1⟩'], probabilities, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Probabilities: {label}')
        ax2.set_ylim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('module2_01_quantum_amplitudes.png', dpi=300, bbox_inches='tight')
    plt.show()


def explore_phase_relationships():
    """Explore how phase affects quantum states."""
    print("=== PHASE RELATIONSHIPS ===")
    print()
    
    # Create states with different phases
    phases = np.linspace(0, 2*np.pi, 8)
    
    print("Exploring |+⟩ state with different global phases:")
    print("State: (|0⟩ + e^(iφ)|1⟩)/√2")
    print()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, phase in enumerate(phases):
        alpha = 1/np.sqrt(2)
        beta = cmath.exp(1j * phase) / np.sqrt(2)
        
        print(f"φ = {phase:.2f} rad ({np.degrees(phase):.0f}°):")
        print(f"  β = e^(i{phase:.2f})/√2 = {beta:.3f}")
        print(f"  |β|² = {abs(beta)**2:.3f}")
        print()
        
        # Plot amplitude on complex plane
        color = plt.cm.viridis(i / len(phases))
        ax1.arrow(0, 0, beta.real, beta.imag, head_width=0.02, head_length=0.02,
                 fc=color, ec=color, alpha=0.8, length_includes_head=True)
        ax1.plot(beta.real, beta.imag, 'o', color=color, markersize=6,
                label=f'φ={np.degrees(phase):.0f}°')
    
    # Unit circle
    circle = plt.Circle((0, 0), 1/np.sqrt(2), fill=False, linestyle='--', alpha=0.5)
    ax1.add_patch(circle)
    
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.set_title('β amplitude with different phases')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # Plot probabilities (should all be the same)
    probabilities = [0.5] * len(phases)
    ax2.plot(np.degrees(phases), probabilities, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Phase (degrees)')
    ax2.set_ylabel('P(|1⟩)')
    ax2.set_title('Probability vs Phase')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('module2_01_phase_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Key insight: Global phase doesn't affect measurement probabilities!")
    print("But relative phases between amplitudes do matter.")
    print()


def demonstrate_euler_formula():
    """Demonstrate Euler's formula and its quantum applications."""
    print("=== EULER'S FORMULA IN QUANTUM COMPUTING ===")
    print()
    
    print("Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)")
    print()
    
    # Demonstrate for specific angles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    print("Special angles:")
    for angle in angles:
        euler_form = cmath.exp(1j * angle)
        cartesian = complex(np.cos(angle), np.sin(angle))
        
        print(f"θ = {angle:.3f} rad ({np.degrees(angle):.0f}°):")
        print(f"  e^(iθ) = {euler_form:.3f}")
        print(f"  cos(θ) + i*sin(θ) = {cartesian:.3f}")
        print(f"  Match: {np.allclose(euler_form, cartesian)}")
        print()
        
        # Plot on unit circle
        ax1.plot(euler_form.real, euler_form.imag, 'ro', markersize=8)
        ax1.annotate(f'{np.degrees(angle):.0f}°', 
                    (euler_form.real, euler_form.imag),
                    xytext=(10, 10), textcoords='offset points')
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3, linewidth=2)
    ax1.set_xlabel('Real (cos θ)')
    ax1.set_ylabel('Imaginary (sin θ)')
    ax1.set_title('Euler\'s Formula on Unit Circle')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Show quantum gate phases
    gate_phases = {
        'I': 0,
        'Z': np.pi,
        'S': np.pi/2,
        'T': np.pi/4,
        'S†': -np.pi/2,
        'T†': -np.pi/4
    }
    
    gate_names = list(gate_phases.keys())
    phases = list(gate_phases.values())
    
    ax2.bar(gate_names, phases, alpha=0.7, color='purple')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('Common Quantum Gate Phases')
    ax2.grid(True, alpha=0.3)
    
    # Add phase values on bars
    for name, phase in gate_phases.items():
        height = phase if phase >= 0 else 0
        ax2.text(gate_names.index(name), height + 0.1, f'{phase:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('module2_01_euler_formula.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='Complex Numbers and Quantum Amplitudes Demo')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--show-math', action='store_true',
                       help='Show detailed mathematical calculations')
    args = parser.parse_args()
    
    print("🚀 Quantum Computing 101 - Module 2, Example 1")
    print("Complex Numbers and Quantum Amplitudes")
    print("=" * 50)
    print()
    
    try:
        # Demonstrate complex number basics
        complex_numbers = demonstrate_complex_basics()
        
        # Visualize on complex plane
        visualize_complex_plane(complex_numbers)
        
        # Quantum amplitudes
        quantum_states = demonstrate_quantum_amplitudes()
        
        # Visualize quantum amplitudes
        visualize_quantum_amplitudes(quantum_states)
        
        # Explore phase relationships
        explore_phase_relationships()
        
        # Euler's formula
        if args.show_math:
            demonstrate_euler_formula()
        
        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module2_01_complex_plane.png - Complex numbers visualization")
        print("• module2_01_quantum_amplitudes.png - Quantum amplitude analysis")
        print("• module2_01_phase_relationships.png - Phase effect on states")
        if args.show_math:
            print("• module2_01_euler_formula.png - Euler's formula applications")
        print()
        print("🎯 Key takeaways:")
        print("• Complex numbers encode both magnitude and phase information")
        print("• Quantum amplitudes are complex numbers with physical meaning")
        print("• Probabilities come from |amplitude|², phases affect interference")
        print("• Euler's formula connects exponential and trigonometric forms")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
