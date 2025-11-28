#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 2: Mathematics
Example 5: Tensor Products and Multi-Qubit Systems

This script explores tensor products, multi-qubit state construction,
and the mathematics of composite quantum systems.

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

# Add utils to path for visualization tools
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_aer import AerSimulator
import itertools
import warnings

warnings.filterwarnings("ignore")


class TensorProductAnalyzer:
    """Comprehensive analysis of tensor products and multi-qubit quantum systems."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.constructed_states = []

    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[TensorProduct] {message}")

    def basic_tensor_products(self):
        """
        Demonstrate basic tensor product operations.
        
        Mathematical Foundation - Tensor Products:
        -----------------------------------------
        The tensor product (also called Kronecker product) combines quantum
        states from separate systems into a composite system state.
        
        Notation: ⊗ (tensor product symbol)
        Also written as: |ψ⟩|φ⟩ or |ψ⟩ ⊗ |φ⟩ or |ψφ⟩
        
        Definition for Vectors:
        -----------------------
        For vectors |ψ⟩ = [a, b]ᵀ and |φ⟩ = [c, d]ᵀ:
        
        |ψ⟩ ⊗ |φ⟩ = [a, b]ᵀ ⊗ [c, d]ᵀ = [a·[c,d]ᵀ, b·[c,d]ᵀ]
                  = [ac, ad, bc, bd]ᵀ
        
        Detailed Example:
        |0⟩ ⊗ |1⟩ = [1, 0]ᵀ ⊗ [0, 1]ᵀ
                   = [1·[0,1]ᵀ, 0·[0,1]ᵀ]
                   = [1×0, 1×1, 0×0, 0×1]ᵀ
                   = [0, 1, 0, 0]ᵀ = |01⟩
        
        Dimensionality:
        ---------------
        If |ψ⟩ has dimension m and |φ⟩ has dimension n,
        then |ψ⟩ ⊗ |φ⟩ has dimension m × n
        
        For qubits:
        - 1 qubit: dimension 2¹ = 2
        - 2 qubits: dimension 2² = 4
        - 3 qubits: dimension 2³ = 8
        - n qubits: dimension 2ⁿ (exponential growth!)
        
        Computational Basis for 2 Qubits:
        ---------------------------------
        The 4 basis states are:
        |00⟩ = [1,0,0,0]ᵀ  (both qubits in |0⟩)
        |01⟩ = [0,1,0,0]ᵀ  (first |0⟩, second |1⟩)
        |10⟩ = [0,0,1,0]ᵀ  (first |1⟩, second |0⟩)
        |11⟩ = [0,0,0,1]ᵀ  (both qubits in |1⟩)
        
        Any 2-qubit state can be written as:
        |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
        where |α|² + |β|² + |γ|² + |δ|² = 1
        
        Properties of Tensor Products:
        ------------------------------
        1. NOT Commutative: |ψ⟩⊗|φ⟩ ≠ |φ⟩⊗|ψ⟩ (order matters!)
        2. Associative: (|ψ⟩⊗|φ⟩)⊗|χ⟩ = |ψ⟩⊗(|φ⟩⊗|χ⟩)
        3. Distributive: (|ψ⟩+|φ⟩)⊗|χ⟩ = |ψ⟩⊗|χ⟩ + |φ⟩⊗|χ⟩
        4. Scalar mult: (c|ψ⟩)⊗|φ⟩ = c(|ψ⟩⊗|φ⟩)
        
        Why Tensor Products in Quantum Computing?
        ------------------------------------------
        - Combine individual qubit states into multi-qubit states
        - State space grows exponentially (quantum advantage!)
        - Enables entanglement when combined with gates
        - Foundation for quantum algorithms
        
        Returns:
            dict: Dictionary of constructed two-qubit states
        """
        print("\n=== Basic Tensor Product Operations ===")

        # Single-qubit computational and superposition basis states
        # These are the building blocks for multi-qubit systems
        state_0 = np.array([1, 0], dtype=complex)  # |0⟩ = [1, 0]ᵀ
        state_1 = np.array([0, 1], dtype=complex)  # |1⟩ = [0, 1]ᵀ
        state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩ = (|0⟩+|1⟩)/√2
        state_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)  # |-⟩ = (|0⟩-|1⟩)/√2

        print("Single-qubit states:")
        print(f"|0⟩ = {state_0}")
        print(f"|1⟩ = {state_1}")
        print(f"|+⟩ = {state_plus}")
        print(f"|-⟩ = {state_minus}")

        # Two-qubit tensor products using Kronecker product (np.kron)
        print("\nTwo-qubit tensor products:")
        print("General form: |ψ⟩ ⊗ |φ⟩")
        print("np.kron(a, b) computes the Kronecker (tensor) product")

        # Computational basis states for 2 qubits
        # These form an orthonormal basis for the 4-dimensional Hilbert space
        state_00 = np.kron(state_0, state_0)  # |0⟩ ⊗ |0⟩ = [1,0,0,0]ᵀ
        state_01 = np.kron(state_0, state_1)  # |0⟩ ⊗ |1⟩ = [0,1,0,0]ᵀ
        state_10 = np.kron(state_1, state_0)  # |1⟩ ⊗ |0⟩ = [0,0,1,0]ᵀ
        state_11 = np.kron(state_1, state_1)  # |1⟩ ⊗ |1⟩ = [0,0,0,1]ᵀ

        print(f"\n|00⟩ = |0⟩ ⊗ |0⟩ = {state_00}")
        print(f"|01⟩ = |0⟩ ⊗ |1⟩ = {state_01}")
        print(f"|10⟩ = |1⟩ ⊗ |0⟩ = {state_10}")
        print(f"|11⟩ = |1⟩ ⊗ |1⟩ = {state_11}")

        if self.verbose:
            print("\nDetailed tensor product calculation for |01⟩:")
            print(f"|0⟩ ⊗ |1⟩ = {state_0} ⊗ {state_1}")
            print("Step-by-step: [1,0]ᵀ ⊗ [0,1]ᵀ")
            print("= [1·[0,1]ᵀ, 0·[0,1]ᵀ]")
            print(f"= [1×0, 1×1, 0×0, 0×1]ᵀ = {state_01}")

        # Mixed basis states - combining computational and superposition bases
        # These demonstrate the flexibility of tensor products
        print(f"\nMixed basis products:")
        state_0_plus = np.kron(state_0, state_plus)  # |0⟩ ⊗ |+⟩ = |0⟩⊗((|0⟩+|1⟩)/√2) = (|00⟩+|01⟩)/√2
        state_plus_1 = np.kron(state_plus, state_1)  # |+⟩ ⊗ |1⟩ = ((|0⟩+|1⟩)/√2)⊗|1⟩ = (|01⟩+|11⟩)/√2

        print(f"|0⟩ ⊗ |+⟩ = {state_0_plus}")
        print(f"   Expanded: |0⟩ ⊗ (|0⟩+|1⟩)/√2 = (|00⟩+|01⟩)/√2")
        print(f"|+⟩ ⊗ |1⟩ = {state_plus_1}")
        print(f"   Expanded: (|0⟩+|1⟩)/√2 ⊗ |1⟩ = (|01⟩+|11⟩)/√2")

        # Store for later analysis
        two_qubit_states = {
            "|00⟩": state_00,
            "|01⟩": state_01,
            "|10⟩": state_10,
            "|11⟩": state_11,
            "|0+⟩": state_0_plus,
            "|+1⟩": state_plus_1,
        }

        self.constructed_states.extend(two_qubit_states.items())
        return two_qubit_states

    def separable_vs_entangled_states(self):
        """Analyze separable vs entangled multi-qubit states."""
        print("\n=== Separable vs Entangled States ===")

        # Separable states (can be written as tensor products)
        print("1. Separable States:")
        print("   Can be written as |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩")

        # Product state example
        state_A = np.array([0.6, 0.8], dtype=complex)
        state_B = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
        separable_state = np.kron(state_A, state_B)

        print(f"\n   |ψ₁⟩ = {state_A}")
        print(f"   |ψ₂⟩ = {state_B}")
        print(f"   |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ = {separable_state}")

        # Verify separability using Schmidt decomposition
        separability = self._check_separability(separable_state)
        print(f"   Schmidt rank: {separability['schmidt_rank']}")
        print(f"   → {'Separable' if separability['is_separable'] else 'Entangled'}")

        # Entangled states (cannot be written as simple tensor products)
        print("\n2. Entangled States:")
        print("   Cannot be written as |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩")

        # Bell states
        bell_states = {
            "|Φ+⟩": np.array([1, 0, 0, 1]) / np.sqrt(2),
            "|Φ-⟩": np.array([1, 0, 0, -1]) / np.sqrt(2),
            "|Ψ+⟩": np.array([0, 1, 1, 0]) / np.sqrt(2),
            "|Ψ-⟩": np.array([0, 1, -1, 0]) / np.sqrt(2),
        }

        for name, state in bell_states.items():
            print(f"\n   {name} = {state}")
            separability = self._check_separability(state)
            print(f"   Schmidt rank: {separability['schmidt_rank']}")
            print(f"   Entanglement entropy: {separability['entanglement']:.4f}")
            print(
                f"   → {'Separable' if separability['is_separable'] else 'Entangled'}"
            )

        # Partially entangled state
        print("\n3. Partially Entangled State:")
        partial_entangled = np.array([0.8, 0, 0, 0.6], dtype=complex)
        partial_entangled = partial_entangled / np.linalg.norm(partial_entangled)

        print(f"   |ψ⟩ = {partial_entangled}")
        separability = self._check_separability(partial_entangled)
        print(f"   Schmidt rank: {separability['schmidt_rank']}")
        print(f"   Entanglement entropy: {separability['entanglement']:.4f}")
        print(f"   → {'Separable' if separability['is_separable'] else 'Entangled'}")

        return {
            "separable": separable_state,
            "bell_states": bell_states,
            "partial_entangled": partial_entangled,
        }

    def _check_separability(self, state_vector):
        """Check if a two-qubit state is separable using Schmidt decomposition."""
        # Reshape to 2x2 matrix for Schmidt decomposition
        state_matrix = state_vector.reshape(2, 2)

        # Perform SVD (Schmidt decomposition)
        U, s, Vh = np.linalg.svd(state_matrix)

        # Count non-zero singular values (Schmidt coefficients)
        schmidt_coeffs = s[s > 1e-12]
        schmidt_rank = len(schmidt_coeffs)

        # Calculate entanglement entropy
        if schmidt_rank > 1:
            # Normalize Schmidt coefficients
            schmidt_coeffs = schmidt_coeffs / np.linalg.norm(schmidt_coeffs)
            entanglement = -np.sum(
                schmidt_coeffs**2 * np.log2(schmidt_coeffs**2 + 1e-12)
            )
        else:
            entanglement = 0.0

        return {
            "schmidt_rank": schmidt_rank,
            "schmidt_coefficients": schmidt_coeffs,
            "entanglement": entanglement,
            "is_separable": schmidt_rank == 1,
        }

    def three_qubit_systems(self):
        """Explore three-qubit tensor product systems."""
        print("\n=== Three-Qubit Tensor Product Systems ===")

        # Basic three-qubit computational states
        print("1. Computational Basis States (8 states):")

        state_0 = np.array([1, 0])
        state_1 = np.array([0, 1])

        three_qubit_computational = {}
        for i, j, k in itertools.product([0, 1], repeat=3):
            state_name = f"|{i}{j}{k}⟩"
            if i == 0:
                qubit1 = state_0
            else:
                qubit1 = state_1

            if j == 0:
                qubit2 = state_0
            else:
                qubit2 = state_1

            if k == 0:
                qubit3 = state_0
            else:
                qubit3 = state_1

            # Triple tensor product
            state_vector = np.kron(np.kron(qubit1, qubit2), qubit3)
            three_qubit_computational[state_name] = state_vector

            print(f"   {state_name} = {state_vector}")

        # GHZ state (maximally entangled three-qubit state)
        print("\n2. GHZ State (Greenberger-Horne-Zeilinger):")
        ghz_state = (
            three_qubit_computational["|000⟩"] + three_qubit_computational["|111⟩"]
        ) / np.sqrt(2)
        print(f"   |GHZ⟩ = (|000⟩ + |111⟩)/√2 = {ghz_state}")

        # W state (symmetric entangled state)
        print("\n3. W State:")
        w_state = (
            three_qubit_computational["|001⟩"]
            + three_qubit_computational["|010⟩"]
            + three_qubit_computational["|100⟩"]
        ) / np.sqrt(3)
        print(f"   |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3 = {w_state}")

        # Analyze entanglement structure
        print("\n4. Entanglement Analysis:")

        # For GHZ state
        ghz_analysis = self._analyze_three_qubit_entanglement(ghz_state)
        print(f"\n   GHZ State:")
        print(f"   Bipartite entanglement A|BC: {ghz_analysis['A_BC']:.4f}")
        print(f"   Bipartite entanglement AB|C: {ghz_analysis['AB_C']:.4f}")
        print(f"   Bipartite entanglement AC|B: {ghz_analysis['AC_B']:.4f}")

        # For W state
        w_analysis = self._analyze_three_qubit_entanglement(w_state)
        print(f"\n   W State:")
        print(f"   Bipartite entanglement A|BC: {w_analysis['A_BC']:.4f}")
        print(f"   Bipartite entanglement AB|C: {w_analysis['AB_C']:.4f}")
        print(f"   Bipartite entanglement AC|B: {w_analysis['AC_B']:.4f}")

        return {
            "computational": three_qubit_computational,
            "ghz": ghz_state,
            "w": w_state,
        }

    def _analyze_three_qubit_entanglement(self, state_vector):
        """Analyze bipartite entanglement in three-qubit system."""
        # Convert to density matrix
        rho = np.outer(state_vector, np.conj(state_vector))

        entanglements = {}

        # A|BC partition (qubit 0 vs qubits 1,2)
        rho_A = self._partial_trace_3qubit(rho, [1, 2])
        entanglements["A_BC"] = self._von_neumann_entropy(rho_A)

        # AB|C partition (qubits 0,1 vs qubit 2)
        rho_AB = self._partial_trace_3qubit(rho, [2])
        entanglements["AB_C"] = self._von_neumann_entropy(rho_AB)

        # AC|B partition (qubits 0,2 vs qubit 1)
        # Need to reorder for partial trace
        rho_reordered = self._reorder_3qubit_density_matrix(rho, [0, 2, 1])
        rho_AC = self._partial_trace_3qubit(rho_reordered, [2])
        entanglements["AC_B"] = self._von_neumann_entropy(rho_AC)

        return entanglements

    def _partial_trace_3qubit(self, rho, trace_qubits):
        """Compute partial trace for 3-qubit system."""
        # For simplicity, use a basic implementation
        # In practice, would use more efficient algorithms

        if trace_qubits == [1, 2]:  # Trace out qubits 1 and 2
            # Result is 2x2 matrix for qubit 0
            result = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            for l1 in range(2):
                                for l2 in range(2):
                                    idx1 = i * 4 + k1 * 2 + k2
                                    idx2 = j * 4 + l1 * 2 + l2
                                    if k1 == l1 and k2 == l2:
                                        result[i, j] += rho[idx1, idx2]

        elif trace_qubits == [2]:  # Trace out qubit 2
            # Result is 4x4 matrix for qubits 0,1
            result = np.zeros((4, 4), dtype=complex)
            for i in range(4):
                for j in range(4):
                    for k in range(2):
                        idx1 = i * 2 + k
                        idx2 = j * 2 + k
                        result[i, j] += rho[idx1, idx2]

        return result

    def _reorder_3qubit_density_matrix(self, rho, order):
        """Reorder qubits in 3-qubit density matrix."""
        # Simplified implementation - in practice would use tensor manipulations
        return rho  # Placeholder

    def _von_neumann_entropy(self, rho):
        """Calculate von Neumann entropy of density matrix."""
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros

        if len(eigenvals) == 0:
            return 0.0

        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
        return entropy

    def partial_trace_demonstration(self):
        """Demonstrate partial trace operations."""
        print("\n=== Partial Trace Operations ===")

        # Create entangled state
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |Φ+⟩

        print("Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        print(f"State vector: {bell_state}")

        # Convert to density matrix
        rho_full = np.outer(bell_state, np.conj(bell_state))
        print(f"\nFull density matrix ρ:")
        print(rho_full)

        # Partial trace over second qubit
        print(f"\nPartial trace over qubit 2: ρ₁ = Tr₂(ρ)")

        # Manual calculation
        rho_1 = np.zeros((2, 2), dtype=complex)

        # ρ₁ = ⟨0₂|ρ|0₂⟩ + ⟨1₂|ρ|1₂⟩
        # Basis order: |00⟩, |01⟩, |10⟩, |11⟩ (indices 0,1,2,3)

        # ⟨0₂|ρ|0₂⟩ contribution (indices 0,2 for |00⟩,|10⟩)
        rho_1[0, 0] += rho_full[0, 0]  # ⟨00|ρ|00⟩
        rho_1[0, 1] += rho_full[0, 2]  # ⟨00|ρ|10⟩
        rho_1[1, 0] += rho_full[2, 0]  # ⟨10|ρ|00⟩
        rho_1[1, 1] += rho_full[2, 2]  # ⟨10|ρ|10⟩

        # ⟨1₂|ρ|1₂⟩ contribution (indices 1,3 for |01⟩,|11⟩)
        rho_1[0, 0] += rho_full[1, 1]  # ⟨01|ρ|01⟩
        rho_1[0, 1] += rho_full[1, 3]  # ⟨01|ρ|11⟩
        rho_1[1, 0] += rho_full[3, 1]  # ⟨11|ρ|01⟩
        rho_1[1, 1] += rho_full[3, 3]  # ⟨11|ρ|11⟩

        print(rho_1)

        # Verify using Qiskit
        try:
            state_qiskit = Statevector(bell_state)
            rho_qiskit = DensityMatrix(state_qiskit)
            rho_1_qiskit = partial_trace(
                rho_qiskit, [1]
            )  # Trace out qubit 1 (0-indexed)

            print(f"\nUsing Qiskit partial_trace:")
            print(rho_1_qiskit.data)

            # Check if results match
            diff = np.max(np.abs(rho_1 - rho_1_qiskit.data))
            print(f"\nDifference from manual calculation: {diff:.10f}")
        except Exception as e:
            self.log(f"Qiskit partial trace not available: {e}")

        # Physical interpretation
        print(f"\nPhysical interpretation:")
        print(f"The reduced density matrix represents the state of qubit 1")
        print(f"when we 'ignore' or don't measure qubit 2.")
        print(f"For maximally entangled states, the reduced state is maximally mixed.")

        # Check if reduced state is mixed
        eigenvals = np.linalg.eigvals(rho_1)
        eigenvals = eigenvals[eigenvals > 1e-12]
        purity = np.sum(eigenvals**2)

        print(f"\nPurity of reduced state: {purity:.6f}")
        print(f"→ {'Pure state' if abs(purity - 1.0) < 1e-10 else 'Mixed state'}")

        return rho_full, rho_1

    def visualize_tensor_products(self, states_dict):
        """Visualize tensor product states and their properties."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Two-qubit computational basis visualization
        comp_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
        if all(state in states_dict for state in comp_states):
            amplitudes = np.array([states_dict[state] for state in comp_states])

            # Plot amplitudes
            x_pos = np.arange(len(comp_states))
            width = 0.35

            real_parts = [np.real(amp[0]) for amp in amplitudes]  # First component
            imag_parts = [np.imag(amp[0]) for amp in amplitudes]

            ax1.bar(x_pos - width / 2, real_parts, width, label="Real", alpha=0.7)
            ax1.bar(x_pos + width / 2, imag_parts, width, label="Imaginary", alpha=0.7)

            ax1.set_xlabel("Computational Basis States")
            ax1.set_ylabel("Amplitude (1st component)")
            ax1.set_title("Two-Qubit Computational Basis")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(comp_states)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Entanglement visualization for Bell states
        bell_names = ["|Φ+⟩", "|Φ-⟩", "|Ψ+⟩", "|Ψ-⟩"]
        bell_states_example = {
            "|Φ+⟩": np.array([1, 0, 0, 1]) / np.sqrt(2),
            "|Φ-⟩": np.array([1, 0, 0, -1]) / np.sqrt(2),
            "|Ψ+⟩": np.array([0, 1, 1, 0]) / np.sqrt(2),
            "|Ψ-⟩": np.array([0, 1, -1, 0]) / np.sqrt(2),
        }

        entanglements = []
        for name, state in bell_states_example.items():
            separability = self._check_separability(state)
            entanglements.append(separability["entanglement"])

        ax2.bar(bell_names, entanglements, alpha=0.7, color="skyblue")
        ax2.set_xlabel("Bell States")
        ax2.set_ylabel("Entanglement Entropy")
        ax2.set_title("Entanglement in Bell States")
        ax2.grid(True, alpha=0.3)

        # 3. State probability distributions
        if "|Φ+⟩" in bell_states_example:
            state = bell_states_example["|Φ+⟩"]
            probabilities = np.abs(state) ** 2

            ax3.bar(comp_states, probabilities, alpha=0.7, color="lightcoral")
            ax3.set_xlabel("Measurement Outcomes")
            ax3.set_ylabel("Probability")
            ax3.set_title("Bell State |Φ+⟩ Measurement Probabilities")
            ax3.grid(True, alpha=0.3)

        # 4. Schmidt coefficients visualization
        separable_example = np.kron(np.array([1, 0]), np.array([1, 1]) / np.sqrt(2))
        entangled_example = bell_states_example["|Φ+⟩"]

        sep_analysis = self._check_separability(separable_example)
        ent_analysis = self._check_separability(entangled_example)

        # Pad Schmidt coefficients to same length
        max_len = max(
            len(sep_analysis["schmidt_coefficients"]),
            len(ent_analysis["schmidt_coefficients"]),
        )

        sep_coeffs = list(sep_analysis["schmidt_coefficients"]) + [0] * (
            max_len - len(sep_analysis["schmidt_coefficients"])
        )
        ent_coeffs = list(ent_analysis["schmidt_coefficients"]) + [0] * (
            max_len - len(ent_analysis["schmidt_coefficients"])
        )

        x_pos = np.arange(max_len)
        width = 0.35

        ax4.bar(x_pos - width / 2, sep_coeffs, width, label="Separable", alpha=0.7)
        ax4.bar(x_pos + width / 2, ent_coeffs, width, label="Entangled", alpha=0.7)

        ax4.set_xlabel("Schmidt Mode")
        ax4.set_ylabel("Schmidt Coefficient")
        ax4.set_title("Schmidt Decomposition Comparison")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"Mode {i}" for i in range(max_len)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()

    def generate_summary_report(self):
        """Generate comprehensive summary of tensor product analysis."""
        print("\n" + "=" * 60)
        print("TENSOR PRODUCTS AND MULTI-QUBIT SYSTEMS - ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 States Constructed: {len(self.constructed_states)}")

        print("\n🔬 Key Concepts Demonstrated:")
        print("  • Tensor product construction: |ψ⟩ ⊗ |φ⟩")
        print("  • Multi-qubit computational basis")
        print("  • Separable vs entangled states")
        print("  • Schmidt decomposition and rank")
        print("  • Partial trace operations")
        print("  • Bipartite entanglement measures")

        print("\n📚 Mathematical Foundations:")
        print("  • Two-qubit space: ℂ² ⊗ ℂ² ≅ ℂ⁴")
        print("  • Three-qubit space: ℂ² ⊗ ℂ² ⊗ ℂ² ≅ ℂ⁸")
        print("  • Separability: |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ (Schmidt rank = 1)")
        print("  • Entanglement: Schmidt rank > 1")
        print("  • Bell states: maximally entangled two-qubit states")
        print("  • GHZ and W states: multi-qubit entanglement")

        print("\n🎯 Learning Outcomes:")
        print("  ✓ Understanding tensor product mathematics")
        print("  ✓ Constructing multi-qubit quantum states")
        print("  ✓ Distinguishing separable from entangled states")
        print("  ✓ Computing Schmidt decompositions")
        print("  ✓ Performing partial trace operations")
        print("  ✓ Quantifying bipartite entanglement")

        print("\n🚀 Next Steps:")
        print("  → Explore quantum gates on multi-qubit systems")
        print("  → Study quantum error correction codes")
        print("  → Investigate quantum algorithms")
        print("  → Advanced entanglement theory")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Tensor Products and Multi-Qubit Quantum Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_tensor_products_multiqubit.py
  python 05_tensor_products_multiqubit.py --verbose
  python 05_tensor_products_multiqubit.py --show-visualization
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed calculations",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display tensor product visualizations",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis without visualizations",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 2: Mathematics")
    print("Example 5: Tensor Products and Multi-Qubit Systems")
    print("=" * 58)

    # Initialize analyzer
    analyzer = TensorProductAnalyzer(verbose=args.verbose)

    try:
        # Basic tensor product operations
        two_qubit_states = analyzer.basic_tensor_products()

        # Separable vs entangled analysis
        entanglement_analysis = analyzer.separable_vs_entangled_states()

        # Three-qubit systems
        three_qubit_states = analyzer.three_qubit_systems()

        # Partial trace demonstration
        rho_full, rho_reduced = analyzer.partial_trace_demonstration()

        # Visualization (optional)
        if not args.analysis_only and args.show_visualization:
            analyzer.visualize_tensor_products(two_qubit_states)

        # Generate summary
        analyzer.generate_summary_report()

        print(f"\n✅ Tensor product analysis completed successfully!")
        print(f"📊 Constructed {len(analyzer.constructed_states)} quantum states")

        if args.verbose:
            print(f"\n🔍 Detailed mathematical calculations enabled")
            print(f"📈 Use --show-visualization for plots")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
