#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 4: Quantum Algorithms
Example 5: Variational Quantum Eigensolver (VQE)

Implementation of VQE for finding ground state energies.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# Handle different Qiskit versions for primitives
try:
    from qiskit.primitives import Estimator
except ImportError:
    # For older Qiskit versions or when primitives are not available
    try:
        from qiskit_aer.primitives import Estimator

        print("ℹ️  Using Aer primitives for VQE")
    except ImportError:
        # Fallback: create a simple estimator-like class
        print("ℹ️  Using fallback estimator implementation")

        class Estimator:
            def __init__(self):
                self.backend = AerSimulator()

            def run(self, circuits, observables, parameter_values=None):
                # Simple implementation for educational purposes
                from qiskit.quantum_info import Statevector

                results = []
                for circuit, observable in zip(circuits, observables):
                    if parameter_values:
                        # This is a simplified version
                        state = Statevector.from_instruction(circuit)
                        expectation = state.expectation_value(observable).real
                        results.append(expectation)
                return type("Result", (), {"values": results})

        Estimator = Estimator
import warnings

warnings.filterwarnings("ignore")


class VariationalQuantumEigensolver:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.history = []

    def build_ansatz(self, n_qubits, depth, parameters):
        """
        Build parameterized quantum circuit ansatz.
        
        Mathematical Foundation - Variational Ansatz:
        --------------------------------------------
        An ansatz is a parameterized quantum circuit that can represent
        a family of quantum states by varying its parameters.
        
        General Form:
        |ψ(θ)⟩ = U(θ)|0⟩
        
        where:
        - θ = [θ₁, θ₂, ..., θₙ] are classical parameters
        - U(θ) is a unitary circuit built from parameterized gates
        - |0⟩ is the initial state (all qubits in |0⟩)
        
        Ansatz Design Goals:
        -------------------
        1. Expressivity: Can represent target state |ψ_target⟩
        2. Efficiency: Uses minimal gates and parameters
        3. Trainability: Gradients don't vanish (no barren plateaus)
        4. Hardware compatibility: Gates available on real devices
        
        This Ansatz Structure (Hardware-Efficient):
        ------------------------------------------
        Alternating layers of:
        
        Layer k:
        1. Single-qubit rotations: R_y(θ_{k,i}) on each qubit i
           R_y(θ) = [[cos(θ/2), -sin(θ/2)],
                     [sin(θ/2),  cos(θ/2)]]
        
        2. Entangling gates: CNOT between adjacent qubits
           Creates correlations, enables non-separable states
        
        3. More rotations: R_y(θ_{k+1,i}) for additional flexibility
        
        Mathematical Analysis:
        --------------------
        
        Total parameters: 2 × n_qubits × depth
        - depth controls expressivity
        - More depth ≈ more complex states possible
        - But deeper ≈ more noise on hardware!
        
        State after layer k:
        |ψ_k⟩ = CNOT_{adj} · R_y(θ_k) · |ψ_{k-1}⟩
        
        Why R_y Gates?
        -------------
        - R_y creates real superpositions (no global phase)
        - R_y(π/2) transforms |0⟩ ↔ |+⟩
        - R_y(π) is X gate: |0⟩ ↔ |1⟩
        - Universal: R_y + CNOT = universal gate set
        
        Why CNOT Entanglers?
        -------------------
        - Creates entanglement when qubits in superposition
        - Native gate on superconducting hardware
        - Linear connectivity (adjacent qubits only)
        - Efficient: O(n) gates per layer
        
        Alternative Ansätze:
        -------------------
        - UCCSD: Chemistry-inspired (Unitary Coupled Cluster)
        - Hamiltonian variational: Problem-specific structure
        - Low-depth: For NISQ devices with high noise
        - All-to-all: Better expressivity but more gates
        
        Expressivity vs. Trainability Trade-off:
        ---------------------------------------
        - More depth → more expressivity → harder to train
        - Barren plateaus: gradients vanish exponentially with depth
        - Optimal depth depends on problem and hardware noise
        
        Args:
            n_qubits (int): Number of qubits in circuit
            depth (int): Number of ansatz layers
            parameters (list): Variational parameters θ
            
        Returns:
            QuantumCircuit: Parameterized ansatz circuit
        """
        circuit = QuantumCircuit(n_qubits)
        param_idx = 0

        # Repeat layers for specified depth
        # Each layer adds expressivity but also noise
        for layer in range(depth):
            # Layer 1: Single-qubit rotations (parameterized)
            # R_y(θ) gates create superposition
            # Each qubit gets independent rotation angle
            for qubit in range(n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1

            # Layer 2: Entangling gates (CNOT chain)
            # Creates quantum correlations between qubits
            # Linear connectivity: adjacent qubits only
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            # Layer 3: Additional rotations for more flexibility
            # Allows correction/fine-tuning after entanglement
            for qubit in range(n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1

        return circuit

    def create_h2_hamiltonian(self, distance=0.735):
        """Create H2 molecule Hamiltonian."""
        # Simplified H2 Hamiltonian in computational basis
        # Real coefficients for H2 at equilibrium distance
        coeffs = [
            -1.0523732,  # Identity
            0.39793742,  # Z0
            -0.39793742,  # Z1
            -0.01128010,  # Z0*Z1
            0.18093120,  # X0*X1
        ]

        pauli_strings = ["I", "Z", "Z", "ZZ", "XX"]
        pauli_list = []

        for coeff, pauli in zip(coeffs, pauli_strings):
            if pauli == "I":
                pauli_list.append((pauli + "I", coeff))
            elif pauli == "Z":
                pauli_list.append(("Z" + "I", coeff))
                pauli_list.append(("I" + "Z", coeff))
            elif pauli == "ZZ":
                pauli_list.append(("ZZ", coeff))
            elif pauli == "XX":
                pauli_list.append(("XX", coeff))

        # Create properly formatted Pauli list
        formatted_paulis = []
        coefficients = []

        formatted_paulis.append("II")
        coefficients.append(coeffs[0])

        formatted_paulis.append("ZI")
        coefficients.append(coeffs[1])

        formatted_paulis.append("IZ")
        coefficients.append(coeffs[2])

        formatted_paulis.append("ZZ")
        coefficients.append(coeffs[3])

        formatted_paulis.append("XX")
        coefficients.append(coeffs[4])

        hamiltonian = SparsePauliOp(formatted_paulis, coefficients)
        return hamiltonian

    def create_simple_hamiltonian(self, n_qubits=2):
        """Create a simple Hamiltonian for testing."""
        # Simple Ising model: H = -Z0 - Z1 - 0.5*Z0*Z1
        pauli_strings = ["I" * n_qubits]
        coeffs = [0.0]  # Identity term

        # Single qubit terms
        for i in range(n_qubits):
            pauli = ["I"] * n_qubits
            pauli[i] = "Z"
            pauli_strings.append("".join(pauli))
            coeffs.append(-1.0)

        # Two-qubit interaction
        if n_qubits >= 2:
            pauli = ["I"] * n_qubits
            pauli[0] = "Z"
            pauli[1] = "Z"
            pauli_strings.append("".join(pauli))
            coeffs.append(-0.5)

        return SparsePauliOp(pauli_strings, coeffs)

    def compute_expectation(self, circuit, hamiltonian):
        """Compute expectation value of Hamiltonian."""
        # Get the statevector
        statevector = Statevector.from_instruction(circuit)

        # Compute expectation value
        expectation = statevector.expectation_value(hamiltonian)

        return np.real(expectation)

    def cost_function(self, parameters, ansatz_func, hamiltonian):
        """Cost function for VQE optimization."""
        circuit = ansatz_func(parameters)
        energy = self.compute_expectation(circuit, hamiltonian)

        self.history.append(energy)

        if self.verbose:
            print(f"Iteration {len(self.history)}: Energy = {energy:.6f}")

        return energy

    def classical_optimization(
        self, initial_params, ansatz_func, hamiltonian, max_iter=100
    ):
        """Simple gradient-free optimization."""
        best_params = initial_params.copy()
        best_energy = self.cost_function(best_params, ansatz_func, hamiltonian)

        step_size = 0.1

        for iteration in range(max_iter):
            # Try random perturbations
            for _ in range(10):
                # Random perturbation
                perturbation = np.random.normal(0, step_size, len(best_params))
                test_params = best_params + perturbation

                test_energy = self.cost_function(test_params, ansatz_func, hamiltonian)

                if test_energy < best_energy:
                    best_energy = test_energy
                    best_params = test_params
                    break

            # Reduce step size
            step_size *= 0.99

            if iteration % 20 == 0 and self.verbose:
                print(f"Optimization step {iteration}: Best energy = {best_energy:.6f}")

        return best_params, best_energy

    def run_vqe(self, hamiltonian, n_qubits, depth=2, max_iter=100):
        """Run complete VQE algorithm."""
        # Initialize parameters
        n_params = 2 * n_qubits * depth
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        # Define ansatz function
        def ansatz_func(params):
            return self.build_ansatz(n_qubits, depth, params)

        # Clear history
        self.history = []

        # Run optimization
        print(f"Starting VQE optimization with {n_params} parameters...")
        optimal_params, optimal_energy = self.classical_optimization(
            initial_params, ansatz_func, hamiltonian, max_iter
        )

        # Build optimal circuit
        optimal_circuit = ansatz_func(optimal_params)

        return {
            "optimal_params": optimal_params,
            "optimal_energy": optimal_energy,
            "optimal_circuit": optimal_circuit,
            "energy_history": self.history.copy(),
            "n_iterations": len(self.history),
        }

    def analyze_ground_state(self, hamiltonian):
        """Compute exact ground state for comparison."""
        # Convert to matrix and find eigenvalues
        matrix = hamiltonian.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        ground_energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]

        return ground_energy, ground_state

    def visualize_results(self, vqe_result, exact_energy=None):
        """Visualize VQE results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Energy convergence
        history = vqe_result["energy_history"]
        ax1.plot(history, "b-", alpha=0.7, linewidth=2)
        ax1.set_title("VQE Energy Convergence")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Energy")
        ax1.grid(True, alpha=0.3)

        if exact_energy is not None:
            ax1.axhline(
                y=exact_energy,
                color="r",
                linestyle="--",
                label=f"Exact: {exact_energy:.6f}",
            )
            ax1.legend()

        # Parameter distribution
        params = vqe_result["optimal_params"]
        ax2.hist(params, bins=15, alpha=0.7, color="green", edgecolor="black")
        ax2.set_title("Optimal Parameter Distribution")
        ax2.set_xlabel("Parameter Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Circuit properties
        circuit = vqe_result["optimal_circuit"]
        metrics = ["Depth", "Gates", "Parameters"]
        values = [circuit.depth(), circuit.size(), len(params)]

        ax3.bar(metrics, values, alpha=0.7, color="orange")
        ax3.set_title("Circuit Properties")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3)

        # Energy comparison
        energies = ["VQE Result"]
        values = [vqe_result["optimal_energy"]]
        colors = ["blue"]

        if exact_energy is not None:
            energies.append("Exact")
            values.append(exact_energy)
            colors.append("red")

            # Add error
            error = abs(vqe_result["optimal_energy"] - exact_energy)
            ax4.text(
                0.5,
                0.7,
                f"Error: {error:.6f}",
                transform=ax4.transAxes,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        ax4.bar(energies, values, alpha=0.7, color=colors)
        ax4.set_title("Energy Comparison")
        ax4.set_ylabel("Energy")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Variational Quantum Eigensolver")
    parser.add_argument("--qubits", type=int, default=2, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=2, help="Ansatz depth")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Optimization iterations"
    )
    parser.add_argument(
        "--molecule",
        choices=["h2", "simple"],
        default="simple",
        help="Hamiltonian type",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 4: Quantum Algorithms")
    print("Example 5: Variational Quantum Eigensolver (VQE)")
    print("=" * 48)

    vqe = VariationalQuantumEigensolver(verbose=args.verbose)

    try:
        # Create Hamiltonian
        if args.molecule == "h2":
            hamiltonian = vqe.create_h2_hamiltonian()
            print("Using H2 molecule Hamiltonian")
        else:
            hamiltonian = vqe.create_simple_hamiltonian(args.qubits)
            print(f"Using simple {args.qubits}-qubit Hamiltonian")

        print(f"Hamiltonian terms: {len(hamiltonian.paulis)}")

        # Compute exact ground state
        print("\nComputing exact ground state...")
        exact_energy, exact_state = vqe.analyze_ground_state(hamiltonian)
        print(f"Exact ground state energy: {exact_energy:.6f}")

        # Run VQE
        print(f"\nRunning VQE with {args.depth}-layer ansatz...")
        result = vqe.run_vqe(hamiltonian, args.qubits, args.depth, args.iterations)

        print(f"\n🎯 VQE Results:")
        print(f"   Optimal energy: {result['optimal_energy']:.6f}")
        print(f"   Exact energy:   {exact_energy:.6f}")
        print(f"   Error:          {abs(result['optimal_energy'] - exact_energy):.6f}")
        print(f"   Iterations:     {result['n_iterations']}")

        # Check convergence
        if len(result["energy_history"]) > 10:
            final_energies = result["energy_history"][-10:]
            convergence = np.std(final_energies)
            print(f"   Convergence:    {convergence:.6f} (std of last 10)")

        if args.show_visualization:
            vqe.visualize_results(result, exact_energy)

        print(f"\n✅ VQE algorithm completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
