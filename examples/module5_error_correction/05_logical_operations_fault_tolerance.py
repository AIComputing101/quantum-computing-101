#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction
Example 5: Logical Operations and Fault Tolerance

Implementation of fault-tolerant logical operations and analysis of fault tolerance.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings

warnings.filterwarnings("ignore")


class FaultTolerantOperations:
    """
    Fault-Tolerant Logical Operations - Computing on Protected Qubits!
    
    MATHEMATICAL CONCEPT (For Beginners):
    ======================================
    THE CHALLENGE: How do we compute on ERROR-CORRECTED qubits?
    
    If we have |ψ⟩_L (logical qubit encoded in 7 physical qubits),
    how do we apply gates like X, Z, H, CNOT without breaking the encoding?
    
    NAIVE APPROACH (WRONG):
    1. Decode: |ψ⟩_L → |ψ⟩ (1 qubit)
    2. Apply gate: X|ψ⟩
    3. Re-encode: X|ψ⟩ → X|ψ⟩_L
    Problem: During step 2, no error protection!
    
    FAULT-TOLERANT APPROACH (RIGHT):
    Apply operations DIRECTLY on encoded state
    WITHOUT decoding!
    
    TRANSVERSAL GATES - The Gold Standard
    ======================================
    DEFINITION: A gate is TRANSVERSAL if it can be implemented by
    applying single-qubit (or two-qubit) gates to corresponding
    qubits in each code block, with no interaction between blocks.
    
    MATHEMATICAL FORMULA:
    Ū = U⊗U⊗...⊗U (tensor product of U on each qubit)
    
    WHY TRANSVERSAL IS GOOD:
    1. Errors cannot spread: One error on one qubit stays on one qubit
    2. Simple to implement: Just apply gate to each physical qubit
    3. Naturally fault-tolerant: Preserves error correction properties
    
    STEANE CODE TRANSVERSAL GATES:
    ===============================
    The [[7,1,3]] Steane code supports:
    
    ✓ Logical X̄ = X⊗X⊗X⊗X⊗X⊗X⊗X (Apply X to all 7 qubits)
    ✓ Logical Z̄ = Z⊗Z⊗Z⊗Z⊗Z⊗Z⊗Z (Apply Z to all 7 qubits)
    ✓ Logical H̄ = H⊗H⊗H⊗H⊗H⊗H⊗H (Apply H to all 7 qubits)
    ✓ Logical CNOT: CNOT⊗...⊗CNOT between corresponding qubits
    
    These are the CLIFFORD gates - very important subset!
    
    NON-TRANSVERSAL GATES:
    ======================
    ❌ T gate (π/8 rotation) is NOT transversal for Steane code
    
    PROBLEM: Transversal gates alone cannot achieve universal
    quantum computation! We need T gate (or similar) for universality.
    
    SOLUTION: Magic State Distillation
    1. Prepare noisy "magic states" |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
    2. Use error correction to "distill" high-fidelity magic states
    3. Inject magic state + measurement → Implements T gate!
    
    EASTIN-KNILL THEOREM:
    =====================
    "No quantum error correcting code can have a universal set
    of transversal gates."
    
    CONSEQUENCE: Must use non-transversal operations (like magic state
    injection) for universal computation. This is more expensive but necessary!
    
    FAULT TOLERANCE GUARANTEE:
    ==========================
    An operation is FAULT-TOLERANT if:
    "Any single fault (error) during the operation causes at most one
    error in each code block."
    
    INTUITION: Errors don't cascade or spread uncontrollably
    
    KEY METRICS:
    ============
    - Transversal depth: How many layers of transversal gates
    - Magic state overhead: How many ancilla qubits for T gates
    - Circuit depth: Total depth including error correction
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def logical_pauli_x_steane(self):
        """
        Logical X operation for Steane code - Transversal bit-flip!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        LOGICAL X̄ OPERATION:
        Flips logical |0̄⟩ ↔ |1̄⟩
        
        ENCODING REMINDER:
        |0̄⟩ = superposition including |0000000⟩
        |1̄⟩ = superposition including |1111111⟩
        
        TRANSVERSAL IMPLEMENTATION:
        Apply X to ALL 7 physical qubits simultaneously
        
        MATHEMATICAL VERIFICATION:
        X̄|0̄⟩ = (X⊗X⊗X⊗X⊗X⊗X⊗X)|0̄⟩ = |1̄⟩ ✓
        
        WHY IT WORKS:
        The Steane code is constructed such that applying X to all
        qubits maps valid codewords to valid codewords!
        
        FAULT TOLERANCE:
        If ONE X gate fails (error), it affects only ONE qubit.
        The code can still correct this single error!
        """
        # ==============================================================
        # TRANSVERSAL X: Apply X gate to each of the 7 qubits
        # ==============================================================
        # MATHEMATICAL OPERATION: X̄ = X^⊗7 = X⊗X⊗X⊗X⊗X⊗X⊗X
        # EFFECT: |0̄⟩ → |1̄⟩ and |1̄⟩ → |0̄⟩ (logical bit-flip)
        logical_x = QuantumCircuit(7, name="Logical_X_Steane")
        for i in range(7):
            logical_x.x(i)  # Apply X to physical qubit i
        return logical_x

    def logical_pauli_z_steane(self):
        """
        Logical Z operation for Steane code - Transversal phase-flip!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        LOGICAL Z̄ OPERATION:
        Applies phase to logical |1̄⟩: α|0̄⟩ + β|1̄⟩ → α|0̄⟩ - β|1̄⟩
        
        TRANSVERSAL IMPLEMENTATION:
        Apply Z to ALL 7 physical qubits simultaneously
        
        MATHEMATICAL FORMULA:
        Z̄ = Z⊗Z⊗Z⊗Z⊗Z⊗Z⊗Z
        
        EFFECT:
        Z̄|0̄⟩ = |0̄⟩ (no change)
        Z̄|1̄⟩ = -|1̄⟩ (global phase flip)
        
        WHY TRANSVERSAL:
        CSS code property: Z errors on physical qubits map to
        Z error on logical qubit!
        """
        # ==============================================================
        # TRANSVERSAL Z: Apply Z gate to each of the 7 qubits
        # ==============================================================
        logical_z = QuantumCircuit(7, name="Logical_Z_Steane")
        for i in range(7):
            logical_z.z(i)  # Apply Z to physical qubit i
        return logical_z

    def logical_hadamard_steane(self):
        """
        Logical Hadamard for Steane code - Basis transformation!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        LOGICAL H̄ OPERATION:
        Swaps X̄ ↔ Z̄ (switches between computational and Hadamard basis)
        
        MATHEMATICAL FORMULA:
        H̄ = H⊗H⊗H⊗H⊗H⊗H⊗H
        
        EFFECT:
        H̄|0̄⟩ = |+̄⟩ = (|0̄⟩ + |1̄⟩)/√2
        H̄|1̄⟩ = |-̄⟩ = (|0̄⟩ - |1̄⟩)/√2
        
        WHY IT WORKS FOR STEANE:
        Steane code is CSS (Calderbank-Shor-Steane) code
        Property: Symmetric under X ↔ Z exchange
        Hadamard implements this symmetry on logical level!
        
        TRANSVERSAL PROPERTY:
        Applying H to each physical qubit implements H̄ on logical qubit
        """
        # ==============================================================
        # TRANSVERSAL H: Apply H gate to each of the 7 qubits
        # ==============================================================
        logical_h = QuantumCircuit(7, name="Logical_H_Steane")
        for i in range(7):
            logical_h.h(i)  # Apply H to physical qubit i
        return logical_h

    def logical_cnot_steane(self, control_block=0, target_block=1):
        """
        Logical CNOT between two Steane code blocks - Two-qubit gate!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        LOGICAL CNOT:
        If control logical qubit is |1̄⟩, flip target logical qubit
        
        CIRCUIT STRUCTURE:
        Need TWO Steane code blocks (14 physical qubits total)
        Block 1 (control): Qubits 0-6
        Block 2 (target): Qubits 7-13
        
        TRANSVERSAL IMPLEMENTATION:
        Apply CNOT between CORRESPONDING qubits in each block:
        CNOT(qubit_i in control, qubit_i in target) for i = 0...6
        
        MATHEMATICAL FORMULA:
        CNOT̄ = CNOT⊗CNOT⊗...⊗CNOT (7 times)
        
        WHY IT'S FAULT-TOLERANT:
        - Errors in control block stay in control block
        - Errors in target block stay in target block
        - Each physical CNOT affects only one qubit in each block
        - Total: At most 1 error per block (correctable!)
        
        EXAMPLE:
        |0̄⟩_control|ψ̄⟩_target → |0̄⟩_control|ψ̄⟩_target (no change)
        |1̄⟩_control|ψ̄⟩_target → |1̄⟩_control X̄|ψ̄⟩_target (target flipped)
        """
        # ==============================================================
        # TRANSVERSAL CNOT: Apply CNOT between corresponding qubits
        # ==============================================================
        # CIRCUIT SIZE: 14 qubits (two 7-qubit code blocks)
        logical_cnot = QuantumCircuit(14, name="Logical_CNOT_Steane")

        # Loop over all 7 qubit positions
        for i in range(7):
            # Calculate absolute qubit indices
            # Control block: qubits 0-6 or 7-13 depending on control_block parameter
            # Target block: qubits 0-6 or 7-13 depending on target_block parameter
            control_qubit = control_block * 7 + i
            target_qubit = target_block * 7 + i
            
            # Apply CNOT between qubit i in control and qubit i in target
            # MATHEMATICAL EFFECT: If control qubit i is |1⟩, flip target qubit i
            logical_cnot.cx(control_qubit, target_qubit)

        return logical_cnot

    def non_transversal_t_gate(self):
        """
        Non-transversal T gate - Required for universal quantum computing!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE T GATE:
        T = [[1, 0], [0, e^(iπ/4)]]
        Applies π/8 phase rotation to |1⟩ state
        
        WHY WE NEED IT:
        Clifford gates (X, Z, H, CNOT) alone CANNOT achieve universal
        quantum computation! Need at least one non-Clifford gate.
        T gate + Clifford gates = Universal! (Can approximate any gate)
        
        THE PROBLEM:
        T gate is NOT transversal for Steane code
        Applying T to each qubit does NOT implement logical T̄
        
        EASTIN-KNILL THEOREM:
        "No quantum code can have a universal set of transversal gates"
        CONSEQUENCE: Must use non-transversal methods for universality
        
        THE SOLUTION: Magic State Distillation
        ========================================
        MAGIC STATE:
        |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
        
        PROCESS:
        1. Prepare many noisy magic states (easy but low fidelity)
        2. Use error correction to "distill" into few high-fidelity states
        3. Inject magic state into computation via measurement
        4. Result: Implements T gate on logical qubit!
        
        MATHEMATICAL PROTOCOL (Simplified):
        Step 1: Prepare ancilla in |T⟩ state
        Step 2: Entangle with logical qubit using CNOTs
        Step 3: Measure ancilla
        Step 4: Apply correction based on measurement outcome
        Result: T gate applied to logical qubit!
        
        OVERHEAD:
        - Resource intensive: Need ~10-100 noisy magic states per good one
        - Dominates fault-tolerant quantum computation cost
        - Active research area: Improving distillation efficiency
        
        TRADE-OFF:
        Transversal gates: Fast, simple, low overhead
        Non-transversal (T) gates: Slow, complex, high overhead
        But NECESSARY for universal computation!
        """
        # ==============================================================
        # MAGIC STATE INJECTION (Simplified implementation)
        # ==============================================================
        # CIRCUIT STRUCTURE:
        # - 7 qubits: Logical qubit (Steane-encoded)
        # - 1 ancilla: Magic state |T⟩
        # - 1 classical bit: Measurement outcome
        #
        # NOTE: This is a simplified version. Real implementation requires:
        # - Magic state distillation protocol
        # - Multiple ancilla qubits
        # - Error correction on ancilla
        # - Classical feedback for correction
        
        t_gate = QuantumCircuit(8, 1, name="T_Gate_Magic_State")

        # --- Step 1: Prepare magic state (simplified) ---
        # MATHEMATICAL STATE: |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
        # APPROXIMATION: Use RY gate to create similar superposition
        # IDEAL: Would use distilled magic state from ancilla factory
        t_gate.ry(np.pi / 4, 7)  # Ancilla qubit (qubit 7)

        # --- Step 2: Entangle with logical qubit ---
        # Apply CNOTs from each logical qubit to magic state ancilla
        # This "injects" the T rotation into the logical qubit
        for i in range(7):
            t_gate.cx(i, 7)  # Control: logical qubit i, Target: ancilla

        # --- Step 3: Measure and apply correction ---
        # Measure ancilla to project logical qubit appropriately
        # Measurement outcome determines if correction needed
        t_gate.measure(7, 0)
        
        # NOTE: In full implementation, would apply conditional
        # corrections based on measurement outcome using c_if()

        return t_gate

    def fault_tolerant_preparation(self, logical_state="0"):
        """Fault-tolerant logical state preparation."""
        prep_circuit = QuantumCircuit(7, name=f"FT_Prep_{logical_state}")

        if logical_state == "0":
            # |0⟩_L preparation (already in computational basis)
            pass
        elif logical_state == "1":
            # |1⟩_L = X_L|0⟩_L
            logical_x = self.logical_pauli_x_steane()
            prep_circuit.compose(logical_x, inplace=True)
        elif logical_state == "+":
            # |+⟩_L = H_L|0⟩_L
            logical_h = self.logical_hadamard_steane()
            prep_circuit.compose(logical_h, inplace=True)
        elif logical_state == "-":
            # |-⟩_L = X_L H_L|0⟩_L
            logical_h = self.logical_hadamard_steane()
            logical_x = self.logical_pauli_x_steane()
            prep_circuit.compose(logical_h, inplace=True)
            prep_circuit.compose(logical_x, inplace=True)

        return prep_circuit

    def fault_tolerant_measurement(self, measurement_basis="Z"):
        """Fault-tolerant logical measurement."""
        if measurement_basis == "Z":
            # Z basis measurement - measure all qubits
            meas_circuit = QuantumCircuit(7, 7, name="FT_Meas_Z")
            meas_circuit.measure_all()

        elif measurement_basis == "X":
            # X basis measurement - Hadamard then measure
            meas_circuit = QuantumCircuit(7, 7, name="FT_Meas_X")
            logical_h = self.logical_hadamard_steane()
            meas_circuit.compose(logical_h, inplace=True)
            meas_circuit.measure_all()

        else:
            # Default Z measurement
            meas_circuit = QuantumCircuit(7, 7, name="FT_Meas")
            meas_circuit.measure_all()

        return meas_circuit

    def analyze_fault_tolerance(self, operation_type="logical_x", error_rate=0.001):
        """Analyze fault tolerance of logical operations."""
        results = {
            "operation": operation_type,
            "error_rate": error_rate,
            "fault_tolerance_metrics": {},
        }

        # Get the logical operation circuit
        if operation_type == "logical_x":
            logical_op = self.logical_pauli_x_steane()
        elif operation_type == "logical_z":
            logical_op = self.logical_pauli_z_steane()
        elif operation_type == "logical_h":
            logical_op = self.logical_hadamard_steane()
        elif operation_type == "logical_cnot":
            logical_op = self.logical_cnot_steane()
        else:
            logical_op = self.logical_pauli_x_steane()  # Default

        # Create full circuit with preparation and measurement
        full_circuit = QuantumCircuit(logical_op.num_qubits, logical_op.num_qubits)

        # Prepare |0⟩_L state (simplified - in practice would use encoding)
        # Apply logical operation
        full_circuit.compose(logical_op, inplace=True)

        # Add measurement
        full_circuit.measure_all()

        # Analyze fault paths
        fault_analysis = self.analyze_fault_paths(full_circuit, error_rate)
        results["fault_tolerance_metrics"] = fault_analysis

        return results

    def analyze_fault_paths(self, circuit, error_rate):
        """Analyze potential fault paths in circuit."""
        metrics = {
            "circuit_depth": circuit.depth(),
            "total_gates": circuit.size(),
            "fault_locations": [],
            "estimated_logical_error_rate": 0,
        }

        # Count different gate types
        gate_counts = circuit.count_ops()
        metrics["gate_counts"] = gate_counts

        # Estimate fault tolerance (simplified analysis)
        # In practice, would require detailed analysis of all fault paths

        # Single gate failure analysis
        single_qubit_gates = (
            gate_counts.get("h", 0) + gate_counts.get("x", 0) + gate_counts.get("z", 0)
        )
        two_qubit_gates = gate_counts.get("cx", 0)

        # Rough estimate: logical error if 2+ physical errors in same block
        # Probability of 2+ errors in n gates with error rate p
        n_total_ops = single_qubit_gates + two_qubit_gates
        logical_error_rate = 0

        # Use binomial approximation for small error rates
        for k in range(2, min(n_total_ops + 1, 10)):  # 2 or more errors
            prob_k_errors = (
                np.math.comb(n_total_ops, k)
                * (error_rate**k)
                * ((1 - error_rate) ** (n_total_ops - k))
            )
            logical_error_rate += prob_k_errors

        metrics["estimated_logical_error_rate"] = logical_error_rate

        return metrics

    def concatenated_code_analysis(self, levels=2, base_error_rate=0.01):
        """Analyze concatenated codes for improved fault tolerance."""
        results = {
            "levels": levels,
            "base_error_rate": base_error_rate,
            "level_analysis": [],
        }

        current_error_rate = base_error_rate

        for level in range(levels + 1):
            level_info = {
                "level": level,
                "physical_qubits": 7**level if level > 0 else 1,
                "logical_error_rate": current_error_rate,
                "improvement_factor": (
                    base_error_rate / current_error_rate
                    if current_error_rate > 0
                    else float("inf")
                ),
            }

            results["level_analysis"].append(level_info)

            # Update error rate for next level (simplified threshold theorem)
            # Assumes error rate below threshold
            if level < levels and current_error_rate < 0.001:  # Simplified threshold
                current_error_rate = current_error_rate**2  # Quadratic improvement
            else:
                current_error_rate = min(
                    1.0, current_error_rate * 10
                )  # Above threshold

        return results

    def benchmark_logical_operations(
        self, operations=["X", "Z", "H", "CNOT"], shots=1000
    ):
        """Benchmark different logical operations."""
        results = {}

        for op_name in operations:
            if op_name == "X":
                op_circuit = self.logical_pauli_x_steane()
            elif op_name == "Z":
                op_circuit = self.logical_pauli_z_steane()
            elif op_name == "H":
                op_circuit = self.logical_hadamard_steane()
            elif op_name == "CNOT":
                op_circuit = self.logical_cnot_steane()
            else:
                continue

            # Create test circuit
            test_circuit = QuantumCircuit(op_circuit.num_qubits, op_circuit.num_qubits)

            # Apply operation
            test_circuit.compose(op_circuit, inplace=True)
            test_circuit.measure_all()

            # Simulate
            simulator = AerSimulator()
            job = simulator.run(test_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Analyze results
            operation_stats = {
                "circuit_depth": test_circuit.depth(),
                "gate_count": test_circuit.size(),
                "measurement_counts": counts,
                "most_probable_outcome": max(counts, key=counts.get),
                "outcome_probability": max(counts.values()) / shots,
            }

            results[op_name] = operation_stats

        return results

    def visualize_fault_tolerance(
        self, ft_analysis, concat_analysis, benchmark_results
    ):
        """Visualize fault tolerance analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Fault tolerance metrics
        metrics = ft_analysis["fault_tolerance_metrics"]

        gate_types = list(metrics["gate_counts"].keys())
        gate_counts = list(metrics["gate_counts"].values())

        ax1.bar(gate_types, gate_counts, alpha=0.7, color="steelblue")
        ax1.set_title(f'Gate Composition: {ft_analysis["operation"]}')
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.3)

        # Error rate comparison
        physical_rate = ft_analysis["error_rate"]
        logical_rate = metrics["estimated_logical_error_rate"]

        ax2.bar(
            ["Physical", "Logical"],
            [physical_rate, logical_rate],
            alpha=0.7,
            color=["red", "blue"],
        )
        ax2.set_title("Error Rate Comparison")
        ax2.set_ylabel("Error Rate")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Add improvement factor
        if logical_rate > 0:
            improvement = physical_rate / logical_rate
            ax2.text(
                0.5,
                (physical_rate + logical_rate) / 2,
                f"{improvement:.1f}x\nimprovement",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

        # Concatenated code analysis
        levels = [info["level"] for info in concat_analysis["level_analysis"]]
        error_rates = [
            info["logical_error_rate"] for info in concat_analysis["level_analysis"]
        ]

        ax3.semilogy(
            levels, error_rates, "o-", linewidth=2, markersize=8, color="green"
        )
        ax3.set_title("Concatenated Code Performance")
        ax3.set_xlabel("Concatenation Level")
        ax3.set_ylabel("Logical Error Rate")
        ax3.grid(True, alpha=0.3)

        # Benchmark results
        operations = list(benchmark_results.keys())
        depths = [benchmark_results[op]["circuit_depth"] for op in operations]
        gate_counts = [benchmark_results[op]["gate_count"] for op in operations]

        x = np.arange(len(operations))
        width = 0.35

        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(
            x - width / 2, depths, width, label="Depth", alpha=0.7, color="blue"
        )
        bars2 = ax4_twin.bar(
            x + width / 2, gate_counts, width, label="Gates", alpha=0.7, color="red"
        )

        ax4.set_xlabel("Logical Operation")
        ax4.set_ylabel("Circuit Depth", color="blue")
        ax4_twin.set_ylabel("Gate Count", color="red")
        ax4.set_title("Logical Operation Complexity")
        ax4.set_xticks(x)
        ax4.set_xticklabels(operations)
        ax4.grid(True, alpha=0.3)

        # Add legends
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fault-Tolerant Logical Operations")
    parser.add_argument(
        "--operation",
        choices=["logical_x", "logical_z", "logical_h", "logical_cnot"],
        default="logical_x",
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.001, help="Physical error rate"
    )
    parser.add_argument("--concatenation-levels", type=int, default=2)
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark all operations"
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 5: Error Correction")
    print("Example 5: Logical Operations and Fault Tolerance")
    print("=" * 51)

    ft_ops = FaultTolerantOperations(verbose=args.verbose)

    try:
        # Analyze specific logical operation
        print(f"\n🔧 Analyzing {args.operation}...")
        ft_analysis = ft_ops.analyze_fault_tolerance(args.operation, args.error_rate)

        metrics = ft_analysis["fault_tolerance_metrics"]
        print(f"   Circuit depth: {metrics['circuit_depth']}")
        print(f"   Total gates: {metrics['total_gates']}")
        print(f"   Gate composition: {metrics['gate_counts']}")
        print(
            f"   Estimated logical error rate: {metrics['estimated_logical_error_rate']:.2e}"
        )

        # Calculate improvement factor
        if metrics["estimated_logical_error_rate"] > 0:
            improvement = args.error_rate / metrics["estimated_logical_error_rate"]
            print(f"   Error suppression factor: {improvement:.1f}x")

        # Concatenated code analysis
        print(f"\n📚 Concatenated Code Analysis (Level {args.concatenation_levels})...")
        concat_analysis = ft_ops.concatenated_code_analysis(
            args.concatenation_levels, args.error_rate
        )

        for level_info in concat_analysis["level_analysis"]:
            print(
                f"   Level {level_info['level']}: "
                f"{level_info['physical_qubits']} qubits, "
                f"error rate {level_info['logical_error_rate']:.2e}"
            )

        # Benchmark operations
        benchmark_results = None
        if args.benchmark:
            print(f"\n⚡ Benchmarking Logical Operations...")
            benchmark_results = ft_ops.benchmark_logical_operations()

            for op_name, stats in benchmark_results.items():
                print(
                    f"   {op_name}: depth={stats['circuit_depth']}, "
                    f"gates={stats['gate_count']}, "
                    f"success_prob={stats['outcome_probability']:.3f}"
                )

        # Fault-tolerant state preparation demo
        print(f"\n🎯 Fault-Tolerant Operations Demo:")
        prep_circuit = ft_ops.fault_tolerant_preparation("+")
        meas_circuit = ft_ops.fault_tolerant_measurement("X")

        print(
            f"   |+⟩_L preparation: {prep_circuit.depth()} depth, {prep_circuit.size()} gates"
        )
        print(
            f"   X-basis measurement: {meas_circuit.depth()} depth, {meas_circuit.size()} gates"
        )

        # Magic state T gate
        t_gate = ft_ops.non_transversal_t_gate()
        print(f"   T gate (magic state): {t_gate.depth()} depth, {t_gate.size()} gates")

        if args.show_visualization and benchmark_results:
            ft_ops.visualize_fault_tolerance(
                ft_analysis, concat_analysis, benchmark_results
            )

        print(f"\n📊 Key Insights:")
        print(f"   • Transversal gates are naturally fault-tolerant")
        print(f"   • Non-transversal gates require special techniques (magic states)")
        print(f"   • Concatenation provides exponential error suppression")
        print(f"   • Trade-off between fault tolerance and resource overhead")

        print(f"\n🎓 Fault Tolerance Principles:")
        print(f"   • Errors must not propagate uncontrollably")
        print(f"   • At most one error per code block from single fault")
        print(f"   • Universal set: {{Clifford + T}} gates sufficient")
        print(f"   • Threshold theorem enables arbitrarily reliable computation")

        print(f"\n✅ Fault-tolerant operation analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
