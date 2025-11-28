#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction
Example 4: Quantum Error Correction Protocols

Implementation of quantum error correction protocols and fault-tolerant operations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector, random_statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings

warnings.filterwarnings("ignore")


class QuantumErrorCorrectionProtocols:
    """
    Quantum Error Correction Protocols - Building Blocks for Fault Tolerance
    
    MATHEMATICAL CONCEPT (For Beginners):
    ======================================
    WHAT IS QUANTUM ERROR CORRECTION (QEC)?
    Protecting quantum information by encoding it across multiple qubits
    so that errors can be detected and corrected WITHOUT measuring the data!
    
    THREE FUNDAMENTAL CODES (From Simple to Complex):
    ==================================================
    
    1. BIT-FLIP CODE (3 qubits) - Protects against X errors
       Encoding: |0⟩ → |000⟩, |1⟩ → |111⟩
       Distance: 3 (can correct 1 X error)
       
    2. PHASE-FLIP CODE (3 qubits) - Protects against Z errors
       Encoding: |0⟩ → |+++⟩, |1⟩ → |---⟩ (in X-basis)
       Distance: 3 (can correct 1 Z error)
       
    3. SHOR CODE (9 qubits) - Protects against ANY single-qubit error!
       Combines bit-flip + phase-flip codes
       Distance: 3 (can correct 1 arbitrary error: X, Y, or Z)
    
    KEY PRINCIPLE: REDUNDANCY
    ==========================
    Classical: Copy bits |0⟩ → |000⟩
    Problem: No-cloning theorem forbids quantum copying!
    
    Quantum Solution: Use ENTANGLEMENT instead of copying
    |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
    
    WHY IT WORKS:
    - Errors affect individual qubits
    - Redundancy allows majority vote (for bit flips)
    - Or parity checks (for phase flips)
    - Detect error WITHOUT measuring the encoded state!
    
    SYNDROME MEASUREMENT:
    ====================
    KEY IDEA: Extract error information without destroying data
    
    Method: Use ancilla (helper) qubits
    1. Entangle ancilla with code qubits
    2. Measure ancilla → Get "syndrome" (error signature)
    3. Syndrome tells us which qubit has error
    4. Apply correction based on syndrome
    5. Data qubits remain unmeasured!
    
    MATHEMATICAL FORMULA:
    Syndrome s = H·e (mod 2)
    where H = parity check matrix, e = error vector
    
    CODE DISTANCE:
    =============
    Definition: Minimum weight of non-trivial logical operator
    
    INTERPRETATION:
    Distance d means:
    - Can DETECT up to d-1 errors
    - Can CORRECT up to ⌊(d-1)/2⌋ errors
    
    Examples:
    - Distance 3: Correct 1 error
    - Distance 5: Correct 2 errors
    - Distance 7: Correct 3 errors
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def three_qubit_bit_flip_code(self):
        """
        Implementation of 3-qubit bit-flip code - The Simplest QEC Code!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE PROBLEM: Bit-flip errors (X gates)
        |0⟩ --X--> |1⟩ (unwanted!)
        |1⟩ --X--> |0⟩ (unwanted!)
        
        THE SOLUTION: Encode 1 qubit into 3 qubits
        
        ENCODING:
        |0⟩_L = |000⟩  (logical zero = all qubits in |0⟩)
        |1⟩_L = |111⟩  (logical one = all qubits in |1⟩)
        
        General state: α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
        
        WHY IT WORKS:
        If ONE qubit flips due to error:
        - |000⟩ → |100⟩, |010⟩, or |001⟩ (detectable!)
        - |111⟩ → |011⟩, |101⟩, or |110⟩ (detectable!)
        
        CORRECTION: Majority vote
        - 2 or more |0⟩'s → Correct to |0⟩
        - 2 or more |1⟩'s → Correct to |1⟩
        
        ENCODING CIRCUIT:
        =================
        Input: |ψ⟩|0⟩|0⟩ where |ψ⟩ = α|0⟩ + β|1⟩
        
        Step 1: CNOT(0→1): Copy qubit 0 to qubit 1
                Result: α|000⟩ + β|110⟩
        
        Step 2: CNOT(0→2): Copy qubit 0 to qubit 2
                Result: α|000⟩ + β|111⟩ ✓
        
        MATHEMATICAL PROPERTY:
        The encoded state is:
        |ψ⟩_L = α|0⟩_L + β|1⟩_L = α|000⟩ + β|111⟩
        
        This is an ENTANGLED state! The qubits are correlated.
        
        SYNDROME MEASUREMENT:
        =====================
        Use 2 ancilla qubits to check parity:
        
        Ancilla 1: Checks if qubits 0 and 1 match
                   Result: 0 (match) or 1 (mismatch)
        
        Ancilla 2: Checks if qubits 1 and 2 match
                   Result: 0 (match) or 1 (mismatch)
        
        SYNDROME TABLE:
        (Ancilla1, Ancilla2) → Error Location
        (0, 0) → No error
        (1, 0) → Error on qubit 0
        (1, 1) → Error on qubit 1
        (0, 1) → Error on qubit 2
        
        LIMITATION: Only corrects X (bit-flip) errors, not Z (phase-flip)!
        """
        # ==============================================================
        # ENCODING CIRCUIT: |ψ⟩|0⟩|0⟩ → α|000⟩ + β|111⟩
        # ==============================================================
        # MATHEMATICAL OPERATION: Copy qubit 0 to qubits 1 and 2
        # RESULT: Creates entanglement (cannot be factored)
        encoding = QuantumCircuit(3, name="3-qubit_encode")
        encoding.cx(0, 1)  # If qubit 0 is |1⟩, flip qubit 1
        encoding.cx(0, 2)  # If qubit 0 is |1⟩, flip qubit 2
        # EFFECT: |0⟩|0⟩|0⟩ stays |000⟩, |1⟩|0⟩|0⟩ becomes |111⟩

        # ==============================================================
        # DECODING CIRCUIT: Inverse of encoding (same gates!)
        # ==============================================================
        # MATHEMATICAL PROPERTY: CNOT gates are self-inverse
        # CNOT·CNOT = Identity
        # Applying encoding twice gives back the original state
        decoding = QuantumCircuit(3, name="3-qubit_decode")
        decoding.cx(0, 1)  # Undo the first CNOT
        decoding.cx(0, 2)  # Undo the second CNOT

        # ==============================================================
        # SYNDROME MEASUREMENT: Detect errors without destroying data
        # ==============================================================
        # CIRCUIT STRUCTURE:
        # Qubits 0,1,2: Data qubits (encoded state)
        # Qubits 3,4: Ancilla qubits (for parity checks)
        # Classical bits 0,1: Store syndrome measurement results
        syndrome = QuantumCircuit(5, 2, name="syndrome_measure")
        
        # --- Parity Check 1: Do qubits 0 and 1 have same value? ---
        # MATHEMATICAL OPERATION: XOR of qubits 0 and 1 into ancilla 3
        # If qubit 0 = qubit 1 → ancilla 3 stays |0⟩ (parity even)
        # If qubit 0 ≠ qubit 1 → ancilla 3 becomes |1⟩ (parity odd)
        syndrome.cx(0, 3)  # Copy qubit 0 parity to ancilla 3
        syndrome.cx(1, 3)  # XOR with qubit 1 parity
        
        # --- Parity Check 2: Do qubits 1 and 2 have same value? ---
        syndrome.cx(1, 4)  # Copy qubit 1 parity to ancilla 4
        syndrome.cx(2, 4)  # XOR with qubit 2 parity
        
        # --- Measure ancillas (NOT the data qubits!) ---
        # CRITICAL: This preserves the quantum superposition in qubits 0,1,2
        syndrome.measure([3, 4], [0, 1])

        return {
            "encoding": encoding,
            "decoding": decoding,
            "syndrome": syndrome,
            "code_distance": 3,           # Minimum weight of error detectable
            "correctable_errors": 1,      # Can correct ⌊(3-1)/2⌋ = 1 error
        }

    def three_qubit_phase_flip_code(self):
        """
        Implementation of 3-qubit phase-flip code - Protects against Z errors!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE PROBLEM: Phase-flip errors (Z gates)
        |+⟩ --Z--> |-⟩  (superposition signs flip)
        α|0⟩ + β|1⟩ --Z--> α|0⟩ - β|1⟩ (phase changed!)
        
        CHALLENGE: Can't just copy qubits (no-cloning theorem)
        
        THE CLEVER SOLUTION: Work in X-BASIS instead of Z-basis!
        
        BASIS TRANSFORMATION:
        =====================
        Z-basis: |0⟩, |1⟩ (computational basis)
        X-basis: |+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2
        
        KEY INSIGHT: Hadamard gate swaps the bases
        H: |0⟩ ↔ |+⟩, |1⟩ ↔ |-⟩
        
        TRANSFORMATION OF ERRORS:
        Z error in Z-basis = X error in X-basis!
        
        MATHEMATICAL PROOF:
        In Z-basis: Z|ψ⟩ causes phase flip
        Transform: H·Z·H = X (Hadamard conjugates Z to X)
        In X-basis: Same error acts like bit-flip!
        
        ENCODING STRATEGY:
        ==================
        |0⟩ → H → |+⟩ → bit-flip encode → |+++⟩ → H → |0̄⟩
        |1⟩ → H → |-⟩ → bit-flip encode → |---⟩ → H → |1̄⟩
        
        WHERE:
        |+++⟩ = |+⟩⊗|+⟩⊗|+⟩ (all three qubits in |+⟩)
        |---⟩ = |-⟩⊗|-⟩⊗|-⟩ (all three qubits in |-⟩)
        
        EXPLICIT ENCODING:
        |0̄⟩ = (1/2√2)[|000⟩+|011⟩+|101⟩+|110⟩+|001⟩+|010⟩+|100⟩+|111⟩]
        (Superposition with EVEN number of |1⟩'s have + sign)
        
        |1̄⟩ = (1/2√2)[|000⟩-|011⟩-|101⟩-|110⟩+|001⟩+|010⟩+|100⟩-|111⟩]
        (Different phase pattern)
        
        ERROR CORRECTION:
        If one Z error occurs:
        |+++⟩ → |++-⟩, |+-+⟩, or |-++⟩ (detectable!)
        Use X-basis parity checks to identify which qubit
        Apply Z correction to fix it
        
        SYNDROME MEASUREMENT (X-basis):
        X₀X₁ = +1 or -1 (parity of qubits 0,1 in X-basis)
        X₁X₂ = +1 or -1 (parity of qubits 1,2 in X-basis)
        
        ANALOGY:
        Bit-flip code protects against X by checking Z parities
        Phase-flip code protects against Z by checking X parities
        They're DUAL to each other!
        """
        # ==============================================================
        # ENCODING CIRCUIT: Transform to X-basis, encode, transform back
        # ==============================================================
        encoding = QuantumCircuit(3, name="phase_flip_encode")
        
        # STEP 1: Transform to X-basis (where Z errors become X errors)
        # MATH: H|0⟩ = |+⟩, H|1⟩ = |-⟩
        encoding.h([0, 1, 2])
        
        # STEP 2: Apply bit-flip encoding in X-basis
        # EFFECT: |+⟩|0⟩|0⟩ → |+++⟩ (or |-⟩|0⟩|0⟩ → |---⟩)
        encoding.cx(0, 1)  # Copy qubit 0 to qubit 1
        encoding.cx(0, 2)  # Copy qubit 0 to qubit 2
        
        # STEP 3: Transform back to Z-basis
        # MATH: H|+⟩ = |0⟩, H|-⟩ = |1⟩
        encoding.h([0, 1, 2])
        
        # RESULT: Encoded state in Z-basis, protected against Z errors!

        # ==============================================================
        # DECODING CIRCUIT: Reverse of encoding
        # ==============================================================
        # MATHEMATICAL PROPERTY: Encoding is self-inverse (unitary)
        # Applying same gates in reverse order decodes
        decoding = QuantumCircuit(3, name="phase_flip_decode")
        decoding.h([0, 1, 2])       # To X-basis
        decoding.cx(0, 1)            # Undo entanglement
        decoding.cx(0, 2)
        decoding.h([0, 1, 2])       # Back to Z-basis

        return {
            "encoding": encoding,
            "decoding": decoding,
            "code_distance": 3,           # Can detect 2 errors, correct 1
            "correctable_errors": 1,      # ⌊(3-1)/2⌋ = 1 error
        }

    def shor_nine_qubit_code(self):
        """
        Implementation of Shor's 9-qubit code - First Universal QEC Code!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE ACHIEVEMENT: First code to protect against ANY single-qubit error!
        - X errors (bit flips): ✓ Corrected
        - Z errors (phase flips): ✓ Corrected
        - Y errors (both): ✓ Corrected (since Y = iXZ)
        
        THE BRILLIANT IDEA: Concatenation
        ==================================
        STEP 1: Protect against phase flips (Z errors)
               |0⟩ → |+++⟩ (3 qubits in superposition)
               |1⟩ → |---⟩ (3 qubits in superposition)
        
        STEP 2: Protect each of those 3 qubits against bit flips (X errors)
               Each |+⟩ → |000⟩+|111⟩ (3 more qubits)
               Total: 3 × 3 = 9 qubits!
        
        MATHEMATICAL STRUCTURE:
        =======================
        Logical |0⟩ encoding:
        |0̄⟩ = (1/2√2)[|000⟩+|111⟩] ⊗ [|000⟩+|111⟩] ⊗ [|000⟩+|111⟩]
        
        Breaking this down:
        - 3 blocks of 3 qubits each
        - Each block: Bit-flip protected |+⟩ state
        - Blocks together: Phase-flip protected
        
        Logical |1⟩ encoding:
        |1̄⟩ = (1/2√2)[|000⟩-|111⟩] ⊗ [|000⟩-|111⟩] ⊗ [|000⟩-|111⟩]
        (Note the minus signs!)
        
        QUBIT ORGANIZATION:
        ===================
        Block 1: Qubits 0,1,2  } 
        Block 2: Qubits 3,4,5  } Phase-flip protected
        Block 3: Qubits 6,7,8  }
        
        Within each block: Bit-flip protected
        
        ERROR CORRECTION CAPABILITY:
        ============================
        Can correct ANY single error on ANY single qubit:
        - X on any qubit: Detected by bit-flip syndrome
        - Z on any qubit: Detected by phase-flip syndrome
        - Y on any qubit: Detected by both syndromes (Y = iXZ)
        
        ENCODING PROCESS:
        =================
        LAYER 1 (Bit-flip protection):
        For each of 3 blocks, encode |ψ⟩ → |ψψψ⟩
        Input: |ψ⟩|0⟩⁸
        After Layer 1: Each block has 3 identical qubits
        
        LAYER 2 (Phase-flip protection):
        Create superposition across the 3 blocks
        Apply in Hadamard (X) basis
        
        ANALOGY: 
        Layer 1 = Protecting each word by writing it 3 times
        Layer 2 = Protecting the sentence structure
        Together = Full protection!
        
        CODE PARAMETERS:
        ================
        [[9,1,3]] code:
        - 9 physical qubits
        - 1 logical qubit
        - Distance 3 (corrects 1 arbitrary error)
        
        HISTORICAL SIGNIFICANCE:
        Peter Shor, 1995 - Proved quantum error correction is possible!
        Before this: Many thought quantum computing was impossible due to errors
        """
        # ==============================================================
        # ENCODING: Two-layer concatenated code
        # ==============================================================
        encoding = QuantumCircuit(9, name="shor_9_encode")

        # ==============================================================
        # LAYER 1: Bit-flip protection (within each block)
        # ==============================================================
        # GOAL: Encode each qubit using 3-qubit bit-flip code
        # MATHEMATICAL OPERATION: |ψ⟩ → |ψψψ⟩ for each block
        #
        # We have 3 blocks:
        # Block 1: Qubits {0,1,2}
        # Block 2: Qubits {3,4,5}
        # Block 3: Qubits {6,7,8}
        #
        # For each block, apply: CNOT(base → base+1), CNOT(base → base+2)
        # This creates: |0⟩ → |000⟩, |1⟩ → |111⟩ (bit-flip protected)
        
        for i in range(3):  # Loop over 3 blocks
            base = i * 3    # Starting qubit of this block
            encoding.cx(base, base + 1)  # Copy to second qubit in block
            encoding.cx(base, base + 2)  # Copy to third qubit in block
        # RESULT: Each block now has redundancy against bit flips

        # ==============================================================
        # LAYER 2: Phase-flip protection (across blocks)
        # ==============================================================
        # GOAL: Protect against phase (Z) errors by working in X-basis
        # MATHEMATICAL OPERATION: Create superposition across 3 blocks
        #
        # STRATEGY:
        # 1. Transform to X-basis using Hadamard gates
        # 2. Apply bit-flip code logic (which protects against X errors in Z-basis
        #    = Z errors in X-basis!)
        # 3. Transform back to Z-basis
        #
        # For each qubit position within blocks (0, 1, 2):
        #   - Apply Hadamard to qubits at that position in all 3 blocks
        #   - Create entanglement across blocks
        #   - Apply Hadamard back
        
        # --- Process qubit position 0 in each block ---
        # QUBITS: 0 (block 1), 3 (block 2), 6 (block 3)
        encoding.h([0, 3, 6])         # Transform to X-basis
        encoding.cx(0, 3)              # Entangle block 1 with block 2
        encoding.cx(0, 6)              # Entangle block 1 with block 3
        encoding.h([0, 3, 6])         # Back to Z-basis
        # RESULT: Qubits 0,3,6 now in phase-flip protected state

        # --- Repeat for qubit positions 1 and 2 ---
        # This ensures ALL qubits in each block are phase-protected
        for offset in [1, 2]:
            qubits = [offset, offset + 3, offset + 6]
            encoding.h(qubits)        # To X-basis
            encoding.cx(offset, offset + 3)  # Entangle
            encoding.cx(offset, offset + 6)
            encoding.h(qubits)        # Back to Z-basis
        
        # ==============================================================
        # FINAL STATE: Doubly-protected quantum state
        # ==============================================================
        # The 9 qubits now encode: α|0̄⟩ + β|1̄⟩
        # where |0̄⟩ and |1̄⟩ are the logical codewords
        # Protected against ANY single X, Y, or Z error!

        return {
            "encoding": encoding,
            "logical_qubits": 1,
            "physical_qubits": 9,
            "code_distance": 3,
            "correctable_errors": 1,
        }

    def fault_tolerant_cnot(self, code_type="steane"):
        """Implement fault-tolerant CNOT gate."""
        if code_type == "steane":
            # Steane code transversal CNOT
            ft_cnot = QuantumCircuit(14, name="FT_CNOT_Steane")
            for i in range(7):
                ft_cnot.cx(i, i + 7)  # Transversal CNOT

        elif code_type == "shor":
            # Shor code CNOT (requires additional operations)
            ft_cnot = QuantumCircuit(18, name="FT_CNOT_Shor")
            for i in range(9):
                ft_cnot.cx(i, i + 9)

        else:
            # Simple 3-qubit code
            ft_cnot = QuantumCircuit(6, name="FT_CNOT_3qubit")
            for i in range(3):
                ft_cnot.cx(i, i + 3)

        return ft_cnot

    def error_correction_cycle(self, code_qubits, syndrome_qubits, error_type="x"):
        """Complete error correction cycle."""
        total_qubits = len(code_qubits) + len(syndrome_qubits)

        # Initialize circuit
        circuit = QuantumCircuit(total_qubits, len(syndrome_qubits))

        # Syndrome extraction
        if error_type == "x":
            # X-type stabilizers
            for i, anc in enumerate(syndrome_qubits):
                for j, code in enumerate(code_qubits):
                    if self.get_stabilizer_matrix(len(code_qubits))[i][j]:
                        circuit.cx(code, anc)

        elif error_type == "z":
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
            return np.array([[1, 1, 0], [0, 1, 1]])
        elif n_qubits == 7:
            # Steane code stabilizers (simplified)
            return np.array(
                [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]
            )
        else:
            # Default identity-like matrix
            return np.eye(min(n_qubits - 1, 3), n_qubits)

    def simulate_error_correction(
        self, code_type="3-qubit", error_rate=0.01, n_cycles=5
    ):
        """Simulate error correction over multiple cycles."""
        results = {
            "code_type": code_type,
            "error_rate": error_rate,
            "cycles": n_cycles,
            "cycle_results": [],
        }

        # Get code parameters
        if code_type == "3-qubit":
            code_info = self.three_qubit_bit_flip_code()
            n_code_qubits = 3
            n_syndrome_qubits = 2
        elif code_type == "steane":
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
            cycle_result["cycle"] = cycle
            results["cycle_results"].append(cycle_result)

        # Calculate statistics
        results["success_rate"] = np.mean(
            [r["correction_success"] for r in results["cycle_results"]]
        )
        results["average_syndrome_weight"] = np.mean(
            [r["syndrome_weight"] for r in results["cycle_results"]]
        )

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
            code_qubits, syndrome_qubits, "x"
        )
        circuit.compose(error_correction_circuit, inplace=True)

        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()

        # Analyze syndrome
        syndrome = list(counts.keys())[0]
        syndrome_weight = syndrome.count("1")

        # Determine if correction was successful (simplified)
        correction_success = n_errors <= 1  # Can correct single errors

        return {
            "n_errors": n_errors,
            "syndrome": syndrome,
            "syndrome_weight": syndrome_weight,
            "correction_success": correction_success,
        }

    def threshold_analysis(self, code_type="3-qubit", error_rates=None):
        """Analyze error threshold for given code."""
        if error_rates is None:
            error_rates = np.logspace(-4, -1, 10)  # 10^-4 to 10^-1

        results = {
            "error_rates": error_rates,
            "logical_error_rates": [],
            "code_type": code_type,
        }

        for p in error_rates:
            # Simulate many cycles at this error rate
            simulation = self.simulate_error_correction(
                code_type, error_rate=p, n_cycles=100
            )

            # Calculate logical error rate
            logical_error_rate = 1 - simulation["success_rate"]
            results["logical_error_rates"].append(logical_error_rate)

        # Find threshold (where logical error rate equals physical error rate)
        physical_rates = error_rates
        logical_rates = results["logical_error_rates"]

        # Find crossing point
        threshold_idx = None
        for i in range(len(physical_rates) - 1):
            if (
                logical_rates[i] <= physical_rates[i]
                and logical_rates[i + 1] > physical_rates[i + 1]
            ):
                threshold_idx = i
                break

        if threshold_idx is not None:
            results["threshold"] = physical_rates[threshold_idx]
        else:
            results["threshold"] = None

        return results

    def visualize_protocols(self, simulation_results, threshold_results=None):
        """Visualize error correction protocol results."""
        fig = plt.figure(figsize=(16, 12))

        # Cycle-by-cycle results
        ax1 = plt.subplot(2, 3, 1)
        cycles = [r["cycle"] for r in simulation_results["cycle_results"]]
        errors = [r["n_errors"] for r in simulation_results["cycle_results"]]
        successes = [
            1 if r["correction_success"] else 0
            for r in simulation_results["cycle_results"]
        ]

        ax1.scatter(cycles, errors, c=successes, cmap="RdYlGn", alpha=0.7, s=50)
        ax1.set_title("Error Correction Performance")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Number of Errors")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2 = plt.subplot(2, 3, 2)
        success_rate = simulation_results["success_rate"]
        ax2.pie(
            [success_rate, 1 - success_rate],
            labels=["Success", "Failure"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
            startangle=90,
        )
        ax2.set_title(f'Overall Success Rate\n({simulation_results["code_type"]} code)')

        # Syndrome weight distribution
        ax3 = plt.subplot(2, 3, 3)
        syndrome_weights = [
            r["syndrome_weight"] for r in simulation_results["cycle_results"]
        ]
        ax3.hist(
            syndrome_weights,
            bins=range(max(syndrome_weights) + 2),
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax3.set_title("Syndrome Weight Distribution")
        ax3.set_xlabel("Syndrome Weight")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        # Error rate vs cycle
        ax4 = plt.subplot(2, 3, 4)
        error_counts = [r["n_errors"] for r in simulation_results["cycle_results"]]
        ax4.plot(cycles, error_counts, "b-o", alpha=0.7, markersize=4)
        ax4.set_title("Errors per Cycle")
        ax4.set_xlabel("Cycle")
        ax4.set_ylabel("Number of Errors")
        ax4.grid(True, alpha=0.3)

        # Threshold plot
        if threshold_results:
            ax5 = plt.subplot(2, 3, 5)
            physical_rates = threshold_results["error_rates"]
            logical_rates = threshold_results["logical_error_rates"]

            ax5.loglog(physical_rates, logical_rates, "b-o", label="Logical error rate")
            ax5.loglog(
                physical_rates, physical_rates, "r--", label="Physical error rate"
            )

            if threshold_results["threshold"]:
                ax5.axvline(
                    x=threshold_results["threshold"],
                    color="green",
                    linestyle=":",
                    label=f'Threshold ≈ {threshold_results["threshold"]:.2e}',
                )

            ax5.set_title("Error Threshold Analysis")
            ax5.set_xlabel("Physical Error Rate")
            ax5.set_ylabel("Logical Error Rate")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Code comparison
        ax6 = plt.subplot(2, 3, 6)
        codes = [
            "3-qubit\nBit Flip",
            "3-qubit\nPhase Flip",
            "Shor\n9-qubit",
            "Steane\n7-qubit",
        ]
        distances = [3, 3, 3, 3]
        rates = [1 / 3, 1 / 3, 1 / 9, 1 / 7]

        x = np.arange(len(codes))
        width = 0.35

        ax6_twin = ax6.twinx()
        bars1 = ax6.bar(
            x - width / 2, distances, width, label="Distance", alpha=0.7, color="blue"
        )
        bars2 = ax6_twin.bar(
            x + width / 2, rates, width, label="Rate", alpha=0.7, color="red"
        )

        ax6.set_xlabel("Code Type")
        ax6.set_ylabel("Distance", color="blue")
        ax6_twin.set_ylabel("Encoding Rate", color="red")
        ax6.set_title("Code Comparison")
        ax6.set_xticks(x)
        ax6.set_xticklabels(codes, rotation=45, ha="right")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Error Correction Protocols")
    parser.add_argument(
        "--code", choices=["3-qubit", "steane", "shor"], default="3-qubit"
    )
    parser.add_argument(
        "--cycles", type=int, default=20, help="Number of correction cycles"
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.01, help="Physical error rate"
    )
    parser.add_argument("--threshold-analysis", action="store_true")
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 5: Error Correction")
    print("Example 4: Quantum Error Correction Protocols")
    print("=" * 47)

    protocols = QuantumErrorCorrectionProtocols(verbose=args.verbose)

    try:
        # Get code information
        if args.code == "3-qubit":
            code_info = protocols.three_qubit_bit_flip_code()
            print(f"\n📋 3-Qubit Bit Flip Code:")
            print(f"   Distance: {code_info['code_distance']}")
            print(f"   Correctable errors: {code_info['correctable_errors']}")

        elif args.code == "steane":
            print(f"\n📋 Steane 7-Qubit Code:")
            print(f"   Parameters: [7, 1, 3]")
            print(f"   Encodes 1 logical qubit in 7 physical qubits")
            print(f"   Can correct any single-qubit error")

        elif args.code == "shor":
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
        print(
            f"   Average syndrome weight: {simulation['average_syndrome_weight']:.2f}"
        )
        print(f"   Error rate: {args.error_rate}")

        # Analyze specific cycles
        failed_cycles = [
            r for r in simulation["cycle_results"] if not r["correction_success"]
        ]
        if failed_cycles:
            print(f"   Failed corrections: {len(failed_cycles)}")
            avg_errors_failed = np.mean([r["n_errors"] for r in failed_cycles])
            print(f"   Average errors in failed cycles: {avg_errors_failed:.1f}")

        # Threshold analysis
        threshold_results = None
        if args.threshold_analysis:
            print(f"\n🎯 Analyzing error threshold...")
            threshold_results = protocols.threshold_analysis(args.code)

            if threshold_results["threshold"]:
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
        print(
            f"   • Syndrome measurement reveals error information without destroying data"
        )

        print(f"\n✅ Error correction protocol analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
