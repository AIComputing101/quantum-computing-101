# Module 4: Core Quantum Algorithms
*Intermediate Tier*

## Learning Objectives
By the end of this module, you will be able to:
- Explain the intuition and problem each core quantum algorithm targets
- Implement Deutsch–Jozsa, Grover, and Quantum Fourier Transform (QFT) in code
- Describe how QFT enables period finding and Shor’s algorithm
- Understand Shor’s algorithm structure (high level) and resource constraints
- Implement a small composite-number factoring demo using classical post-processing
- Build and run a Variational Quantum Eigensolver (VQE) workflow (ansatz, measurement, optimizer loop)
- Compare algorithm categories: oracle-based, Fourier-based, and variational/hybrid
- Evaluate## 4.5 Shor's Algorithm (High-Level)

### Goal
Factor composite N = p·q efficiently (exponential speedup over best known classical general methods).

### Hybrid Structure
1. Pick random a co-prime to N
2. Use quantum order-finding to get period r of a^x mod N
3. If r even & a^{r/2} ≠ −1 mod N, compute gcd(a^{r/2} ± 1, N) → non-trivial factors
4. Else retry with new a

### Quantum Phase Estimation and Period Finding

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT

def quantum_period_finding_demo(a=7, N=15):
    """
    Demonstrate the quantum part of Shor's algorithm
    Find the period r where a^r ≡ 1 (mod N)
    """
    
    print(f"Quantum Period Finding: a = {a}, N = {N}")
    print("=" * 40)
    
    # For this demo, we know that 7^4 ≡ 1 (mod 15), so period r = 4
    # We'll use 4 qubits for the counting register
    n_counting = 4
    
    # Counting register (for superposition over periods)
    counting_reg = QuantumRegister(n_counting, 'counting')
    # Ancilla register (would hold modular exponentiation results)
    ancilla_reg = QuantumRegister(4, 'ancilla')
    
    # Classical register for measurement
    c_reg = ClassicalRegister(n_counting)
    
    qc = QuantumCircuit(counting_reg, ancilla_reg, c_reg)
    
    # Step 1: Create superposition in counting register
    print("Step 1: Creating superposition in counting register")
    for i in range(n_counting):
        qc.h(counting_reg[i])
    
    # Step 2: Controlled modular exponentiation
    # For demo purposes, we'll simulate the effect of U^j|1⟩ = |a^j mod N⟩
    print("Step 2: Simulating controlled modular exponentiation")
    
    # In a real implementation, this would be controlled modular arithmetic
    # Here we'll create a simplified demonstration
    simulate_controlled_modular_exp(qc, counting_reg, ancilla_reg, a, N)
    
    # Step 3: Apply inverse QFT to extract period information
    print("Step 3: Applying inverse QFT")
    qc.append(QFT(n_counting, inverse=True), counting_reg)
    
    # Step 4: Measure counting register
    qc.measure(counting_reg, c_reg)
    
    print("\nQuantum Circuit for Period Finding:")
    print(qc.draw())
    
    return qc

def simulate_controlled_modular_exp(qc, counting_reg, ancilla_reg, a, N):
    """
    Simulate the controlled modular exponentiation U^j|y⟩ = |ay mod N⟩
    This is simplified for demonstration - real implementation requires 
    sophisticated quantum arithmetic circuits
    """
    
    # Initialize ancilla to |1⟩ (starting value for exponentiation)
    qc.x(ancilla_reg[0])
    
    # For each power of 2 in the counting register, apply controlled operations
    for j in range(len(counting_reg)):
        # This represents controlled application of U^(2^j)
        # where U|y⟩ = |ay mod N⟩
        
        # Simplified: just add controlled rotations to simulate phase kickback
        power = 2**j
        phase = 2 * np.pi * power / 4  # Assuming period r=4
        qc.cp(phase, counting_reg[j], ancilla_reg[0])

def demonstrate_phase_estimation():
    """Demonstrate quantum phase estimation principle"""
    
    print("\nQuantum Phase Estimation Principle")
    print("=" * 34)
    
    # Simple 2-qubit example
    # Let's estimate the phase of the T gate (phase π/4)
    n = 3  # Number of counting qubits
    
    counting_reg = QuantumRegister(n, 'counting')
    target_reg = QuantumRegister(1, 'target')
    c_reg = ClassicalRegister(n)
    
    qc = QuantumCircuit(counting_reg, target_reg, c_reg)
    
    # Prepare eigenstate of T gate: |+⟩ = (|0⟩ + |1⟩)/√2
    qc.h(target_reg[0])
    
    # Create superposition in counting register
    for i in range(n):
        qc.h(counting_reg[i])
    
    # Controlled applications of T gate
    for i in range(n):
        # T^(2^i) controlled by counting_reg[i]
        for _ in range(2**i):
            qc.ct(counting_reg[i], target_reg[0])  # Controlled-T
    
    # Inverse QFT on counting register
    qc.append(QFT(n, inverse=True), counting_reg)
    
    # Measure
    qc.measure(counting_reg, c_reg)
    
    print("Phase Estimation Circuit:")
    print(qc.draw())
    
    # Run simulation
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print("\nMeasurement Results:")
    for outcome, count in sorted(counts.items()):
        probability = count / 1024
        # Convert binary to phase estimate
        phase_estimate = int(outcome, 2) / (2**n)
        print(f"  {outcome}: {count:4d} ({probability:.3f}) → phase ≈ {phase_estimate:.3f}π")
    
    # The T gate has phase π/4 = 0.125π, so we expect peak around 001 = 1/8
    print(f"\nExpected phase: π/4 = 0.125π")
    print(f"With {n} counting qubits, resolution = 1/{2**n} = {1/(2**n):.3f}")

def factoring_demo():
    """High-level demonstration of classical post-processing"""
    
    print("\nFactoring with Shor's Algorithm: Classical Post-processing")
    print("=" * 56)
    
    N = 15
    a = 7
    
    print(f"Goal: Factor N = {N}")
    print(f"Chosen coprime: a = {a}")
    
    # Step 1: Find period r using quantum period finding
    print(f"\nStep 1: Find period r where {a}^r ≡ 1 (mod {N})")
    r = 4  # We know this from quantum period finding
    print(f"Quantum computer found: r = {r}")
    
    # Verify the period
    print(f"Verification: {a}^{r} mod {N} = {pow(a, r, N)}")
    
    # Step 2: Check if r is even and a^(r/2) ≢ -1 (mod N)
    if r % 2 != 0:
        print("Period is odd - try different 'a'")
        return
    
    x = pow(a, r//2, N)
    print(f"\nStep 2: Check a^(r/2) mod N = {a}^{r//2} mod {N} = {x}")
    
    if x == N - 1:
        print("Got -1 mod N - try different 'a'")
        return
    
    # Step 3: Compute factors using gcd
    factor1 = math.gcd(x - 1, N)
    factor2 = math.gcd(x + 1, N)
    
    print(f"\nStep 3: Compute factors using GCD")
    print(f"gcd({x} - 1, {N}) = gcd({x-1}, {N}) = {factor1}")
    print(f"gcd({x} + 1, {N}) = gcd({x+1}, {N}) = {factor2}")
    
    if factor1 > 1 and factor1 < N:
        print(f"\nSuccess! {N} = {factor1} × {N//factor1}")
    elif factor2 > 1 and factor2 < N:
        print(f"\nSuccess! {N} = {factor2} × {N//factor2}")
    else:
        print("Factorization failed - try different 'a'")

def complexity_comparison():
    """Compare classical vs quantum factoring complexity"""
    
    print("\nComplexity Analysis: Classical vs Quantum Factoring")
    print("=" * 50)
    
    bit_lengths = [64, 128, 256, 512, 1024, 2048]
    
    print(f"{'Bits':<6} {'Classical (General NFE)':<25} {'Quantum (Shor)':<20}")
    print("-" * 50)
    
    for n in bit_lengths:
        # Classical: sub-exponential but super-polynomial
        # General Number Field Sieve: O(exp((64/9)^(1/3) * (ln N)^(1/3) * (ln ln N)^(2/3)))
        classical_ops = math.exp((64/9)**(1/3) * (n * math.log(2))**(1/3) * (math.log(n * math.log(2)))**(2/3))
        
        # Quantum: polynomial in number of bits
        # O(n^3) for modular exponentiation, O(n^2) repetitions
        quantum_ops = n**3
        
        print(f"{n:<6} {classical_ops:.2e}{'':14} {quantum_ops:.2e}")
    
    print("\nKey Insight: Exponential quantum speedup for large numbers!")
    print("This threatens RSA security for sufficiently large quantum computers.")

# Run demonstrations
quantum_period_finding_demo()
demonstrate_phase_estimation()
factoring_demo()
complexity_comparison()
```hoose a variational (NISQ-friendly) algorithm vs a fault‑tolerant one

## Prerequisites
- Completion of Module 3 (circuit construction, simulators, basic Qiskit)
- Comfort with linear algebra & complex numbers (Module 2)
- Understanding of superposition, interference, entanglement (Module 1)
- Basic Python programming & familiarity with NumPy/Qiskit

---

## 4.1 Algorithm Landscape: Why These Matter

Modern quantum algorithms fall into a few archetypes:
- Oracle / Amplitude Amplification (Deutsch–Jozsa, Grover)
- Fourier / Phase Estimation Based (QFT, Shor)
- Variational / Hybrid (VQE, QAOA)

Think of them as three *design patterns*:
- Pattern 1 (Oracle): Query a black-box function in superposition to reveal global properties faster than classical enumeration.
- Pattern 2 (Fourier/Phase): Convert periodic structure hidden in amplitudes into measurable peaks.
- Pattern 3 (Variational): Use a parameterized quantum circuit + classical optimizer loop.

> Developer analogy: Oracle algorithms = parallel map + clever reduce. Fourier algorithms = frequency-domain analysis for hidden periods. Variational = gradient-free (or assisted) search over circuit parameter space.

| Algorithm | Core Trick | Speedup Type | NISQ Friendly? | Typical Use |
|-----------|------------|--------------|----------------|-------------|
| Deutsch–Jozsa | Global property in 1 query | Exponential vs worst classical | Yes | Didactic intro |
| Grover | Amplitude amplification | Quadratic | Partially | Unstructured search |
| QFT | Basis change to phase domain | Enables exponential (via phase) | Yes (small) | Period extraction |
| Shor | Period finding + classical math | Exponential | No (needs scale) | Factoring RSA moduli |
| VQE | Hybrid optimization | Practical (chemistry) | Yes | Ground states / energies |

---

## 4.2 Deutsch–Jozsa Algorithm

### Problem Statement
Given a boolean function f: {0,1}^n → {0,1} promised to be either:
- Constant (same output for all inputs), or
- Balanced (returns 0 for half the inputs, 1 for half)
Determine which category f belongs to.

Classically: Need up to 2^{n-1} + 1 evaluations in worst case.
Quantum: Exactly 1 oracle evaluation.

### Intuition
Put input register into equal superposition → oracle writes phase pattern → apply Hadamards → measure. Interference collapses all amplitude into |0...0⟩ if constant; otherwise amplitude shifts away, yielding some non-zero pattern.

### Circuit Shape
1. Initialize n input qubits in |0⟩ and 1 output qubit in |1⟩ (phase kickback helper)
2. Apply H to all qubits
3. Apply oracle Uf
4. Apply H to input register
5. Measure input qubits

### Key Mechanism: Phase Kickback
The output qubit prepared in (|0⟩ - |1⟩)/√2 causes Uf to imprint (-1)^{f(x)} phase on the input branch rather than changing output amplitude distribution.

### Minimal Qiskit Implementation (Example n=3)
```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Example balanced oracle: f(x) = x0 XOR x1 (ignoring x2 for simplicity)
def deutsch_jozsa_oracle_balanced(qc, n):
    qc.cx(0, n)  # control x0 to output
    qc.cx(1, n)

# Example constant oracle: f(x)=0 (do nothing)

def deutsch_jozsa(n=3, balanced=True):
    qc = QuantumCircuit(n+1, n)
    # Output qubit init to |1>
    qc.x(n)
    qc.h(range(n+1))

    if balanced:
        deutsch_jozsa_oracle_balanced(qc, n)
    else:
        pass  # constant zero oracle

    qc.h(range(n))
    qc.measure(range(n), range(n))
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    return qc, counts

qc_bal, counts_bal = deutsch_jozsa(balanced=True)
qc_const, counts_const = deutsch_jozsa(balanced=False)

print("Balanced oracle results:", counts_bal)
print("Constant oracle results:", counts_const)
```

### Enhanced Deutsch-Jozsa with Step-by-Step Analysis

```python
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def detailed_deutsch_jozsa_demo():
    """Comprehensive Deutsch-Jozsa demonstration with analysis"""
    
    print("Deutsch-Jozsa Algorithm: Step-by-Step Analysis")
    print("=" * 48)
    
    # Test multiple oracle types
    oracles = {
        "Constant-0": lambda qc, n: None,  # Do nothing
        "Constant-1": lambda qc, n: qc.x(n),  # Flip output
        "Balanced XOR": lambda qc, n: [qc.cx(0, n), qc.cx(1, n)],  # x0 XOR x1
        "Balanced Single": lambda qc, n: qc.cx(0, n)  # Just x0
    }
    
    for oracle_name, oracle_func in oracles.items():
        print(f"\nTesting {oracle_name} Oracle:")
        
        # Build circuit
        qc = QuantumCircuit(4, 3)  # 3 input qubits + 1 output
        
        # Step 1: Initialize output qubit to |1⟩
        qc.x(3)
        qc.barrier()
        
        # Step 2: Create superposition
        qc.h(range(4))
        qc.barrier()
        
        # Step 3: Apply oracle
        if oracle_func:
            if callable(oracle_func):
                oracle_func(qc, 3)
            else:
                for gate in oracle_func:
                    pass  # Already applied
        qc.barrier()
        
        # Step 4: Final Hadamards on input qubits
        qc.h(range(3))
        qc.barrier()
        
        # Step 5: Measure input qubits
        qc.measure(range(3), range(3))
        
        # Execute
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        counts = job.result().get_counts()
        
        # Analyze results
        zero_prob = counts.get('000', 0) / 1000
        
        print(f"  Results: {counts}")
        print(f"  P(000) = {zero_prob:.3f}")
        
        if zero_prob > 0.9:
            print(f"  → CONSTANT function (high probability on 000)")
        else:
            print(f"  → BALANCED function (distributed outcomes)")
        
        print(f"  Circuit depth: {qc.depth()}")

# Run the detailed demo
detailed_deutsch_jozsa_demo()
```

### Visual Circuit Analysis

```python
def visualize_deutsch_jozsa_effect():
    """Visualize how Deutsch-Jozsa algorithm works"""
    
    # Create circuits for visualization
    n_qubits = 2
    
    # Constant oracle circuit
    qc_const = QuantumCircuit(n_qubits + 1, n_qubits)
    qc_const.x(n_qubits)  # Output to |1⟩
    qc_const.h(range(n_qubits + 1))
    # No oracle operation (constant 0)
    qc_const.h(range(n_qubits))
    qc_const.measure(range(n_qubits), range(n_qubits))
    
    # Balanced oracle circuit  
    qc_balanced = QuantumCircuit(n_qubits + 1, n_qubits)
    qc_balanced.x(n_qubits)  # Output to |1⟩
    qc_balanced.h(range(n_qubits + 1))
    qc_balanced.cx(0, n_qubits)  # Balanced oracle: f(x) = x0
    qc_balanced.h(range(n_qubits))
    qc_balanced.measure(range(n_qubits), range(n_qubits))
    
    print("Constant Oracle Circuit:")
    print(qc_const.draw())
    print("\nBalanced Oracle Circuit:")
    print(qc_balanced.draw())
    
    # Execute both
    backend = Aer.get_backend('qasm_simulator')
    
    job_const = execute(qc_const, backend, shots=1000)
    job_balanced = execute(qc_balanced, backend, shots=1000)
    
    counts_const = job_const.result().get_counts()
    counts_balanced = job_balanced.result().get_counts()
    
    print(f"\nConstant Oracle Results: {counts_const}")
    print(f"Balanced Oracle Results: {counts_balanced}")

visualize_deutsch_jozsa_effect()
```

### Result Interpretation
- Constant: Only 000 observed (with high probability = 1 ideally)
- Balanced: Any non-000 pattern(s)

### Why It Matters
Proof-of-principle that querying in superposition plus interference can collapse global properties faster than classical sampling.

---

## 4.3 Grover’s Algorithm

### Problem
Search an unstructured database of N = 2^n entries to find any x such that f(x)=1.
Classical average queries: O(N). Quantum: O(√N) oracle calls.

### Core Steps
1. Create uniform superposition
2. Oracle marks solution(s) via phase flip
3. Diffusion (inversion about mean) amplifies marked amplitudes
4. Repeat ~π/4 * √N times
5. Measure

### Intuition
Each iteration “rotates” the state vector in a two-dimensional subspace spanned by |solution⟩ and |rest⟩ increasing success probability.

### Circuit Skeleton
- Hadamards on all qubits
- Repeat: [Oracle → Diffusion]

### Simple 2-Solution Example (Educational)
```python
from qiskit import QuantumCircuit, Aer, execute
import math

def grover_oracle(qc, marked, n):
    for pattern in marked:
        bits = format(pattern, f'0{n}b')
        # Flip qubits where bit=0 to target |111..> pattern
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
        # Multi-controlled Z (simplified for small n)
        if n == 2:
            qc.cz(0,1)
        elif n == 3:
            qc.h(n-1); qc.ccx(0,1,n-1); qc.h(n-1)
        # Uncompute flips
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)


def diffusion(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    if n == 2:
        qc.h(1); qc.cz(0,1); qc.h(1)
    elif n == 3:
        qc.h(n-1); qc.ccx(0,1,n-1); qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))


def grover(n=3, marked=[5]):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    iterations = int(math.pi/4 * math.sqrt(2**n / len(marked)))
    for _ in range(iterations):
        grover_oracle(qc, marked, n)
        diffusion(qc, n)
    qc.measure(range(n), range(n))
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots=1024).result().get_counts()
    return qc, counts

qc_grover, counts_grover = grover(n=3, marked=[5])

print("Grover results:", counts_grover)
```

### Grover Algorithm Visualization and Analysis

```python
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

def analyze_grover_iterations():
    """Analyze how Grover's algorithm amplifies target amplitude"""
    
    print("Grover's Algorithm: Amplitude Evolution Analysis")
    print("=" * 49)
    
    n = 3
    marked_item = 5  # Binary: 101
    
    # Calculate optimal iterations
    N = 2**n
    optimal_iterations = int(math.pi/4 * math.sqrt(N))
    print(f"Search space size: {N}")
    print(f"Marked item: {marked_item} (binary: {format(marked_item, '0'+str(n)+'b')})")
    print(f"Optimal iterations: {optimal_iterations}")
    
    # Track amplitudes through iterations
    amplitudes_over_time = []
    success_probs = []
    
    for num_iter in range(optimal_iterations + 2):
        # Build circuit with specified iterations
        qc = QuantumCircuit(n)
        qc.h(range(n))  # Initialize superposition
        
        # Apply Grover iterations
        for _ in range(num_iter):
            # Oracle
            grover_oracle(qc, [marked_item], n)
            # Diffusion
            diffusion(qc, n)
        
        # Get statevector (before measurement)
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        statevector = job.result().get_statevector()
        
        # Extract amplitude of marked state
        marked_amplitude = abs(statevector[marked_item])**2
        amplitudes_over_time.append(statevector.copy())
        success_probs.append(marked_amplitude)
        
        print(f"Iteration {num_iter}: P(marked) = {marked_amplitude:.3f}")
    
    # Plot success probability evolution
    iterations = list(range(len(success_probs)))
    
    print(f"\nSuccess Probability Evolution:")
    for i, prob in enumerate(success_probs):
        bar = "█" * int(prob * 20)
        print(f"Iter {i}: {bar:<20} {prob:.3f}")
    
    return success_probs

def grover_with_multiple_targets():
    """Demonstrate Grover with multiple marked items"""
    
    print("\nGrover's Algorithm with Multiple Targets")
    print("=" * 39)
    
    n = 3
    marked_items = [3, 5, 7]  # Multiple targets
    
    # Adjust iteration count for multiple targets
    N = 2**n
    num_marked = len(marked_items)
    optimal_iterations = int(math.pi/4 * math.sqrt(N / num_marked))
    
    print(f"Marked items: {marked_items}")
    print(f"Fraction marked: {num_marked}/{N} = {num_marked/N:.2f}")
    print(f"Adjusted optimal iterations: {optimal_iterations}")
    
    # Build circuit
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    
    for _ in range(optimal_iterations):
        grover_oracle(qc, marked_items, n)
        diffusion(qc, n)
    
    qc.measure_all()
    
    # Execute
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    counts = job.result().get_counts()
    
    print(f"\nResults after {optimal_iterations} iterations:")
    for outcome, count in sorted(counts.items()):
        state_int = int(outcome, 2)
        prob = count / 1000
        marker = "★" if state_int in marked_items else " "
        print(f"{outcome} (int: {state_int:2d}): {count:3d} times ({prob:.1%}) {marker}")
    
    # Calculate success rate
    total_marked_count = sum(counts.get(format(item, '03b'), 0) for item in marked_items)
    success_rate = total_marked_count / 1000
    print(f"\nTotal success rate: {success_rate:.1%}")

# Run Grover analysis
probs = analyze_grover_iterations()
grover_with_multiple_targets()
```

### Understanding Grover's Geometric Picture

```python
def grover_geometric_visualization():
    """Visualize Grover's algorithm as rotation in 2D space"""
    
    print("Grover's Algorithm: Geometric Visualization")
    print("=" * 42)
    
    n = 2  # Small example for clarity
    N = 2**n
    marked_item = 3  # |11⟩
    
    print(f"2-qubit example: marked item is |{format(marked_item, '02b')}⟩")
    
    # Initial state: uniform superposition
    initial_amplitude_marked = 1/math.sqrt(N)
    initial_amplitude_others = math.sqrt((N-1)/N)
    
    print(f"Initial amplitude on marked state: {initial_amplitude_marked:.3f}")
    print(f"Initial amplitude on other states: {initial_amplitude_others:.3f}")
    
    # Calculate rotation angle
    sin_theta = math.sqrt(1/N)  # sine of half-angle between |s⟩ and |w⟩
    theta = 2 * math.asin(sin_theta)
    
    print(f"Rotation angle per iteration: {theta:.3f} radians ({math.degrees(theta):.1f}°)")
    
    # Optimal iterations for maximum overlap with marked state
    optimal_iter = math.pi / (2 * theta) - 0.5
    print(f"Optimal iterations (continuous): {optimal_iter:.2f}")
    print(f"Optimal iterations (integer): {round(optimal_iter)}")
    
    # Show amplitude evolution geometrically
    print(f"\nAmplitude evolution (geometric view):")
    for i in range(int(optimal_iter) + 2):
        angle = (2*i + 1) * theta / 2
        prob_marked = (math.sin(angle))**2
        print(f"Iteration {i}: angle = {angle:.3f}, P(marked) = {prob_marked:.3f}")

grover_geometric_visualization()
```

### Tuning Notes
- Too few iterations: amplitude not sufficiently amplified
- Too many: overshoot and success probability declines (rotation analogy)

### Practical Constraints
Real hardware noise degrades iterative amplification; error mitigation or fewer iterations may be required.

---

## 4.4 Quantum Fourier Transform (QFT)

### Role
The QFT maps computational basis amplitudes into a phase-encoded frequency basis. Critical for period finding (Shor), phase estimation, and some quantum signal processing.

### Difference vs Classical FFT
- QFT acts on quantum amplitudes in superposition, enabling parallel extraction of periodic structure.
- Implementation cost O(n^2) (ideal) vs classical FFT O(n 2^n) when you think of full vector; but comparison is nuanced — QFT’s advantage emerges inside specific algorithmic contexts.

### Mathematical Form
For |x⟩ (n qubits, value x): QFT|x⟩ = (1/√2^n) Σ_{k=0}^{2^n-1} e^{2πi xk / 2^n} |k⟩

### Circuit Pattern
- Sequence of Hadamards + controlled phase rotations
- Bit-reversal (swap layer) at end

### Reference Implementation
```python
from math import pi
from qiskit import QuantumCircuit

def qft(circ, n):
    for j in range(n):
        circ.h(j)
        for k in range(2, n-j+1):
            circ.cp(pi/2**(k-1), j+k-1, j)
    # Bit reversal
    for i in range(n//2):
        circ.swap(i, n-1-i)
    return circ

qc_qft = QuantumCircuit(3)
qft(qc_qft, 3)

print("QFT Circuit:")
print(qc_qft.draw())
```

### QFT Visualization and Testing

```python
from qiskit.quantum_info import Statevector
import numpy as np

def demonstrate_qft_action():
    """Demonstrate what QFT does to different input states"""
    
    print("Quantum Fourier Transform: Action on Different States")
    print("=" * 54)
    
    n = 3
    
    # Test different computational basis states
    test_states = [0, 1, 2, 3, 4, 5, 6, 7]
    
    for state in test_states:
        print(f"\nInput state: |{state}⟩ = |{format(state, '03b')}⟩")
        
        # Create QFT circuit
        qc = QuantumCircuit(n)
        
        # Prepare input state
        if state > 0:
            for i in range(n):
                if (state >> i) & 1:
                    qc.x(i)
        
        # Apply QFT
        qft(qc, n)
        
        # Get output statevector
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        output_state = job.result().get_statevector()
        
        # Show amplitudes
        print("Output amplitudes:")
        for k in range(2**n):
            amplitude = output_state[k]
            magnitude = abs(amplitude)
            phase = np.angle(amplitude)
            if magnitude > 1e-10:  # Only show significant amplitudes
                print(f"  |{k}⟩: {amplitude:.3f} (mag: {magnitude:.3f}, phase: {phase:.2f})")

def qft_period_finding_demo():
    """Demonstrate how QFT helps in period finding"""
    
    print("\nQFT for Period Finding: Toy Example")
    print("=" * 35)
    
    # Simulate a periodic function with period r=3 in a 3-qubit system
    n = 3
    period = 3
    
    print(f"Simulating periodic function with period {period}")
    
    # Create a state with period-3 structure
    # |ψ⟩ = (|0⟩ + |3⟩ + |6⟩)/√3 (every 3rd state)
    qc = QuantumCircuit(n)
    
    # Manually prepare periodic superposition
    # For demo purposes, we'll create |0⟩ + e^(2πi/3)|3⟩ + e^(4πi/3)|6⟩
    qc.initialize([1/math.sqrt(3), 0, 0, 1/math.sqrt(3), 0, 0, 1/math.sqrt(3), 0], range(n))
    
    print("Before QFT - periodic structure in computational basis:")
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    input_state = job.result().get_statevector()
    
    for k in range(2**n):
        amplitude = input_state[k]
        if abs(amplitude) > 1e-10:
            print(f"  |{k}⟩: {amplitude:.3f}")
    
    # Apply QFT
    qft(qc, n)
    
    job = execute(qc, backend)
    output_state = job.result().get_statevector()
    
    print("\nAfter QFT - period information extracted:")
    for k in range(2**n):
        amplitude = output_state[k]
        probability = abs(amplitude)**2
        if probability > 1e-10:
            print(f"  |{k}⟩: prob = {probability:.3f}")
            # The peaks should appear at multiples of 2^n/period

def compare_qft_classical_fft():
    """Compare QFT with classical FFT"""
    
    print("\nQuantum vs Classical Fourier Transform")
    print("=" * 37)
    
    # Create a simple periodic signal
    n = 3
    N = 2**n
    
    # Classical signal with period 2
    signal = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    
    print("Classical signal:", signal)
    
    # Classical FFT
    classical_fft = np.fft.fft(signal) / math.sqrt(N)
    
    print("Classical FFT result:")
    for k in range(N):
        print(f"  k={k}: {classical_fft[k]:.3f}")
    
    # Quantum equivalent: prepare signal as quantum amplitudes
    qc = QuantumCircuit(n)
    qc.initialize(signal / np.linalg.norm(signal), range(n))
    
    # Apply QFT
    qft(qc, n)
    
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    quantum_result = job.result().get_statevector()
    
    print("\nQuantum QFT result:")
    for k in range(N):
        print(f"  k={k}: {quantum_result[k]:.3f}")

# Run QFT demonstrations
demonstrate_qft_action()
qft_period_finding_demo()
compare_qft_classical_fft()
```

### Approximate QFT Implementation

```python
def approximate_qft(circ, n, approximation_degree=None):
    """QFT with approximation (omit small rotation angles)"""
    
    if approximation_degree is None:
        approximation_degree = n  # No approximation
    
    for j in range(n):
        circ.h(j)
        for k in range(2, min(approximation_degree + 1, n - j + 1)):
            circ.cp(pi/2**(k-1), j+k-1, j)
    
    # Bit reversal
    for i in range(n//2):
        circ.swap(i, n-1-i)
    
    return circ

def compare_exact_vs_approximate_qft():
    """Compare exact and approximate QFT implementations"""
    
    print("Exact vs Approximate QFT Comparison")
    print("=" * 35)
    
    n = 4
    
    # Test on a specific input state |5⟩
    input_state = 5
    
    for approx_level in [n, 3, 2]:  # n=exact, lower=more approximation
        qc = QuantumCircuit(n)
        
        # Prepare input state |5⟩
        for i in range(n):
            if (input_state >> i) & 1:
                qc.x(i)
        
        # Apply approximate QFT
        approximate_qft(qc, n, approx_level)
        
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result().get_statevector()
        
        print(f"\nApproximation degree {approx_level}:")
        print(f"Circuit depth: {qc.depth()}")
        print(f"Two-qubit gates: {qc.count_ops().get('cp', 0)}")
        
        # Show significant amplitudes
        for k in range(2**n):
            amplitude = result[k]
            prob = abs(amplitude)**2
            if prob > 0.01:  # Only significant probabilities
                print(f"  |{k:2d}⟩: prob = {prob:.3f}")

compare_exact_vs_approximate_qft()
```

### Inversion (IQFT)
Same circuit reversed with conjugated phase angles (−θ). Required in phase estimation & Shor to unpack period information.

### Intuition Aids
- Think of controlled rotations as accumulating fractional phase contributions of lower significance bits.
- The swap layer reorders from little-endian phase accumulation order to standard binary order.

### Optimization Notes
- Many small-angle controlled rotations may be pruned when target precision is relaxed (approximate QFT).
- Depth reduction matters for noisy hardware.

---

## 4.5 Shor’s Algorithm (High-Level)

### Goal
Factor composite N = p·q efficiently (exponential speedup over best known classical general methods).

### Hybrid Structure
1. Pick random a co-prime to N
2. Use quantum order-finding to get period r of a^x mod N
3. If r even & a^{r/2} ≠ −1 mod N, compute gcd(a^{r/2} ± 1, N) → non-trivial factors
4. Else retry with new a

### Quantum Core = Order Finding
- Prepare superposition over exponents |x⟩
- Compute a^x mod N into second register (modular exponentiation via repeated squaring controlled by x bits)
- Apply QFT (inverse) to exponent register
- Measure → value close to k·(2^n / r); continued fractions recover r

### Why QFT Helps
Hidden periodicity in a^x mod N becomes a sharp frequency peak after QFT, enabling classical extraction of period.

### Resource Reality
Full Shor needs many logical qubits (for modular exponentiation workspace) + error correction for large N (e.g., RSA-2048 far beyond NISQ). We demonstrate only tiny toy N.

### Toy Demonstration (Order Finding Sketch)
```python
from fractions import Fraction
from math import gcd
import random

# Classical helpers for a tiny composite N
N = 15

def attempt_shor_demo(N=15):
    while True:
        a = random.randrange(2, N)
        if gcd(a, N) != 1:
            return gcd(a, N), N//gcd(a, N)
        # Find period r classically (stand-in for quantum subroutine)
        r = 1
        val = pow(a, r, N)
        while val != 1:
            r += 1
            val = pow(a, r, N)
        if r % 2 == 0 and pow(a, r//2, N) != N - 1:
            p = gcd(pow(a, r//2, N) - 1, N)
            q = gcd(pow(a, r//2, N) + 1, N)
            if p*q == N and p not in (1,N):
                return p, q

p, q = attempt_shor_demo()
```

### Takeaways
- QFT-enabled order finding is the only genuinely quantum-heavy part
- Classical number theory wraps around it
- Practical factoring (large N) awaits fault-tolerant machines

---

## 4.6 Variational Quantum Eigensolver (VQE)

### Motivation
Exact diagonalization of large Hamiltonians is classically hard. VQE approximates ground state energy using a parameterized circuit (ansatz) + classical optimizer minimizing expectation value.

### Workflow
1. Choose Hamiltonian H (e.g., molecular or model system)
2. Select ansatz circuit U(θ) producing |ψ(θ)⟩
3. Loop: evaluate E(θ)=⟨ψ(θ)|H|ψ(θ)⟩ via repeated measurements
4. Classical optimizer updates θ
5. Converge when energy stabilizes

### Comprehensive VQE Implementation and Analysis

```python
from qiskit import QuantumCircuit, transpile, execute
from qiskit.providers.aer import Aer
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def create_hamiltonian_examples():
    """Create different Hamiltonians for VQE demonstration"""
    
    hamiltonians = {}
    
    # 1. Simple 2-qubit Hamiltonian: Z⊗I + I⊗Z + 0.5*X⊗X
    hamiltonians['simple'] = SparsePauliOp(['ZI', 'IZ', 'XX'], [1.0, 1.0, 0.5])
    
    # 2. Transverse-field Ising model: -J*Z⊗Z - h*X⊗I - h*I⊗X
    hamiltonians['ising'] = SparsePauliOp(['ZZ', 'XI', 'IX'], [-1.0, -0.5, -0.5])
    
    # 3. Heisenberg model (simplified): X⊗X + Y⊗Y + Z⊗Z
    hamiltonians['heisenberg'] = SparsePauliOp(['XX', 'YY', 'ZZ'], [1.0, 1.0, 1.0])
    
    return hamiltonians

def vqe_ansatz_library():
    """Library of different ansatz circuits"""
    
    def hardware_efficient_ansatz(n_qubits, depth, params):
        """Hardware-efficient ansatz with RY rotations and CNOT entanglers"""
        qc = QuantumCircuit(n_qubits)
        param_idx = 0
        
        for d in range(depth):
            # Single-qubit rotations
            for qubit in range(n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
            
            # Entangling layer
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        
        # Final rotation layer
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1
            
        return qc
    
    def unitary_coupled_cluster_ansatz(n_qubits, params):
        """Simplified UCC ansatz for demonstration"""
        qc = QuantumCircuit(n_qubits)
        
        # Hartree-Fock reference state (|01⟩ for 2 electrons in 2 orbitals)
        qc.x(0)
        
        # Single excitation: |01⟩ ↔ |10⟩
        qc.ry(params[0], 0)
        qc.cx(0, 1)
        qc.ry(-params[0], 1)
        qc.cx(0, 1)
        
        return qc
    
    def custom_ansatz(n_qubits, params):
        """Custom ansatz with specific structure"""
        qc = QuantumCircuit(n_qubits)
        
        # Initial state preparation
        qc.h(0)
        qc.cx(0, 1)
        
        # Parameterized gates
        qc.rz(params[0], 0)
        qc.rz(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 0)
        qc.ry(params[3], 1)
        
        return qc
    
    return {
        'hardware_efficient': hardware_efficient_ansatz,
        'ucc': unitary_coupled_cluster_ansatz,
        'custom': custom_ansatz
    }

def comprehensive_vqe(hamiltonian, ansatz_func, initial_params, 
                     optimizer_method='COBYLA', max_iterations=100):
    """
    Comprehensive VQE implementation with detailed tracking
    """
    
    print(f"VQE Optimization using {optimizer_method}")
    print("=" * 40)
    
    # Store optimization history
    energy_history = []
    param_history = []
    iteration_count = [0]  # Use list for mutable counter
    
    def cost_function(params):
        """Cost function: expectation value of Hamiltonian"""
        
        # Create ansatz circuit
        n_qubits = hamiltonian.num_qubits
        qc = ansatz_func(n_qubits, params)
        
        # Calculate expectation value
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        statevector = job.result().get_statevector()
        
        # Expectation value ⟨ψ|H|ψ⟩
        expectation_value = statevector.conjugate().T @ (hamiltonian @ statevector)
        energy = expectation_value.real
        
        # Track progress
        iteration_count[0] += 1
        energy_history.append(energy)
        param_history.append(params.copy())
        
        if iteration_count[0] % 10 == 1:
            print(f"Iteration {iteration_count[0]:3d}: Energy = {energy:.6f}")
        
        return energy
    
    # Run optimization
    print(f"Starting optimization with {len(initial_params)} parameters")
    print(f"Initial energy estimate: {cost_function(initial_params):.6f}")
    
    result = minimize(cost_function, initial_params, method=optimizer_method,
                     options={'maxiter': max_iterations})
    
    print(f"\nOptimization completed!")
    print(f"Final energy: {result.fun:.6f}")
    print(f"Total iterations: {iteration_count[0]}")
    print(f"Convergence: {'Yes' if result.success else 'No'}")
    
    return {
        'energy': result.fun,
        'optimal_params': result.x,
        'energy_history': energy_history,
        'param_history': param_history,
        'success': result.success
    }

def analyze_vqe_landscapes():
    """Analyze VQE energy landscapes for different problems"""
    
    print("\nVQE Energy Landscape Analysis")
    print("=" * 31)
    
    hamiltonians = create_hamiltonian_examples()
    ansatz_lib = vqe_ansatz_library()
    
    # Test simple Hamiltonian with different ansätze
    hamiltonian = hamiltonians['simple']
    
    print("\nTesting different ansätze on simple Hamiltonian:")
    print(f"H = ZI + IZ + 0.5*XX")
    
    # Classical ground state for comparison
    classical_matrix = hamiltonian.to_matrix()
    classical_eigenvals = np.linalg.eigvals(classical_matrix)
    classical_ground_energy = np.min(classical_eigenvals.real)
    print(f"Classical ground state energy: {classical_ground_energy:.6f}")
    
    results = {}
    
    # Test hardware-efficient ansatz
    print("\n1. Hardware-Efficient Ansatz:")
    n_params = 6  # depth=1, 2 qubits: 2+2 = 4 rotation + 2 final = 6
    initial_params = np.random.uniform(0, 2*np.pi, n_params)
    
    def he_ansatz_wrapper(n_qubits, params):
        return ansatz_lib['hardware_efficient'](n_qubits, depth=1, params=params)
    
    results['hardware_efficient'] = comprehensive_vqe(
        hamiltonian, he_ansatz_wrapper, initial_params
    )
    
    # Test custom ansatz
    print("\n2. Custom Ansatz:")
    initial_params = np.random.uniform(0, 2*np.pi, 4)
    results['custom'] = comprehensive_vqe(
        hamiltonian, ansatz_lib['custom'], initial_params
    )
    
    # Compare results
    print("\nComparison of Results:")
    print(f"{'Ansatz':<20} {'Final Energy':<15} {'Error':<15} {'Iterations'}")
    print("-" * 65)
    
    for name, result in results.items():
        error = abs(result['energy'] - classical_ground_energy)
        iterations = len(result['energy_history'])
        print(f"{name:<20} {result['energy']:<15.6f} {error:<15.6f} {iterations}")
    
    return results, classical_ground_energy

def parameter_landscape_visualization():
    """Visualize the parameter landscape for a simple 2-parameter problem"""
    
    print("\nParameter Landscape Visualization")
    print("=" * 33)
    
    # Simple 1-qubit problem for easy visualization
    hamiltonian = SparsePauliOp(['Z'], [1.0])  # Just σ_z
    
    def simple_ansatz(n_qubits, params):
        qc = QuantumCircuit(n_qubits)
        qc.ry(params[0], 0)  # Only one parameter
        return qc
    
    # Create parameter sweep
    theta_range = np.linspace(0, 2*np.pi, 100)
    energies = []
    
    for theta in theta_range:
        qc = simple_ansatz(1, [theta])
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        statevector = job.result().get_statevector()
        energy = statevector.conjugate().T @ (hamiltonian @ statevector)
        energies.append(energy.real)
    
    print("Parameter sweep completed for H = Z")
    print(f"Energy range: [{min(energies):.3f}, {max(energies):.3f}]")
    print(f"Minimum at θ = {theta_range[np.argmin(energies)]:.3f}")
    print("Theoretical minimum: θ = π (energy = -1)")
    
    # Show key points
    key_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    print("\nKey points:")
    for theta in key_points:
        idx = np.argmin(np.abs(theta_range - theta))
        print(f"  θ = {theta:.3f}: E = {energies[idx]:.3f}")

def vqe_with_noise_analysis():
    """Analyze VQE performance under different noise conditions"""
    
    print("\nVQE Noise Analysis")
    print("=" * 18)
    
    # This would require noise models - simplified demonstration
    print("Noise effects on VQE:")
    print("- Shot noise: Affects expectation value estimation accuracy")
    print("- Gate errors: Introduce systematic bias in energy estimates") 
    print("- Coherence limits: Restrict maximum circuit depth")
    print("- Mitigation strategies: Error extrapolation, readout calibration")

# Run comprehensive VQE analysis
hamiltonians = create_hamiltonian_examples()
ansatz_lib = vqe_ansatz_library()

print("Available Hamiltonians:")
for name, H in hamiltonians.items():
    print(f"  {name}: {H}")

results, classical_energy = analyze_vqe_landscapes()
parameter_landscape_visualization()
vqe_with_noise_analysis()
```

### Advanced VQE Techniques

```python
def adaptive_vqe_demo():
    """Demonstrate adaptive VQE that builds ansatz iteratively"""
    
    print("\nAdaptive VQE: Building Ansatz Iteratively")
    print("=" * 40)
    
    # Start with simple Hamiltonian
    hamiltonian = SparsePauliOp(['ZI', 'IZ', 'XX'], [1.0, 1.0, 0.5])
    
    # Pool of operators to add to ansatz
    operator_pool = ['X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XY', 'YX']
    
    print("Operator pool:", operator_pool)
    print("Strategy: Add operators that reduce energy most effectively")
    
    # This would require gradient calculations in full implementation
    print("\nSimplified adaptive procedure:")
    print("1. Start with minimal ansatz")
    print("2. Calculate gradients w.r.t. adding each pool operator")
    print("3. Add operator with largest gradient magnitude")
    print("4. Optimize parameters")
    print("5. Repeat until convergence")

def quantum_natural_gradients():
    """Demonstrate quantum natural gradient optimization"""
    
    print("\nQuantum Natural Gradients")
    print("=" * 26)
    
    print("Standard gradients: ∇E = ∂E/∂θ")
    print("Natural gradients: ∇_nat E = g^(-1) ∇E")
    print("where g is the quantum Fisher information matrix")
    print("\nAdvantages:")
    print("- Accounts for parameter space geometry")
    print("- Can avoid barren plateaus")
    print("- Often faster convergence")

# Run advanced demonstrations
adaptive_vqe_demo()
quantum_natural_gradients()
```

### Key VQE Concepts

#### Ansatz Design Principles
- **Hardware-efficient**: Use native gates, minimize depth
- **Chemistry-inspired**: UCC for molecular problems  
- **Problem-specific**: Leverage known symmetries

#### Optimization Challenges
- **Barren plateaus**: Gradients vanish exponentially with system size
- **Local minima**: Classical optimizers can get stuck
- **Shot noise**: Finite sampling affects gradient estimation

#### Practical Considerations
- **Circuit depth vs accuracy**: Deeper circuits more expressive but noisier
- **Parameter initialization**: Smart initialization can help convergence
- **Observable grouping**: Reduce measurement overhead by grouping commuting terms
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import numpy as np

backend = Aer.get_backend('qasm_simulator')

# Ansatz: Two-qubit rotation with entangling CX
theta = Parameter('θ')
ansatz = QuantumCircuit(2)
ansatz.ry(theta, 0)
ansatz.cx(0,1)

# Measure ZI + IZ expectation

def expectation(theta_val, shots=2048):
    qc = ansatz.bind_parameters({theta: theta_val})
    qc.measure_all()
    result = execute(qc, backend, shots=shots).result()
    counts = result.get_counts()
    exp_ZI = 0; exp_IZ = 0
    for bitstring, c in counts.items():
        z0 = 1 if bitstring[-1]=='0' else -1
        z1 = 1 if bitstring[-2]=='0' else -1
        exp_ZI += z0 * c / shots
        exp_IZ += z1 * c / shots
    return exp_ZI + exp_IZ

# Simple parameter sweep
vals = np.linspace(0, 2*np.pi, 25)
energies = [expectation(v) for v in vals]
min_energy = min(energies)
opt_theta = vals[energies.index(min_energy)]
```

### Optimization Considerations
- Shot noise introduces stochastic objective → robust optimizers (COBYLA, SPSA)
- Ansatz expressivity vs depth trade-off (barren plateaus risk)
- Measurement reduction via grouping commuting Pauli terms

### Why VQE Is NISQ-Friendly
Keeps quantum depth shallow (offloading heavy optimization to classical CPU). More resilient to current noise levels than long coherent algorithms like Shor.

---

## 4.7 Comparing Algorithm Paradigms

| Dimension | Oracle / Grover | Fourier / Shor | Variational / VQE |
|-----------|-----------------|----------------|-------------------|
| Quantum Depth | Moderate (iterations) | High (phase estimation + modular arithmetic) | Low–Moderate |
| Noise Sensitivity | Medium | High | Medium (measurement noise) |
| Classical Post-Processing | Minimal | Heavy (continued fractions, gcd) | Heavy (optimization loop) |
| Scalability Today | Small n demos | Limited (toy only) | Active research / deployed |
| Core Resource | Oracle construction | Clean coherent qubits | Quality measurements |
| Advantage Type | Quadratic | Exponential | Practical / heuristic |

Takeaway: Choose the paradigm matching hardware constraints + problem structure. Hybrid variational methods dominate near-term practical exploration; Fourier-based dominate long-term fault-tolerant roadmaps.

---

## 4.8 Practice & Mini Projects

### Exercises
1. Modify Deutsch–Jozsa oracle to implement a different balanced function and verify result distribution.
2. Instrument Grover code to log amplitude vector after each iteration (simulator) and plot growth of marked state probability.
3. Replace exact QFT with an approximate version (omit smallest rotations) and measure impact on order-finding precision (simulation).
4. Implement a tiny phase estimation circuit and connect it to controlled-U examples (e.g., controlled-Z rotations) to see binary phase readout.
5. Extend VQE example with an additional entangling layer and compare convergence / minimum energy.
6. Experiment with parameter shift vs finite difference to approximate gradients (if using an analytic-friendly backend).

### Mini Project Ideas
- Build a generalized Grover framework allowing multiple marked items and iteration auto-tuning.
- Implement a visual QFT step tracer that logs phase contributions per qubit for a given |x⟩.
- Compose a hybrid routine: Use Grover as subroutine inside a variational outer loop (meta-optimization experiment).
- Benchmark approximate vs exact QFT in a mini period-finding sketch.
- Implement grouped Pauli term measurement reduction for a slightly larger VQE Hamiltonian.

---

## 4.9 Summary & Forward Look

### Key Concepts Recap
- Superposition + interference harnessed by oracles (Deutsch–Jozsa, Grover)
- Phase as encoded structure unlocked by QFT
- Period finding + classical number theory yields Shor’s exponential promise
- Hybrid quantum-classical loops (VQE) extract near-term value despite noise
- Algorithm paradigm choice guided by hardware era & problem nature

### What You Can Now Do
- Prototype canonical algorithms in Qiskit
- Reason about when amplitude amplification vs phase estimation is appropriate
- Evaluate feasibility of Shor-like workflows on given hardware resources
- Implement a basic variational loop and interpret convergence behavior

### Next Module Preview (Module 5: Quantum Error Correction & Noise)
You will learn why these idealized circuits degrade on real hardware, how to characterize noise channels, apply mitigation techniques, and explore the fundamentals of error-correcting codes that protect fragile quantum information.

---

*End of Module 4: Core Quantum Algorithms*
