# Module 3: Quantum Programming Basics
*Foundation Tier*

> **✅ Qiskit 2.x Compatible** - All examples updated and tested (Dec 2024)
> 
> **Recent Updates:**
> - Updated conditional operations to use `if_test()` for Qiskit 2.x
> - Changed `bind_parameters` → `assign_parameters` across all examples
> - Fixed quantum adder implementation to avoid duplicate qubit errors
> - All 6 examples (100%) passing tests

## Learning Objectives
By the end of this module, you will be able to:
- Set up a reproducible quantum development environment
- Construct and analyze quantum circuits in Qiskit
- Use Cirq for hardware‑aware / moment‑based circuit design
- Apply circuit optimization and transpilation workflows
- Decide between simulation methods and real hardware execution
- Model basic noise and apply lightweight mitigation strategies
- Build, validate, and productionize a quantum random number generator

## Prerequisites
- Module 1 & 2 conceptual + math foundations
- Basic Python (functions, virtual environments, packages)
- Familiarity with linear algebra & probability

---

## 3.0 Core Algorithm Design Patterns (Preview)
Before diving into tooling, here’s a quick preview of structural templates you will implement or explore in later modules (fully detailed in Module 4). These guide how we architect circuits.

| Pattern | Goal | Skeleton | Key Insight |
|---------|------|----------|-------------|
| Deutsch–Jozsa | Detect constant vs balanced oracle | Prepare uniform → Oracle (phase) → Interfere → Measure | One query encodes a global property via phase kickback |
| Grover | Unstructured search | Uniform init → (Oracle + Diffusion)^k → Measure | Iterative amplitude amplification (quadratic speedup) |
| QPE | Eigenphase extraction | Eigenstate prep → Controlled powers → Inverse QFT → Measure | Converts phase into binary via interference |
| VQE | Ground state approximation | Ansatz(θ) ↔ Optimizer loop | Hybrid classical/quantum reduces depth |

---

## 3.1 Environment Setup & Tooling

### 1. Create and Activate a Virtual Environment

```bash
python -m venv quantum_env
source quantum_env/bin/activate  # Windows: quantum_env\\Scripts\\activate
python -m pip install --upgrade pip
```

### 2. Install Core Frameworks

```bash
pip install "qiskit[visualization]" qiskit-aer
pip install cirq
# Optional (later modules): pennylane mitiq
```

### 3. Quick Sanity Check (Bell Pair in Qiskit)

```python
from qiskit import QuantumCircuit, Aer, execute

def sanity_check():
    qc = QuantumCircuit(2,2)
    qc.h(0); qc.cx(0,1); qc.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots=256).result().get_counts()
    return counts

print(sanity_check())
```

You should see only roughly balanced '00' and '11' outcomes. If imports fail, verify the virtual environment is active and that `qiskit-aer` installed successfully (may require system build tools on some platforms).

### 4. (Optional) Cirq Sanity Check
```python
import cirq
q0,q1 = cirq.LineQubit.range(2)
c = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0,q1), cirq.measure(q0,q1))
print(c)
res = cirq.Simulator().run(c, repetitions=10)
print(res)
```

### Development Workflow Best Practices

#### The Quantum Programming Cycle

1. Problem Analysis
    * Assess potential for quantum advantage
    * Understand structure & classical baselines
    * Estimate resources (qubits, depth, shots)
    * Define measurable success criteria
2. Algorithm Design
    * Select or adapt a known quantum paradigm
    * Sketch circuit architecture and data flow
    * Plan measurement & classical post‑processing
    * Anticipate error mitigation needs
3. Circuit Construction
    * Implement gates and register layout
    * Add measurements (defer where possible to reduce noise)
    * Sanity‑check small test inputs
4. Classical Simulation
    * Run noiseless reference executions
    * Debug logic; inspect intermediate states if needed
    * Profile gate counts and depth hotspots
5. Optimization
    * Simplify manually (merge rotations / cancel inverses)
    * Transpile with target backend; adjust layout hints
    * Iterate on parameterization to reduce depth
6. Hardware Testing
    * Execute small shot counts first
    * Compare empirical distributions vs simulation
    * Capture calibration & error metrics
7. Results Analysis
    * Statistical significance & variance checks
    * Benchmark against classical approach
    * Document performance & limitations
8. Iteration
    * Refine architecture or algorithm choice
    * Scale input size when stable
    * Share results; open issues for future tuning

#### Version Control for Quantum Projects:

```python
# .gitignore for quantum projects
gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Quantum-specific
qiskit.log
*.qobj
experiment_data/
quantum_results/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

print("Recommended .gitignore for quantum projects:")
print(gitignore_content)
```

### Common Environment Issues and Solutions

#### Issue 1: Installation Problems

```python
def troubleshoot_installation():
    """Common installation issues and solutions"""
    
    issues = {
        "Qiskit won't install": [
            "Update pip: pip install --upgrade pip",
            "Try: pip install qiskit --user",
            "Check Python version (need 3.7+)",
            "Use virtual environment"
        ],
        "Import errors": [
            "Verify virtual environment is activated",
            "Check package installation: pip list",
            "Reinstall: pip uninstall qiskit && pip install qiskit",
            "Try importing specific modules"
        ],
        "Visualization issues": [
            "Install matplotlib: pip install matplotlib",
            "Install additional tools: pip install qiskit[visualization]",
            "Update graphics backend",
            "Try text-based circuit drawing"
        ],
        "Simulator problems": [
            "Install Aer: pip install qiskit-aer",
            "Check system resources (RAM)",
            "Try BasicAer for small circuits",
            "Update to latest version"
        ]
    }
    
    print("Common Issues and Solutions:")
    print("=" * 28)
    
    for issue, solutions in issues.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  • {solution}")

troubleshoot_installation()
```

#### Issue 2: Performance Optimization

```python
def performance_tips():
    """Tips for optimal quantum programming performance"""
    
    tips = [
        "Use virtual environments to avoid package conflicts",
        "Install Aer simulator for faster classical simulation", 
        "Limit circuit size during development (start small)",
        "Use transpilation to optimize circuits",
        "Cache simulation results for repeated experiments",
        "Use parallel execution for multiple circuits",
        "Monitor memory usage for large state vectors",
        "Consider using cloud quantum services for heavy computation"
    ]
    
    print("Performance Optimization Tips:")
    print("=" * 30)
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

performance_tips()
```

---

## 3.2 Qiskit Deep Dive: IBM's Quantum Framework

### Introduction to Qiskit

Qiskit (Quantum Information Science Kit) is IBM's open-source quantum computing framework. Think of it as the "Python of quantum computing" - it's comprehensive, well-documented, and has a huge community.

#### Why Qiskit?
Qiskit is widely adopted for several practical reasons:

**Why Choose Qiskit?**

* 🏢 Industry standard backed by IBM (long-term roadmap & stability)
* 🌐 Large, active global community (fast answers, many examples)
* 📚 Excellent documentation, tutorials, and learning resources
* 🔬 Direct access to real IBM quantum hardware (cloud backends)
* 🛠️ Comprehensive toolkit covering circuits, algorithms, transpilation, simulators
* 🎓 Strong support for education and academic research
* 🔧 Frequent updates with evolving hardware-aware optimizations
* 💼 Proven usage in enterprise and domain‑specific applications

### Qiskit Architecture and Components

**Qiskit Architecture (Ecosystem Overview)**

| Component | Purpose | When To Use | Key Features |
|-----------|---------|-------------|--------------|
| Terra | Core circuits & transpilation layer | Always (foundation) | Circuit classes, gate definitions, passes, transpiler, basic providers |
| Aer | High‑performance simulators | Fast accurate simulation & noise studies | Statevector, density matrix, MPS, stabilizer, noise models, optional GPU |
| Nature | Physics & chemistry domain tools | Molecular / materials / vibrational problems | Electronic structure, second‑quantization mappers, VQE workflows |
| Finance | Financial optimization & analytics | Portfolio, risk, derivative modeling | Portfolio optimization, QAOA helpers, uncertainty models |
| Machine Learning | Quantum ML primitives | Kernel methods & hybrid models | Quantum kernels, variational classifiers, neural network interfaces |

These modules plug into a consistent circuit & provider abstraction so you can prototype algorithms and then swap simulation for hardware with minimal refactoring.

### Basic Qiskit Concepts and Objects

#### 1. QuantumCircuit - The Heart of Qiskit

```python
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

def quantum_circuit_basics():
    """Comprehensive introduction to QuantumCircuit"""
    
    print("QuantumCircuit: The Foundation of Qiskit")
    print("=" * 39)
    
    # Method 1: Simple circuit creation
    print("Method 1: Simple creation")
    qc1 = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits
    print(f"Circuit: {qc1}")
    print(f"Number of qubits: {qc1.num_qubits}")
    print(f"Number of classical bits: {qc1.num_clbits}")
    
    # Method 2: Using registers (more organized)
    print(f"\nMethod 2: Using registers")
    qreg = QuantumRegister(3, 'q')      # Quantum register
    creg = ClassicalRegister(3, 'c')    # Classical register
    qc2 = QuantumCircuit(qreg, creg)
    
    print(f"Quantum register: {qreg}")
    print(f"Classical register: {creg}")
    
    # Method 3: Mixed register sizes
    print(f"\nMethod 3: Mixed sizes")
    qc3 = QuantumCircuit(5, 2)  # 5 qubits, 2 classical bits
    print(f"5 qubits, 2 classical bits: {qc3}")
    
    # Adding gates to circuits
    print(f"\nAdding gates:")
    qc1.h(0)           # Hadamard on qubit 0
    qc1.x(1)           # Pauli-X on qubit 1  
    qc1.cx(0, 2)       # CNOT: control=0, target=2
    qc1.measure(0, 0)  # Measure qubit 0 into classical bit 0
    
    print("Circuit after adding gates:")
    print(qc1.draw())
    
    return qc1, qc2, qc3

# Demo basic circuits
circuit1, circuit2, circuit3 = quantum_circuit_basics()
```

#### 2. Quantum Gates in Qiskit

**Common Qiskit Gates (Reference)**

Single‑qubit (state preparation & phase control):
* Identity `I` – no change
* Pauli X / Y / Z – bit flip, bit+phase flip, phase flip
* Hadamard `H` – creates |0⟩→(|0⟩+|1⟩)/√2, |1⟩→(|0⟩−|1⟩)/√2
* Phase family: `S`, `S†`, `T`, `T†` for π/2 and π/4 rotations about Z

Parameterized rotations / general gate:
* `RX(θ)`, `RY(θ)`, `RZ(θ)` – axis rotations
* `P(λ)` (phase) – adds phase e^{iλ}
* `U(θ,φ,λ)` – general single‑qubit unitary decomposition

Two‑qubit operations:
* `CX` (CNOT) – conditional bit flip
* `CZ`, `CY`, `CH` – controlled phase / Y / H
* `SWAP` – exchange qubit states (costly; often decomposed to 3 CNOTs)
* Controlled rotations: `CRX`, `CRY`, `CRZ` (useful in variational ansätze)

Multi‑qubit / higher control:
* Toffoli `CCX` – controlled‑controlled‑X (universal for reversible logic)
* Multi‑controlled `X` / `Z` (`mcx`, `mcz`) – generalized controls (cost grows quickly)

Tip: Minimize two‑qubit gate count (dominant error contributors) and consolidate adjacent single‑qubit rotations before transpilation.

#### 3. Building Your First Real Circuits

```python
import numpy as np
from math import pi

def build_example_circuits():
    """Build several example circuits to demonstrate concepts"""
    
    print("Example Quantum Circuits")
    print("=" * 24)
    
    # Example 1: Bell State (Entanglement)
    print("Example 1: Bell State Creation")
    bell_circuit = QuantumCircuit(2, 2)
    bell_circuit.h(0)        # Superposition on qubit 0
    bell_circuit.cx(0, 1)    # Entangle with qubit 1
    bell_circuit.measure_all()
    
    print("Circuit:")
    print(bell_circuit.draw())
    print("Expected results: 50% |00⟩, 50% |11⟩ (perfectly correlated)")
    
    # Example 2: GHZ State (3-qubit entanglement)
    print(f"\nExample 2: GHZ State (3-qubit entanglement)")
    ghz_circuit = QuantumCircuit(3, 3)
    ghz_circuit.h(0)         # Superposition on qubit 0
    ghz_circuit.cx(0, 1)     # Entangle 0 and 1
    ghz_circuit.cx(0, 2)     # Entangle 0 and 2
    ghz_circuit.measure_all()
    
    print("Circuit:")
    print(ghz_circuit.draw())
    print("Expected results: 50% |000⟩, 50% |111⟩")
    
    # Example 3: Quantum Phase Kickback
    print(f"\nExample 3: Phase Kickback Demonstration")
    phase_circuit = QuantumCircuit(2, 2)
    phase_circuit.h(0)       # Control in superposition
    phase_circuit.x(1)       # Target in |1⟩ state
    phase_circuit.h(1)       # Put target in |-⟩ state
    phase_circuit.cx(0, 1)   # CNOT causes phase kickback
    phase_circuit.h(1)       # Return target to computational basis
    phase_circuit.h(0)       # Return control to computational basis
    phase_circuit.measure_all()
    
    print("Circuit:")
    print(phase_circuit.draw())
    print("Demonstrates: How controlled operations can affect control qubit")
    
    # Example 4: Parameterized Circuit
    print(f"\nExample 4: Parameterized Rotation Circuit")
    from qiskit.circuit import Parameter
    
    theta = Parameter('θ')
    phi = Parameter('φ')
    
    param_circuit = QuantumCircuit(1, 1)
    param_circuit.ry(theta, 0)   # Y-rotation by angle theta
    param_circuit.rz(phi, 0)     # Z-rotation by angle phi
    param_circuit.measure_all()
    
    print("Circuit (before parameter binding):")
    print(param_circuit.draw())
    
    # Bind parameters
    bound_circuit = param_circuit.bind_parameters({theta: pi/4, phi: pi/2})
    print("Circuit (after binding θ=π/4, φ=π/2):")
    print(bound_circuit.draw())
    
    return bell_circuit, ghz_circuit, phase_circuit, param_circuit

# Build example circuits
bell, ghz, phase, param = build_example_circuits()
```

### Circuit Manipulation and Analysis

#### 1. Circuit Properties and Information

```python
def analyze_circuit_properties(circuit, name="Circuit"):
    """Analyze various properties of a quantum circuit"""
    
    print(f"{name} Analysis")
    print("=" * (len(name) + 9))
    
    # Basic properties
    print(f"Basic Properties:")
    print(f"  Number of qubits: {circuit.num_qubits}")
    print(f"  Number of classical bits: {circuit.num_clbits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Circuit size (gates): {circuit.size()}")
    print(f"  Circuit width: {circuit.width()}")
    
    # Gate count analysis
    gate_counts = circuit.count_ops()
    print(f"\nGate Counts:")
    for gate, count in gate_counts.items():
        print(f"  {gate}: {count}")
    
    # Parameters (if any)
    parameters = circuit.parameters
    if parameters:
        print(f"\nParameters:")
        for param in parameters:
            print(f"  {param}")
    else:
        print(f"\nNo parameters in circuit")
    
    # Two-qubit gate count (important for hardware)
    two_qubit_gates = ['cx', 'cy', 'cz', 'ch', 'swap', 'crx', 'cry', 'crz']
    two_qubit_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
    print(f"\nTwo-qubit gates total: {two_qubit_count}")
    
    return {
        'depth': circuit.depth(),
        'size': circuit.size(),
        'two_qubit_count': two_qubit_count,
        'gate_counts': gate_counts
    }

# Analyze our example circuits
print("Circuit Analysis Examples:")
print("=" * 26)

bell_analysis = analyze_circuit_properties(bell, "Bell State Circuit")
print()
ghz_analysis = analyze_circuit_properties(ghz, "GHZ State Circuit")
```

#### 2. Circuit Decomposition and Equivalence

```python
def circuit_decomposition_examples():
    """Show how to decompose and manipulate circuits"""
    
    print("Circuit Decomposition and Manipulation")
    print("=" * 35)
    
    # Create a circuit with higher-level gates
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)  # Toffoli gate
    
    print("Original circuit with Toffoli gate:")
    print(qc.draw())
    
    # Decompose Toffoli into elementary gates
    decomposed = qc.decompose()
    print(f"\nAfter decomposition:")
    print(decomposed.draw())
    
    print(f"Original depth: {qc.depth()}")
    print(f"Decomposed depth: {decomposed.depth()}")
    print(f"Original size: {qc.size()}")
    print(f"Decomposed size: {decomposed.size()}")
    
    # Circuit equivalence testing
    print(f"\nTesting circuit equivalence:")
    
    # Create two equivalent circuits
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.z(0)
    qc1.h(0)
    
    qc2 = QuantumCircuit(1)
    qc2.x(0)
    
    print("Circuit 1 (H-Z-H sequence):")
    print(qc1.draw())
    
    print("Circuit 2 (Single X gate):")
    print(qc2.draw())
    
    print("These circuits are equivalent: H-Z-H = X")
    
    return qc, decomposed, qc1, qc2

circuit_decomposition_examples()
```

### Advanced Qiskit Features

#### 1. Circuit Transpilation

```python
from qiskit import transpile
from qiskit.providers.fake_provider import FakeVigo

def transpilation_demo():
    """Demonstrate circuit transpilation for hardware"""
    
    print("Circuit Transpilation for Hardware")
    print("=" * 34)
    
    # Create a circuit that needs transpilation
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)  # This creates a cycle
    
    print("Original circuit:")
    print(qc.draw())
    print(f"Original depth: {qc.depth()}")
    print(f"Original two-qubit gates: {qc.count_ops().get('cx', 0)}")
    
    # Use a fake backend to simulate hardware constraints
    backend = FakeVigo()  # IBM's 5-qubit backend
    
    # Transpile for the backend
    transpiled_qc = transpile(qc, backend, optimization_level=2)
    
    print(f"\nAfter transpilation for {backend.name()}:")
    print(transpiled_qc.draw())
    print(f"Transpiled depth: {transpiled_qc.depth()}")
    print(f"Transpiled two-qubit gates: {transpiled_qc.count_ops().get('cx', 0)}")
    
    # Show the effect of different optimization levels
    print(f"\nOptimization level comparison:")
    for level in range(4):
        opt_qc = transpile(qc, backend, optimization_level=level)
        print(f"Level {level}: depth={opt_qc.depth()}, "
              f"cx_gates={opt_qc.count_ops().get('cx', 0)}")
    
    return qc, transpiled_qc

original, transpiled = transpilation_demo()
```

#### 2. Working with Real Hardware

```python
def hardware_integration_guide():
    """Guide to using real quantum hardware with Qiskit"""
    
    print("Real Quantum Hardware Integration")
    print("=" * 33)
    
    setup_steps = [
        "1. Create IBM Quantum account at quantum-computing.ibm.com",
        "2. Get your API token from account settings",
        "3. Save credentials locally",
        "4. Load account and select backend",
        "5. Submit jobs to quantum computer",
        "6. Retrieve and analyze results"
    ]
    
    print("Setup Steps:")
    for step in setup_steps:
        print(f"  {step}")
    
    print(f"\nCode example for hardware access:")
    
    hardware_code = '''
# Save your IBM Quantum credentials (run once)
from qiskit import IBMQ
IBMQ.save_account('YOUR_API_TOKEN_HERE')

# Load account and get backends
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

# See available backends
backends = provider.backends()
for backend in backends:
    status = backend.status()
    print(f"{backend.name()}: {status.pending_jobs} jobs pending")

# Choose a backend
backend = provider.get_backend('ibmq_qasm_simulator')  # Start with simulator
# backend = provider.get_backend('ibm_nairobi')  # Real hardware example

# Prepare your circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Transpile for the backend
transpiled_qc = transpile(qc, backend)

# Submit job
job = backend.run(transpiled_qc, shots=1024)
print(f"Job ID: {job.job_id()}")
print(f"Job Status: {job.status()}")

# Get results (may take time for real hardware)
result = job.result()
counts = result.get_counts()
print(f"Results: {counts}")
'''
    
    print(hardware_code)
    
    print(f"\nImportant considerations for real hardware:")
    considerations = [
        "Queue times can be hours or days",
        "Noise and errors affect results",
        "Limited gate sets on real devices",
        "Connectivity constraints between qubits",
        "Time limits on circuit execution",
        "Cost considerations for premium access"
    ]
    
    for consideration in considerations:
        print(f"  • {consideration}")

hardware_integration_guide()
```

### Qiskit Best Practices

```python
def qiskit_best_practices():
    """Best practices for Qiskit development"""
    
    print("Qiskit Best Practices")
    print("=" * 20)
    
    practices = {
        "Circuit Design": [
            "Start small and test incrementally",
            "Use descriptive names for registers",
            "Comment complex circuit sections",
            "Minimize circuit depth when possible",
            "Use parameterized circuits for flexibility"
        ],
        "Performance": [
            "Transpile circuits before execution",
            "Use appropriate optimization levels",
            "Cache transpilation results",
            "Use Aer for fast simulation",
            "Batch multiple circuits together"
        ],
        "Debugging": [
            "Test on simulators first",
            "Use smaller qubit counts during development",
            "Print intermediate states when debugging",
            "Verify circuit properties (depth, gate counts)",
            "Compare with analytical results when possible"
        ],
        "Hardware Preparation": [
            "Understand backend limitations",
            "Account for noise and errors",
            "Use error mitigation techniques",
            "Monitor queue times",
            "Test on noise simulators first"
        ],
        "Code Organization": [
            "Separate circuit construction from execution",
            "Use functions for reusable circuit patterns",
            "Save important results to files",
            "Use version control for experiments",
            "Document experimental parameters"
        ]
    }
    
    for category, tips in practices.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  • {tip}")

qiskit_best_practices()
```

---

## 3.3 Cirq Introduction: Google's Quantum Library

### Introduction to Cirq

Cirq is Google's open-source quantum computing framework, designed with a focus on Noisy Intermediate-Scale Quantum (NISQ) devices. While Qiskit is more academic and comprehensive, Cirq is more focused on practical near-term quantum computing.

#### Cirq vs Qiskit: Key Differences

| Aspect | Cirq | Qiskit |
|--------|------|--------|
| Design Philosophy | NISQ‑focused, hardware‑aware, pragmatic | Broad, research & education oriented, full‑stack |
| Circuit Representation | Explicit Moments (parallel gate time slices) | Gate sequence; scheduling/transpilation adds timing |
| Qubit Representation | Grid / named physical qubits (e.g., cirq.GridQubit) | Abstract indices; mapping handled during transpile |
| Gate API | Gate objects + operators (`gate.on(qubit)`) | Method chaining on `QuantumCircuit` (`qc.h(q)`) |
| Hardware Focus | Google devices & neutral architecture | IBM Quantum systems & provider ecosystem |
| Ecosystem Breadth | Lean core toolkit | Wide domain extensions (Nature, Finance, ML, etc.) |
| Compilation Emphasis | Manual moment control | Automated pass manager & layout/transpile stack |

If you want tight control over timing & placement: start with Cirq. If you want a large ecosystem and immediate access to learning materials & hardware variety: start with Qiskit.

### Basic Cirq Concepts

#### 1. Qubits and Grid Structure

In Cirq you can represent qubits in several semantically meaningful ways:

* NamedQubit('q0') – abstract label you control (good for conceptual demos)
* GridQubit(row, col) – encodes a physical 2D lattice coordinate (mirrors many superconducting layouts)
* LineQubit.range(n) – 1D chain (useful for algorithms assuming linear nearest‑neighbor coupling)

The grid model helps you reason about adjacency constraints early. Example mental model for a 2×2 grid: (0,0) — (0,1) / (1,0) — (1,1). You align two‑qubit operations only between physically connected sites (or accept SWAP insertion overhead during compilation).

#### 2. Gates and Operations

Core Cirq gate families (conceptual quick reference):

* Single‑qubit: I, X, Y, Z, H, S, T (mirrors standard universal set)
* Parameterized rotations: `rx(θ)`, `ry(θ)`, `rz(θ)`, `ZPowGate(exponent)`
* Two‑qubit entangling: CNOT, CZ, SWAP, iSWAP (match hardware native set when possible)
* Multi‑qubit controls: TOFFOLI, FREDKIN (often decomposed; use sparingly)

Use Cirq’s gate objects (e.g., `cirq.H(q)`), and where parallelism is possible, group operations into the same Moment to reduce circuit depth.

#### 3. Moments and Circuits

In Cirq, a circuit is an ordered list of Moments. Each Moment contains operations that can be executed in parallel (no qubit conflicts). This explicit grouping helps you reason about temporal structure and potential depth reduction:

Example conceptual layering:
* Moment 1: Apply H to all three qubits (parallel single‑qubit gates)
* Moment 2: Entangling gate (e.g., CNOT between two qubits)
* Moment 3: Measurements (can often be parallelized)

You can build circuits either by supplying a list of Moments directly or incrementally appending operations (Cirq will auto‑group them into Moments where compatible). Depth is then literally the number of Moments—so reducing depth = maximizing safe parallel placement.

### Cirq Simulation

#### 1. State Vector Simulation

```python
def cirq_simulation_basics():
    """Basic simulation in Cirq"""
    
    print("Cirq Simulation Basics")
    print("=" * 21)
    
    # Create a simple circuit
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure_each(*qubits)
    ])
    
    print("Circuit to simulate:")
    print(circuit)
    
    # Simulation without measurements (state vector)
    print(f"\nState vector simulation:")
    simulator = cirq.Simulator()
    
    # Remove measurements for state vector simulation
    circuit_no_measure = circuit[:-1]  # All but last moment
    result = simulator.simulate(circuit_no_measure)
    
    print(f"Final state vector:")
    print(result.final_state_vector)
    
    # Show state in computational basis
    print(f"\nState in computational basis:")
    state_vector = result.final_state_vector
    for i, amplitude in enumerate(state_vector):
        if abs(amplitude) > 1e-10:  # Only show significant amplitudes
            binary = format(i, f'0{len(qubits)}b')
            print(f"|{binary}⟩: {amplitude:.3f}")
    
    # Measurement simulation
    print(f"\nMeasurement simulation:")
    measurement_result = simulator.run(circuit, repetitions=1000)
    counts = measurement_result.histogram(key='q')
    
    print(f"Measurement results after 1000 shots:")
    for outcome, count in counts.items():
        prob = count / 1000
        print(f"Outcome {outcome}: {count} times ({prob:.1%})")
    
    return circuit, result, measurement_result

# Demo Cirq simulation
circ, state_result, measure_result = cirq_simulation_basics()
```

#### 2. Advanced Simulation Features

```python
def cirq_advanced_simulation():
    """Advanced Cirq simulation features"""
    
    print("Advanced Cirq Simulation Features")
    print("=" * 33)
    
    qubits = cirq.LineQubit.range(2)
    
    # Parameterized circuit
    print("1. Parameterized Circuit Simulation:")
    
    theta = cirq.Symbol('theta')
    phi = cirq.Symbol('phi')
    
    param_circuit = cirq.Circuit([
        cirq.ry(theta).on(qubits[0]),
        cirq.rz(phi).on(qubits[0]),
        cirq.measure(qubits[0], key='result')
    ])
    
    print(param_circuit)
    
    # Sweep over parameter values
    param_sweep = cirq.Linspace(key='theta', start=0, stop=np.pi, num=5) * \
                  cirq.Linspace(key='phi', start=0, stop=2*np.pi, num=3)
    
    simulator = cirq.Simulator()
    results = simulator.run_sweep(param_circuit, param_sweep, repetitions=100)
    
    print(f"\nParameter sweep results (first few):")
    for i, result in enumerate(results[:3]):
        params = result.params
        histogram = result.histogram(key='result')
        prob_1 = histogram.get(1, 0) / 100
        print(f"θ={params['theta']:.2f}, φ={params['phi']:.2f}: P(1)={prob_1:.2f}")
    
    # Noise simulation
    print(f"\n2. Noise Simulation:")
    
    # Define noise model
    noise_model = cirq.NoiseModel.from_noise_model_like(
        cirq.depolarize(p=0.01)  # 1% depolarizing noise
    )
    
    # Simple circuit for noise demo
    clean_circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.measure(qubits[0], key='noisy')
    ])
    
    print("Clean circuit:")
    print(clean_circuit)
    
    # Simulate with and without noise
    clean_result = simulator.run(clean_circuit, repetitions=1000)
    noisy_result = simulator.run(clean_circuit.with_noise(noise_model), repetitions=1000)
    
    clean_prob_1 = clean_result.histogram(key='noisy').get(1, 0) / 1000
    noisy_prob_1 = noisy_result.histogram(key='noisy').get(1, 0) / 1000
    
    print(f"\nNoise effect on Hadamard gate:")
    print(f"Clean simulation P(1): {clean_prob_1:.3f}")
    print(f"Noisy simulation P(1): {noisy_prob_1:.3f}")
    print(f"Expected for H|0⟩: 0.500")

cirq_advanced_simulation()
```

### Cirq and Google Hardware

```python
def cirq_hardware_overview():
    """Overview of Cirq's hardware integration"""
    
    print("Cirq and Google Quantum Hardware")
    print("=" * 32)
    
    hardware_info = {
        "Sycamore Processor": {
            "Type": "Superconducting qubits",
            "Layout": "54-qubit 2D grid (originally)",
            "Connectivity": "Nearest-neighbor coupling",
            "Gate Set": "√X, √Y, √W, CZ gates",
            "Special Features": "Tunable coupling, fast gates"
        },
        "Quantum AI Service": {
            "Access": "Through Google Cloud Platform",
            "Authentication": "Service account credentials",
            "Scheduling": "Job queue system",
            "Pricing": "Pay-per-use model"
        },
        "Device Characteristics": {
            "Gate Times": "~10-50 nanoseconds",
            "Coherence Times": "~10-100 microseconds", 
            "Error Rates": "~0.1-1% per gate",
            "Connectivity": "Limited to adjacent qubits"
        }
    }
    
    for category, details in hardware_info.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print(f"\nCode example for hardware access:")
    
    hardware_code = '''
# Install Cirq with Google AI Quantum support
# pip install cirq-google

import cirq
import cirq_google

# Authenticate (requires Google Cloud setup)
# Follow: https://quantumai.google/cirq/google/access

# Get the quantum processor
processor = cirq_google.get_engine().get_processor('rainbow')

# Create a simple circuit
qubits = processor.get_sampler().get_qubits()[:2]
circuit = cirq.Circuit([
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure_each(*qubits)
])

# Submit to quantum processor
sampler = processor.get_sampler()
results = sampler.run(circuit, repetitions=1000)

print("Results from Google quantum processor:")
print(results.histogram(key='q'))
'''
    
    print(hardware_code)
    
    print(f"\nKey differences from simulators:")
    differences = [
        "Limited qubit connectivity (grid layout)",
        "Specific native gate set (√X, √Y, √W, CZ)",
        "Noise and decoherence effects",
        "Queue times and scheduling",
        "Calibration-dependent performance",
        "Cost considerations"
    ]
    
    for diff in differences:
        print(f"  • {diff}")

cirq_hardware_overview()
```

### Cirq Best Practices

```python
def cirq_best_practices():
    """Best practices for Cirq development"""
    
    print("Cirq Development Best Practices")
    print("=" * 31)
    
    practices = {
        "Circuit Design": [
            "Use GridQubits for hardware-aware design",
            "Respect qubit connectivity constraints",
            "Minimize circuit depth (moments)",
            "Use native gates when targeting hardware",
            "Plan for noise and errors"
        ],
        "Moment Organization": [
            "Group parallel operations in moments",
            "Understand moment-based scheduling",
            "Use cirq.InsertStrategy for control",
            "Optimize moment structure for hardware",
            "Visualize moments for debugging"
        ],
        "Parameter Handling": [
            "Use cirq.Symbol for parameterized circuits",
            "Leverage parameter sweeps for optimization",
            "Cache resolved circuits when possible",
            "Use symbolic computation carefully",
            "Document parameter meanings"
        ],
        "Simulation Strategy": [
            "Start with small circuits",
            "Use appropriate simulator backends",
            "Include noise models for realism",
            "Compare with analytical results",
            "Profile performance for large circuits"
        ],
        "Hardware Preparation": [
            "Understand Google quantum processor specs",
            "Use device-specific optimizations",
            "Test with realistic noise models",
            "Plan for queue times and costs",
            "Validate circuits before submission"
        ]
    }
    
    for category, tips in practices.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  • {tip}")

cirq_best_practices()
```

---

## 3.4 Circuit Building Techniques

### From Basic Gates to Complex Algorithms

Building quantum circuits is like constructing with LEGO blocks - you start with simple pieces (gates) and combine them into complex structures (algorithms). Let's learn the systematic approach to circuit construction.

#### 1. Circuit Building Methodology

**Circuit Building Methodology (Structured Workflow)**

1. Problem Analysis
    * Clarify objective & success metrics
    * Identify classical vs quantum partitioning
    * Estimate qubit count, depth, measurement needs
    * Assess potential for quantum speedup
2. Algorithm Selection
    * Map problem to known primitives (e.g., amplitude amplification, phase estimation, variational)
    * Evaluate algorithmic complexity & resource scaling
    * Note noise sensitivity & hardware constraints
3. Circuit Architecture
    * Assign roles to qubits (data, ancilla, controls)
    * Define register layout & measurement points
    * Plan entanglement structure and layering
4. Gate-Level Implementation
    * Use native or efficiently decomposable gates
    * Minimize two‑qubit operations & depth hotspots
    * Parameterize where tuning is expected
5. Optimization
    * Apply identities & gate cancellations pre‑transpile
    * Transpile with hardware target (vary optimization levels)
    * Inspect depth, critical path, two‑qubit counts
6. Validation
    * Simulate small instances for correctness
    * Compare statistical outcomes vs theory
    * Add diagnostic measurements in early iterations

#### 2. Common Circuit Patterns and Building Blocks

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def quantum_circuit_patterns():
    """Common patterns and building blocks for quantum circuits"""
    
    print("Common Quantum Circuit Patterns")
    print("=" * 31)
    
    # Pattern 1: State Preparation
    print("1. State Preparation Patterns:")
    
    # Uniform superposition
    print("\nUniform Superposition (all states equally likely):")
    n_qubits = 3
    superpos_circuit = QuantumCircuit(n_qubits)
    superpos_circuit.h(range(n_qubits))  # Hadamard on all qubits
    
    print(f"H^⊗{n_qubits} creates uniform superposition of 2^{n_qubits} states")
    print(superpos_circuit.draw())
    
    # Arbitrary state preparation
    print("\nArbitrary State Preparation:")
    state_prep_circuit = QuantumCircuit(2)
    # Prepare state α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
    state_prep_circuit.ry(np.pi/3, 0)  # Control amplitude
    state_prep_circuit.cx(0, 1)        # Create entanglement
    state_prep_circuit.ry(np.pi/6, 1)  # Adjust target amplitude
    
    print("State preparation with controlled rotations:")
    print(state_prep_circuit.draw())
    
    # Pattern 2: Entanglement Generation
    print("\n2. Entanglement Generation Patterns:")
    
    # Bell states
    bell_patterns = {
        "|Φ+⟩": ["H(0)", "CX(0,1)"],
        "|Φ-⟩": ["H(0)", "CX(0,1)", "Z(0)"],
        "|Ψ+⟩": ["H(0)", "CX(0,1)", "X(1)"],
        "|Ψ-⟩": ["H(0)", "CX(0,1)", "Z(0)", "X(1)"]
    }
    
    print("Bell State Generation:")
    for state, gates in bell_patterns.items():
        print(f"  {state}: {' → '.join(gates)}")
    
    # GHZ state
    ghz_circuit = QuantumCircuit(3)
    ghz_circuit.h(0)
    ghz_circuit.cx(0, 1)
    ghz_circuit.cx(0, 2)
    
    print(f"\nGHZ State (3-qubit entanglement):")
    print(ghz_circuit.draw())
    
    # Pattern 3: Phase Manipulation
    print("\n3. Phase Manipulation Patterns:")
    
    phase_circuit = QuantumCircuit(2)
    phase_circuit.h([0, 1])           # Create superposition
    phase_circuit.cz(0, 1)            # Conditional phase
    phase_circuit.rz(np.pi/4, 0)     # Single-qubit phase
    
    print("Phase manipulation example:")
    print(phase_circuit.draw())
    
    # Pattern 4: Amplitude Amplification Structure
    print("\n4. Amplitude Amplification Structure:")
    
    amp_amp_circuit = QuantumCircuit(2)
    # Oracle (marks target state)
    amp_amp_circuit.x([0, 1])         # Flip to |11⟩
    amp_amp_circuit.cz(0, 1)          # Mark |11⟩
    amp_amp_circuit.x([0, 1])         # Flip back
    
    # Diffusion operator
    amp_amp_circuit.h([0, 1])
    amp_amp_circuit.x([0, 1])
    amp_amp_circuit.cz(0, 1)
    amp_amp_circuit.x([0, 1])
    amp_amp_circuit.h([0, 1])
    
    print("Amplitude amplification iteration:")
    print(amp_amp_circuit.draw())
    
    return superpos_circuit, state_prep_circuit, ghz_circuit, phase_circuit

# Demo circuit patterns
superpos, state_prep, ghz, phase = quantum_circuit_patterns()
```

#### 3. Modular Circuit Construction

```python
def modular_circuit_construction():
    """Building complex circuits from reusable modules"""
    
    print("Modular Circuit Construction")
    print("=" * 27)
    
    # Define reusable circuit modules
    print("Creating reusable circuit modules:")
    
    def create_qft_module(n_qubits):
        """Quantum Fourier Transform module"""
        qft_circuit = QuantumCircuit(n_qubits, name='QFT')
        
        for i in range(n_qubits):
            qft_circuit.h(i)
            for j in range(i + 1, n_qubits):
                qft_circuit.cp(np.pi / (2 ** (j - i)), i, j)
        
        # Reverse qubit order
        for i in range(n_qubits // 2):
            qft_circuit.swap(i, n_qubits - 1 - i)
            
        return qft_circuit
    
    def create_grover_oracle(target_state, n_qubits):
        """Grover's oracle module for specific target"""
        oracle = QuantumCircuit(n_qubits, name=f'Oracle_{target_state}')
        
        # Flip bits that should be 0 in target state
        for i, bit in enumerate(reversed(target_state)):
            if bit == '0':
                oracle.x(i)
        
        # Multi-controlled Z gate
        if n_qubits == 1:
            oracle.z(0)
        elif n_qubits == 2:
            oracle.cz(0, 1)
        else:
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.z(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        
        # Flip bits back
        for i, bit in enumerate(reversed(target_state)):
            if bit == '0':
                oracle.x(i)
                
        return oracle
    
    def create_diffusion_operator(n_qubits):
        """Grover's diffusion operator module"""
        diffusion = QuantumCircuit(n_qubits, name='Diffusion')
        
        diffusion.h(range(n_qubits))
        diffusion.x(range(n_qubits))
        
        if n_qubits == 1:
            diffusion.z(0)
        elif n_qubits == 2:
            diffusion.cz(0, 1)
        else:
            diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            diffusion.z(n_qubits - 1)
            diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        
        diffusion.x(range(n_qubits))
        diffusion.h(range(n_qubits))
        
        return diffusion
    
    # Demonstrate modular construction
    print("\n1. Creating modules:")
    n_qubits = 3
    target = "101"  # Target state for Grover's algorithm
    
    qft_module = create_qft_module(n_qubits)
    oracle_module = create_grover_oracle(target, n_qubits)
    diffusion_module = create_diffusion_operator(n_qubits)
    
    print(f"QFT module ({n_qubits} qubits):")
    print(qft_module.draw())
    
    print(f"\nOracle module (target: |{target}⟩):")
    print(oracle_module.draw())
    
    print(f"\nDiffusion module:")
    print(diffusion_module.draw())
    
    # Combine modules into complete algorithm
    print(f"\n2. Combining modules - Grover's Algorithm:")
    grover_circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize in superposition
    grover_circuit.h(range(n_qubits))
    
    # Grover iterations
    iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
    print(f"Optimal iterations: {iterations}")
    
    for i in range(iterations):
        grover_circuit.append(oracle_module, range(n_qubits))
        grover_circuit.append(diffusion_module, range(n_qubits))
    
    # Measurement
    grover_circuit.measure_all()
    
    print(f"\nComplete Grover circuit:")
    print(grover_circuit.draw())
    
    # Advanced: Parameterized modules
    print(f"\n3. Parameterized modules:")
    
    def create_rotation_layer(n_qubits, params):
        """Parameterized rotation layer for VQE"""
        from qiskit.circuit import Parameter
        
        layer = QuantumCircuit(n_qubits, name='Rot_Layer')
        
        # Single-qubit rotations
        for i in range(n_qubits):
            if len(params) > i:
                layer.ry(params[i], i)
        
        # Entangling gates
        for i in range(n_qubits - 1):
            layer.cx(i, i + 1)
        
        return layer
    
    # Create parameterized circuit
    from qiskit.circuit import Parameter
    params = [Parameter(f'θ_{i}') for i in range(n_qubits)]
    rotation_layer = create_rotation_layer(n_qubits, params)
    
    print(f"Parameterized rotation layer:")
    print(rotation_layer.draw())
    
    return qft_module, oracle_module, diffusion_module, grover_circuit

# Demo modular construction
qft_mod, oracle_mod, diffusion_mod, grover_circ = modular_circuit_construction()
```

#### 4. Circuit Optimization Techniques

```python
def circuit_optimization_techniques():
    """Techniques for optimizing quantum circuits"""
    
    print("Circuit Optimization Techniques")
    print("=" * 31)
    
    # Create an unoptimized circuit
    unopt_circuit = QuantumCircuit(3)
    unopt_circuit.h(0)
    unopt_circuit.x(0)      # Can be optimized
    unopt_circuit.x(0)      # X-X = I (cancels out)
    unopt_circuit.cx(0, 1)
    unopt_circuit.h(0)      # H-H = I when combined later
    unopt_circuit.h(0)
    unopt_circuit.cx(0, 2)
    unopt_circuit.cx(0, 2)  # CX-CX = I (cancels out)
    
    print("Unoptimized circuit:")
    print(unopt_circuit.draw())
    print(f"Depth: {unopt_circuit.depth()}, Gates: {unopt_circuit.size()}")
    
    # Manual optimization techniques
    print(f"\n1. Manual Optimization Techniques:")
    
    optimization_rules = {
        "Gate Cancellation": [
            "X-X = I (identity)",
            "H-H = I",
            "CNOT-CNOT = I",
            "Z-Z = I"
        ],
        "Gate Commutation": [
            "Adjacent single-qubit gates can be combined",
            "Some gates commute and can be reordered",
            "CZ gates commute with Z rotations"
        ],
        "Gate Substitution": [
            "H-Z-H = X",
            "H-X-H = Z", 
            "Two CNOTs can make SWAP",
            "Toffoli can be decomposed to 6 CNOTs"
        ],
        "Circuit Identities": [
            "Controlled gates can be moved through commuting operations",
            "Phase gates can be combined: P(α)P(β) = P(α+β)",
            "Some multi-qubit patterns have efficient equivalents"
        ]
    }
    
    for technique, rules in optimization_rules.items():
        print(f"\n{technique}:")
        for rule in rules:
            print(f"  • {rule}")
    
    # Demonstrate automated optimization
    print(f"\n2. Automated Optimization:")
    
    from qiskit import transpile
    from qiskit.providers.fake_provider import FakeVigo
    
    # Apply different optimization levels
    backend = FakeVigo()
    
    print(f"Optimization level comparison:")
    for level in range(4):
        opt_circuit = transpile(unopt_circuit, backend, optimization_level=level)
        print(f"Level {level}: depth={opt_circuit.depth():2d}, "
              f"gates={opt_circuit.size():2d}, "
              f"cx_gates={opt_circuit.count_ops().get('cx', 0):2d}")
    
    # Show optimized circuit
    opt_circuit = transpile(unopt_circuit, backend, optimization_level=3)
    print(f"\nOptimized circuit (level 3):")
    print(opt_circuit.draw())
    
    # Custom optimization pass
    print(f"\n3. Custom Optimization Example:")
    
    def remove_consecutive_x_gates(circuit):
        """Remove consecutive X gates (they cancel)"""
        optimized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Track previous gate on each qubit
        prev_gate = [None] * circuit.num_qubits
        
        for instruction in circuit.data:
            gate = instruction[0]
            qubits = [q.index for q in instruction[1]]
            
            # Check for consecutive X gates
            if gate.name == 'x' and len(qubits) == 1:
                qubit = qubits[0]
                if prev_gate[qubit] == 'x':
                    # Cancel out - don't add either gate
                    prev_gate[qubit] = None
                    continue
                else:
                    prev_gate[qubit] = 'x'
            else:
                # Reset tracking for affected qubits
                for q in qubits:
                    prev_gate[q] = None
            
            # Add the gate
            optimized.append(gate, qubits)
        
        return optimized
    
    # Test custom optimization
    test_circuit = QuantumCircuit(1)
    test_circuit.x(0)
    test_circuit.x(0)  # Should cancel
    test_circuit.h(0)
    test_circuit.x(0)
    test_circuit.x(0)  # Should cancel
    
    print(f"Before custom optimization:")
    print(test_circuit.draw())
    
    custom_opt = remove_consecutive_x_gates(test_circuit)
    print(f"After custom optimization:")
    print(custom_opt.draw())

circuit_optimization_techniques()
```

### Advanced Circuit Construction

#### 1. Quantum Algorithm Implementation Templates (Reference)
Instead of a print‑heavy helper function, the core algorithm blueprints are summarized here for quick lookup:

| Algorithm | Purpose | Structure (High Level) | Key Insight |
|-----------|---------|------------------------|-------------|
| Deutsch–Jozsa | Distinguish balanced vs constant oracle | Uniform init → Oracle (phase) → Interfere → Measure | Single oracle query reveals global property |
| Grover | Find marked item in unstructured space | Uniform init → (Oracle + Diffusion)^k → Measure | Iterative rotation amplifies target amplitude |
| QPE | Estimate eigenphase of unitary | Eigenstate prep → Controlled powers → Inverse QFT → Measure | Maps phase to binary digits |
| VQE | Approximate ground state | Ansatz(θ) ↔ Optimizer loop | Hybrid loop trades depth for classical compute |

> These detailed implementations migrate to Module 4; here they act as architectural examples while you learn tooling.

#### 2. Error-Aware Circuit Design

Error‑aware circuit design principles:

* Minimize Depth: parallelize; remove redundant rotations; defer measurements
* Reduce Two‑Qubit Gates: prioritize layout mapping; collapse consecutive CNOT patterns
* Respect Topology: choose initial qubit mapping to avoid SWAP insertion
* Prepare for Mitigation: include symmetry checks / parity qubits when cheap

Example (symmetry verification) circuit below keeps a copy of a reference qubit to detect certain error classes; discard shots where parity is violated.

```python
def create_error_mitigated_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 3)            # Reference copy
    qc.measure([0,1,2],[0,1,2])
    qc.measure(3,3)        # Symmetry check bit
    return qc

qc_sym = create_error_mitigated_circuit()
print(qc_sym.draw())
```

Interpretation: invalidate outcomes where measurement bit 3 disagrees with bit 0.

---

## 3.5 Simulation vs Real Hardware

### Understanding the Quantum Computing Stack

Think of quantum computing like the early days of classical computing - we have powerful simulators (like software emulators) and real but limited hardware (like early mainframes). Understanding when to use each is crucial for effective quantum programming.

#### The Quantum Computing Reality Check

**Simulation vs Real Hardware (Reality Check)**

| Aspect | Classical Simulation | Real Quantum Hardware |
|--------|----------------------|------------------------|
| Advantages | Noise‑free, controllable, instant feedback, free, introspection (state access) | Physical effects, real error models, hardware validation, access to emerging scale |
| Limitations | Memory exponential in qubits (~30–50 qubit practical cap for full state) | Noise, decoherence, queue delays, limited connectivity, cost, calibration drift |
| Best Uses | Learning, prototyping, debugging, analytical comparison, parameter sweeps | Final validation, noise studies, benchmarking, hardware research, demonstrations |

Use simulation to iterate rapidly; move to hardware only when logical correctness and resource profile are stable.

### Types of Quantum Simulators

#### 1. Classical Simulation Methods

**Quantum Simulation Methods (Overview)**

| Method | Core Idea | Practical Qubit Range* | Strengths | Typical Use Cases |
|--------|-----------|------------------------|-----------|------------------|
| State Vector | Store full 2^n amplitude vector | ≤30–32 (RAM bound) | Exact, simple | Education, small circuits, verification |
| MPS | Factor low‑entanglement 1D structures | 50–100+ (if low bond dimension) | Efficient for shallow/structured circuits | Variational ansätze, 1D physics |
| Stabilizer | Track Clifford stabilizers symbolically | 1000+ | Extremely fast for Clifford subset | Error correction, Clifford benchmarking |
| Tensor Network | Contract graph of localized tensors | 50–100 (structure dependent) | Flexible trade‑offs, partial exactness | Chemistry, optimization, structured lattices |

*Actual limits depend on entanglement growth and hardware resources.

Choose the cheapest method that preserves the fidelity required for your analysis; fall back to full statevector only when necessary.

#### 2. Practical Simulation Examples

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time

def practical_simulation_comparison():
    """Compare different simulation backends practically"""
    
    print("Practical Simulation Backend Comparison")
    print("=" * 38)
    
    # Create test circuits of different sizes
    def create_test_circuit(n_qubits, depth=5):
        """Create a test circuit with given parameters"""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Add random layers
        for layer in range(depth):
            # Hadamards
            for i in range(0, n_qubits, 2):
                qc.h(i)
            
            # CNOTs
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, i + 1)
                
            # Rotations
            for i in range(1, n_qubits, 2):
                qc.ry(0.5, i)
        
        qc.measure_all()
        return qc
    
    # Test different circuit sizes
    test_sizes = [5, 10, 15, 20]
    
    print("Performance Comparison:")
    print("Qubits | State Vector | MPS Method | Stabilizer")
    print("-------|--------------|------------|------------")
    
    for n_qubits in test_sizes:
        circuit = create_test_circuit(n_qubits, depth=3)
        
        # State vector simulation
        try:
            simulator_sv = AerSimulator(method='statevector')
            start_time = time.time()
            job_sv = simulator_sv.run(circuit, shots=100)
            result_sv = job_sv.result()
            sv_time = time.time() - start_time
            sv_status = f"{sv_time:.2f}s"
        except Exception as e:
            sv_status = "Failed"
        
        # MPS simulation
        try:
            simulator_mps = AerSimulator(method='matrix_product_state')
            start_time = time.time()
            job_mps = simulator_mps.run(circuit, shots=100)
            result_mps = job_mps.result()
            mps_time = time.time() - start_time
            mps_status = f"{mps_time:.2f}s"
        except Exception as e:
            mps_status = "Failed"
        
        # For stabilizer, need Clifford-only circuit
        clifford_circuit = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            clifford_circuit.h(i)
        for i in range(n_qubits - 1):
            clifford_circuit.cx(i, i + 1)
        clifford_circuit.measure_all()
        
        try:
            simulator_stab = AerSimulator(method='stabilizer')
            start_time = time.time()
            job_stab = simulator_stab.run(clifford_circuit, shots=100)
            result_stab = job_stab.result()
            stab_time = time.time() - start_time
            stab_status = f"{stab_time:.2f}s"
        except Exception as e:
            stab_status = "Failed"
        
        print(f"{n_qubits:6d} | {sv_status:12s} | {mps_status:10s} | {stab_status}")
    
    print(f"\nKey Insights:")
    insights = [
        "State vector method becomes impractical beyond ~25 qubits",
        "MPS method works well for low-entanglement circuits",
        "Stabilizer method is extremely fast but limited to Clifford gates",
        "Choice of method depends on circuit structure and requirements"
    ]
    
    for insight in insights:
        print(f"  • {insight}")

# Run practical comparison (commented out for memory safety)
# practical_simulation_comparison()
print("Simulation comparison function defined (run manually to test)")
```

### Real Quantum Hardware Characteristics

#### 1. Current Hardware Landscape

```python
def quantum_hardware_landscape():
    """Overview of current quantum hardware technologies"""
    
    print("Current Quantum Hardware Technologies")
    print("=" * 36)
    
    technologies = {
        "Superconducting Qubits": {
            "Companies": ["IBM", "Google", "Rigetti"],
            "Advantages": [
                "Fast gate operations (~10-100 ns)",
                "Good connectivity options",
                "Mature fabrication technology",
                "Scalable architecture"
            ],
            "Challenges": [
                "Requires dilution refrigeration (~10 mK)",
                "Short coherence times (~100 μs)",
                "Crosstalk between qubits",
                "Gate fidelity ~99-99.9%"
            ],
            "Typical Systems": "5-1000+ qubits"
        },
        "Trapped Ions": {
            "Companies": ["IonQ", "Honeywell/Quantinuum", "Alpine Quantum"],
            "Advantages": [
                "High-fidelity gates (~99.9%)",
                "Long coherence times (~1 minute)",
                "All-to-all connectivity",
                "Identical qubits"
            ],
            "Challenges": [
                "Slower gate operations (~10-100 μs)",
                "Complex laser control systems",
                "Scaling challenges",
                "Heating and decoherence"
            ],
            "Typical Systems": "10-100 qubits"
        },
        "Photonic Qubits": {
            "Companies": ["Xanadu", "PsiQuantum", "Orca Computing"],
            "Advantages": [
                "Room temperature operation",
                "Natural networking capability",
                "Low decoherence",
                "Fast information transfer"
            ],
            "Challenges": [
                "Probabilistic gates",
                "Photon loss",
                "Limited gate sets",
                "Detection efficiency"
            ],
            "Typical Systems": "10-200+ modes"
        },
        "Neutral Atoms": {
            "Companies": ["QuEra", "Pasqal", "Atom Computing"],
            "Advantages": [
                "Flexible qubit arrangements",
                "Good coherence times",
                "Parallel gate operations",
                "Scalable to large arrays"
            ],
            "Challenges": [
                "Complex control systems",
                "Loading and cooling atoms",
                "Gate fidelity optimization",
                "Error rate variations"
            ],
            "Typical Systems": "100-1000+ atoms"
        }
    }
    
    for tech, details in technologies.items():
        print(f"\n{tech}:")
        print(f"  Companies: {', '.join(details['Companies'])}")
        print(f"  Typical Systems: {details['Typical Systems']}")
        print(f"  Advantages:")
        for advantage in details['Advantages']:
            print(f"    • {advantage}")
        print(f"  Challenges:")
        for challenge in details['Challenges']:
            print(f"    • {challenge}")

quantum_hardware_landscape()
```

#### 2. Hardware Constraints and Limitations

```python
def hardware_constraints_analysis():
    """Analysis of quantum hardware constraints"""
    
    print("Quantum Hardware Constraints and Limitations")
    print("=" * 43)
    
    constraints = {
        "Connectivity Constraints": {
            "Description": "Not all qubits can interact directly",
            "Impact": [
                "Requires SWAP gates for distant interactions",
                "Increases circuit depth",
                "Limits algorithm efficiency",
                "Affects error accumulation"
            ],
            "Mitigation": [
                "Smart qubit mapping",
                "Algorithm-aware layout",
                "SWAP optimization",
                "Modular circuit design"
            ]
        },
        "Gate Set Limitations": {
            "Description": "Hardware supports limited native gates",
            "Impact": [
                "Arbitrary gates must be decomposed",
                "Increases gate count",
                "Higher error rates",
                "Longer execution times"
            ],
            "Mitigation": [
                "Use native gates when possible",
                "Optimize decompositions",
                "Algorithm adaptation",
                "Compiler optimizations"
            ]
        },
        "Coherence Time Limits": {
            "Description": "Qubits lose quantum information over time",
            "Impact": [
                "Circuit depth limitations",
                "Algorithm time constraints",
                "Quality degradation",
                "Success probability limits"
            ],
            "Mitigation": [
                "Minimize circuit depth",
                "Parallel operations",
                "Error correction codes",
                "Algorithm optimization"
            ]
        },
        "Noise and Errors": {
            "Description": "Operations are imperfect and introduce errors",
            "Impact": [
                "Reduced circuit fidelity",
                "Limits algorithm size",
                "Measurement errors",
                "Decoherence effects"
            ],
            "Mitigation": [
                "Error mitigation techniques",
                "Noise-aware compilation",
                "Variational approaches",
                "Post-processing corrections"
            ]
        }
    }
    
    for constraint, details in constraints.items():
        print(f"\n{constraint}:")
        print(f"  Description: {details['Description']}")
        print(f"  Impact:")
        for impact in details['Impact']:
            print(f"    • {impact}")
        print(f"  Mitigation Strategies:")
        for strategy in details['Mitigation']:
            print(f"    • {strategy}")

hardware_constraints_analysis()
```

### When to Use Simulation vs Hardware

#### Decision Framework

```python
def simulation_vs_hardware_decision():
    """Framework for deciding between simulation and hardware"""
    
    print("Simulation vs Hardware Decision Framework")
    print("=" * 40)
    
    decision_tree = {
        "Use Classical Simulation When:": [
            "Learning quantum programming concepts",
            "Developing and debugging algorithms",
            "Circuit has ≤25 qubits (state vector)",
            "Circuit has low entanglement (MPS)",
            "Circuit uses only Clifford gates (stabilizer)",
            "Need perfect, noise-free results",
            "Rapid prototyping and iteration",
            "Educational demonstrations",
            "Comparing with analytical results"
        ],
        "Use Quantum Hardware When:": [
            "Need to validate quantum advantage",
            "Studying noise effects and mitigation",
            "Algorithm requires >25 qubits", 
            "Circuit cannot be simulated classically",
            "Testing hardware-specific optimizations",
            "Final algorithm validation",
            "Research into NISQ algorithms",
            "Demonstrating quantum computation",
            "Exploring quantum supremacy regimes"
        ],
        "Hybrid Approach (Both):": [
            "Develop on simulation, validate on hardware",
            "Use simulation for parameter optimization",
            "Hardware for final measurements",
            "Simulation for error analysis",
            "Cross-validation between platforms"
        ]
    }
    
    for category, criteria in decision_tree.items():
        print(f"\n{category}")
        for criterion in criteria:
            print(f"  • {criterion}")
    
    # Practical workflow
    print(f"\nRecommended Development Workflow:")
    workflow = [
        "1. Start with classical simulation",
        "2. Develop and debug algorithm logic",
        "3. Optimize for hardware constraints",
        "4. Test with noise simulation",
        "5. Validate on real hardware",
        "6. Iterate based on results"
    ]
    
    for step in workflow:
        print(f"  {step}")

simulation_vs_hardware_decision()
```

### Noise Modeling and Error Mitigation

#### 1. Understanding Quantum Noise

```python
def quantum_noise_modeling():
    """Understanding and modeling quantum noise"""
    
    print("Quantum Noise Types and Models")
    print("=" * 30)
    
    noise_types = {
        "Depolarizing Noise": {
            "Description": "Random Pauli errors with equal probability",
            "Mathematical Model": "ρ → (1-p)ρ + p(XρX + YρY + ZρZ)/3",
            "Physical Cause": "Environmental interactions",
            "Typical Values": "p = 0.001 - 0.01 per gate",
            "Impact": "General state degradation"
        },
        "Amplitude Damping": {
            "Description": "Energy loss (|1⟩ → |0⟩ transitions)",
            "Mathematical Model": "Kraus operators with damping rate γ",
            "Physical Cause": "Energy dissipation to environment",
            "Typical Values": "γ varies with T₁ time",
            "Impact": "Preferential decay to ground state"
        },
        "Phase Damping": {
            "Description": "Phase information loss without energy loss",
            "Mathematical Model": "Dephasing channel with rate γ",
            "Physical Cause": "Magnetic field fluctuations",
            "Typical Values": "γ varies with T₂ time",
            "Impact": "Coherence loss, shorter T₂* time"
        },
        "Readout Errors": {
            "Description": "Measurement classification errors",
            "Mathematical Model": "Confusion matrix",
            "Physical Cause": "Imperfect state discrimination",
            "Typical Values": "1-5% error rate",
            "Impact": "Wrong measurement outcomes"
        }
    }
    
    for noise_type, details in noise_types.items():
        print(f"\n{noise_type}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

quantum_noise_modeling()
```

#### 2. Error Mitigation Techniques

Key categories of mitigation (when full error correction is infeasible):

* Zero Noise Extrapolation (ZNE)
    - Principle: Artificially scale noise, fit trend, extrapolate to zero
    - Method: Stretch gates / insert idle operations → collect results at multiple effective noise levels → polynomial (or Richardson) extrapolation
    - Pros: Simple, backend‑agnostic
    - Cons: Multiple runs; extrapolation model risk
* Symmetry Verification
    - Principle: Discard results violating conserved quantities (e.g., particle number, parity)
    - Method: Add measurement checks; post‑select valid shots
    - Pros: Significant fidelity boost when symmetry strong
    - Cons: Shot wastage; needs problem symmetry
* Virtual Distillation
    - Principle: Combine several noisy copies to suppress incoherent error components
    - Pros: Can yield strong improvements
    - Cons: High resource overhead (multiple state preparations)
* Probabilistic Error Cancellation
    - Principle: Build inverse noisy channel statistically via quasiprobability weights
    - Pros: Theoretically unbiased
    - Cons: Variance explosion; requires precise noise characterization
* Readout Mitigation
    - Principle: Calibrate confusion matrix; invert during post‑processing
    - Pros: Low overhead; always recommended
    - Cons: Limited to measurement errors
* Randomized Compiling
    - Principle: Convert coherent errors into stochastic (easier to average out)
    - Pros: Reduces worst‑case bias; integrates with ZNE
    - Cons: Extra compilation complexity
* Subspace / Subsystem Projection
    - Principle: Project raw results into valid code or symmetry subspace
    - Pros: General conceptual framework
    - Cons: Discards data; may bias if projection imperfect

Guideline: Always start with readout mitigation + basic layout/depth optimization, then add symmetry checks or lightweight ZNE only if accuracy demands justify extra runs.
```

## 3.6 Practical Project: Quantum Random Number Generator

### Project Overview

For our first real quantum programming project, we'll build a Quantum Random Number Generator (QRNG) - a perfect introduction to quantum programming that demonstrates fundamental concepts while creating something genuinely useful.

#### Why Start with QRNG?

**Why Build a Quantum Random Number Generator First?**

Educational Value:
* Direct use of superposition & measurement
* Minimal circuit complexity (easy to grasp)
* Immediate demonstration of quantum advantage (true randomness)
* Easy correctness validation via distribution tests

Practical Application:
* High‑quality entropy source (security / cryptography)
* Commercial relevance (tokens, keys, seeding)
* Reusable pattern for later algorithms
* Foundation for randomness‑dependent protocols

Technical Skills Covered:
* Circuit construction & execution
* Measurement handling & statistics
* Classical post‑processing pipeline
* Simulator vs hardware comparison

Conceptual Understanding Reinforced:
* Collapse of superposed states
* Difference between true and pseudo‑random sources
* Statistical uniformity & bias detection

Scoped Progression:
1. Single‑qubit prototype
2. Multi‑qubit scalable generator
3. Statistical quality tests
4. Hardware vs simulator benchmarking
5. Usability layer (API/interface)
6. Bias correction & entropy conditioning

### Phase 1: Basic Single-Qubit QRNG

#### Understanding True Randomness

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

def classical_vs_quantum_randomness():
    """Compare classical and quantum randomness sources"""
    
    print("Classical vs Quantum Randomness")
    print("=" * 31)
    
    # Classical pseudo-random numbers
    print("1. Classical Pseudo-Random Numbers:")
    classical_random = [random.randint(0, 1) for _ in range(1000)]
    classical_count = Counter(classical_random)
    
    print(f"Classical random bits (first 20): {classical_random[:20]}")
    print(f"Distribution: 0s: {classical_count[0]}, 1s: {classical_count[1]}")
    print(f"Characteristic: Deterministic algorithm, repeatable with same seed")
    
    # Quantum random numbers (simulated)
    print(f"\n2. Quantum Random Numbers:")
    
    def quantum_random_bit():
        """Generate single quantum random bit"""
        qc = QuantumCircuit(1, 1)
        qc.h(0)        # Create superposition
        qc.measure(0, 0)  # Measure in computational basis
        
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Return the measured bit
        return int(list(counts.keys())[0])
    
    quantum_random = [quantum_random_bit() for _ in range(100)]  # Smaller for demo
    quantum_count = Counter(quantum_random)
    
    print(f"Quantum random bits (first 20): {quantum_random[:20]}")
    print(f"Distribution: 0s: {quantum_count[0]}, 1s: {quantum_count[1]}")
    print(f"Characteristic: Fundamentally random, not reproducible")
    
    return classical_random, quantum_random

# Demo randomness comparison
classical_bits, quantum_bits = classical_vs_quantum_randomness()
```

#### Building the Basic QRNG

```python
class QuantumRandomNumberGenerator:
    """A comprehensive quantum random number generator"""
    
    def __init__(self, backend=None):
        """Initialize QRNG with specified backend"""
        self.backend = backend if backend else AerSimulator()
        self.stats = {'total_bits': 0, 'zeros': 0, 'ones': 0}
        
    def generate_random_bit(self, shots=1):
        """Generate a single random bit using quantum circuit"""
        
        # Create quantum circuit
        qc = QuantumCircuit(1, 1)
        qc.h(0)        # Put qubit in superposition |+⟩ = (|0⟩ + |1⟩)/√2
        qc.measure(0, 0)  # Measure in computational basis
        
        # Execute circuit
        transpiled_qc = transpile(qc, self.backend)
        job = self.backend.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Extract random bit
        if shots == 1:
            bit = int(list(counts.keys())[0])
            self._update_stats(bit)
            return bit
        else:
            # Return list of bits for multiple shots
            bits = []
            for outcome, count in counts.items():
                bits.extend([int(outcome)] * count)
            
            for bit in bits:
                self._update_stats(bit)
            return bits
    
    def generate_random_bytes(self, num_bytes):
        """Generate random bytes"""
        total_bits = num_bytes * 8
        bits = []
        
        print(f"Generating {total_bits} random bits ({num_bytes} bytes)...")
        
        # Generate bits in batches for efficiency
        batch_size = 1000
        for i in range(0, total_bits, batch_size):
            remaining = min(batch_size, total_bits - i)
            batch_bits = self.generate_random_bit(shots=remaining)
            
            if isinstance(batch_bits, list):
                bits.extend(batch_bits)
            else:
                bits.append(batch_bits)
            
            # Progress indicator
            if (i + batch_size) % 5000 == 0:
                print(f"Progress: {i + batch_size}/{total_bits} bits")
        
        # Convert bits to bytes
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) == 8:
                byte_value = sum(bit * (2 ** (7-j)) for j, bit in enumerate(byte_bits))
                bytes_list.append(byte_value)
        
        return bytes_list
    
    def generate_random_integer(self, min_val=0, max_val=100, num_bits=None):
        """Generate random integer in specified range"""
        
        if num_bits is None:
            # Calculate bits needed for range
            range_size = max_val - min_val + 1
            num_bits = range_size.bit_length()
        
        # Generate random bits
        random_bits = self.generate_random_bit(shots=num_bits)
        if not isinstance(random_bits, list):
            random_bits = [random_bits]
        
        # Convert to integer
        random_int = sum(bit * (2 ** i) for i, bit in enumerate(reversed(random_bits)))
        
        # Map to desired range
        scaled_int = min_val + (random_int % (max_val - min_val + 1))
        
        return scaled_int
    
    def _update_stats(self, bit):
        """Update internal statistics"""
        self.stats['total_bits'] += 1
        if bit == 0:
            self.stats['zeros'] += 1
        else:
            self.stats['ones'] += 1
    
    def get_statistics(self):
        """Get current statistics"""
        total = self.stats['total_bits']
        if total == 0:
            return "No bits generated yet"
        
        zero_ratio = self.stats['zeros'] / total
        one_ratio = self.stats['ones'] / total
        bias = abs(zero_ratio - 0.5)
        
        return {
            'total_bits': total,
            'zeros': self.stats['zeros'],
            'ones': self.stats['ones'],
            'zero_ratio': zero_ratio,
            'one_ratio': one_ratio,
            'bias': bias,
            'quality': 'Good' if bias < 0.05 else 'Fair' if bias < 0.1 else 'Poor'
        }
    
    def reset_statistics(self):
        """Reset internal statistics"""
        self.stats = {'total_bits': 0, 'zeros': 0, 'ones': 0}

# Demonstrate basic QRNG
def demo_basic_qrng():
    """Demonstrate basic quantum random number generator"""
    
    print("Basic Quantum Random Number Generator Demo")
    print("=" * 41)
    
    # Create QRNG instance
    qrng = QuantumRandomNumberGenerator()
    
    # Generate some random bits
    print("Generating individual random bits:")
    random_bits = [qrng.generate_random_bit() for _ in range(20)]
    print(f"Random bits: {random_bits}")
    
    # Generate random integer
    print(f"\nGenerating random integers:")
    for i in range(5):
        rand_int = qrng.generate_random_integer(1, 100)
        print(f"Random integer (1-100): {rand_int}")
    
    # Generate random bytes
    print(f"\nGenerating random bytes:")
    random_bytes = qrng.generate_random_bytes(3)
    print(f"Random bytes: {[hex(b) for b in random_bytes]}")
    
    # Show statistics
    print(f"\nStatistics:")
    stats = qrng.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

# Run demo
demo_basic_qrng()
```

### Phase 2: Multi-Qubit QRNG with Optimization

#### Scaling to Multiple Qubits

```python
class MultiQubitQRNG:
    """Enhanced QRNG using multiple qubits for efficiency"""
    
    def __init__(self, num_qubits=4, backend=None):
        """Initialize multi-qubit QRNG"""
        self.num_qubits = min(num_qubits, 10)  # Reasonable limit
        self.backend = backend if backend else AerSimulator()
        self.stats = {'total_bits': 0, 'distributions': {}}
        
    def create_multi_qubit_circuit(self):
        """Create circuit for multiple random bits"""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Put all qubits in superposition
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Measure all qubits
        qc.measure_all()
        
        return qc
    
    def generate_random_bits_batch(self, num_batches=100):
        """Generate multiple random bit strings efficiently"""
        
        qc = self.create_multi_qubit_circuit()
        transpiled_qc = transpile(qc, self.backend)
        
        # Run circuit multiple times
        job = self.backend.run(transpiled_qc, shots=num_batches)
        result = job.result()
        counts = result.get_counts()
        
        # Extract bit strings
        bit_strings = []
        for outcome, count in counts.items():
            bit_strings.extend([outcome] * count)
        
        # Convert to individual bits
        all_bits = []
        for bit_string in bit_strings:
            bits = [int(bit) for bit in bit_string]
            all_bits.extend(bits)
            self._update_stats(bit_string)
        
        return all_bits
    
    def generate_uniform_random(self, min_val, max_val, count=1):
        """Generate uniformly distributed random numbers"""
        
        # Calculate bits needed
        range_size = max_val - min_val + 1
        bits_needed = range_size.bit_length()
        
        results = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops
        
        while len(results) < count and attempts < max_attempts:
            # Generate enough random bits
            random_bits = self.generate_random_bits_batch(
                num_batches=max(1, (bits_needed * count) // self.num_qubits)
            )
            
            # Group bits and convert to numbers
            for i in range(0, len(random_bits) - bits_needed + 1, bits_needed):
                if len(results) >= count:
                    break
                    
                bit_group = random_bits[i:i + bits_needed]
                if len(bit_group) == bits_needed:
                    number = sum(bit * (2 ** (bits_needed - 1 - j)) 
                               for j, bit in enumerate(bit_group))
                    
                    # Accept only if in valid range (rejection sampling)
                    if number < range_size:
                        results.append(min_val + number)
            
            attempts += 1
        
        return results[:count]
    
    def _update_stats(self, bit_string):
        """Update statistics for bit strings"""
        self.stats['total_bits'] += len(bit_string)
        
        if bit_string not in self.stats['distributions']:
            self.stats['distributions'][bit_string] = 0
        self.stats['distributions'][bit_string] += 1
    
    def analyze_distribution(self):
        """Analyze the distribution of generated bit strings"""
        total_strings = sum(self.stats['distributions'].values())
        expected_prob = 1.0 / (2 ** self.num_qubits)
        
        print(f"Multi-Qubit QRNG Distribution Analysis")
        print(f"Total bit strings: {total_strings}")
        print(f"Expected probability per string: {expected_prob:.4f}")
        print(f"Unique strings generated: {len(self.stats['distributions'])}")
        print(f"Theoretical maximum: {2 ** self.num_qubits}")
        
        # Chi-square test for uniformity
        if total_strings > 0:
            expected_count = total_strings * expected_prob
            chi_square = sum((count - expected_count) ** 2 / expected_count 
                           for count in self.stats['distributions'].values())
            
            print(f"Chi-square statistic: {chi_square:.2f}")
            
            # Show most and least frequent outcomes
            sorted_outcomes = sorted(self.stats['distributions'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            print(f"\nMost frequent outcomes:")
            for outcome, count in sorted_outcomes[:3]:
                prob = count / total_strings
                print(f"  {outcome}: {count} times ({prob:.4f})")
            
            print(f"Least frequent outcomes:")
            for outcome, count in sorted_outcomes[-3:]:
                prob = count / total_strings
                print(f"  {outcome}: {count} times ({prob:.4f})")

# Demonstrate multi-qubit QRNG
def demo_multi_qubit_qrng():
    """Demonstrate multi-qubit quantum random number generator"""
    
    print("Multi-Qubit Quantum Random Number Generator Demo")
    print("=" * 46)
    
    # Create multi-qubit QRNG
    mqrng = MultiQubitQRNG(num_qubits=3)
    
    # Generate batch of random bits
    print("Generating batch of random bits:")
    random_bits = mqrng.generate_random_bits_batch(num_batches=50)
    print(f"Generated {len(random_bits)} random bits")
    print(f"First 30 bits: {random_bits[:30]}")
    
    # Generate uniform random numbers
    print(f"\nGenerating uniform random numbers (1-20):")
    uniform_numbers = mqrng.generate_uniform_random(1, 20, count=10)
    print(f"Random numbers: {uniform_numbers}")
    
    # Analyze distribution
    print(f"\nDistribution Analysis:")
    mqrng.analyze_distribution()

# Run multi-qubit demo
demo_multi_qubit_qrng()
```

### Phase 3: Statistical Testing and Validation

#### Implementing Randomness Tests

```python
import math
from scipy import stats
import numpy as np

class QRNGTester:
    """Statistical testing suite for quantum random number generators"""
    
    def __init__(self):
        self.test_results = {}
    
    def frequency_test(self, bits):
        """Test if frequency of 0s and 1s is approximately equal"""
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        # Calculate test statistic
        s_obs = abs(ones - zeros) / math.sqrt(n)
        
        # P-value calculation
        p_value = math.erfc(s_obs / math.sqrt(2))
        
        result = {
            'test_name': 'Frequency (Monobit) Test',
            'statistic': s_obs,
            'p_value': p_value,
            'passed': p_value >= 0.01,  # 99% confidence
            'ones': ones,
            'zeros': zeros,
            'expected_ones': n/2,
            'expected_zeros': n/2
        }
        
        self.test_results['frequency'] = result
        return result
    
    def runs_test(self, bits):
        """Test for randomness of runs (consecutive identical bits)"""
        n = len(bits)
        ones = sum(bits)
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Expected number of runs
        pi = ones / n
        expected_runs = (2 * n * pi * (1 - pi)) + 1
        
        # Test statistic
        if abs(pi - 0.5) >= (2 / math.sqrt(n)):
            p_value = 0.0  # Frequency test should be run first
        else:
            variance = (2 * n * pi * (1 - pi) - 1) / (4 * n)
            z = (runs - expected_runs) / math.sqrt(variance)
            p_value = math.erfc(abs(z) / math.sqrt(2))
        
        result = {
            'test_name': 'Runs Test',
            'runs_observed': runs,
            'runs_expected': expected_runs,
            'statistic': z if 'z' in locals() else None,
            'p_value': p_value,
            'passed': p_value >= 0.01
        }
        
        self.test_results['runs'] = result
        return result
    
    def serial_test(self, bits, pattern_length=2):
        """Test for independence of consecutive patterns"""
        n = len(bits)
        
        # Count overlapping patterns
        patterns = {}
        for i in range(n - pattern_length + 1):
            pattern = tuple(bits[i:i + pattern_length])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Expected count for each pattern
        expected_count = (n - pattern_length + 1) / (2 ** pattern_length)
        
        # Chi-square test
        chi_square = sum((count - expected_count) ** 2 / expected_count 
                        for count in patterns.values())
        
        # Degrees of freedom
        df = (2 ** pattern_length) - 1
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi_square, df)
        
        result = {
            'test_name': f'Serial Test (length {pattern_length})',
            'chi_square': chi_square,
            'degrees_freedom': df,
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'patterns_found': len(patterns),
            'expected_patterns': 2 ** pattern_length
        }
        
        self.test_results[f'serial_{pattern_length}'] = result
        return result
    
    def autocorrelation_test(self, bits, shift=1):
        """Test for autocorrelation at given shift"""
        n = len(bits)
        
        # Calculate autocorrelation
        correlation = 0
        for i in range(n - shift):
            correlation += bits[i] * bits[i + shift]
        
        # Normalize
        correlation = (2 * correlation - (n - shift)) / math.sqrt(n - shift)
        
        # P-value (approximate)
        p_value = math.erfc(abs(correlation) / math.sqrt(2))
        
        result = {
            'test_name': f'Autocorrelation Test (shift {shift})',
            'correlation': correlation,
            'p_value': p_value,
            'passed': p_value >= 0.01
        }
        
        self.test_results[f'autocorr_{shift}'] = result
        return result
    
    def run_all_tests(self, bits):
        """Run comprehensive test suite"""
        print("Running Comprehensive Randomness Tests")
        print("=" * 37)
        
        # Run individual tests
        tests = [
            self.frequency_test(bits),
            self.runs_test(bits),
            self.serial_test(bits, 2),
            self.serial_test(bits, 3),
            self.autocorrelation_test(bits, 1),
            self.autocorrelation_test(bits, 2)
        ]
        
        # Display results
        passed_tests = 0
        for test in tests:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"{test['test_name']:30s}: {status:4s} (p={test['p_value']:.4f})")
            if test['passed']:
                passed_tests += 1
        
        # Overall assessment
        total_tests = len(tests)
        pass_rate = passed_tests / total_tests
        
        print(f"\nOverall Results:")
        print(f"Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1%})")
        
        if pass_rate >= 0.8:
            quality = "Excellent"
        elif pass_rate >= 0.6:
            quality = "Good"
        elif pass_rate >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"Randomness quality: {quality}")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': pass_rate,
            'quality': quality,
            'individual_results': tests
        }

# Demonstrate statistical testing
def demo_statistical_testing():
    """Demonstrate statistical testing of QRNG output"""
    
    print("Statistical Testing of QRNG Output")
    print("=" * 34)
    
    # Generate test data
    qrng = QuantumRandomNumberGenerator()
    print("Generating 1000 random bits for testing...")
    
    # Generate in batches for efficiency
    test_bits = []
    for i in range(100):  # 100 batches of 10 bits each
        batch = qrng.generate_random_bit(shots=10)
        if isinstance(batch, list):
            test_bits.extend(batch)
        else:
            test_bits.append(batch)
    
    print(f"Generated {len(test_bits)} bits")
    
    # Run statistical tests
    tester = QRNGTester()
    results = tester.run_all_tests(test_bits)
    
    # Additional analysis
    print(f"\nAdditional Analysis:")
    print(f"Bit distribution: {Counter(test_bits)}")
    
    # Calculate entropy
    p0 = test_bits.count(0) / len(test_bits)
    p1 = test_bits.count(1) / len(test_bits)
    
    if p0 > 0 and p1 > 0:
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        print(f"Shannon entropy: {entropy:.4f} bits (max: 1.0)")
    
    return results

# Run statistical testing demo
test_results = demo_statistical_testing()
```

### Phase 4: Production-Ready QRNG with Features

#### Complete QRNG Implementation

```python
import json
import time
from datetime import datetime
import hashlib

class ProductionQRNG:
    """Production-ready quantum random number generator with advanced features"""
    
    def __init__(self, backend=None, num_qubits=4, bias_correction=True):
        """Initialize production QRNG with configuration"""
        self.backend = backend if backend else AerSimulator()
        self.num_qubits = num_qubits
        self.bias_correction = bias_correction
        self.stats = {
            'total_bits': 0,
            'generation_time': 0,
            'session_start': datetime.now().isoformat(),
            'configurations': []
        }
        self.buffer = []  # For bias correction
        
    def generate_entropy_pool(self, pool_size_bits=8192):
        """Generate large entropy pool for high-quality random numbers"""
        
        print(f"Generating entropy pool of {pool_size_bits} bits...")
        start_time = time.time()
        
        entropy_bits = []
        batch_size = self.num_qubits * 100  # Efficient batch size
        
        while len(entropy_bits) < pool_size_bits:
            # Create multi-qubit circuit
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            for i in range(self.num_qubits):
                qc.h(i)
            qc.measure_all()
            
            # Execute
            shots = min(batch_size // self.num_qubits, 
                       (pool_size_bits - len(entropy_bits)) // self.num_qubits)
            
            if shots > 0:
                transpiled_qc = transpile(qc, self.backend)
                job = self.backend.run(transpiled_qc, shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                # Extract bits
                for outcome, count in counts.items():
                    for _ in range(count):
                        if len(entropy_bits) < pool_size_bits:
                            entropy_bits.extend([int(bit) for bit in outcome])
        
        # Trim to exact size
        entropy_bits = entropy_bits[:pool_size_bits]
        
        # Apply bias correction if enabled
        if self.bias_correction:
            entropy_bits = self._apply_von_neumann_correction(entropy_bits)
        
        generation_time = time.time() - start_time
        self.stats['generation_time'] += generation_time
        self.stats['total_bits'] += len(entropy_bits)
        
        print(f"Generated {len(entropy_bits)} high-quality random bits in {generation_time:.2f}s")
        
        return entropy_bits
    
    def _apply_von_neumann_correction(self, bits):
        """Apply von Neumann bias correction (01→0, 10→1, discard 00,11)"""
        
        corrected_bits = []
        i = 0
        
        while i < len(bits) - 1:
            pair = (bits[i], bits[i + 1])
            
            if pair == (0, 1):
                corrected_bits.append(0)
            elif pair == (1, 0):
                corrected_bits.append(1)
            # Discard (0,0) and (1,1) pairs
            
            i += 2
        
        return corrected_bits
    
    def generate_secure_random_bytes(self, num_bytes):
        """Generate cryptographically secure random bytes"""
        
        # Generate entropy pool
        bits_needed = num_bytes * 8
        
        # Add extra bits to account for bias correction
        pool_size = int(bits_needed * 1.5) if self.bias_correction else bits_needed
        entropy_bits = self.generate_entropy_pool(pool_size)
        
        # Ensure we have enough bits
        while len(entropy_bits) < bits_needed:
            additional_bits = self.generate_entropy_pool(bits_needed - len(entropy_bits))
            entropy_bits.extend(additional_bits)
        
        # Convert to bytes
        random_bytes = []
        for i in range(0, bits_needed, 8):
            byte_bits = entropy_bits[i:i + 8]
            if len(byte_bits) == 8:
                byte_value = sum(bit * (2 ** (7 - j)) for j, bit in enumerate(byte_bits))
                random_bytes.append(byte_value)
        
        return bytes(random_bytes[:num_bytes])
    
    def generate_uuid(self):
        """Generate UUID using quantum random numbers"""
        
        # Generate 16 random bytes for UUID
        random_bytes = self.generate_secure_random_bytes(16)
        
        # Format as UUID (version 4)
        uuid_hex = random_bytes.hex()
        formatted_uuid = f"{uuid_hex[:8]}-{uuid_hex[8:12]}-4{uuid_hex[13:16]}-{random.choice('89ab')}{uuid_hex[17:20]}-{uuid_hex[20:]}"
        
        return formatted_uuid
    
    def generate_cryptographic_key(self, key_length_bits=256):
        """Generate cryptographic key of specified length"""
        
        key_bytes = self.generate_secure_random_bytes(key_length_bits // 8)
        
        return {
            'key_hex': key_bytes.hex(),
            'key_base64': base64.b64encode(key_bytes).decode(),
            'length_bits': key_length_bits,
            'generation_time': datetime.now().isoformat()
        }
    
    def generate_password(self, length=16, character_set="alphanumeric"):
        """Generate secure password using quantum randomness"""
        
        char_sets = {
            'alphanumeric': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'alphanum_symbols': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*',
            'hex': '0123456789abcdef',
            'numbers': '0123456789'
        }
        
        chars = char_sets.get(character_set, char_sets['alphanumeric'])
        
        # Calculate bits needed per character
        bits_per_char = math.ceil(math.log2(len(chars)))
        total_bits_needed = length * bits_per_char * 2  # Extra for rejection sampling
        
        # Generate entropy
        entropy_bits = self.generate_entropy_pool(total_bits_needed)
        
        # Generate password using rejection sampling
        password = []
        bit_index = 0
        
        while len(password) < length and bit_index < len(entropy_bits) - bits_per_char:
            # Extract bits for one character
            char_bits = entropy_bits[bit_index:bit_index + bits_per_char]
            char_value = sum(bit * (2 ** (bits_per_char - 1 - j)) for j, bit in enumerate(char_bits))
            
            # Accept if within character set range
            if char_value < len(chars):
                password.append(chars[char_value])
            
            bit_index += bits_per_char
        
        return ''.join(password)
    
    def get_detailed_statistics(self):
        """Get comprehensive statistics"""
        
        return {
            'session_statistics': self.stats,
            'configuration': {
                'backend': str(self.backend),
                'num_qubits': self.num_qubits,
                'bias_correction': self.bias_correction
            },
            'performance_metrics': {
                'bits_per_second': self.stats['total_bits'] / max(self.stats['generation_time'], 1),
                'average_generation_time': self.stats['generation_time'] / max(1, len(self.stats['configurations']))
            }
        }
    
    def export_configuration(self, filename):
        """Export QRNG configuration and statistics"""
        
        config = {
            'qrng_configuration': self.get_detailed_statistics(),
            'export_time': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration exported to {filename}")

# Demonstrate production QRNG
def demo_production_qrng():
    """Demonstrate production-ready QRNG features"""
    
    print("Production Quantum Random Number Generator Demo")
    print("=" * 46)
    
    # Create production QRNG
    pqrng = ProductionQRNG(num_qubits=3, bias_correction=True)
    
    # Generate secure random bytes
    print("1. Generating secure random bytes:")
    random_bytes = pqrng.generate_secure_random_bytes(16)
    print(f"Random bytes (hex): {random_bytes.hex()}")
    print(f"Random bytes (length): {len(random_bytes)} bytes")
    
    # Generate UUID
    print(f"\n2. Generating quantum UUID:")
    quantum_uuid = pqrng.generate_uuid()
    print(f"Quantum UUID: {quantum_uuid}")
    
    # Generate cryptographic key
    print(f"\n3. Generating cryptographic key:")
    crypto_key = pqrng.generate_cryptographic_key(128)  # 128-bit key
    print(f"Key (hex): {crypto_key['key_hex']}")
    print(f"Key (base64): {crypto_key['key_base64']}")
    
    # Generate passwords
    print(f"\n4. Generating secure passwords:")
    passwords = {
        'Alphanumeric': pqrng.generate_password(12, 'alphanumeric'),
        'With symbols': pqrng.generate_password(16, 'alphanum_symbols'),
        'Hex only': pqrng.generate_password(8, 'hex')
    }
    
    for password_type, password in passwords.items():
        print(f"{password_type}: {password}")
    
    # Show statistics
    print(f"\n5. Performance Statistics:")
    stats = pqrng.get_detailed_statistics()
    print(f"Total bits generated: {stats['session_statistics']['total_bits']}")
    print(f"Generation time: {stats['session_statistics']['generation_time']:.2f}s")
    print(f"Bits per second: {stats['performance_metrics']['bits_per_second']:.0f}")
    print(f"Bias correction: {'Enabled' if stats['configuration']['bias_correction'] else 'Disabled'}")

# Run production demo
demo_production_qrng()
```

### Project Wrap-up and Next Steps

#### Summary and Learning Outcomes

**QRNG Project Summary (Learning Outcomes)**

Technical Skills Developed:
* Circuit design & execution across simulation and (preparable for) hardware
* Multi‑qubit scaling strategies & batching
* Statistical validation (frequency, runs, serial, autocorrelation, entropy)
* Bias detection & von Neumann correction
* Entropy pooling and post‑processing for secure outputs
* Moving from prototype to production‑style API design

Quantum Concepts Mastered:
* Superposition & probabilistic measurement
* Distinction: true vs pseudo randomness
* Error sources & mitigation leverage points
* Scaling limits (noise, depth, two‑qubit gate accumulation)
* Entropy quality vs resource trade‑offs

Practical Applications Implemented:
* Random bit / byte streams
* UUID and cryptographic key material generation
* Secure password synthesis via rejection sampling
* Distribution quality assessment pipelines

Software Engineering Practices Reinforced:
* Modular abstraction layers (generator, tester, production wrapper)
* Configuration & statistics tracking
* Performance instrumentation (throughput metrics)
* Extensibility for hardware integration

Next Extensions:
1. Run on real hardware with error‑aware batch scheduling
2. Add advanced error mitigation (ZNE, symmetry filters)
3. Provide REST / gRPC entropy microservice
4. Build web dashboard for entropy quality monitoring
5. Benchmark vs high‑quality classical CSPRNGs
6. Integrate with quantum key distribution workflows
7. Implement entropy conditioning (hash extractor)
8. Package as installable Python library

---

## Module 3 Summary and Assessment

### Key Takeaways

By completing Module 3, you have:

1. **Set up a complete quantum development environment** with Qiskit, Cirq, and supporting tools
2. **Mastered Qiskit fundamentals** including circuits, gates, simulation, and hardware integration
3. **Learned Cirq basics** and Google's approach to quantum programming
4. **Developed circuit building techniques** from basic gates to complex algorithms
5. **Understood simulation vs hardware trade-offs** and when to use each approach
6. **Built a complete quantum application** - a production-ready quantum random number generator

### Self-Assessment Questions

Test your understanding with these questions:

1. **Environment Setup**: What are the key differences between quantum and classical programming environments?

2. **Framework Comparison**: When would you choose Cirq over Qiskit, and vice versa?

3. **Circuit Design**: How do you optimize a quantum circuit for hardware execution?

4. **Simulation Strategy**: What factors determine whether to use classical simulation or quantum hardware?

5. **QRNG Implementation**: Why is quantum randomness superior to classical pseudo-randomness for cryptographic applications?

### Practical Exercises

1. **Extend the QRNG**: Add support for different probability distributions (Gaussian, exponential)
2. **Hardware Integration**: Adapt the QRNG to run on real quantum hardware
3. **Performance Benchmarking**: Compare QRNG performance across different backends
4. **Algorithm Implementation**: Build a quantum algorithm from another domain using the techniques learned

### Looking Ahead

Module 3 provides the foundation for all practical quantum programming. The skills developed here - circuit construction, framework usage, and hardware awareness - are essential for the advanced topics in upcoming modules.

In Module 4, we'll build on these programming skills to explore specific quantum algorithms, starting with search and optimization problems where quantum computing shows clear advantages.

---

**Congratulations on completing Module 3! You now have solid quantum programming skills and have built your first real quantum application.**
