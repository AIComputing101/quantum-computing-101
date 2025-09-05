# Module 2: Mathematical Foundations
*Foundation Tier*

## Learning Objectives
By the end of this module, you will be able to:
- Understand linear algebra concepts through a programmer's lens
- Work with complex numbers and see why quantum computing needs them
- Grasp probability theory differences between classical and quantum
- Represent quantum states as vectors and visualize them
- Perform matrix operations on quantum gates
- Use interactive demos to visualize quantum states

## Prerequisites
- Completion of Module 1: Quantum Fundamentals
- Basic high school algebra
- Programming experience (any language)
- Willingness to think about math like data structures!

---

## 2.1 Linear Algebra Refresher: Vectors and Matrices for Programmers

### Why Linear Algebra for Quantum Computing?

If you're a programmer, you already know linear algebra - you just might not realize it! Every time you work with:
- **Arrays** → These are vectors
- **2D arrays/matrices** → These are matrices  
- **Coordinate systems** → These are vector spaces
- **Transformations in graphics** → These are matrix operations

Quantum computing uses the same mathematical tools, but for representing and manipulating quantum states instead of pixels or game objects.

> Developer intuition: Treat a quantum state like a structured data record whose fields (the amplitudes) must obey strict invariants (normalization) and are transformed only by special, reversible functions (unitary matrices). If you keep that framing, almost every math rule below will feel like a constraint system or type rule rather than abstract algebra.

### Vectors: The Building Blocks

We start with vectors because they are the simplest container that can hold a quantum state. A single qubit is a 2‑component vector; two qubits need 4 components; n qubits need 2ⁿ. Think of this as an exponentially growing array whose indices correspond to bit patterns (|00…0⟩ … |11…1⟩) and whose stored values are complex amplitudes. Operations we perform (gates) are just carefully structured transformations of that array.

#### What Programmers Already Know About Vectors:

```python
# You've seen vectors before - they're just lists of numbers!
position_2d = [3, 4]        # 2D position vector
rgb_color = [255, 128, 0]   # RGB color vector  
player_stats = [100, 50, 75, 90]  # Health, mana, strength, speed

# Vector operations you might know:
def add_vectors(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def scale_vector(vector, scalar):
    return [scalar * component for component in vector]

# Example: Adding velocities in a game
velocity1 = [5, 3]   # Moving right and up
velocity2 = [2, -1]  # Moving right and slightly down  
total_velocity = add_vectors(velocity1, velocity2)  # [7, 2]
print(f"Combined velocity: {total_velocity}")
```

#### Quantum States as Vectors:
Before we introduce code, pin this mental model:

* A classical bit has exactly one valid value at a time (0 or 1). We can encode that as a "one‑hot" vector: |0⟩ = [1, 0], |1⟩ = [0, 1].
* A qubit is allowed to be a weighted, complex combination of those basis vectors: |ψ⟩ = α|0⟩ + β|1⟩, stored as [α, β].
* The squares of the magnitudes (|α|² and |β|²) behave like probabilities and must sum to 1. This is exactly a data integrity constraint — you can think of normalization as a validation step after constructing or mutating a state.

So when you see a state vector below, read it as: "these are the amplitudes the system carries before measurement." Measurement will later collapse that structure to a single basis index according to those probabilities.

```python
import numpy as np
import math

# A classical bit as a vector
classical_0 = np.array([1, 0])  # "I'm definitely 0"
classical_1 = np.array([0, 1])  # "I'm definitely 1"

print("Classical states:")
print(f"|0⟩ = {classical_0}")
print(f"|1⟩ = {classical_1}")

# A quantum bit (qubit) in superposition as a vector
superposition = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
print(f"\nSuperposition state: {superposition}")
print("This represents: (1/√2)|0⟩ + (1/√2)|1⟩")

# The vector tells us the probability amplitudes!
prob_0 = abs(superposition[0])**2
prob_1 = abs(superposition[1])**2
print(f"Probability of measuring 0: {prob_0:.1%}")
print(f"Probability of measuring 1: {prob_1:.1%}")
```

### Vector Properties That Matter for Quantum:

#### 1. Vector Length (Normalization)
Why this matters: If the vector were not normalized, its squared magnitudes would not sum to 1 and we would lose the interpretation of components as probability amplitudes. Practically, many algorithms apply a sequence of gates assuming the invariant still holds; breaking it (e.g., via numerical drift) invalidates downstream expectations much like mutating shared state without updating dependent caches.
```python
def vector_length(vector):
    """Calculate the length (magnitude) of a vector"""
    return math.sqrt(sum(component**2 for component in vector))

def normalize_vector(vector):
    """Make a vector have length 1"""
    length = vector_length(vector)
    return [component/length for component in vector]

# Quantum states must always have length 1!
unnormalized = [3, 4]  # Length = 5
normalized = normalize_vector(unnormalized)
print(f"Unnormalized: {unnormalized}, length = {vector_length(unnormalized)}")
print(f"Normalized: {normalized}, length = {vector_length(normalized):.3f}")

# Why? Because probabilities must add up to 1
print(f"Probability sum: {sum(abs(x)**2 for x in normalized):.3f}")
```

#### 2. Vector Addition (Superposition)
When we add (and re‑normalize) basis vectors we are constructing a state that encodes *potential* outcomes. This is not the same as classical randomness (which would require hidden information); instead, both possibilities co‑exist until a measurement forces a choice. For developers: superposition is like a lazily evaluated branching structure that is kept in a compressed form until an observation requires resolving one branch.
```python
# In graphics: Adding vectors gives combined motion
# In quantum: Adding state vectors gives superposition

# Two possible quantum states
state_0 = np.array([1, 0])  # |0⟩
state_1 = np.array([0, 1])  # |1⟩

# Create superposition by adding them
superposition = (state_0 + state_1) / math.sqrt(2)
print(f"Superposition: {superposition}")
print("This is like being in both states simultaneously!")

# Visualization as arrows in 2D space
print("\nThink of quantum states as arrows pointing in 2D space:")
print("↑ (up) = |0⟩")
print("→ (right) = |1⟩") 
print("↗ (diagonal) = superposition of |0⟩ and |1⟩")
```

#### 3. Inner Product (Dot Product)
Interpretation tip: The inner product ⟨φ|ψ⟩ is a *similarity score* between two quantum states. Its magnitude squared gives the probability that a measurement designed to test for |φ⟩ will affirm when the actual state is |ψ⟩. You can treat it like a projection or cosine similarity from ML — except now the vectors can be complex.
```python
def dot_product(v1, v2):
    """Calculate dot product of two vectors"""
    return sum(v1[i] * v2[i] for i in range(len(v1)))

# In graphics: Dot product tells you angle between vectors
# In quantum: Dot product tells you probability of transition

state_0 = np.array([1, 0])
state_1 = np.array([0, 1])
superposition = np.array([1/math.sqrt(2), 1/math.sqrt(2)])

print("Quantum inner products (overlaps):")
print(f"⟨0|0⟩ = {dot_product(state_0, state_0)}")  # 1 (same state)
print(f"⟨0|1⟩ = {dot_product(state_0, state_1)}")  # 0 (orthogonal)
print(f"⟨0|+⟩ = {dot_product(state_0, superposition):.3f}")  # 0.707 (45° angle)
```

### Multi-Qubit Systems as Higher-Dimensional Vectors:
Scaling intuition: Instead of storing one boolean, you now conceptually store amplitudes for every possible bit string of length n. This is why simulation cost explodes: you cannot (in the general case) factor the full vector into smaller independent pieces once entanglement appears. The tensor product (shown later) is how we *build* composite states; entanglement is what prevents us from *decomposing* them back trivially.

```python
# One qubit = 2D vector
one_qubit = np.array([1, 0])  # 2 components
print(f"1 qubit state: {one_qubit} (2D vector)")

# Two qubits = 4D vector  
two_qubits = np.array([1, 0, 0, 0])  # 4 components
print(f"2 qubit state |00⟩: {two_qubits} (4D vector)")

# Three qubits = 8D vector
three_qubits = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # 8 components
print(f"3 qubit state |000⟩: {three_qubits} (8D vector)")

# The pattern: n qubits = 2^n dimensional vector
def quantum_vector_size(num_qubits):
    return 2**num_qubits

print("\nQuantum state vector sizes:")
for qubits in range(1, 11):
    size = quantum_vector_size(qubits)
    print(f"{qubits} qubit(s): {size:,} dimensional vector")
    
print("\n50 qubits would need a vector with 1,125,899,906,842,624 components!")
print("This is why quantum simulation becomes impossible on classical computers!")
```

### Matrices: Quantum Operations

Matrices are to quantum states what pure functions are to immutable data: they map one valid state to another while preserving invariants (normalization and inner products). In quantum mechanics those matrices must be *unitary* (their inverse is their conjugate transpose) — guaranteeing reversibility. Unlike many classical transformations (e.g., ReLU in neural nets) nothing here "throws information away"; it just redistributes amplitude and phase.

#### What Programmers Know About Matrices:

```python
# 2D arrays are matrices!
game_board = [
    [1, 0, 1],
    [0, 1, 0], 
    [1, 1, 0]
]

# Transformation matrices in graphics
rotation_90 = [
    [0, -1],
    [1,  0]
]

# Apply transformation to a point
def matrix_vector_multiply(matrix, vector):
    """Multiply matrix by vector"""
    result = []
    for row in matrix:
        result.append(sum(row[i] * vector[i] for i in range(len(vector))))
    return result

point = [3, 4]
rotated = matrix_vector_multiply(rotation_90, point)
print(f"Point {point} rotated 90°: {rotated}")
```

#### Quantum Gates as Matrices:
Reading a gate matrix: each column shows how that gate transforms a basis vector. For example, the X gate swaps the basis columns (so |0⟩ → |1⟩ and |1⟩ → |0⟩). The Hadamard (H) gate spreads amplitude evenly while introducing relative phase (a sign) so it both creates and interferes superpositions in later compositions.

```python
import numpy as np

# Quantum gates are just 2x2 matrices (for single qubits)!

# Pauli-X gate (quantum NOT gate)
X_gate = np.array([
    [0, 1],
    [1, 0]
])

# Hadamard gate (creates superposition)  
H_gate = np.array([
    [1/math.sqrt(2),  1/math.sqrt(2)],
    [1/math.sqrt(2), -1/math.sqrt(2)]
])

# Pauli-Z gate (phase flip)
Z_gate = np.array([
    [1,  0],
    [0, -1]
])

print("Quantum gate matrices:")
print(f"X (NOT) gate:\n{X_gate}")
print(f"\nH (Hadamard) gate:\n{H_gate}")
print(f"\nZ (phase) gate:\n{Z_gate}")
```

#### Applying Gates to States:
Mechanically this is plain matrix @ vector multiplication. Conceptually it is a reversible redistribution of the "probability mass" across basis states, with phase adjustments encoded by sign changes or complex factors. Because gates compose by matrix multiplication, circuit synthesis reduces to picking an ordered list of matrices whose product implements the algorithmic transformation you need.

```python
# Apply gates by matrix multiplication
state_0 = np.array([1, 0])  # |0⟩
state_1 = np.array([0, 1])  # |1⟩

# Apply X gate (NOT operation)
result_x0 = X_gate @ state_0  # Matrix multiplication
result_x1 = X_gate @ state_1

print("X gate application:")
print(f"X|0⟩ = {result_x0} = |1⟩")
print(f"X|1⟩ = {result_x1} = |0⟩")

# Apply Hadamard gate (create superposition)
result_h0 = H_gate @ state_0
result_h1 = H_gate @ state_1

print("\nHadamard gate application:")
print(f"H|0⟩ = {result_h0}")
print(f"H|1⟩ = {result_h1}")
print("Both create superposition states!")
```

### Linear Algebra Cheat Sheet for Quantum:
Use this small utility class when prototyping ideas outside a full framework (like Qiskit). Keeping these helpers close reinforces the invariant mindset: normalize early, treat amplitudes carefully, and always compute probabilities from magnitudes squared — never by summing raw complex parts.

```python
# Essential operations for quantum computing
class QuantumMath:
    
    @staticmethod
    def normalize(vector):
        """Ensure quantum state has probability 1"""
        norm = np.linalg.norm(vector)
        return vector / norm
    
    @staticmethod
    def probability(amplitude):
        """Get measurement probability from amplitude"""
        return abs(amplitude)**2
    
    @staticmethod
    def apply_gate(gate_matrix, state_vector):
        """Apply quantum gate to quantum state"""
        return gate_matrix @ state_vector
    
    @staticmethod  
    def tensor_product(state1, state2):
        """Combine two quantum states (for multi-qubit systems)"""
        return np.kron(state1, state2)

# Example usage
qm = QuantumMath()

# Create and normalize a state
unnormalized = np.array([3, 4])
normalized = qm.normalize(unnormalized)
print(f"Normalized state: {normalized}")

# Calculate probabilities
prob_0 = qm.probability(normalized[0])
prob_1 = qm.probability(normalized[1])
print(f"Probabilities: {prob_0:.3f}, {prob_1:.3f}")

# Apply Hadamard gate
superposition = qm.apply_gate(H_gate, np.array([1, 0]))
print(f"After Hadamard: {superposition}")
```

---

## 2.2 Complex Numbers: Why Quantum Needs Imaginary Friends

If linear algebra gave us *structure*, complex numbers give us *behavior*. The imaginary unit introduces a second dimension (phase) perpendicular to magnitude that lets different computational paths cancel or reinforce each other. Without phase, we'd only have non‑negative probabilities that could never destructively interfere — eliminating a core quantum advantage.

### What Are Complex Numbers? (A Programmer's Introduction)

If you've ever worked with 2D graphics or signal processing, you've essentially worked with complex numbers without knowing it!

```python
# Complex numbers are just 2D coordinates in disguise
import cmath

# A complex number has two parts: real and imaginary
z = 3 + 4j  # Python uses 'j' for imaginary unit (mathematicians use 'i')

print(f"Complex number: {z}")
print(f"Real part: {z.real}")      # 3
print(f"Imaginary part: {z.imag}")  # 4

# Think of it as a 2D point
print(f"As coordinates: ({z.real}, {z.imag})")
```

#### Visual Representation:
Think of the complex plane like a 2D coordinate system where radius = magnitude and angle = phase. For quantum developers:
* Magnitude → contributes to measurement probability (after squaring)
* Phase → latent metadata influencing how amplitudes combine later
Two states with identical magnitudes but different phases produce the same *single* measurement statistics but can lead to totally different downstream interference once more gates act.

```python
# Complex numbers live on a 2D plane
print("Complex plane visualization:")
print("     Imaginary axis")
print("          ↑")
print("          |")
print("    2i ---+--- (real part)")
print("          |")
print("          |")
print("----------+----------→ Real axis")
print("          |")
print("         -2i")

# Examples plotted
examples = [
    (1 + 0j, "real number"),
    (0 + 1j, "purely imaginary"), 
    (3 + 4j, "general complex"),
    (1 + 1j, "45° angle"),
    (-1 + 0j, "negative real")
]

print("\nComplex number examples:")
for num, description in examples:
    print(f"{num}: {description}")
```

### Why Quantum Computing Needs Complex Numbers:
High‑level rationale:
1. Phases encode timing/orientation so multiple evolution paths can add or cancel.
2. Euler's formula e^{iθ} allows rotational gates to be expressed compactly as exponentials (critical for hardware calibration and algorithm design).
3. Complex conjugation underlies inner products, ensuring probabilities remain real and non‑negative.

#### 1. Amplitude and Phase Information
Global vs relative phase: Multiplying an entire state vector by e^{iφ} (a *global* phase) yields an observationally indistinguishable state — all measurement probabilities remain identical. Only *relative* phase between components (e.g., α vs β) affects interference. Treat global phase like a harmless formatting change; treat relative phase like semantic meaning.

```python
import math

# Real numbers can only store magnitude
real_amplitude = 0.707  # Just tells us "how much"

# Complex numbers store magnitude AND direction
complex_amplitude = 0.707 + 0j      # Points right (0° phase)
complex_amplitude2 = 0 + 0.707j     # Points up (90° phase)  
complex_amplitude3 = -0.707 + 0j    # Points left (180° phase)

print("Complex amplitudes store direction (phase):")
for i, amp in enumerate([complex_amplitude, complex_amplitude2, complex_amplitude3]):
    magnitude = abs(amp)
    phase = cmath.phase(amp) * 180 / math.pi
    print(f"Amplitude {i+1}: magnitude={magnitude:.3f}, phase={phase:.1f}°")
```

#### 2. Quantum Interference
Interference is the computational lever: algorithms engineer constructive interference toward correct answers and destructive interference away from incorrect ones. Code below shows addition and cancellation; scale that pattern up with carefully chosen gate sequences and you get Grover's quadratic speedup or the period-finding at Shor's core.

```python
# Complex numbers enable quantum interference
def quantum_interference_demo():
    """Show how complex numbers create interference"""
    
    # Two paths a quantum particle can take
    path1_amplitude = 0.5 + 0.5j     # Path 1: 45° phase
    path2_amplitude = 0.5 - 0.5j     # Path 2: -45° phase
    
    # Constructive interference (amplitudes add)
    constructive = path1_amplitude + path1_amplitude
    print(f"Constructive interference: {constructive}")
    print(f"Probability: {abs(constructive)**2:.3f}")
    
    # Destructive interference (amplitudes cancel)
    destructive = path1_amplitude + (-path1_amplitude)
    print(f"Destructive interference: {destructive}")
    print(f"Probability: {abs(destructive)**2:.3f}")
    
    print("This is impossible with just real numbers!")

quantum_interference_demo()
```

#### 3. Rotation and Phase Gates
Phase and rotation gates manipulate only the angle of amplitude vectors in the complex plane. Even when a probability (|amplitude|²) does not change immediately, the *future* probability distribution can change after combining with other paths. Think of it like scheduling relative timing offsets in a distributed system so messages (amplitudes) collide constructively later.

```python
# Phase gates rotate quantum states in complex plane
def phase_rotation_demo():
    """Show how complex numbers enable phase rotations"""
    
    # A quantum state
    initial_state = 1 + 0j  # Real, pointing right
    
    # Rotation by different angles
    angles = [0, 90, 180, 270]
    
    print("Rotating quantum state:")
    for angle in angles:
        # Convert angle to radians and create rotation
        rad = math.radians(angle)
        rotation = cmath.exp(1j * rad)  # e^(i*θ) = cos(θ) + i*sin(θ)
        rotated_state = initial_state * rotation
        
        print(f"{angle:3d}°: {rotated_state:.3f}")
        
    print("These rotations represent different quantum phases!")

phase_rotation_demo()
```

### Complex Number Operations for Quantum:
You rarely implement these from scratch in production (libraries handle it), but seeing them explicitly helps map mental models: multiplication combines rotations (adds phases) and scales magnitudes, conjugation mirrors across the real axis (used in inner products), and polar form exposes the separation of concerns (magnitude vs phase) crucial for reasoning about interference.

#### Basic Operations:

```python
# Addition and subtraction
z1 = 3 + 4j
z2 = 1 + 2j

addition = z1 + z2
subtraction = z1 - z2

print(f"Addition: {z1} + {z2} = {addition}")
print(f"Subtraction: {z1} - {z2} = {subtraction}")

# Multiplication (important for quantum gates!)
multiplication = z1 * z2
print(f"Multiplication: {z1} * {z2} = {multiplication}")

# Complex conjugate (flips imaginary part sign)
conjugate = z1.conjugate()
print(f"Conjugate: {z1}* = {conjugate}")

# Magnitude (absolute value)
magnitude = abs(z1)
print(f"Magnitude: |{z1}| = {magnitude:.3f}")
```

#### Polar Form (Magnitude and Phase):

```python
def complex_to_polar(z):
    """Convert complex number to polar form"""
    magnitude = abs(z)
    phase = cmath.phase(z)
    return magnitude, phase

def polar_to_complex(magnitude, phase):
    """Convert polar form to complex number"""
    return magnitude * cmath.exp(1j * phase)

# Example conversion
z = 3 + 4j
mag, phase = complex_to_polar(z)

print(f"Rectangular: {z}")
print(f"Polar: magnitude={mag:.3f}, phase={phase:.3f} rad ({math.degrees(phase):.1f}°)")

# Convert back
z_reconstructed = polar_to_complex(mag, phase)
print(f"Reconstructed: {z_reconstructed}")
```

### Euler's Formula: The Bridge Between Exponentials and Trigonometry
Takeaway: rotations, oscillations, and phase shifts all unify under e^{iθ}. Quantum Hamiltonians exponentiate to unitary evolution operators using exactly this identity, so mastering Euler's formula repays attention later when reading algorithm derivations or hardware calibration notes.

```python
# Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
# This is the mathematical foundation of quantum rotations!

def eulers_formula_demo():
    """Demonstrate Euler's formula"""
    
    angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
    
    print("Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)")
    print("θ (rad)  | θ (deg) | e^(iθ)           | cos(θ)+i*sin(θ)")
    print("-" * 60)
    
    for theta in angles:
        exponential = cmath.exp(1j * theta)
        trigonometric = math.cos(theta) + 1j * math.sin(theta)
        degrees = math.degrees(theta)
        
        print(f"{theta:7.3f} | {degrees:7.1f} | {exponential:12.3f} | {trigonometric:12.3f}")

eulers_formula_demo()

# This formula is why quantum gates can be written as exponentials!
print(f"\nQuantum rotation gate: R(θ) = e^(iθ) rotates by angle θ")
print(f"Hadamard gate involves rotations by π/4 and 3π/4")
```

### Complex Numbers in Quantum State Representation:
Notice how only the *phase difference* between components changes interference potential. The examples juxtapose real, sign‑flipped, and imaginary second components — each yields identical normalization yet encodes different future interaction behavior with gates like Hadamard or CNOT.

```python
# Quantum states with complex amplitudes
import numpy as np

def quantum_state_examples():
    """Show quantum states with complex amplitudes"""
    
    # Various quantum states
    states = {
        "Real superposition": np.array([1/math.sqrt(2), 1/math.sqrt(2)]),
        "Complex superposition": np.array([1/math.sqrt(2), 1j/math.sqrt(2)]),
        "Phase difference": np.array([1/math.sqrt(2), -1/math.sqrt(2)]),
        "General complex": np.array([0.6, 0.8j])
    }
    
    print("Quantum states with complex amplitudes:")
    print("-" * 50)
    
    for name, state in states.items():
        print(f"\n{name}:")
        print(f"State vector: {state}")
        
        # Calculate probabilities
        prob_0 = abs(state[0])**2
        prob_1 = abs(state[1])**2
        
        print(f"P(0) = |{state[0]}|² = {prob_0:.3f}")
        print(f"P(1) = |{state[1]}|² = {prob_1:.3f}")
        print(f"Total probability: {prob_0 + prob_1:.3f}")
        
        # Calculate phases
        phase_0 = cmath.phase(state[0]) * 180 / math.pi
        phase_1 = cmath.phase(state[1]) * 180 / math.pi
        
        print(f"Phase(0): {phase_0:.1f}°, Phase(1): {phase_1:.1f}°")

quantum_state_examples()
```

### Why Complex Numbers Make Quantum "Work":
Below is a quick list rendered programmatically, but conceptually emphasize that complex phase is *not* an implementation detail — it's the substrate enabling algorithmic amplification/suppression patterns.

```python
# The key insights about complex numbers in quantum mechanics
insights = [
    "Enable quantum interference (constructive and destructive)",
    "Store both probability and phase information",
    "Allow continuous rotations in quantum space", 
    "Make quantum gates reversible (unitary transformations)",
    "Enable quantum algorithms like Shor's and Grover's",
    "Provide the mathematical foundation for quantum field theory"
]

print("Why quantum computing needs complex numbers:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

print("\nWithout complex numbers:")
print("- No quantum interference → No quantum speedup")
print("- No phase information → No quantum algorithms")  
print("- No continuous rotations → Limited quantum operations")
print("- Quantum mechanics would be fundamentally different!")
```

### Complex Numbers Cheat Sheet for Quantum:
Practical tip: When debugging unexpected probabilities, first print amplitudes in polar form (magnitude & phase). Often an "incorrect probability" stems from an unintended relative phase introduced earlier rather than a raw magnitude bug.

```python
class ComplexQuantumMath:
    """Useful complex number operations for quantum computing"""
    
    @staticmethod
    def amplitude_to_probability(amplitude):
        """Convert complex amplitude to measurement probability"""
        return abs(amplitude)**2
    
    @staticmethod
    def normalize_amplitudes(amplitudes):
        """Normalize complex amplitudes so probabilities sum to 1"""
        total_prob = sum(abs(amp)**2 for amp in amplitudes)
        normalization = math.sqrt(total_prob)
        return [amp / normalization for amp in amplitudes]
    
    @staticmethod
    def quantum_phase_gate(angle):
        """Create a phase rotation gate matrix"""
        return np.array([
            [1, 0],
            [0, cmath.exp(1j * angle)]
        ])
    
    @staticmethod
    def global_phase(state, phase):
        """Apply global phase to quantum state"""
        phase_factor = cmath.exp(1j * phase)
        return [phase_factor * amplitude for amplitude in state]

# Example usage
cqm = ComplexQuantumMath()

# Complex amplitudes
amplitudes = [0.6 + 0.8j, 0.3 - 0.4j]
normalized = cqm.normalize_amplitudes(amplitudes)
print(f"Normalized amplitudes: {normalized}")

# Create phase gate
phase_gate = cqm.quantum_phase_gate(math.pi/4)  # 45° rotation
print(f"Phase gate matrix:\n{phase_gate}")

# Apply global phase
state = [1/math.sqrt(2), 1/math.sqrt(2)]
phased_state = cqm.global_phase(state, math.pi/2)
print(f"State with global phase: {phased_state}")
```

---

## 2.3 Probability Theory: From Classical to Quantum Probabilities

Probability in quantum land is *derived* not *stored*. We never carry a probability table inside the state vector; we carry amplitudes and extract probabilities only when needed (e.g., predicting measurement, computing expectation values). This indirection is what allows interference to reshape outcome likelihoods mid‑algorithm — something classical probability tables cannot emulate efficiently.

### Classical Probability (What You Already Know)
We start grounded: classical probability is frequency or belief about mutually exclusive outcomes. All rules you know (add, multiply, conditional) still apply — quantum doesn't throw them away; it nests them underneath amplitude algebra via the Born rule.

Probability in classical computing and everyday life follows intuitive rules:

```python
import random
import matplotlib.pyplot as plt
from collections import Counter

# Classical coin flip
def classical_coin_flip(num_flips=1000):
    """Simulate classical coin flips"""
    results = []
    for _ in range(num_flips):
        result = random.choice(['H', 'T'])  # Heads or Tails
        results.append(result)
    
    counts = Counter(results)
    prob_heads = counts['H'] / num_flips
    prob_tails = counts['T'] / num_flips
    
    print(f"Classical coin flip results ({num_flips} flips):")
    print(f"Heads: {counts['H']} ({prob_heads:.1%})")
    print(f"Tails: {counts['T']} ({prob_tails:.1%})")
    
    return results

# Classical probability rules
def classical_probability_rules():
    """Demonstrate classical probability rules"""
    
    print("Classical Probability Rules:")
    print("1. Probabilities are real numbers between 0 and 1")
    print("2. All probabilities sum to 1")
    print("3. P(A or B) = P(A) + P(B) - P(A and B)")
    print("4. Independent events: P(A and B) = P(A) × P(B)")
    
    # Example: Rolling two dice
    outcomes = []
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            total = die1 + die2
            outcomes.append(total)
    
    counts = Counter(outcomes)
    print(f"\nTwo dice example:")
    print(f"P(sum=7) = {counts[7]/36:.3f}")
    print(f"P(sum=2) = {counts[2]/36:.3f}")
    print(f"P(sum=12) = {counts[12]/36:.3f}")

classical_coin_flip()
classical_probability_rules()
```

### Quantum Probability: The Weird Cousin
Key shift: instead of summing probabilities directly, we sum *amplitudes* (complex numbers) for indistinguishable paths and then square magnitude. This "sum‑then‑square" pipeline unlocks new constructive/destructive patterns. Treat the provided comparison table as a translation dictionary between mental models.

Quantum probability behaves very differently:

```python
import numpy as np
import cmath

def quantum_vs_classical_comparison():
    """Compare classical and quantum probability"""
    
    print("CLASSICAL vs QUANTUM Probability:")
    print("=" * 50)
    
    comparisons = [
        ("Probabilities", "Real numbers (0 to 1)", "Come from complex amplitudes"),
        ("Superposition", "One outcome at a time", "Multiple outcomes simultaneously"),
        ("Measurement", "Reveals existing value", "Creates the outcome"),
        ("Interference", "Not possible", "Constructive and destructive"),
        ("Combination rule", "P(A or B) = P(A) + P(B)", "Add amplitudes, then square"),
        ("Information", "Can be copied", "Cannot be cloned"),
        ("Correlation", "Local (hidden variables)", "Non-local (entanglement)")
    ]
    
    print(f"{'Property':<15} | {'Classical':<25} | {'Quantum'}")
    print("-" * 70)
    for prop, classical, quantum in comparisons:
        print(f"{prop:<15} | {classical:<25} | {quantum}")

quantum_vs_classical_comparison()
```

### Amplitude vs Probability:
Guideline: Never square intermediate results twice. Compute probabilities only at the measurement boundary or when deriving expectation values. Keeping data in amplitude form preserves phase information for later steps.

The key difference is that quantum mechanics uses **probability amplitudes** (complex numbers) instead of direct probabilities:

```python
def amplitude_vs_probability():
    """Show the relationship between amplitudes and probabilities"""
    
    print("Quantum Amplitudes → Probabilities")
    print("=" * 40)
    
    # Various amplitude examples
    amplitudes = [
        (1 + 0j, "Real positive"),
        (-1 + 0j, "Real negative"),  
        (0 + 1j, "Imaginary positive"),
        (0 - 1j, "Imaginary negative"),
        (1/math.sqrt(2) + 1j/math.sqrt(2), "Complex diagonal"),
        (0.6 + 0.8j, "General complex")
    ]
    
    print(f"{'Amplitude':<20} | {'|Amplitude|²':<12} | {'Probability'}")
    print("-" * 50)
    
    for amp, description in amplitudes:
        probability = abs(amp)**2
        print(f"{amp:<20} | {probability:<12.3f} | {probability:.1%}")
    
    print("\nKey insight: Probability = |Amplitude|²")
    print("The phase (angle) of amplitude affects interference, not probability directly")

amplitude_vs_probability()
```

### Quantum Interference: The Magic of Amplitude Addition
Engineer the relative phases → engineer the final probability landscape. Most algorithm design questions reduce (implicitly) to: "How do I prepare a state where the correct answers' path amplitudes line up while incorrect ones cancel?"

```python
def quantum_interference_examples():
    """Demonstrate quantum interference through amplitude addition"""
    
    print("Quantum Interference Examples")
    print("=" * 35)
    
    # Two-path interference
    print("Scenario: Particle can take two paths to reach detector")
    
    # Path amplitudes
    path1 = 0.5 + 0.5j      # 45° phase
    path2_constructive = 0.5 + 0.5j      # Same phase
    path2_destructive = 0.5 - 0.5j       # Opposite phase
    
    print(f"\nPath 1 amplitude: {path1}")
    
    # Constructive interference
    total_constructive = path1 + path2_constructive
    prob_constructive = abs(total_constructive)**2
    
    print(f"\nConstructive interference:")
    print(f"Path 2 amplitude: {path2_constructive}")
    print(f"Total amplitude: {path1} + {path2_constructive} = {total_constructive}")
    print(f"Detection probability: |{total_constructive}|² = {prob_constructive:.3f}")
    
    # Destructive interference  
    total_destructive = path1 + path2_destructive
    prob_destructive = abs(total_destructive)**2
    
    print(f"\nDestructive interference:")
    print(f"Path 2 amplitude: {path2_destructive}")
    print(f"Total amplitude: {path1} + {path2_destructive} = {total_destructive}")
    print(f"Detection probability: |{total_destructive}|² = {prob_destructive:.3f}")
    
    print(f"\nClassical expectation: 50% + 50% = 100%")
    print(f"Quantum reality: Can be 0% to 200% depending on phase!")

quantum_interference_examples()
```

### Born Rule: The Bridge Between Quantum and Classical
The Born rule is the adapter layer converting amplitude-space (quantum evolution) into probability-space (classical observation). It is *postulated* (not derived in basic QM) and every predictive use of a quantum circuit ends with applying it conceptually, even if a simulator or hardware backend does it for you.

```python
def born_rule_detailed():
    """Detailed explanation of the Born rule"""
    
    print("Born Rule: The Foundation of Quantum Measurement")
    print("=" * 50)
    
    print("If a quantum system is in state |ψ⟩ = Σ cᵢ|i⟩")
    print("Then P(measuring outcome i) = |cᵢ|²")
    print()
    
    # Example quantum state
    c0 = 0.6 + 0.8j  # Amplitude for |0⟩
    c1 = 0.3 - 0.4j  # Amplitude for |1⟩
    
    # Normalize the state
    norm = math.sqrt(abs(c0)**2 + abs(c1)**2)
    c0_norm = c0 / norm
    c1_norm = c1 / norm
    
    print(f"Example state: |ψ⟩ = {c0_norm:.3f}|0⟩ + {c1_norm:.3f}|1⟩")
    
    # Calculate probabilities using Born rule
    p0 = abs(c0_norm)**2
    p1 = abs(c1_norm)**2
    
    print(f"\nBorn rule application:")
    print(f"P(0) = |{c0_norm:.3f}|² = {p0:.3f}")
    print(f"P(1) = |{c1_norm:.3f}|² = {p1:.3f}")
    print(f"Total: {p0 + p1:.3f} ✓")
    
    # Phase information
    phase0 = cmath.phase(c0_norm) * 180 / math.pi
    phase1 = cmath.phase(c1_norm) * 180 / math.pi
    
    print(f"\nPhase information (affects interference):")
    print(f"Phase of |0⟩ component: {phase0:.1f}°")
    print(f"Phase of |1⟩ component: {phase1:.1f}°")
    print(f"Relative phase: {phase1 - phase0:.1f}°")

born_rule_detailed()
```

### Conditional Probability in Quantum Systems:
Important nuance: the second measurement's distribution *depends* on the first because the first physically changes the state (collapse). Classical conditional probability updates knowledge; quantum measurement updates *reality* (of the system). Keep that philosophical difference in mind when modeling sequential experiments.

```python
def quantum_conditional_probability():
    """Explore conditional probability in quantum mechanics"""
    
    print("Conditional Probability: Classical vs Quantum")
    print("=" * 45)
    
    # Classical conditional probability
    print("Classical example: Drawing cards")
    print("P(King | Red card) = P(King AND Red) / P(Red)")
    print("= (2/52) / (26/52) = 2/26 = 1/13")
    
    # Quantum conditional probability
    print("\nQuantum example: Sequential measurements")
    print("If we measure a qubit in superposition:")
    
    # Initial superposition state
    initial_state = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    print(f"Initial state: {initial_state}")
    print(f"P(0) = P(1) = 50%")
    
    print(f"\nAfter measuring 0 (state collapses):")
    collapsed_state = np.array([1, 0])
    print(f"New state: {collapsed_state}")
    print(f"P(0 on second measurement | first was 0) = 100%")
    
    print(f"\nKey difference:")
    print(f"Classical: Conditional probability reveals hidden information")
    print(f"Quantum: First measurement creates the condition for the second")

quantum_conditional_probability()
```

### Probability Distributions in Multi-Qubit Systems:
Entangled states exhibit correlations that cannot be reproduced by any classical joint distribution over hidden variables. When you compute marginals (like P(first qubit=0)) you may see perfectly balanced distributions even though joint outcomes are highly constrained together (e.g., Bell state always matches both bits). This is the resource algorithms exploit.

```python
def multi_qubit_probabilities():
    """Show probability distributions for multi-qubit systems"""
    
    print("Multi-Qubit Probability Distributions")
    print("=" * 40)
    
    # Two-qubit examples
    states = {
        "Separable": np.array([0.5, 0.5, 0.5, 0.5]),  # Product state
        "Entangled": np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]),  # Bell state
        "W-state": np.array([0, 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
    }
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    for name, state in states.items():
        print(f"\n{name} state:")
        print(f"State vector: {state}")
        
        # Calculate probabilities for each basis state
        probabilities = [abs(amplitude)**2 for amplitude in state]
        
        print("Measurement probabilities:")
        for label, prob in zip(basis_labels, probabilities):
            print(f"  P({label}) = {prob:.3f}")
        
        # Check normalization
        total_prob = sum(probabilities)
        print(f"  Total: {total_prob:.3f}")
        
        # Calculate marginal probabilities
        p_first_0 = probabilities[0] + probabilities[1]  # |00⟩ + |01⟩
        p_first_1 = probabilities[2] + probabilities[3]  # |10⟩ + |11⟩
        
        print(f"Marginal probabilities:")
        print(f"  P(first qubit = 0) = {p_first_0:.3f}")
        print(f"  P(first qubit = 1) = {p_first_1:.3f}")

multi_qubit_probabilities()
```

### Quantum vs Classical Information:
Think of a qubit as a continuous object we can only query through a narrow, lossy interface (measurement). Designing algorithms is the art of shaping that continuous internal structure so that one lossy read still extracts something computationally valuable (period, marked item index, etc.).

```python
def information_comparison():
    """Compare classical and quantum information theory"""
    
    print("Information Theory: Classical vs Quantum")
    print("=" * 42)
    
    # Classical information
    print("Classical Information:")
    print("- Stored in bits (0 or 1)")
    print("- Can be copied perfectly")
    print("- Can be read without disturbance")
    print("- Probability represents ignorance")
    print("- Information is 'out there' waiting to be discovered")
    
    # Quantum information  
    print("\nQuantum Information:")
    print("- Stored in qubits (superposition states)")
    print("- Cannot be cloned (no-cloning theorem)")
    print("- Reading disturbs the system (measurement)")
    print("- Probability is fundamental to reality")
    print("- Information is created by measurement")
    
    # Quantify information content
    print(f"\nInformation content:")
    print(f"Classical bit: 1 bit of information")
    print(f"Qubit: Infinite information (continuous amplitudes)")
    print(f"But only 1 bit extractable by measurement!")
    
    # No-cloning demonstration
    print(f"\nNo-cloning theorem consequences:")
    print(f"- Quantum copy machines are impossible")
    print(f"- Perfect quantum error correction is challenging")  
    print(f"- Quantum cryptography is fundamentally secure")
    print(f"- Quantum teleportation is the only way to 'move' quantum states")

information_comparison()
```

### Probability Cheat Sheet for Quantum Computing:
Entropy hint: von Neumann entropy for a *pure* state (described by a single state vector) is 0 — all uncertainty is potential, not mixed. Mixed states (density matrices) appear once we trace out subsystems or include noise; they're coming in later modules.

```python
class QuantumProbability:
    """Essential probability operations for quantum computing"""
    
    @staticmethod
    def amplitudes_to_probabilities(amplitudes):
        """Convert complex amplitudes to measurement probabilities"""
        return [abs(amp)**2 for amp in amplitudes]
    
    @staticmethod
    def normalize_probabilities(probabilities):
        """Ensure probabilities sum to 1"""
        total = sum(probabilities)
        return [p/total for p in probabilities]
    
    @staticmethod
    def quantum_expectation_value(state, observable):
        """Calculate expectation value ⟨ψ|O|ψ⟩"""
        # observable is a matrix, state is a vector
        return np.real(np.conj(state) @ observable @ state)
    
    @staticmethod
    def fidelity(state1, state2):
        """Calculate fidelity between two quantum states"""
        return abs(np.vdot(state1, state2))**2
    
    @staticmethod
    def von_neumann_entropy(probabilities):
        """Calculate quantum entropy"""
        entropy = 0
        for p in probabilities:
            if p > 0:  # Avoid log(0)
                entropy -= p * math.log2(p)
        return entropy

# Example usage
qp = QuantumProbability()

# Complex amplitudes
amplitudes = [0.6 + 0.8j, 0.3 - 0.4j]
probs = qp.amplitudes_to_probabilities(amplitudes)
normalized_probs = qp.normalize_probabilities(probs)

print(f"Amplitudes: {amplitudes}")
print(f"Raw probabilities: {probs}")
print(f"Normalized probabilities: {normalized_probs}")

# Calculate entropy
entropy = qp.von_neumann_entropy(normalized_probs)
print(f"Quantum entropy: {entropy:.3f} bits")

# Compare with classical entropy
classical_probs = [0.5, 0.5]  # Fair coin
classical_entropy = qp.von_neumann_entropy(classical_probs)
print(f"Classical entropy (fair coin): {classical_entropy:.3f} bits")
```

---

## 2.4 State Vectors: Representing Quantum Information

So far we've mixed states, amplitudes, and probabilities informally. This section formalizes the state vector concept and introduces geometrical intuition (Bloch sphere) plus scaling realities.

### From Classical States to Quantum State Vectors

In classical computing, we represent information as simple bits. In quantum computing, we need a much richer mathematical structure to capture superposition, entanglement, and phase relationships.

#### Classical Information Representation:
Use this as a control group: classical structures enumerate *actual* values; quantum vectors enumerate *potential* values with weights. Bridging them requires measurement.

```python
# Classical information is simple
classical_bit_0 = 0
classical_bit_1 = 1
classical_byte = [0, 1, 1, 0, 1, 0, 0, 1]

print("Classical representation:")
print(f"Bit: {classical_bit_0} or {classical_bit_1}")
print(f"Byte: {classical_byte}")
print("Each bit is definitely 0 or definitely 1")
```

#### Quantum State Vector Representation:
Every quantum programming framework (Qiskit, Cirq, etc.) ultimately manipulates this vector (or an equivalent abstraction). When debugging, inspecting intermediate state vectors in a simulator can reveal unintuitive phase issues before trying on hardware.

```python
import numpy as np
import math

# Quantum states are vectors in complex vector space
quantum_0 = np.array([1, 0])        # |0⟩ state
quantum_1 = np.array([0, 1])        # |1⟩ state
quantum_superposition = np.array([1/math.sqrt(2), 1/math.sqrt(2)])  # |+⟩ state

print("\nQuantum representation:")
print(f"|0⟩ = {quantum_0}")
print(f"|1⟩ = {quantum_1}")
print(f"|+⟩ = {quantum_superposition}")
print("Each state is a vector with complex amplitudes")
```

### The State Vector Formalism:

#### Mathematical Foundation:
Notice how the normalization constraint acts like an invariant enforcing a "probability budget" of exactly 1. Any valid sequence of unitary gates preserves this automatically — so if normalization drifts in simulation, suspect numerical precision or a non‑unitary bug.

```python
def state_vector_basics():
    """Explain the basics of quantum state vectors"""
    
    print("Quantum State Vector Basics")
    print("=" * 30)
    
    # State vector properties
    print("A quantum state |ψ⟩ is represented as a column vector:")
    print("|ψ⟩ = α|0⟩ + β|1⟩ = α[1] + β[0] = [α]")
    print("                        [0]   [1]   [β]")
    
    # Normalization constraint
    print("\nNormalization constraint:")
    print("|α|² + |β|² = 1")
    print("This ensures total probability = 1")
    
    # Examples
    examples = {
        "|0⟩": np.array([1, 0]),
        "|1⟩": np.array([0, 1]),
        "|+⟩": np.array([1/math.sqrt(2), 1/math.sqrt(2)]),
        "|-⟩": np.array([1/math.sqrt(2), -1/math.sqrt(2)]),
        "|i⟩": np.array([1/math.sqrt(2), 1j/math.sqrt(2)])
    }
    
    print(f"\nCommon single-qubit states:")
    for name, state in examples.items():
        norm = np.linalg.norm(state)
        print(f"{name:4s} = {state} (norm = {norm:.3f})")

state_vector_basics()
```

#### Bloch Sphere Representation:
Interpretation tips:
* θ controls how much weight is on |0⟩ vs |1⟩ (north vs south component)
* φ controls the relative phase in the equatorial plane
Rotations about X/Y/Z axes correspond to moving along great circles. This geometric view makes reasoning about single‑qubit gate sequences dramatically faster than grinding matrix multiplication.

The Bloch sphere is a geometric way to visualize single-qubit states:

```python
def bloch_sphere_explanation():
    """Explain the Bloch sphere representation"""
    
    print("Bloch Sphere: Geometric Representation of Qubits")
    print("=" * 48)
    
    print("The Bloch sphere maps qubit states to points on a unit sphere:")
    print("                   ┌─ |0⟩ (North pole)")
    print("                   |")
    print("        |+⟩ ──────○────── |-⟩")
    print("               /   |   \\")
    print("           |+i⟩    |    |-i⟩")
    print("                   |")
    print("                   └─ |1⟩ (South pole)")
    
    # Mathematical mapping
    print("\nMathematical representation:")
    print("Any qubit state can be written as:")
    print("|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print("where θ ∈ [0,π] and φ ∈ [0,2π)")
    
    # Examples with Bloch coordinates
    states = {
        "|0⟩": (0, 0),           # θ=0, φ=0
        "|1⟩": (math.pi, 0),     # θ=π, φ=0
        "|+⟩": (math.pi/2, 0),   # θ=π/2, φ=0
        "|-⟩": (math.pi/2, math.pi),  # θ=π/2, φ=π
        "|+i⟩": (math.pi/2, math.pi/2)  # θ=π/2, φ=π/2
    }
    
    print(f"\nBloch sphere coordinates (θ, φ):")
    for name, (theta, phi) in states.items():
        print(f"{name:5s}: θ={theta:.3f}, φ={phi:.3f}")

bloch_sphere_explanation()
```

### Multi-Qubit State Vectors:

#### Tensor Products and Composite Systems:
Tensor product growth is multiplicative: kron(stateA, stateB). If a multi‑qubit state *can* be factored into single-qubit states it is separable; if not, it is entangled. Algorithms often generate entanglement early, redistribute phase/information, then sometimes disentangle partly before measurement to localize answers.

```python
def multi_qubit_states():
    """Explain multi-qubit state representation"""
    
    print("Multi-Qubit State Vectors")
    print("=" * 25)
    
    # Single qubit states
    state_0 = np.array([1, 0])
    state_1 = np.array([0, 1])
    state_plus = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    
    print("Building multi-qubit states using tensor products:")
    
    # Two-qubit states
    state_00 = np.kron(state_0, state_0)  # |00⟩
    state_01 = np.kron(state_0, state_1)  # |01⟩
    state_10 = np.kron(state_1, state_0)  # |10⟩
    state_11 = np.kron(state_1, state_1)  # |11⟩
    
    print(f"\nTwo-qubit computational basis states:")
    print(f"|00⟩ = {state_00}")
    print(f"|01⟩ = {state_01}")
    print(f"|10⟩ = {state_10}")
    print(f"|11⟩ = {state_11}")
    
    # Superposition of two qubits
    state_plus_plus = np.kron(state_plus, state_plus)  # |++⟩
    print(f"\n|++⟩ = |+⟩ ⊗ |+⟩ = {state_plus_plus}")
    print("This state has equal probability for all computational basis states")
    
    # Verify probabilities
    probabilities = [abs(amp)**2 for amp in state_plus_plus]
    print(f"Probabilities: {probabilities}")
    
    # Entangled state (cannot be written as tensor product)
    bell_state = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)])  # |Φ+⟩
    print(f"\nBell state |Φ+⟩ = {bell_state}")
    print("This CANNOT be written as a tensor product of single-qubit states!")

multi_qubit_states()
```

#### State Vector Dimensionality:
Memory warning: doubling qubits doubles vector length — which doubles memory — which often doubles runtime for naive operations. Practical simulation tricks (sparsity, tensor networks) try to avoid holding the fully dense vector when structure permits.

```python
def state_vector_scaling():
    """Show how state vector size grows with qubit number"""
    
    print("State Vector Dimensionality Scaling")
    print("=" * 35)
    
    print("Number of qubits → State vector dimension → Memory required")
    print("-" * 60)
    
    for n_qubits in range(1, 21):
        dimension = 2**n_qubits
        
        # Assuming complex128 (16 bytes per complex number)
        memory_bytes = dimension * 16
        
        if memory_bytes < 1024:
            memory_str = f"{memory_bytes} B"
        elif memory_bytes < 1024**2:
            memory_str = f"{memory_bytes/1024:.1f} KB"
        elif memory_bytes < 1024**3:
            memory_str = f"{memory_bytes/1024**2:.1f} MB"
        elif memory_bytes < 1024**4:
            memory_str = f"{memory_bytes/1024**3:.1f} GB"
        else:
            memory_str = f"{memory_bytes/1024**4:.1f} TB"
        
        print(f"{n_qubits:12d} → {dimension:18,d} → {memory_str:>12s}")
        
        if n_qubits == 10:
            print("     ↑ Still manageable on laptop")
        elif n_qubits == 15:
            print("     ↑ Needs workstation/server")
        elif n_qubits == 20:
            print("     ↑ Needs supercomputer")
    
    print("\nThis exponential scaling is why quantum simulation is hard!")

state_vector_scaling()
```

### Operations on State Vectors:

#### Inner Products and Overlaps:
Overlap matrix (Gram matrix) intuition: diagonals are 1 (self similarity), zeros indicate orthogonality (mutually exclusive outcomes), intermediate magnitudes indicate partial alignment (non‑zero sampling probability under that basis test).

```python
def state_vector_operations():
    """Demonstrate operations on quantum state vectors"""
    
    print("Operations on Quantum State Vectors")
    print("=" * 35)
    
    # Define some states
    states = {
        "|0⟩": np.array([1, 0]),
        "|1⟩": np.array([0, 1]),
        "|+⟩": np.array([1/math.sqrt(2), 1/math.sqrt(2)]),
        "|-⟩": np.array([1/math.sqrt(2), -1/math.sqrt(2)]),
        "|ψ⟩": np.array([0.6, 0.8])  # Custom state
    }
    
    print("Inner products (overlaps) between states:")
    print("⟨φ|ψ⟩ tells us how 'similar' two states are")
    print()
    
    state_names = list(states.keys())
    
    # Calculate all pairwise inner products
    print(f"{'':6s}", end="")
    for name in state_names:
        print(f"{name:>8s}", end="")
    print()
    
    for i, name1 in enumerate(state_names):
        print(f"{name1:6s}", end="")
        for j, name2 in enumerate(state_names):
            overlap = np.vdot(states[name1], states[name2])
            if abs(overlap.imag) < 1e-10:  # Essentially real
                print(f"{overlap.real:8.3f}", end="")
            else:
                print(f"{overlap:8.3f}", end="")
        print()
    
    print("\nInterpretation:")
    print("- Diagonal elements = 1 (state overlaps perfectly with itself)")
    print("- ⟨0|1⟩ = 0 (orthogonal states)")
    print("- ⟨+|-⟩ = 0 (orthogonal superposition states)")
    print("- |⟨φ|ψ⟩|² = probability of measuring |φ⟩ when system is in |ψ⟩")

state_vector_operations()
```

#### State Vector Evolution (Unitary Transformations):
All valid closed-system evolution is unitary. Noise channels (coming later) extend this with non‑unitary operations when we model the environment. For now, interpret "unitary" as: preserves both normalization and pairwise overlaps (i.e., quantum information not lost).

```python
def unitary_evolution():
    """Show how quantum states evolve under unitary operations"""
    
    print("Quantum State Evolution")
    print("=" * 23)
    
    # Define quantum gates as unitary matrices
    gates = {
        "I": np.array([[1, 0], [0, 1]]),                    # Identity
        "X": np.array([[0, 1], [1, 0]]),                    # Pauli-X (NOT)
        "Y": np.array([[0, -1j], [1j, 0]]),                 # Pauli-Y
        "Z": np.array([[1, 0], [0, -1]]),                   # Pauli-Z
        "H": np.array([[1, 1], [1, -1]]) / math.sqrt(2),    # Hadamard
        "S": np.array([[1, 0], [0, 1j]])                    # S gate (phase)
    }
    
    # Initial state
    initial_state = np.array([1, 0])  # |0⟩
    print(f"Initial state: |0⟩ = {initial_state}")
    
    print(f"\nApplying different gates:")
    for gate_name, gate_matrix in gates.items():
        final_state = gate_matrix @ initial_state
        
        # Check if state is still normalized
        norm = np.linalg.norm(final_state)
        
        print(f"{gate_name} gate: {final_state} (norm = {norm:.3f})")
        
        # Calculate measurement probabilities
        prob_0 = abs(final_state[0])**2
        prob_1 = abs(final_state[1])**2
        print(f"         P(0) = {prob_0:.3f}, P(1) = {prob_1:.3f}")
    
    print("\nKey properties of unitary evolution:")
    print("1. Preserves normalization (total probability = 1)")
    print("2. Reversible (unitary matrices have inverses)")
    print("3. Preserves inner products (angles between states)")

unitary_evolution()
```

### Visualization of State Vectors:

#### Complex Amplitude Visualization:
Tables and simple text bar charts are fast cognitive aids. Use them before reaching for full graphical tooling; they also paste cleanly into code reviews or issues when discussing algorithm behavior.

```python
def visualize_state_vectors():
    """Create visualizations for quantum state vectors"""
    
    print("Visualizing Quantum State Vectors")
    print("=" * 33)
    
    # Define states to visualize
    states = {
        "|0⟩": np.array([1, 0]),
        "|1⟩": np.array([0, 1]),
        "|+⟩": np.array([1/math.sqrt(2), 1/math.sqrt(2)]),
        "|-⟩": np.array([1/math.sqrt(2), -1/math.sqrt(2)]),
        "|+i⟩": np.array([1/math.sqrt(2), 1j/math.sqrt(2)]),
        "|-i⟩": np.array([1/math.sqrt(2), -1j/math.sqrt(2)])
    }
    
    print("State vector components (real and imaginary parts):")
    print(f"{'State':6s} | {'α (|0⟩ coeff)':15s} | {'β (|1⟩ coeff)':15s} | {'Prob |0⟩':9s} | {'Prob |1⟩':9s}")
    print("-" * 70)
    
    for name, state in states.items():
        alpha = state[0]
        beta = state[1]
        prob_0 = abs(alpha)**2
        prob_1 = abs(beta)**2
        
        alpha_str = f"{alpha:.3f}" if abs(alpha.imag) < 1e-10 else f"{alpha}"
        beta_str = f"{beta:.3f}" if abs(beta.imag) < 1e-10 else f"{beta}"
        
        print(f"{name:6s} | {alpha_str:15s} | {beta_str:15s} | {prob_0:9.3f} | {prob_1:9.3f}")

    # Bar chart representation
    print(f"\nProbability bar charts:")
    for name, state in states.items():
        prob_0 = abs(state[0])**2
        prob_1 = abs(state[1])**2
        
        bar_0 = "█" * int(prob_0 * 20)
        bar_1 = "█" * int(prob_1 * 20)
        
        print(f"{name:6s}: |0⟩ {bar_0:<20s} {prob_0:.3f}")
        print(f"      |1⟩ {bar_1:<20s} {prob_1:.3f}")
        print()

visualize_state_vectors()
```

### State Vector Cheat Sheet:
This class abstracts recurring patterns (normalize, apply gate, compute overlaps). When you move to library frameworks you'll see analogous methods; internalizing them here reduces on‑ramp friction.

```python
class QuantumStateVector:
    """Utility class for quantum state vector operations"""
    
    def __init__(self, amplitudes):
        """Initialize quantum state from amplitudes"""
        self.state = np.array(amplitudes, dtype=complex)
        self.normalize()
    
    def normalize(self):
        """Ensure state vector has unit norm"""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    def probabilities(self):
        """Get measurement probabilities"""
        return [abs(amp)**2 for amp in self.state]
    
    def apply_gate(self, gate_matrix):
        """Apply unitary gate to state"""
        self.state = gate_matrix @ self.state
        return self
    
    def measure(self, basis_state_index):
        """Calculate probability of measuring specific basis state"""
        return abs(self.state[basis_state_index])**2
    
    def overlap_with(self, other_state):
        """Calculate overlap ⟨other|self⟩"""
        return np.vdot(other_state.state, self.state)
    
    def fidelity_with(self, other_state):
        """Calculate fidelity |⟨other|self⟩|²"""
        return abs(self.overlap_with(other_state))**2
    
    def __repr__(self):
        """String representation of state"""
        return f"QuantumState({self.state})"

# Example usage
print("Quantum State Vector Class Example:")
print("=" * 35)

# Create states
psi = QuantumStateVector([0.6, 0.8])
phi = QuantumStateVector([1/math.sqrt(2), 1/math.sqrt(2)])

print(f"State |ψ⟩: {psi.state}")
print(f"State |φ⟩: {phi.state}")

# Calculate properties
print(f"\nProbabilities for |ψ⟩: {psi.probabilities()}")
print(f"Overlap ⟨φ|ψ⟩: {phi.overlap_with(psi):.3f}")
print(f"Fidelity: {phi.fidelity_with(psi):.3f}")

# Apply Hadamard gate to |ψ⟩
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
psi.apply_gate(H)
print(f"\nAfter Hadamard: {psi.state}")
print(f"New probabilities: {psi.probabilities()}")
```

---

## 2.5 Matrix Operations: How Quantum Gates Work Mathematically

We now treat gates as first-class mathematical objects. Mastering their properties (unitarity, eigenstructure, composition) lets you reason about whole circuit templates (e.g., why H X H = Z) without brute-force simulation.

### Matrices in Programming vs Quantum Computing

If you've done any graphics programming, game development, or machine learning, you've worked with matrices. Quantum computing uses matrices in a similar way, but for transforming quantum states instead of 3D objects or data.

#### What Programmers Know About Matrices:
Draw a parallel: stacking graphical transformations vs stacking quantum gates. Order matters in both; matrix multiplication is the composition operator. Debugging an unexpected result often reduces to a mistaken ordering or an unintended extra gate — analogous to a misplaced rotation in a graphics pipeline.

```python
import numpy as np

# 2D arrays are matrices!
transformation_2d = np.array([
    [2, 0],    # Scale x by 2
    [0, 3]     # Scale y by 3
])

rotation_90 = np.array([
    [0, -1],   # Rotate 90 degrees counterclockwise
    [1,  0]
])

# Apply transformation to points
point = np.array([3, 4])
scaled_point = transformation_2d @ point
rotated_point = rotation_90 @ point

print("Matrix transformations in graphics:")
print(f"Original point: {point}")
print(f"After scaling: {scaled_point}")
print(f"After rotation: {rotated_point}")

# Matrix multiplication combines transformations
combined = rotation_90 @ transformation_2d
combined_result = combined @ point
print(f"Combined transformation: {combined_result}")
```

#### Quantum Gates as Matrices:
Single-qubit universal set insight: {H, T, CNOT} (plus some phase rotations) can approximate any unitary to arbitrary precision. Recognizing common small matrices speeds mental decoding of circuit diagrams.

In quantum computing, matrices represent operations on quantum states (rather than geometric transformations):

```python
# Common quantum gate matrices
quantum_gates = {
    "I": np.array([[1, 0], [0, 1]]),                        # Identity (do nothing)
    "X": np.array([[0, 1], [1, 0]]),                        # Pauli-X (bit flip)  
    "Y": np.array([[0, -1j], [1j, 0]]),                     # Pauli-Y (bit + phase flip)
    "Z": np.array([[1, 0], [0, -1]]),                       # Pauli-Z (phase flip)
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),          # Hadamard (superposition)
    "S": np.array([[1, 0], [0, 1j]]),                       # S gate (90° phase)
    "T": np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])        # T gate (45° phase)
}

print("Quantum gate matrices:")
for name, matrix in quantum_gates.items():
    print(f"\n{name} gate:")
    print(matrix)
```

### Matrix Properties for Quantum Gates:

#### 1. Unitary Property (Reversibility):
If a matrix fails the unitarity test it cannot represent an ideal quantum gate. In simulation that usually means: (a) a numerical precision threshold too strict or (b) an arithmetic error constructing the gate. Always test custom-constructed gates.

```python
def check_unitary(matrix, name):
    """Check if a matrix is unitary (reversible)"""
    # A matrix U is unitary if U† @ U = I (where U† is conjugate transpose)
    conjugate_transpose = np.conj(matrix.T)
    product = conjugate_transpose @ matrix
    identity = np.eye(len(matrix))
    
    is_unitary = np.allclose(product, identity)
    
    print(f"{name} gate:")
    print(f"U = \n{matrix}")
    print(f"U† @ U = \n{product}")
    print(f"Is unitary: {is_unitary}")
    
    if is_unitary:
        print(f"This means {name} is reversible!")
    print()

# Check all quantum gates
for name, gate in quantum_gates.items():
    check_unitary(gate, name)
```

#### 2. Determinant Property (Probability Preservation):
For 2×2 single-qubit gates, determinant magnitude 1 follows from unitarity; looking at it can still give quick sanity signals. A pure global phase e^{iφ} multiplies determinant by e^{i2φ} (for 2×2) but leaves measurement behavior unchanged.

```python
def analyze_determinant(matrix, name):
    """Analyze determinant to check probability preservation"""
    det = np.linalg.det(matrix)
    det_magnitude = abs(det)
    
    print(f"{name} gate determinant: {det:.3f}")
    print(f"Magnitude: {det_magnitude:.3f}")
    
    if np.isclose(det_magnitude, 1):
        print("✓ Preserves probability (|det| = 1)")
    else:
        print("✗ Does not preserve probability")
    print()

print("Determinant analysis (probability preservation):")
for name, gate in quantum_gates.items():
    analyze_determinant(gate, name)
```

### Single-Qubit Gate Operations:

#### Step-by-Step Gate Application:
Stepping through component math demystifies the "black box" feeling. Once comfortable, you can jump straight to linear algebra libraries, but return here when explaining concepts to teammates or writing educational documentation.

```python
def demonstrate_gate_application():
    """Show detailed gate application process"""
    
    print("Step-by-Step Quantum Gate Application")
    print("=" * 40)
    
    # Initial states
    states = {
        "|0⟩": np.array([1, 0]),
        "|1⟩": np.array([0, 1]),
        "|+⟩": np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        "|-⟩": np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    }
    
    # Pick Hadamard gate for demonstration
    H = quantum_gates["H"]
    
    print("Applying Hadamard gate to different initial states:")
    print("H = [[1/√2,  1/√2],")
    print("     [1/√2, -1/√2]]")
    print()
    
    for state_name, state_vector in states.items():
        print(f"Initial state: {state_name} = {state_vector}")
        
        # Matrix multiplication step by step
        print("Matrix multiplication:")
        print(f"H @ {state_name} = [[{H[0,0]:.3f}, {H[0,1]:.3f}] @ [{state_vector[0]:.3f}]")
        print(f"                [[{H[1,0]:.3f}, {H[1,1]:.3f}]   [{state_vector[1]:.3f}]")
        
        # Calculate result
        result = H @ state_vector
        
        # Show calculation
        component_0 = H[0,0] * state_vector[0] + H[0,1] * state_vector[1]
        component_1 = H[1,0] * state_vector[0] + H[1,1] * state_vector[1]
        
        print(f"= [{H[0,0]:.3f}×{state_vector[0]:.3f} + {H[0,1]:.3f}×{state_vector[1]:.3f}]")
        print(f"  [{H[1,0]:.3f}×{state_vector[0]:.3f} + {H[1,1]:.3f}×{state_vector[1]:.3f}]")
        print(f"= [{component_0:.3f}]")
        print(f"  [{component_1:.3f}]")
        
        # Final result
        print(f"Final state: {result}")
        
        # Probabilities
        prob_0 = abs(result[0])**2
        prob_1 = abs(result[1])**2
        print(f"Probabilities: P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")
        print("-" * 50)

demonstrate_gate_application()
```

#### Quantum Gate Composition:
Conjugation pattern: H X H = Z, H Z H = X. These identities let you translate between different implementation bases depending on hardware native gates or optimization goals.

```python
def gate_composition_demo():
    """Show how quantum gates compose (multiply)"""
    
    print("Quantum Gate Composition")
    print("=" * 24)
    
    # Individual gates
    X = quantum_gates["X"]
    H = quantum_gates["H"]
    Z = quantum_gates["Z"]
    
    print("Composing gates: Apply H, then X, then H again")
    print("This is equivalent to: H @ X @ H")
    
    # Compose gates (note: rightmost applied first!)
    composition = H @ X @ H
    print(f"\nResulting matrix:")
    print(composition)
    
    # This should equal Z gate!
    print(f"\nZ gate for comparison:")
    print(Z)
    
    if np.allclose(composition, Z):
        print("✓ H @ X @ H = Z (Hadamard conjugates X into Z)")
    
    # Test on |0⟩ state
    state_0 = np.array([1, 0])
    
    # Apply gates sequentially
    after_h1 = H @ state_0
    after_x = X @ after_h1  
    after_h2 = H @ after_x
    
    # Apply composed gate
    direct_result = composition @ state_0
    
    print(f"\nTesting on |0⟩:")
    print(f"Sequential application: {after_h2}")
    print(f"Direct composition: {direct_result}")
    print(f"Results match: {np.allclose(after_h2, direct_result)}")

gate_composition_demo()
```

### Multi-Qubit Gates and Tensor Products:

#### Building Two-Qubit Gates:
Tensor products expand operations to larger registers when acting independently. As soon as a gate cannot be factored into such a product (like CNOT) it has entangling power — a key resource. Entangling gates are typically slower/noisier on hardware; minimize them where possible while retaining algorithmic structure.

```python
def multi_qubit_gates():
    """Demonstrate multi-qubit gate construction"""
    
    print("Multi-Qubit Gates and Tensor Products")
    print("=" * 37)
    
    # Single-qubit gates
    I = quantum_gates["I"]  # Identity
    X = quantum_gates["X"]  # Pauli-X
    H = quantum_gates["H"]  # Hadamard
    
    print("Building two-qubit gates using tensor products:")
    
    # Two-qubit gates from single-qubit gates
    II = np.kron(I, I)  # Identity on both qubits
    IX = np.kron(I, X)  # Identity on first, X on second
    XI = np.kron(X, I)  # X on first, Identity on second  
    HH = np.kron(H, H)  # Hadamard on both qubits
    
    print(f"\nI ⊗ I (4×4 identity):")
    print(II)
    
    print(f"\nI ⊗ X (X on second qubit only):")
    print(IX)
    
    print(f"\nX ⊗ I (X on first qubit only):")
    print(XI)
    
    print(f"\nH ⊗ H (Hadamard on both qubits):")
    print(HH)
    
    # CNOT gate (controlled-X)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    
    print(f"\nCNOT gate (controlled-X):")
    print(CNOT)
    print("This cannot be written as a tensor product!")
    print("It's a genuinely two-qubit gate (creates entanglement)")

multi_qubit_gates()
```

#### CNOT Gate in Detail:
Bell state creation sequence (H on control then CNOT) is *the* canonical demonstration of entanglement. Internalize it — many protocols (teleportation, error detection) start from or produce Bell-like patterns.

```python
def cnot_gate_analysis():
    """Detailed analysis of the CNOT gate"""
    
    print("CNOT Gate: The Entangling Gate")
    print("=" * 30)
    
    CNOT = np.array([
        [1, 0, 0, 0],  # |00⟩ → |00⟩
        [0, 1, 0, 0],  # |01⟩ → |01⟩
        [0, 0, 0, 1],  # |10⟩ → |11⟩
        [0, 0, 1, 0]   # |11⟩ → |10⟩
    ])
    
    # Computational basis states
    basis_states = {
        "|00⟩": np.array([1, 0, 0, 0]),
        "|01⟩": np.array([0, 1, 0, 0]),
        "|10⟩": np.array([0, 0, 1, 0]),
        "|11⟩": np.array([0, 0, 0, 1])
    }
    
    print("CNOT gate action on computational basis:")
    for state_name, state_vector in basis_states.items():
        result = CNOT @ state_vector
        
        # Find which basis state this corresponds to
        for result_name, result_basis in basis_states.items():
            if np.allclose(result, result_basis):
                print(f"CNOT @ {state_name} = {result_name}")
                break
    
    print("\nRule: CNOT flips target qubit IF control qubit is |1⟩")
    
    # Creating Bell state with CNOT
    print("\nCreating Bell state:")
    print("1. Start with |00⟩")
    initial = basis_states["|00⟩"]
    print(f"   Initial state: {initial}")
    
    print("2. Apply H ⊗ I (Hadamard on first qubit)")
    H_I = np.kron(quantum_gates["H"], quantum_gates["I"])
    after_hadamard = H_I @ initial
    print(f"   After Hadamard: {after_hadamard}")
    print(f"   = (1/√2)(|00⟩ + |10⟩)")
    
    print("3. Apply CNOT")
    bell_state = CNOT @ after_hadamard
    print(f"   Bell state: {bell_state}")
    print(f"   = (1/√2)(|00⟩ + |11⟩)")
    
    # Verify entanglement
    print("\nThis is an entangled state!")
    print("Cannot be written as |ψ₁⟩ ⊗ |ψ₂⟩ for any single-qubit states")

cnot_gate_analysis()
```

### Advanced Matrix Operations:

#### Eigenvalues and Eigenvectors:
Why care? Eigenvectors are states that only acquire a phase under the gate (eigenvalue = e^{iθ}); they form a natural basis for understanding repeated applications (e.g., phase estimation algorithms) and time evolution under a Hamiltonian.

```python
def quantum_gate_spectral_analysis():
    """Analyze eigenvalues and eigenvectors of quantum gates"""
    
    print("Spectral Analysis of Quantum Gates")
    print("=" * 34)
    
    gates_to_analyze = ["X", "Y", "Z", "H"]
    
    for gate_name in gates_to_analyze:
        gate = quantum_gates[gate_name]
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(gate)
        
        print(f"\n{gate_name} gate analysis:")
        print(f"Matrix:\n{gate}")
        
        print(f"Eigenvalues: {eigenvalues}")
        
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"Eigenvalue {val:.3f}: eigenvector {vec}")
            
            # Verify: A|v⟩ = λ|v⟩  
            result = gate @ vec
            expected = val * vec
            print(f"  Verification: A|v⟩ = {result}")
            print(f"               λ|v⟩ = {expected}")
            print(f"  Match: {np.allclose(result, expected)}")

quantum_gate_spectral_analysis()
```

#### Matrix Exponentiation and Time Evolution:
In near-term algorithm work you rarely exponentiate large matrices directly; instead you decompose into primitive gates (Trotterization, product formulas, etc.). Still, recognizing exp(-iHt) conceptually helps connect high-level physics descriptions with gate sequences.

```python
def matrix_exponentiation():
    """Demonstrate matrix exponentiation for quantum time evolution"""
    
    print("Matrix Exponentiation and Time Evolution")
    print("=" * 40)
    
    # Time evolution in quantum mechanics: U(t) = exp(-iHt/ℏ)
    # For simplicity, we'll use ℏ = 1
    
    # Pauli-Z as a simple Hamiltonian
    H = quantum_gates["Z"]
    
    print("Time evolution operator: U(t) = exp(-iHt)")
    print(f"Using Hamiltonian H = Z gate:")
    print(H)
    
    # Calculate evolution for different times
    times = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]
    
    print(f"\nTime evolution at different times:")
    for t in times:
        # Calculate exp(-iHt) using matrix exponentiation
        evolution_operator = scipy.linalg.expm(-1j * H * t)
        
        print(f"\nt = {t:.3f}:")
        print(f"U({t:.3f}) = exp(-iZt) =")
        print(evolution_operator)
        
        # Apply to |+⟩ state
        plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        evolved_state = evolution_operator @ plus_state
        
        print(f"Applied to |+⟩: {evolved_state}")
        
        # Calculate probabilities
        prob_0 = abs(evolved_state[0])**2
        prob_1 = abs(evolved_state[1])**2
        print(f"Probabilities: P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")

# Note: This requires scipy, so let's make a simpler version
def simple_time_evolution():
    """Simplified time evolution using direct calculation"""
    
    print("Simplified Time Evolution")
    print("=" * 25)
    
    # For Z gate: exp(-iZt) = cos(t)I - i*sin(t)Z
    times = [0, np.pi/4, np.pi/2, np.pi]
    
    for t in times:
        # Calculate evolution operator analytically
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        
        I = quantum_gates["I"]
        Z = quantum_gates["Z"]
        
        evolution_op = cos_t * I - 1j * sin_t * Z
        
        print(f"\nt = {t:.3f} (= {t/np.pi:.2f}π):")
        print(f"U(t) = cos({t:.3f})I - i*sin({t:.3f})Z")
        print(f"     = {cos_t:.3f}I - i*{sin_t:.3f}Z")
        print(evolution_op)

simple_time_evolution()
```

### Matrix Operations Cheat Sheet:
When writing custom tooling, these helpers encapsulate recurring safety checks (is_unitary) and construction patterns (controlled gates, rotations). Keeping them centralized reduces subtle sign or ordering mistakes.

```python
class QuantumMatrixOps:
    """Utility class for quantum matrix operations"""
    
    @staticmethod
    def is_unitary(matrix, tolerance=1e-10):
        """Check if matrix is unitary"""
        conjugate_transpose = np.conj(matrix.T)
        product = conjugate_transpose @ matrix
        identity = np.eye(len(matrix))
        return np.allclose(product, identity, atol=tolerance)
    
    @staticmethod
    def compose_gates(*gates):
        """Compose multiple gates (rightmost applied first)"""
        result = gates[0]
        for gate in gates[1:]:
            result = gate @ result
        return result
    
    @staticmethod
    def tensor_product(*matrices):
        """Calculate tensor product of multiple matrices"""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result
    
    @staticmethod
    def controlled_gate(control_qubit, target_qubit, gate, num_qubits):
        """Create controlled version of a gate"""
        # This is a simplified version for 2 qubits
        if num_qubits == 2:
            I = np.eye(2)
            if control_qubit == 0 and target_qubit == 1:
                # Control on first, target on second
                proj_0 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
                proj_1 = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
                
                return (np.kron(proj_0, I) + np.kron(proj_1, gate))
        
        raise NotImplementedError("General controlled gates not implemented")
    
    @staticmethod
    def pauli_rotation(angle, axis='z'):
        """Create rotation gate around Pauli axis"""
        if axis.lower() == 'x':
            pauli = np.array([[0, 1], [1, 0]])
        elif axis.lower() == 'y':
            pauli = np.array([[0, -1j], [1j, 0]])
        elif axis.lower() == 'z':
            pauli = np.array([[1, 0], [0, -1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        return np.cos(angle/2) * np.eye(2) - 1j * np.sin(angle/2) * pauli

# Example usage
print("Quantum Matrix Operations Examples:")
print("=" * 35)

qmo = QuantumMatrixOps()

# Check if gates are unitary
for name, gate in quantum_gates.items():
    is_unitary = qmo.is_unitary(gate)
    print(f"{name} gate is unitary: {is_unitary}")

# Compose gates
H = quantum_gates["H"]
X = quantum_gates["X"]
composed = qmo.compose_gates(H, X, H)
print(f"\nH @ X @ H =")
print(composed)

# Create tensor products
HH = qmo.tensor_product(H, H)
print(f"\nH ⊗ H =")
print(HH)

# Create rotation gates
rx_90 = qmo.pauli_rotation(np.pi/2, 'x')
print(f"\nRotation around X-axis by π/2:")
print(rx_90)
```

---

## 2.6 Interactive Demos: Visualizing Quantum States

Visualization accelerates intuition. Use lightweight textual or bar representations in early debugging; escalate to Bloch plots or multi-qubit polar diagrams only when phase relationships become opaque. Remember: visualization is diagnostic overhead — keep it targeted.

### Introduction to Quantum State Visualization

Understanding quantum states becomes much easier when you can see them! Let's explore different ways to visualize quantum information, from simple probability bars to sophisticated 3D representations.

### Basic Amplitude and Probability Visualization:
These bar panels separate the *what* (probabilities) from the *why* (amplitude real & imaginary parts). If two states have identical probability bars but different complex parts, they will behave the same under immediate measurement yet diverge under subsequent interference-inducing gates.

```python
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi

def plot_quantum_state_bars(amplitudes, state_labels=None, title="Quantum State"):
    """Plot quantum state as amplitude and probability bars"""
    
    if state_labels is None:
        state_labels = [f"|{i}⟩" for i in range(len(amplitudes))]
    
    # Calculate probabilities
    probabilities = [abs(amp)**2 for amp in amplitudes]
    
    # Get real and imaginary parts
    real_parts = [amp.real for amp in amplitudes]
    imag_parts = [amp.imag for amp in amplitudes]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    x_pos = range(len(amplitudes))
    
    # Plot 1: Real parts
    ax1.bar(x_pos, real_parts, alpha=0.7, color='blue')
    ax1.set_title('Real Parts of Amplitudes')
    ax1.set_ylabel('Real(amplitude)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(state_labels)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Imaginary parts
    ax2.bar(x_pos, imag_parts, alpha=0.7, color='red')
    ax2.set_title('Imaginary Parts of Amplitudes')
    ax2.set_ylabel('Imag(amplitude)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(state_labels)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Magnitudes
    magnitudes = [abs(amp) for amp in amplitudes]
    ax3.bar(x_pos, magnitudes, alpha=0.7, color='green')
    ax3.set_title('Amplitude Magnitudes')
    ax3.set_ylabel('|amplitude|')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(state_labels)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Probabilities
    ax4.bar(x_pos, probabilities, alpha=0.7, color='purple')
    ax4.set_title('Measurement Probabilities')
    ax4.set_ylabel('Probability')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(state_labels)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Demo with different quantum states
print("Visualizing Different Quantum States")
print("=" * 35)

states_to_visualize = {
    "|0⟩ state": [1, 0],
    "|+⟩ state": [1/sqrt(2), 1/sqrt(2)],
    "|-⟩ state": [1/sqrt(2), -1/sqrt(2)],
    "|i⟩ state": [1/sqrt(2), 1j/sqrt(2)],
    "Custom state": [0.6, 0.8j]
}

for name, amplitudes in states_to_visualize.items():
    print(f"\n{name}: {amplitudes}")
    # plot_quantum_state_bars(amplitudes, title=name)  # Uncomment to show plots
```

### Complex Plane Visualization (Phasor Diagrams):
Phasor diagrams externalize relative phase as literal angle. Use them to spot unintended phase drift or confirm that a sequence of rotations accomplished the intended net rotation.

```python
def plot_complex_amplitudes(amplitudes, state_labels=None, title="Complex Amplitudes"):
    """Plot quantum amplitudes as phasors in the complex plane"""
    
    if state_labels is None:
        state_labels = [f"|{i}⟩" for i in range(len(amplitudes))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Complex plane representation
    for i, (amp, label) in enumerate(zip(amplitudes, state_labels)):
        # Plot arrow from origin to amplitude
        ax1.arrow(0, 0, amp.real, amp.imag, 
                 head_width=0.05, head_length=0.05,
                 fc=f'C{i}', ec=f'C{i}', alpha=0.7, linewidth=2)
        
        # Add label at the tip
        ax1.text(amp.real + 0.1, amp.imag + 0.1, label, 
                fontsize=12, color=f'C{i}')
        
        # Add magnitude circle
        magnitude = abs(amp)
        if magnitude > 0:
            circle = plt.Circle((0, 0), magnitude, fill=False, 
                              color=f'C{i}', alpha=0.3, linestyle='--')
            ax1.add_patch(circle)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Complex Amplitude Phasors')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add unit circle
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='black', alpha=0.5)
    ax1.add_patch(unit_circle)
    
    # Plot 2: Phase and magnitude
    magnitudes = [abs(amp) for amp in amplitudes]
    phases = [np.angle(amp) * 180 / pi for amp in amplitudes]
    
    x_pos = range(len(amplitudes))
    
    # Magnitude bars
    bars1 = ax2.bar([x - 0.2 for x in x_pos], magnitudes, 0.4, 
                    label='Magnitude', alpha=0.7, color='blue')
    
    # Phase on secondary y-axis
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar([x + 0.2 for x in x_pos], phases, 0.4,
                        label='Phase (°)', alpha=0.7, color='red')
    
    ax2.set_ylabel('Magnitude', color='blue')
    ax2_twin.set_ylabel('Phase (degrees)', color='red')
    ax2.set_xlabel('Quantum State')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(state_labels)
    ax2.set_title('Magnitude and Phase')
    
    # Add legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Demo complex visualization
complex_states = {
    "Equal superposition": [1/sqrt(2), 1/sqrt(2)],
    "Phase difference": [1/sqrt(2), 1j/sqrt(2)],
    "Opposite phases": [1/sqrt(2), -1/sqrt(2)],
    "Complex example": [0.6 + 0.2j, -0.3 + 0.7j]
}

for name, amplitudes in complex_states.items():
    print(f"\n{name}:")
    for i, amp in enumerate(amplitudes):
        mag = abs(amp)
        phase = np.angle(amp) * 180 / pi
        print(f"  |{i}⟩: {amp:.3f} (mag={mag:.3f}, phase={phase:.1f}°)")
    # plot_complex_amplitudes(amplitudes, title=name)  # Uncomment to show plots
```

### Bloch Sphere Visualization:
2D projections are a pragmatic compromise when full 3D interactivity isn’t available. Reading them regularly builds a reflexive sense for how X/Y/Z rotations navigate the sphere.

```python
def bloch_sphere_coordinates(alpha, beta):
    """Convert qubit amplitudes to Bloch sphere coordinates"""
    # Normalize if necessary
    norm = sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha_norm = alpha / norm
    beta_norm = beta / norm
    
    # Extract theta and phi from standard form:
    # |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    theta = 2 * np.arccos(abs(alpha_norm))
    
    if abs(beta_norm) > 1e-10:  # Avoid division by zero
        phi = np.angle(beta_norm) - np.angle(alpha_norm)
    else:
        phi = 0
    
    # Convert to Cartesian coordinates on unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return x, y, z, theta, phi

def plot_bloch_sphere_2d(states_dict, title="Bloch Sphere Projection"):
    """Plot Bloch sphere as 2D projections"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Collect all coordinates
    coordinates = {}
    for name, (alpha, beta) in states_dict.items():
        x, y, z, theta, phi = bloch_sphere_coordinates(alpha, beta)
        coordinates[name] = (x, y, z, theta, phi)
    
    # Plot 1: X-Y projection (view from Z axis)
    for i, (name, (x, y, z, theta, phi)) in enumerate(coordinates.items()):
        ax1.scatter(x, y, s=100, c=f'C{i}', alpha=0.8, label=name)
        ax1.arrow(0, 0, x, y, head_width=0.05, head_length=0.05,
                 fc=f'C{i}', ec=f'C{i}', alpha=0.5)
    
    # Add unit circle
    circle1 = plt.Circle((0, 0), 1, fill=False, color='black', alpha=0.3)
    ax1.add_patch(circle1)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel('X (Re[⟨σₓ⟩])')
    ax1.set_ylabel('Y (Re[⟨σᵧ⟩])')
    ax1.set_title('X-Y Projection (equatorial plane)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Plot 2: X-Z projection (view from Y axis)
    for i, (name, (x, y, z, theta, phi)) in enumerate(coordinates.items()):
        ax2.scatter(x, z, s=100, c=f'C{i}', alpha=0.8, label=name)
        ax2.arrow(0, 0, x, z, head_width=0.05, head_length=0.05,
                 fc=f'C{i}', ec=f'C{i}', alpha=0.5)
    
    # Add semicircle
    theta_range = np.linspace(0, 2*pi, 100)
    circle2_x = np.cos(theta_range)
    circle2_z = np.sin(theta_range)
    ax2.plot(circle2_x, circle2_z, 'k-', alpha=0.3)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel('X (Re[⟨σₓ⟩])')
    ax2.set_ylabel('Z (Re[⟨σᵤ⟩])')
    ax2.set_title('X-Z Projection')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Theta-Phi coordinates
    for i, (name, (x, y, z, theta, phi)) in enumerate(coordinates.items()):
        ax3.scatter(phi * 180/pi, theta * 180/pi, s=100, c=f'C{i}', alpha=0.8, label=name)
    
    ax3.set_xlabel('φ (degrees)')
    ax3.set_ylabel('θ (degrees)')
    ax3.set_title('Spherical Coordinates (θ, φ)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-180, 180)
    ax3.set_ylim(0, 180)
    
    # Plot 4: State information table
    ax4.axis('off')
    table_data = []
    headers = ['State', 'α', 'β', 'θ°', 'φ°', 'X', 'Y', 'Z']
    
    for name, (alpha, beta) in states_dict.items():
        x, y, z, theta, phi = coordinates[name]
        row = [name, f"{alpha:.3f}", f"{beta:.3f}", 
               f"{theta*180/pi:.1f}", f"{phi*180/pi:.1f}",
               f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('State Coordinates')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Demo Bloch sphere visualization
bloch_states = {
    "|0⟩": (1, 0),
    "|1⟩": (0, 1),
    "|+⟩": (1/sqrt(2), 1/sqrt(2)),
    "|-⟩": (1/sqrt(2), -1/sqrt(2)),
    "|+i⟩": (1/sqrt(2), 1j/sqrt(2)),
    "|-i⟩": (1/sqrt(2), -1j/sqrt(2))
}

print("Bloch Sphere Coordinates:")
for name, (alpha, beta) in bloch_states.items():
    x, y, z, theta, phi = bloch_sphere_coordinates(alpha, beta)
    print(f"{name:4s}: (x={x:5.2f}, y={y:5.2f}, z={z:5.2f}) "
          f"θ={theta*180/pi:5.1f}° φ={phi*180/pi:5.1f}°")

# plot_bloch_sphere_2d(bloch_states)  # Uncomment to show plot
```

### Multi-Qubit State Visualization:
Multi-qubit visualization quickly becomes dense; focus on: (1) significant amplitudes, (2) probability distribution shape, (3) marginal distributions per qubit, and (4) notable phase patterns (clusters of aligned phases vs scattered).

```python
def plot_multi_qubit_state(amplitudes, title="Multi-Qubit State"):
    """Visualize multi-qubit quantum states"""
    
    n_qubits = int(np.log2(len(amplitudes)))
    n_states = len(amplitudes)
    
    # Generate basis state labels
    basis_labels = []
    for i in range(n_states):
        binary = format(i, f'0{n_qubits}b')
        basis_labels.append(f"|{binary}⟩")
    
    # Calculate probabilities
    probabilities = [abs(amp)**2 for amp in amplitudes]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    x_pos = range(n_states)
    
    # Plot 1: Probability histogram
    bars = ax1.bar(x_pos, probabilities, alpha=0.7, color='purple')
    ax1.set_title(f'Measurement Probabilities ({n_qubits} qubits)')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Basis States')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(basis_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        if prob > 0.01:  # Only label significant probabilities
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Real vs Imaginary parts
    real_parts = [amp.real for amp in amplitudes]
    imag_parts = [amp.imag for amp in amplitudes]
    
    width = 0.35
    ax2.bar([x - width/2 for x in x_pos], real_parts, width, 
           label='Real', alpha=0.7, color='blue')
    ax2.bar([x + width/2 for x in x_pos], imag_parts, width,
           label='Imaginary', alpha=0.7, color='red')
    
    ax2.set_title('Real and Imaginary Parts')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Basis States')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(basis_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phase diagram
    phases = [np.angle(amp) * 180 / pi for amp in amplitudes]
    magnitudes = [abs(amp) for amp in amplitudes]
    
    # Create polar plot
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    
    for i, (mag, phase) in enumerate(zip(magnitudes, phases)):
        if mag > 0.01:  # Only plot significant amplitudes
            ax3.arrow(0, 0, np.radians(phase), mag, 
                     head_width=0.1, head_length=0.05,
                     fc=f'C{i}', ec=f'C{i}', alpha=0.7)
            ax3.text(np.radians(phase), mag + 0.05, basis_labels[i],
                    fontsize=8, ha='center')
    
    ax3.set_title('Phase Diagram (Polar)', pad=20)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Marginal probabilities (for multi-qubit)
    if n_qubits >= 2:
        marginal_probs = {}
        
        for qubit in range(n_qubits):
            prob_0 = 0
            prob_1 = 0
            
            for i, prob in enumerate(probabilities):
                bit_value = (i >> (n_qubits - 1 - qubit)) & 1
                if bit_value == 0:
                    prob_0 += prob
                else:
                    prob_1 += prob
            
            marginal_probs[f'Qubit {qubit}'] = [prob_0, prob_1]
        
        # Plot marginal probabilities
        qubit_names = list(marginal_probs.keys())
        x_margin = range(len(qubit_names))
        
        prob_0_values = [marginal_probs[name][0] for name in qubit_names]
        prob_1_values = [marginal_probs[name][1] for name in qubit_names]
        
        ax4.bar([x - 0.2 for x in x_margin], prob_0_values, 0.4, 
               label='P(0)', alpha=0.7, color='lightblue')
        ax4.bar([x + 0.2 for x in x_margin], prob_1_values, 0.4,
               label='P(1)', alpha=0.7, color='lightcoral')
        
        ax4.set_title('Marginal Probabilities')
        ax4.set_ylabel('Probability')
        ax4.set_xlabel('Qubit')
        ax4.set_xticks(x_margin)
        ax4.set_xticklabels(qubit_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'Marginal probabilities\nonly for multi-qubit states',
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Demo multi-qubit visualization
multi_qubit_states = {
    "2-qubit |00⟩": [1, 0, 0, 0],
    "2-qubit Bell state": [1/sqrt(2), 0, 0, 1/sqrt(2)],
    "2-qubit |++⟩": [0.5, 0.5, 0.5, 0.5],
    "3-qubit W state": [0, 1/sqrt(3), 1/sqrt(3), 0, 1/sqrt(3), 0, 0, 0]
}

for name, amplitudes in multi_qubit_states.items():
    print(f"\n{name}:")
    n_qubits = int(np.log2(len(amplitudes)))
    print(f"  Number of qubits: {n_qubits}")
    print(f"  State vector: {amplitudes}")
    
    # Calculate and display probabilities
    probabilities = [abs(amp)**2 for amp in amplitudes]
    for i, prob in enumerate(probabilities):
        if prob > 0.001:
            binary = format(i, f'0{n_qubits}b')
            print(f"  P(|{binary}⟩) = {prob:.3f}")
    
    # plot_multi_qubit_state(amplitudes, title=name)  # Uncomment to show plots
```

### Interactive Quantum Circuit Simulator:
Rolling a minimal simulator deepens understanding of what frameworks automate: gate expansion (tensoring identities), state updates, and probability extraction. Treat this as a didactic scaffold — for performance and correctness on larger circuits, lean on established libraries.

```python
class QuantumCircuitVisualizer:
    """Interactive quantum circuit simulator with visualization"""
    
    def __init__(self, num_qubits=1):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |00...0⟩
        self.circuit_history = []
        
    def reset(self):
        """Reset to |00...0⟩ state"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1
        self.circuit_history = []
        
    def apply_gate(self, gate_matrix, qubit_indices):
        """Apply a gate to specified qubits"""
        if len(qubit_indices) == 1:
            # Single-qubit gate
            qubit = qubit_indices[0]
            full_gate = self._expand_single_qubit_gate(gate_matrix, qubit)
        elif len(qubit_indices) == 2:
            # Two-qubit gate
            full_gate = self._expand_two_qubit_gate(gate_matrix, qubit_indices)
        else:
            raise ValueError("Only single and two-qubit gates supported")
        
        self.state = full_gate @ self.state
        
    def _expand_single_qubit_gate(self, gate, target_qubit):
        """Expand single-qubit gate to full system"""
        gates = []
        for i in range(self.num_qubits):
            if i == target_qubit:
                gates.append(gate)
            else:
                gates.append(np.eye(2))
        
        # Tensor product in correct order
        result = gates[0]
        for g in gates[1:]:
            result = np.kron(result, g)
        return result
    
    def _expand_two_qubit_gate(self, gate, qubit_indices):
        """Expand two-qubit gate to full system (simplified for 2 qubits)"""
        if self.num_qubits != 2:
            raise NotImplementedError("Multi-qubit gates only implemented for 2 qubits")
        return gate
    
    def get_probabilities(self):
        """Get measurement probabilities"""
        return [abs(amp)**2 for amp in self.state]
    
    def get_state_description(self):
        """Get human-readable state description"""
        description = []
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-10:  # Only include significant amplitudes
                binary = format(i, f'0{self.num_qubits}b')
                if abs(amp.imag) < 1e-10:  # Essentially real
                    description.append(f"{amp.real:.3f}|{binary}⟩")
                else:
                    description.append(f"({amp:.3f})|{binary}⟩")
        return " + ".join(description)
    
    def visualize_current_state(self):
        """Visualize the current quantum state"""
        if self.num_qubits == 1:
            plot_quantum_state_bars(self.state, title="Current Quantum State")
        else:
            plot_multi_qubit_state(self.state, title="Current Quantum State")

# Demo quantum circuit simulator
print("Interactive Quantum Circuit Simulator")
print("=" * 37)

# Single qubit example
sim = QuantumCircuitVisualizer(num_qubits=1)

print("Initial state:")
print(f"State: {sim.get_state_description()}")
print(f"Probabilities: {sim.get_probabilities()}")

# Apply Hadamard gate
H = np.array([[1, 1], [1, -1]]) / sqrt(2)
sim.apply_gate(H, [0])

print("\nAfter Hadamard gate:")
print(f"State: {sim.get_state_description()}")
print(f"Probabilities: {sim.get_probabilities()}")

# Apply Pauli-Z gate
Z = np.array([[1, 0], [0, -1]])
sim.apply_gate(Z, [0])

print("\nAfter Z gate:")
print(f"State: {sim.get_state_description()}")
print(f"Probabilities: {sim.get_probabilities()}")

# Two qubit example
sim2 = QuantumCircuitVisualizer(num_qubits=2)

print(f"\n\nTwo-qubit system:")
print(f"Initial state: {sim2.get_state_description()}")

# Create Bell state
H_I = np.kron(H, np.eye(2))  # Hadamard on first qubit
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0], 
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

sim2.state = H_I @ sim2.state
print(f"After H⊗I: {sim2.get_state_description()}")

sim2.state = CNOT @ sim2.state
print(f"After CNOT: {sim2.get_state_description()}")
print(f"Probabilities: {sim2.get_probabilities()}")
```

### Summary and Next Steps:
The recap below ties visualization modes to specific debugging or learning goals. Reference it when deciding *which* view to generate rather than defaulting to plotting everything.

```python
def visualization_summary():
    """Summary of quantum state visualization techniques"""
    
    print("Quantum State Visualization Summary")
    print("=" * 35)
    
    techniques = {
        "Amplitude Bars": {
            "Use": "Show real/imaginary parts and probabilities",
            "Best for": "Single and multi-qubit states",
            "Pros": "Easy to interpret, shows all information",
            "Cons": "Can be cluttered for many qubits"
        },
        "Complex Plane": {
            "Use": "Show amplitudes as phasors",
            "Best for": "Understanding phase relationships",
            "Pros": "Shows interference clearly",
            "Cons": "Limited to small number of amplitudes"
        },
        "Bloch Sphere": {
            "Use": "Geometric representation of single qubits",
            "Best for": "Visualizing rotations and operations",
            "Pros": "Intuitive geometric picture",
            "Cons": "Only works for single qubits"
        },
        "Probability Histograms": {
            "Use": "Show measurement outcome distributions",
            "Best for": "Understanding measurement results",
            "Pros": "Direct connection to experiments",
            "Cons": "Loses phase information"
        },
        "Phase Diagrams": {
            "Use": "Show relative phases between amplitudes",
            "Best for": "Understanding interference patterns",
            "Pros": "Highlights phase relationships",
            "Cons": "Can be hard to interpret"
        }
    }
    
    for name, info in techniques.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print(f"\nRecommended visualization workflow:")
    print(f"1. Start with amplitude bars to see overall state")
    print(f"2. Use Bloch sphere for single-qubit operations")
    print(f"3. Use complex plane for phase analysis")
    print(f"4. Use probability histograms for measurement predictions")
    print(f"5. Use interactive simulators for circuit design")

visualization_summary()
```

---

## Module 2 Summary and Assessment

At this stage you should recognize that "mathematical foundations" are not abstract prerequisites but active tools: you will manipulate vectors (states), compose matrices (gates), monitor normalization, reason in phase space, and translate amplitude patterns into probability predictions repeatedly in actual quantum programming tasks.

### Key Takeaways from Module 2:

```python
def module_2_summary():
    """Comprehensive summary of Module 2 concepts"""
    
    print("Module 2: Mathematical Foundations Summary")
    print("=" * 42)
    
    concepts = {
        "Linear Algebra": [
            "Quantum states are vectors in complex vector space",
            "Operations are matrix multiplications", 
            "Inner products give state overlaps",
            "Normalization ensures probability conservation"
        ],
        "Complex Numbers": [
            "Enable quantum interference",
            "Store both magnitude and phase information",
            "Essential for quantum gate operations",
            "Born rule: Probability = |amplitude|²"
        ],
        "Probability Theory": [
            "Quantum probabilities come from amplitudes",
            "Superposition enables interference",
            "Measurement collapses quantum states",
            "Non-classical correlations through entanglement"
        ],
        "State Vectors": [
            "Mathematical representation of quantum information",
            "Dimensionality grows exponentially with qubits",
            "Bloch sphere for single-qubit visualization",
            "Tensor products for multi-qubit systems"
        ],
        "Matrix Operations": [
            "Quantum gates are unitary matrices",
            "Composition through matrix multiplication",
            "Eigenvalues and eigenvectors for analysis",
            "Time evolution through matrix exponentiation"
        ]
    }
    
    for topic, points in concepts.items():
        print(f"\n{topic}:")
        for point in points:
            print(f"  • {point}")
    
    print(f"\nPractical Skills Gained:")
    skills = [
        "Calculate quantum state probabilities",
        "Apply quantum gates using matrices",
        "Visualize quantum states in multiple ways",
        "Understand the mathematics behind quantum algorithms",
        "Work with complex amplitudes and phases",
        "Analyze multi-qubit quantum systems"
    ]
    
    for skill in skills:
        print(f"  ✓ {skill}")

module_2_summary()
```

### Practice Exercises:

```python
def module_2_exercises():
    """Practice exercises for Module 2"""
    
    print("Module 2 Practice Exercises")
    print("=" * 27)
    
    exercises = [
        {
            "title": "Exercise 1: State Vector Manipulation",
            "description": "Given |ψ⟩ = (0.6 + 0.8i)|0⟩ + (0.3 - 0.4i)|1⟩",
            "tasks": [
                "Normalize the state vector",
                "Calculate measurement probabilities",
                "Find the complex conjugate",
                "Plot on complex plane"
            ]
        },
        {
            "title": "Exercise 2: Gate Applications", 
            "description": "Apply quantum gates to different initial states",
            "tasks": [
                "Apply X gate to |+⟩ state",
                "Apply H gate to |1⟩ state", 
                "Compose H-X-H gates",
                "Verify results match expected outcomes"
            ]
        },
        {
            "title": "Exercise 3: Multi-Qubit Systems",
            "description": "Work with two-qubit quantum states",
            "tasks": [
                "Create |++⟩ state using tensor products",
                "Apply CNOT to create Bell state",
                "Calculate marginal probabilities",
                "Visualize the entangled state"
            ]
        },
        {
            "title": "Exercise 4: Complex Interference",
            "description": "Explore quantum interference effects",
            "tasks": [
                "Create two interfering paths",
                "Vary relative phase between paths",
                "Observe constructive/destructive interference",
                "Plot interference pattern"
            ]
        },
        {
            "title": "Exercise 5: Bloch Sphere Mapping",
            "description": "Map various quantum states to Bloch sphere",
            "tasks": [
                "Convert amplitude form to Bloch coordinates",
                "Identify Bloch sphere symmetries",
                "Understand geometric gate operations",
                "Trace quantum state evolution"
            ]
        }
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n{exercise['title']}:")
        print(f"Description: {exercise['description']}")
        print("Tasks:")
        for task in exercise['tasks']:
            print(f"  • {task}")

module_2_exercises()
```

### Assessment Quiz:

```python
def module_2_quiz():
    """Assessment quiz for Module 2"""
    
    print("Module 2 Assessment Quiz")
    print("=" * 24)
    
    questions = [
        {
            "q": "What is the relationship between quantum amplitudes and probabilities?",
            "options": [
                "A) Probability = amplitude",
                "B) Probability = |amplitude|²", 
                "C) Probability = amplitude²",
                "D) Probability = Re(amplitude)"
            ],
            "answer": "B"
        },
        {
            "q": "Why are complex numbers essential for quantum computing?",
            "options": [
                "A) They make calculations faster",
                "B) They enable quantum interference",
                "C) They are required by computers",
                "D) They simplify notation"
            ],
            "answer": "B"
        },
        {
            "q": "What property must quantum gate matrices have?",
            "options": [
                "A) They must be symmetric",
                "B) They must be real-valued",
                "C) They must be unitary",
                "D) They must be diagonal"
            ],
            "answer": "C"
        },
        {
            "q": "How does the state vector dimension scale with qubit number?",
            "options": [
                "A) Linearly (n)",
                "B) Quadratically (n²)",
                "C) Exponentially (2ⁿ)",
                "D) Logarithmically (log n)"
            ],
            "answer": "C"
        },
        {
            "q": "What does the Bloch sphere represent?",
            "options": [
                "A) All possible quantum states",
                "B) Single-qubit quantum states",
                "C) Multi-qubit entangled states",
                "D) Classical probability distributions"
            ],
            "answer": "B"
        }
    ]
    
    print("Answer the following questions:\n")
    for i, q_data in enumerate(questions, 1):
        print(f"Question {i}: {q_data['q']}")
        for option in q_data['options']:
            print(f"  {option}")
        print()
    
    print("Answers: 1-B, 2-B, 3-C, 4-C, 5-B")

module_2_quiz()
```

### Looking Ahead to Module 3:

```python
print("Next Module Preview: Quantum Programming Basics")
print("=" * 48)

preview_topics = [
    "Development Environment Setup",
    "Qiskit Deep Dive: IBM's quantum framework", 
    "Cirq Introduction: Google's quantum library",
    "Circuit Building: From gate-level to high-level abstractions",
    "Simulation vs Real Hardware: Understanding the differences",
    "Project: Build a quantum random number generator"
]

print("In Module 3, you will learn:")
for topic in preview_topics:
    print(f"  • {topic}")

print(f"\nPrerequisites for Module 3:")
print(f"  ✓ Understanding of quantum states and gates (Module 1)")
print(f"  ✓ Mathematical foundations covered in this module")
print(f"  ✓ Basic Python programming skills")
print(f"  ✓ Familiarity with Jupyter notebooks (recommended)")

print(f"\nBy the end of Module 3, you'll be able to:")
print(f"  • Set up a quantum development environment")
print(f"  • Build and simulate quantum circuits")
print(f"  • Run quantum programs on real hardware")
print(f"  • Compare different quantum frameworks")
print(f"  • Debug and optimize quantum programs")
```

---

**Module 2 Complete!** 

You now have a solid mathematical foundation for quantum computing. The concepts covered here—linear algebra, complex numbers, probability theory, state vectors, and matrix operations—form the mathematical backbone of all quantum algorithms and applications.

In Module 3, we'll put this mathematical knowledge to practical use by learning how to program quantum computers using industry-standard frameworks like Qiskit and Cirq.

**Next Module**: [Module 3: Quantum Programming Basics](Module3_Quantum_Programming_Basics.md)

---

*This module is part of the Quantum Computing 101 curriculum. For questions or feedback, please refer to the course discussion forum.*
