# Quantum Computing 101 - Reference Guide

A comprehensive reference guide for quantum computing frameworks, libraries, tools, and resources. This reference supports the Quantum Computing 101 curriculum and provides practical guidance for quantum software development.

---

## Table of Contents

1. [Development Frameworks](#development-frameworks)
2. [Quantum Simulators](#quantum-simulators) 
3. [Quantum Compilers & Transpilers](#quantum-compilers--transpilers)
4. [Error Correction & Mitigation](#error-correction--mitigation)
5. [Hardware Control & Optimal Control](#hardware-control--optimal-control)
6. [Quantum Applications & Algorithms](#quantum-applications--algorithms)
7. [Cloud Platforms & Hardware Access](#cloud-platforms--hardware-access)
8. [Educational Resources](#educational-resources)
9. [Research Tools & Utilities](#research-tools--utilities)
10. [Programming Languages & DSLs](#programming-languages--dsls)

---

## Development Frameworks

Full-stack quantum computing frameworks for circuit construction, execution, and algorithm development.

### Major Frameworks

| Framework | Provider | Language | Focus | Hardware Support |
|-----------|----------|----------|-------|------------------|
| **[Qiskit](https://github.com/Qiskit)** | IBM | Python | General purpose, enterprise | IBM Quantum, simulators |
| **[Cirq](https://github.com/quantumlib/Cirq)** | Google | Python | NISQ devices, research | Google Quantum AI |
| **[PennyLane](https://github.com/PennyLaneAI/pennylane)** | Xanadu | Python | Quantum ML, differentiable | Multi-platform |
| **[Braket](https://github.com/amazon-braket/amazon-braket-sdk-python)** | Amazon | Python | Cloud-native | AWS Braket ecosystem |
| **[PyQuil](https://github.com/rigetti/pyquil)** | Rigetti | Python | Quil language | Rigetti processors |

### Specialized Frameworks

* **[qBraid](https://github.com/qBraid/qBraid)** - Platform-agnostic quantum runtime framework for cross-platform development
* **[Ocean](https://github.com/dwavesystems/dwave-ocean-sdk)** - SDK for D-Wave quantum annealers and hybrid algorithms
* **[Bloqade](https://github.com/QuEraComputing/Bloqade.jl)** - Neutral atom quantum computing with QuEra's architecture (Julia)
* **[Qadence](https://github.com/pasqal-io/qadence)** - Digital-analog quantum programming interface for Pasqal
* **[CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** - NVIDIA's C++/Python framework for heterogeneous quantum-classical workflows
* **[ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ)** - High-level quantum programming framework with advanced optimization
* **[Qibo](https://github.com/qiboteam/qibo)** - Multi-backend framework with efficient GPU simulation support

### Framework Selection Guide

| Use Case | Recommended Framework | Rationale |
|----------|----------------------|-----------|
| Learning & Education | Qiskit | Comprehensive documentation, tutorials |
| Research & Prototyping | Cirq, PennyLane | Flexibility, cutting-edge features |
| Production Applications | Qiskit, Braket | Enterprise support, cloud integration |
| Quantum ML | PennyLane | Differentiable programming, ML integration |
| Optimization Problems | Ocean (D-Wave) | Specialized for annealing |

---

## Quantum Simulators

Classical simulation backends for quantum circuit execution and algorithm development.

### High-Performance Simulators

#### State Vector Simulators
* **[Qiskit Aer](https://github.com/Qiskit/qiskit-aer)** - Multi-backend simulator suite (state vector, unitary, noise)
* **[Qsim](https://github.com/quantumlib/qsim)** - Google's high-performance state vector simulator with GPU support
* **[PennyLane Lightning](https://github.com/PennyLaneAI/pennylane-lightning)** - C++ accelerated simulator with differentiation support
* **[QVM](https://github.com/quil-lang/qvm)** - Rigetti's high-performance Quil simulator (Common Lisp)
* **[Qulacs](https://github.com/qulacs/)** - Fast C++/Python simulator for large, noisy, parametric circuits

#### Tensor Network Simulators
* **[CuQuantum](https://github.com/NVIDIA/cuQuantum)** - NVIDIA's GPU-accelerated tensor network simulator
* **[ITensor](https://github.com/ITensor/ITensors.jl)** - Advanced tensor network library (Julia)
* **[Quimb](https://github.com/jcmgray/quimb)** - Python tensor network library for quantum many-body systems

#### Specialized Simulators
* **[Stim](https://github.com/quantumlib/Stim)** - Ultra-fast stabilizer circuit simulator
* **[qHiPSTER](https://github.com/intel/intel-qs)** - Intel's distributed quantum simulator
* **[QuEST](https://github.com/QuEST-Kit/QuEST)** - Multi-threaded, distributed, GPU-accelerated simulator
* **[QRack](https://github.com/unitaryfund/qrack)** - GPU-accelerated universal quantum simulator
* **[Dynamiqs](https://github.com/dynamiqs/dynamiqs)** - JAX-based quantum dynamics simulation

#### Noise & Pulse Simulators
* **[Qutip-qip](https://github.com/qutip/qutip-qip)** - Circuit simulation with advanced noise modeling
* **[Qiskit Dynamics](https://github.com/Qiskit-Extensions/qiskit-dynamics)** - Time-dependent quantum system simulation

### Simulator Comparison

| Simulator | Max Qubits | GPU Support | Noise | Differentiable | Best For |
|-----------|------------|-------------|--------|----------------|----------|
| Qiskit Aer | ~30 | ✓ | ✓ | ✗ | General purpose |
| Qsim | ~40 | ✓ | ✓ | ✗ | High performance |
| Lightning | ~30 | ✓ | ✗ | ✓ | Quantum ML |
| CuQuantum | 50+ | ✓ | ✗ | ✗ | Large circuits |
| Stim | 1000+ | ✗ | ✓ | ✗ | Stabilizer circuits |

---

## Quantum Compilers & Transpilers

Tools for optimizing and transforming quantum circuits for hardware execution.

### Comprehensive Compilers
* **[Qiskit Transpiler](https://github.com/Qiskit/qiskit/tree/main/qiskit/compiler)** - Full-stack compiler with 50+ optimization passes
* **[TKET](https://github.com/CQCL/tket)** - Quantinuum's C++ quantum compiler with advanced optimization
* **[BQSKit](https://github.com/BQSKit/bqskit)** - Berkeley's numerical quantum compiler with synthesis capabilities
* **[QuilC](https://github.com/quil-lang/quilc)** - Rigetti's Quil language compiler (Common Lisp)

### Specialized Optimizers
* **[Cirq Transformers](https://github.com/quantumlib/Cirq/tree/main/cirq-core/cirq/transformers)** - Circuit transformation passes (no routing)
* **[PyZX](https://github.com/zxcalc/pyzx)** - ZX-calculus based circuit optimization

### Compilation Strategies

| Strategy | Description | Tools | Best For |
|----------|-------------|-------|----------|
| **Layout Optimization** | Qubit mapping to hardware topology | SABRE, VF2Layout | Connectivity constraints |
| **Gate Synthesis** | Decomposition to native gates | Solovay-Kitaev, KAK | Hardware gate sets |
| **Circuit Depth Reduction** | Minimize circuit depth | Commutation analysis | Coherence limits |
| **Numerical Optimization** | Approximate synthesis | BQSKit | High-fidelity gates |

---

## Error Correction & Mitigation

Tools and techniques for managing quantum errors in NISQ and fault-tolerant systems.

### Error Mitigation Libraries
* **[Mitiq](https://github.com/unitaryfund/mitiq)** - Comprehensive error mitigation toolkit
  - Zero-noise extrapolation (ZNE)
  - Probabilistic error cancellation (PEC)
  - Clifford data regression (CDR)
  - Digital dynamical decoupling (DDD)

* **[Qermit](https://github.com/CQCL/Qermit)** - Quantinuum's error mitigation protocols for TKET
* **[PyIBU](https://github.com/sidsrinivasan/PyIBU)** - Iterative Bayesian unfolding for measurement errors
* **[AutomatedPERTools](https://github.com/benmcdonough20/AutomatedPERTools)** - Autonomous probabilistic error reduction

### Error Mitigation Techniques

| Technique | Overhead | Accuracy | Scalability | Implementation |
|-----------|----------|----------|-------------|----------------|
| **Zero-Noise Extrapolation** | 2-5x shots | High | Limited | Mitiq, Qermit |
| **Readout Error Mitigation** | Calibration cost | Medium | Good | All frameworks |
| **Dynamical Decoupling** | Circuit depth | Medium | Excellent | Qiskit, Cirq |
| **Probabilistic Error Cancellation** | 100-1000x shots | Very High | Very Limited | Mitiq |

### Error Correction (Experimental)
* **[Stim](https://github.com/quantumlib/Stim)** - Stabilizer code simulation and decoding
* **[PyMatching](https://github.com/oscarhiggott/PyMatching)** - Fast decoder for quantum error correction
* **[Qiskit Experiments](https://github.com/Qiskit-Extensions/qiskit-experiments)** - Characterization and benchmarking protocols

---

## Hardware Control & Optimal Control

Libraries for pulse-level control and quantum system optimization.

### Pulse Control & Calibration
* **[Qiskit Pulse](https://github.com/Qiskit/qiskit)** - Pulse-level programming for IBM hardware
* **[Cirq Pasqal](https://github.com/quantumlib/Cirq/tree/main/cirq-pasqal)** - Neutral atom control
* **[C3](https://github.com/q-optimize/c3)** - Integrated Control, Calibration, and Characterization toolkit
* **[Qibo](https://github.com/qiboteam/qibo)** - Hardware control API with multiple backends

### Optimal Control
* **[Piccolo](https://github.com/kestrelquantum/Piccolo.jl)** - PICO method for quantum optimal control (Julia)
* **[Quandary](https://github.com/LLNL/quandary)** - LLNL's optimization solver for quantum control
* **[Qiskit Dynamics](https://github.com/Qiskit-Extensions/qiskit-dynamics)** - Time-dependent quantum systems

### Hardware Simulation
* **[SCQubits](https://github.com/scqubits/scqubits)** - Superconducting qubit simulation and analysis
* **[QuTiP](https://github.com/qutip/qutip)** - General quantum optics simulation

---

## Quantum Applications & Algorithms

Libraries implementing quantum algorithms for specific application domains.

### Quantum Chemistry & Materials
* **[Qiskit Nature](https://github.com/Qiskit/qiskit-nature)** - Chemistry and physics applications
* **[OpenFermion](https://github.com/quantumlib/OpenFermion)** - Electronic structure calculations
* **[PySCF](https://github.com/pyscf/pyscf)** - Classical quantum chemistry (hybrid algorithms)
* **[VQE Playground](https://github.com/aspuru-guzik-group/VQE-examples)** - Variational quantum eigensolver examples

### Quantum Machine Learning
* **[PennyLane](https://github.com/PennyLaneAI/pennylane)** - Differentiable quantum programming
* **[TensorFlow Quantum](https://github.com/tensorflow/quantum)** - Google's TensorFlow integration
* **[Qiskit Machine Learning](https://github.com/Qiskit/qiskit-machine-learning)** - Quantum ML algorithms
* **[Lambeq](https://github.com/CQCL/lambeq)** - Quantum natural language processing

### Optimization & Finance
* **[Qiskit Optimization](https://github.com/Qiskit/qiskit-optimization)** - QAOA and optimization algorithms
* **[D-Wave Examples](https://github.com/dwave-examples)** - Quantum annealing applications
* **[Qiskit Finance](https://github.com/Qiskit/qiskit-finance)** - Financial modeling and optimization

### Quantum Algorithms
* **[Cirq Examples](https://github.com/quantumlib/Cirq/tree/main/examples)** - Algorithm implementations
* **[Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)** - Comprehensive algorithm reference
* **[QWorld](https://github.com/qworld)** - Educational quantum programming materials

---

## Cloud Platforms & Hardware Access

Quantum computing cloud services and hardware providers.

### Major Cloud Platforms

| Platform | Provider | Hardware | Simulators | Pricing Model |
|----------|----------|----------|------------|---------------|
| **[IBM Quantum](https://quantum-computing.ibm.com/)** | IBM | Superconducting | Qiskit Aer | Free tier + premium |
| **[AWS Braket](https://aws.amazon.com/braket/)** | Amazon | Multi-vendor | SV1, TN1, DM1 | Pay-per-use |
| **[Azure Quantum](https://azure.microsoft.com/en-us/products/quantum)** | Microsoft | Multi-vendor | Azure simulators | Subscription + usage |
| **[Google Quantum AI](https://quantumai.google/)** | Google | Superconducting | Cirq simulators | Research access |

### Hardware Vendors
* **[IonQ](https://ionq.com/)** - Trapped ion quantum computers
* **[Rigetti](https://www.rigetti.com/)** - Superconducting quantum processors
* **[D-Wave](https://www.dwavesys.com/)** - Quantum annealing systems
* **[QuEra](https://www.quera.com/)** - Neutral atom quantum computers
* **[Pasqal](https://www.pasqal.com/)** - Analog quantum processors
* **[Xanadu](https://www.xanadu.ai/)** - Photonic quantum computers
* **[Alpine Quantum Technologies](https://www.aqt.eu/)** - Trapped ion systems

### Access Methods
* **Cloud APIs** - REST/Python SDKs for programmatic access
* **Jupyter Notebooks** - Interactive development environments
* **Queue Systems** - Job scheduling and priority management
* **Hybrid Workflows** - Classical-quantum algorithm orchestration

---

## Educational Resources

Learning materials, tutorials, and educational platforms.

### Interactive Learning
* **[Qiskit Textbook](https://qiskit.org/textbook/)** - Comprehensive quantum computing textbook
* **[IBM Quantum Experience](https://quantum-computing.ibm.com/)** - Hands-on quantum computing platform
* **[Microsoft Quantum Katas](https://github.com/microsoft/QuantumKatas)** - Self-paced programming exercises
* **[QWorld Workshops](https://qworld.net/)** - Global quantum programming workshops

### Online Courses
* **[IBM Qiskit Global Summer School](https://qiskit.org/events/summer-school/)** - Annual intensive program
* **[edX Quantum Computing Courses](https://www.edx.org/learn/quantum-computing)** - University-level courses
* **[Coursera Quantum Computing](https://www.coursera.org/courses?query=quantum%20computing)** - Academic and industry courses

### Books & References
* **[Quantum Computation and Quantum Information](https://www.cambridge.org/core/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)** - Nielsen & Chuang (foundational textbook)
* **[Programming Quantum Computers](https://www.oreilly.com/library/view/programming-quantum-computers/9781492039679/)** - Practical quantum programming
* **[Quantum Computing: An Applied Approach](https://link.springer.com/book/10.1007/978-3-030-23922-0)** - Hidary (applications focus)

---

## Research Tools & Utilities

Specialized tools for quantum computing research and development.

### Benchmarking & Analysis
* **[Qiskit Experiments](https://github.com/Qiskit-Extensions/qiskit-experiments)** - Characterization and benchmarking
* **[True-Q](https://trueq.quantumbenchmark.com/)** - Quantum processor benchmarking
* **[Quantum Benchmark Tools](https://github.com/qiskit-community/quantum-prototype-template)** - Community benchmarking

### Visualization & Analysis
* **[Qiskit Visualization](https://github.com/Qiskit/qiskit/tree/main/qiskit/visualization)** - Circuit and state visualization
* **[PyQuil Visualization](https://github.com/rigetti/forest-benchmarking)** - Rigetti analysis tools
* **[Cirq Contrib](https://github.com/quantumlib/Cirq/tree/main/cirq-core/cirq/contrib)** - Community visualization tools

### Development Tools
* **[Black](https://github.com/psf/black)** - Python code formatting
* **[Pytest](https://github.com/pytest-dev/pytest)** - Testing framework
* **[MyPy](https://github.com/python/mypy)** - Static type checking
* **[Pre-commit](https://github.com/pre-commit/pre-commit)** - Code quality hooks

### Performance Profiling
* **[Qiskit Profiler](https://github.com/Qiskit/qiskit/tree/main/qiskit/tools)** - Circuit analysis tools
* **[Memory Profiler](https://github.com/pythonprofilers/memory_profiler)** - Memory usage analysis
* **[Line Profiler](https://github.com/pyutils/line_profiler)** - Line-by-line performance

---

## Programming Languages & DSLs

Domain-specific languages and programming paradigms for quantum computing.

### Quantum Languages
* **[Quil](https://github.com/quil-lang/quil)** - Rigetti's quantum instruction language
* **[OpenQASM](https://github.com/openqasm/openqasm)** - Open quantum assembly language
* **[Q#](https://github.com/microsoft/qsharp-compiler)** - Microsoft's quantum programming language
* **[Silq](https://github.com/eth-sri/silq)** - High-level quantum programming language

### Hybrid Languages
* **[PyQuil](https://github.com/rigetti/pyquil)** - Python embedding of Quil
* **[Qiskit Terra](https://github.com/Qiskit/qiskit)** - Python quantum circuits
* **[Cirq](https://github.com/quantumlib/Cirq)** - Python-first quantum programming

### Compilation Targets
* **[LLVM](https://llvm.org/)** - Compiler infrastructure (used by several quantum compilers)
* **[WebAssembly](https://webassembly.org/)** - Browser-based quantum simulation
* **[CUDA](https://developer.nvidia.com/cuda-zone)** - GPU acceleration target

---

## Getting Started Guide

### For Beginners
1. **Start with Qiskit Textbook** - Learn fundamentals
2. **Practice with IBM Quantum Experience** - Hands-on exercises
3. **Complete Microsoft Quantum Katas** - Programming practice
4. **Join QWorld workshops** - Community learning

### For Developers
1. **Choose primary framework** - Qiskit (general), PennyLane (ML), Cirq (research)
2. **Set up development environment** - Python, Jupyter, Git
3. **Practice with simulators** - Before using real hardware
4. **Learn transpilation** - Circuit optimization is crucial

### For Researchers
1. **Survey multiple frameworks** - Different strengths for different research areas
2. **Understand hardware limitations** - NISQ constraints and opportunities
3. **Master error mitigation** - Essential for meaningful results
4. **Engage with community** - Conferences, workshops, GitHub

---

## Contributing

This reference guide is maintained as part of the Quantum Computing 101 curriculum. Contributions are welcome:

1. **Submit issues** for outdated information or missing tools
2. **Propose additions** for new libraries or resources
3. **Improve organization** and clarity of information
4. **Add examples** and use cases for better guidance

---

## License & Acknowledgments

This reference guide is provided under the same license as the Quantum Computing 101 curriculum. We acknowledge the quantum computing community for developing and maintaining these excellent open-source tools and resources.

---

*Last updated: September 2025*