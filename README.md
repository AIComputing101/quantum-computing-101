# Quantum Computing 101 🚀⚛️

**The most comprehensive, beginner-friendly quantum computing course** with **48 examples** covering everything from "what is a qubit?" to industry applications in drug discovery and financial optimization.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple.svg)](https://qiskit.org/)
[![Beginner Friendly](https://img.shields.io/badge/beginner-friendly-brightgreen.svg)]()
[![Examples](https://img.shields.io/badge/examples-48_working-brightgreen.svg)]()

## 🎯 Perfect for Complete Beginners

**Never studied quantum mechanics? No problem!** This course is designed for software developers, students, and professionals who want to understand quantum computing without needing a PhD in physics.

### 🌟 What Makes This Course Special
- **🎓 Zero Prerequisites**: Assumes no quantum mechanics or advanced math background
- **🛠️ Hands-On Learning**: Learn by running real quantum programs, not just reading theory
- **📈 Gentle Learning Curve**: Carefully designed progression from basic concepts to advanced applications
- **🐛 Beginner-Focused**: Includes debugging guides, common mistakes, and troubleshooting
- **📊 Rich Visualizations**: Beautiful plots, Bloch spheres, and circuit diagrams make concepts clear
- **⚡ Real-World Ready**: Industry applications across chemistry, finance, cryptography, and AI

### 🚨 Reality Check Included
Unlike other courses that oversell quantum computing, we give you an honest assessment of:
- What quantum computers can and cannot do today
- Realistic timeline for practical applications (hint: we're still early!)
- Current hardware limitations and why they matter
- Why learning quantum computing now still makes sense for your career


## 🚀 Quick Start for Beginners

### 📖 New to Quantum Computing? Start Here!

**👉 [Read the Complete Beginner's Guide](BEGINNERS_GUIDE.md)** - Your roadmap to quantum computing mastery

**Essential First Steps:**
1. **Hardware Reality Check**: Run `python examples/module1_fundamentals/08_hardware_reality_check.py`
2. **Your First Qubit**: Run `python examples/module1_fundamentals/01_classical_vs_quantum_bits.py`
3. **Quantum "Magic"**: Run `python examples/module1_fundamentals/07_no_cloning_theorem.py`

### Prerequisites (Don't Worry - We Teach Everything!)
- Python 3.11 or higher (3.12+ recommended for best performance) 
- Basic programming knowledge (if/else, loops, functions)
- Curiosity about the future of computing!
- **New**: Docker (optional) for containerized environments with GPU support

**You do NOT need:**
- ❌ PhD in quantum physics
- ❌ Advanced linear algebra
- ❌ Expensive quantum computer

### ✅ Qiskit 2.x Compatible & Fully Tested
All examples have been updated and tested for **Qiskit 2.x compatibility** and **headless environment execution** (Docker, SSH, remote servers).

**🎯 Testing Status: 48/48 examples (100%) passing** ✨
- ✅ All modules: 100% passing (48/48 examples)
- ✅ Comprehensive automated test suite included
- ✅ All critical functionality verified and working
- See [Testing Guide](docs/TESTING.md) for details

**Recent Compatibility Fixes (November 2025):**
- Updated all `bind_parameters` → `assign_parameters` (Qiskit 2.x API)
- Fixed noise model configurations for 1-qubit vs 2-qubit gates
- Added measurement circuits where required
- Updated conditional gate syntax (`c_if` → `if_test`)
- Made optional dependencies gracefully degrade (networkx)
- Enhanced decomposition for circuit library objects

### Installation

#### Option 1: Docker (Recommended - Zero Setup!)
```bash
# Clone the repository
git clone https://github.com/AIComputing101/quantum-computing-101.git
cd quantum-computing-101

# Build CPU container using unified build script
cd docker
./build-unified.sh cpu

# Run with docker-compose (recommended)
docker-compose up -d qc101-cpu

# Or run specific example directly
docker run -it --rm \
  -v $(pwd)/../examples:/home/qc101/quantum-computing-101/examples \
  quantum-computing-101:cpu \
  python examples/module1_fundamentals/01_classical_vs_quantum_bits.py
```

#### Option 2: Local Python Installation
```bash
# Clone the repository
git clone https://github.com/AIComputing101/quantum-computing-101.git
cd quantum-computing-101

# Install core dependencies (recommended for beginners)
pip install -r examples/requirements-core.txt

# Test your setup
python examples/module1_fundamentals/01_classical_vs_quantum_bits.py

# Verify all examples work (optional)
python verify_examples.py --quick
```

#### Option 3: GPU-Accelerated (For Advanced Users)
```bash
# NVIDIA GPU acceleration (PyTorch 2.8.0 + CUDA 12.9)
cd docker
./build-unified.sh nvidia
docker-compose up -d qc101-nvidia

# AMD ROCm GPU acceleration (latest ROCm PyTorch)
./build-unified.sh amd
docker-compose up -d qc101-amd

# Run specific example with GPU
docker exec -it qc101-nvidia \
  python /home/qc101/quantum-computing-101/examples/module6_machine_learning/01_quantum_neural_network.py
```


## 📚 Learning Modules

### 🎓 Foundation Tier (Modules 1-3) - NEW BEGINNER FOCUS!
Perfect for complete beginners - now with enhanced explanations and reality checks:

| Module | Topic | Examples | Key New Features |
|--------|-------|----------|------------------|
| **[Module 1](modules/Module1_Quantum_Fundamentals.md)** | Quantum Fundamentals | **8** ⭐ | **NEW:** No-Cloning, Hardware Reality, Enhanced explanations |
| **[Module 2](modules/Module2_Mathematical_Foundations.md)** | Mathematical Foundations | 5 | Enhanced intuitive explanations |
| **[Module 3](modules/Module3_Quantum_Programming_Basics.md)** | Quantum Programming | **6** ⭐ | **NEW:** Complete Debugging Guide for beginners |

**🌟 New Beginner-Essential Examples:**
- `07_no_cloning_theorem.py` - Why quantum is fundamentally different
- `08_hardware_reality_check.py` - What QC can/can't do today  
- `06_quantum_debugging_guide.py` - Essential troubleshooting for beginners

### 🧠 Intermediate Tier (Modules 4-6)  
Build algorithmic expertise:

| Module | Topic | Examples | Lines of Code |
|--------|-------|----------|---------------|
| **[Module 4](modules/Module4_Core_Quantum_Algorithms.md)** | Quantum Algorithms | 5 | 1,843 |
| **[Module 5](modules/Module5_Quantum_Error_Correction_and_Noise.md)** | Error Correction | 5 | 2,111 |
| **[Module 6](modules/Module6_Quantum_Machine_Learning.md)** | Quantum Machine Learning | 5 | 3,157 |

### 🏭 Advanced Tier (Modules 7-8) - NOW WITH MORE APPS!
Real-world applications and quantum cryptography:

| Module | Topic | Examples | Key New Features |
|--------|-------|----------|------------------|
| **[Module 7](modules/Module7_Quantum_Hardware_Cloud_Platforms.md)** | Hardware & Cloud | 5 | Enhanced hardware compatibility fixes |
| **[Module 8](modules/Module8_Advanced_Applications_Industry_Use_Cases.md)** | Industry Applications | **6** ⭐ | **NEW:** BB84 Quantum Cryptography |

**🔐 New Real-World Example:**
- `06_quantum_cryptography_bb84.py` - Secure quantum key distribution protocol

## 💡 Example Highlights

### 🔬 **Quantum Chemistry & Drug Discovery**
```bash
# Local installation
python examples/module8_applications/01_quantum_chemistry_drug_discovery.py

# Docker (with docker-compose)
cd docker && docker-compose up -d qc101-cpu
docker exec -it qc101-cpu python \
  /home/qc101/quantum-computing-101/examples/module8_applications/01_quantum_chemistry_drug_discovery.py
```
Simulate molecular systems for drug discovery using VQE (Variational Quantum Eigensolver).

### 💰 **Financial Portfolio Optimization**  
```bash
# Local installation
python examples/module8_applications/02_financial_portfolio_optimization.py

# Docker with NVIDIA GPU acceleration
cd docker && docker-compose up -d qc101-nvidia
docker exec -it qc101-nvidia python \
  /home/qc101/quantum-computing-101/examples/module8_applications/02_financial_portfolio_optimization.py
```
Optimize investment portfolios using QAOA (Quantum Approximate Optimization Algorithm).

### 🔐 **Quantum Cryptography**
```bash
# Local installation
python examples/module8_applications/04_cryptography_cybersecurity.py

# Docker (with docker-compose)
cd docker && docker-compose up -d qc101-cpu
docker exec -it qc101-cpu python \
  /home/qc101/quantum-computing-101/examples/module8_applications/04_cryptography_cybersecurity.py
```
Implement quantum key distribution protocols (BB84, E91) and post-quantum cryptography.

### 🎯 **Grover's Search Algorithm**
```bash
# Local installation
python examples/module4_algorithms/02_grovers_search_algorithm.py

# Docker (with docker-compose)
cd docker && docker-compose up -d qc101-cpu
docker exec -it qc101-cpu python \
  /home/qc101/quantum-computing-101/examples/module4_algorithms/02_grovers_search_algorithm.py
```
Experience quadratic speedup in unstructured search problems.

### 🐳 **Docker Benefits (Unified v2.0 Architecture!)**
- **🎯 Advanced GPU Support**: NVIDIA CUDA 12.9 + AMD ROCm latest
- **⚡ Zero Setup**: No Python installation required
- **🚀 GPU Acceleration**: 5-8x speedup for large simulations  
- **🔄 Reproducible**: Identical environment across all machines
- **☁️ Cloud Ready**: Easy deployment to AWS/GCP/Azure
- **📊 Three Variants**: CPU, NVIDIA GPU, AMD ROCm
- **🏗️ Latest Hardware**: PyTorch 2.8.0 + CUDA 12.9, ROCm PyTorch latest
- **🖥️ Headless Ready**: All examples work in non-interactive/remote environments
- **🔧 Unified Architecture**: Single Dockerfile with multi-stage builds
- **📦 Volume Mounts**: Live code editing with instant container sync

### 🔄 **Docker Development Workflow**

The Docker setup supports seamless development with volume mounting:

```bash
# Start container with volume mounts (via docker-compose)
cd docker
docker-compose up -d qc101-cpu

# Edit examples on your host machine
cd ../examples/module1_fundamentals
# Edit files in your favorite editor - changes sync instantly!

# Run in container - sees your edits immediately
docker exec -it qc101-cpu python \
  /home/qc101/quantum-computing-101/examples/module1_fundamentals/01_classical_vs_quantum_bits.py

# Output files appear in your host's outputs/ directory
ls ../outputs/
```

**Volume Mount Benefits:**
- ✅ Edit files on host → Changes appear instantly in container
- ✅ Container outputs save to host → Results immediately visible
- ✅ No rebuild needed → Instant feedback loop
- ✅ Use any IDE → VS Code, PyCharm, Vim, etc.

### 📊 **Jupyter Lab Access**

Each Docker variant runs Jupyter Lab on different ports to avoid conflicts:

```bash
# CPU variant - Port 8888
cd docker
docker-compose up qc101-cpu
# Access at: http://localhost:8888

# NVIDIA variant - Port 8889
docker-compose up qc101-nvidia
# Access at: http://localhost:8889

# AMD variant - Port 8890
docker-compose up qc101-amd
# Access at: http://localhost:8890
```

For detailed Docker setup, volume mounting, and advanced configuration, see [docker/README.md](docker/README.md).

## 🛠️ Features

### 🎨 **Rich Visualizations**
- Interactive Bloch sphere representations
- Circuit diagrams with detailed annotations
- Measurement probability histograms
- Algorithm performance comparisons
- Quantum state evolution animations
- **Headless-ready**: All visualizations automatically save to files in Docker/remote environments

### 💻 **Professional Code Quality**
- Comprehensive CLI interfaces with argparse
- Robust error handling and informative messages
- Extensive docstrings and inline comments
- Object-oriented design with reusable components
- Unit tests and validation checks
- **Qiskit 2.x compatible**: Fully tested with latest Qiskit API

### 🌐 **Hardware Integration**
- IBM Quantum cloud platform examples
- AWS Braket integration tutorials
- Real quantum device noise analysis
- Hardware-optimized circuit compilation

## 📖 Documentation Structure

```
quantum-computing-101/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License  
├── CHANGELOG.md                 # Version history and updates
├── modules/                     # Theoretical curriculum
│   ├── Module1_Quantum_Fundamentals.md
│   ├── Module2_Mathematical_Foundations.md
│   ├── ...
│   └── REFERENCE.md            # Comprehensive reference guide
├── examples/                    # Hands-on implementations (45 examples)
│   ├── README.md               # Examples overview
│   ├── requirements-core.txt   # Core dependencies for beginners (Updated v2.0)
│   ├── requirements.txt        # All dependencies (Updated v2.0)
│   ├── requirements-dev.txt    # Development tools
│   ├── module1_fundamentals/   # 8 beginner examples
│   ├── module2_mathematics/    # 5 math examples
│   ├── module3_programming/    # 6 programming examples
│   ├── module4_algorithms/     # 5 algorithm examples
│   ├── module5_error_correction/# 5 error correction examples
│   ├── module6_machine_learning/# 5 ML examples
│   ├── module7_hardware/       # 5 hardware examples
│   ├── module8_applications/   # 6 industry examples
│   └── utils/                  # Shared utilities
├── docker/                      # **v2.0 Unified Architecture** - Complete containerization
│   ├── README.md               # Comprehensive Docker setup guide
│   ├── Dockerfile              # Unified multi-variant Dockerfile
│   ├── build-unified.sh        # New unified build script
│   ├── docker-compose.yml      # Multi-service orchestration (qc101-cpu/nvidia/amd)
│   ├── entrypoint.sh          # Container entry point
│   ├── requirements/           # Modular requirements for Docker
│   │   ├── base.txt            # Core frameworks for all variants
│   │   ├── cpu.txt             # CPU-specific dependencies
│   │   ├── gpu-nvidia.txt      # NVIDIA CUDA 12.9 packages
│   │   └── gpu-amd.txt         # AMD ROCm packages
├── verify_examples.py          # Quality assurance tool
├── BEGINNERS_GUIDE.md          # Complete learning pathway (Updated v2.0)
└── docs/                       # Additional documentation
    ├── COMPATIBILITY.md        # Qiskit 2.x compatibility reference
    ├── CONTRIBUTING.md         # Contribution guidelines
    ├── CODE_OF_CONDUCT.md      # Community standards
    ├── SECURITY.md             # Security policy
    └── TESTING.md              # Testing guide and procedures
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? Let us know!
- ✨ **Feature Requests**: Ideas for new examples or improvements
- 📚 **Documentation**: Help improve explanations and tutorials
- 🧪 **Testing**: Test examples on different platforms
- 🎨 **Visualizations**: Create new ways to visualize quantum concepts
- 🔧 **Performance**: Optimize simulation speed and memory usage

### Development Setup
```bash
# Clone and install development dependencies
git clone https://github.com/AIComputing101/quantum-computing-101.git
cd quantum-computing-101
pip install -r examples/requirements-dev.txt

# Run comprehensive test suite (recommended)
./test-examples.sh --continue

# Or verify all examples work (legacy)
python verify_examples.py

# Run specific module tests
./test-examples.sh --module module1_fundamentals
```

For detailed testing procedures and options, see the **[Testing Guide](docs/TESTING.md)**.

## 🎓 Educational Use

### For Students
- Follow the progressive module structure
- Run examples to reinforce theoretical concepts
- Experiment with parameters to deepen understanding
- Complete exercises at the end of each module

### For Educators
- Comprehensive curriculum ready for classroom use
- Detailed instructor notes in each module
- Exercises and assessment materials
- Flexible module structure for different course lengths

### For Researchers
- Production-ready implementations of quantum algorithms
- Extensible framework for algorithm development
- Benchmarking tools for performance analysis
- Integration with popular quantum computing frameworks

## 📊 Project Stats

- **📚 8 Complete Modules**: Comprehensive learning progression from basics to advanced applications
- **💻 45 Production Examples**: All examples fully implemented and tested
- **🎯 100% Qiskit 2.x Compatible**: All 46 files updated for Qiskit 2.x API compatibility
- **🌍 Multi-Platform**: Linux, macOS, Windows support
- **🐳 Container-Ready**: Full Docker support with headless environment compatibility
- **🔧 Quality Verified**: Automated verification tool ensures all examples work
- **📈 Educational Impact**: Designed for students, professionals, and complete beginners

## 🔗 Related Projects

- **[Qiskit](https://github.com/Qiskit/qiskit)**: IBM's quantum computing framework
- **[Cirq](https://github.com/quantumlib/Cirq)**: Google's quantum computing framework  
- **[PennyLane](https://github.com/PennyLaneAI/pennylane)**: Quantum machine learning framework
- **[Quantum Open Source Foundation](https://github.com/qosf)**: Community-driven quantum software

## 🔧 Troubleshooting & Common Issues

### Qiskit API Changes (November 2025 Update)
If you encounter errors with older code or examples from other sources:

**❌ Common Errors:**
```python
AttributeError: 'QuantumCircuit' object has no attribute 'bind_parameters'
AttributeError: 'TwoLocal' object has no attribute 'bind_parameters'
```

**✅ Solution:** Use `assign_parameters` instead:
```python
# Old Qiskit 0.x syntax (deprecated)
bound_circuit = circuit.bind_parameters(params)

# New Qiskit 2.x syntax (current)
bound_circuit = circuit.assign_parameters(params)
```

**❌ Instruction Errors:**
```python
Error: 'unknown instruction: TwoLocal'
Error: 'unknown instruction: QAOAAnsatz'
```

**✅ Solution:** Decompose circuit library objects before composition:
```python
# Add decompose() when composing library circuits
qc.compose(ansatz.assign_parameters(params).decompose(), inplace=True)
```

**❌ Noise Model Errors:**
```python
Error: '1 qubit QuantumError cannot be applied to 2 qubit instruction "cx"'
```

**✅ Solution:** Use separate error models for different gate types:
```python
# Create separate 1-qubit and 2-qubit error models
error_1q = depolarizing_error(error_rate, 1)
error_2q = depolarizing_error(error_rate, 2)

noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "y", "z"])
noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cy", "cz"])
```

### Performance Tips
- Module 8 examples may take 60-120s due to VQE/QAOA optimization
- Use `--quick` flag when available for faster testing
- Docker containers include all dependencies pre-configured

### Optional Dependencies
Some examples require additional packages:
```bash
# For supply chain optimization examples
pip install networkx

# For machine learning examples  
pip install scikit-learn

# Already included in requirements.txt
```

## 📞 Support & Community

### **When You Need Help:**
- 🐛 **Technical Issues**: Run `./test-examples.sh --continue` to diagnose problems (see [Testing Guide](docs/TESTING.md))
- 📚 **Learning Questions**: Check the [Complete Beginner's Guide](BEGINNERS_GUIDE.md)
- 📖 **Qiskit 2.x Compatibility**: All examples updated for Qiskit 2.x (tested November 2025)
- 🐳 **Docker/Headless Problems**: Examples use matplotlib 'Agg' backend for headless compatibility
- 💬 **Community Support**: Join quantum computing forums and communities
- 🔧 **Installation Problems**: Follow the setup instructions above
- 🔍 **See [Troubleshooting](#-troubleshooting--common-issues)** for common errors and solutions

### **Useful Resources:**
- **[Compatibility Guide](docs/COMPATIBILITY.md)** - Detailed Qiskit 2.x compatibility reference
- **[Qiskit Textbook](https://qiskit.org/textbook/)** - Comprehensive quantum computing resource
- **[IBM Quantum Experience](https://quantum-computing.ibm.com/)** - Run on real quantum computers
- **[Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/)** - Q&A community
- **[Qiskit Documentation](https://docs.quantum.ibm.com/)** - Official Qiskit 2.x documentation

## 📖 Citation

If you use this project in your research, education, or publications, please cite it as:

### BibTeX
```bibtex
@misc{quantum-computing-101,
  title={Quantum Computing 101: A Comprehensive Beginner-Friendly Course},
  author={{Stephen Shao}},
  year={2025},
  howpublished={\url{https://github.com/AIComputing101/quantum-computing-101}},
  note={A complete quantum computing educational resource with production-ready examples covering fundamentals to advanced applications}
}
```

## 📋 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Qiskit Team**: For the excellent quantum computing framework
- **Quantum Computing Community**: For inspiration and feedback
- **Open Source Contributors**: For making this project better

---

## ⭐ Star This Project

If you find Quantum Computing 101 helpful, please give it a star ⭐ to help others discover it!

**Ready to start your quantum journey? [Jump to Quick Start](#-quick-start) or explore the [examples](examples/) directory!**

---

*Quantum Computing 101 - Making quantum computing accessible to everyone* 🚀⚛️🌍
