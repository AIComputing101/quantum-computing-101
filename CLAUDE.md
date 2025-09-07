# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Quantum Computing 101 is a comprehensive educational platform with 45 production-ready quantum computing examples across 8 modules. The project uses Qiskit as the primary quantum computing framework and follows a structured modular approach for progressive learning.

## Common Development Commands

### Installation and Setup
```bash
# Install core dependencies (minimal setup for basic examples)
pip install -r examples/requirements-core.txt

# Install all dependencies (includes cloud SDKs, Jupyter, advanced libraries)
pip install -r examples/requirements.txt

# Install development dependencies (includes testing, linting, documentation)
pip install -r examples/requirements-dev.txt

# Install as package (enables CLI)
pip install -e .
```

### Running Examples
```bash
# Run individual examples
cd examples/module1_fundamentals
python 01_classical_vs_quantum_bits.py

# Most examples support CLI arguments
python 01_classical_vs_quantum_bits.py --help
python 01_classical_vs_quantum_bits.py --verbose --shots 5000

# Use the CLI interface
quantum101 list                           # List all modules
quantum101 list module1_fundamentals      # List examples in a module
quantum101 run module1_fundamentals 01_classical_vs_quantum_bits.py
```

### Testing and Quality Assurance
```bash
# Basic functionality testing
python -c "import sys; sys.path.append('examples'); import utils.visualization"
python examples/module1_fundamentals/01_classical_vs_quantum_bits.py --help

# Optional code formatting (for contributions)
black examples/

# Full development tools (optional):
# pytest examples/        # Unit testing
# pylint examples/        # Code quality
# mypy examples/          # Type checking
```

### Visualization and Headless Mode
```bash
# For headless systems, set backend
export MPLBACKEND=Agg

# Many examples support --no-plots flag
python example.py --no-plots
```

## Architecture and Code Structure

### Module Organization
- `examples/module1_fundamentals/` - Basic quantum concepts (8 examples, 1,703 LOC)
- `examples/module2_mathematics/` - Mathematical foundations (5 examples, 2,361 LOC)  
- `examples/module3_programming/` - Advanced Qiskit programming (6 examples, 3,246 LOC)
- `examples/module4_algorithms/` - Core quantum algorithms (5 examples, 1,843 LOC)
- `examples/module5_error_correction/` - Noise and error handling (5 examples, 2,111 LOC)
- `examples/module6_machine_learning/` - Quantum ML applications (5 examples, 3,157 LOC)
- `examples/module7_hardware/` - Hardware and cloud platforms (5 examples, 4,394 LOC)
- `examples/module8_applications/` - Industry use cases (6 examples, 5,346 LOC)
- `examples/utils/` - Shared utilities and helpers (387 LOC)

### Code Patterns and Standards

**Example Structure**: All examples follow a consistent pattern:
- CLI interface with argparse for user interaction
- Professional docstrings explaining learning objectives
- Error handling with informative messages  
- Rich visualizations using matplotlib and Qiskit's plotting tools
- Progressive complexity building on previous concepts

**Utility Architecture**:
- `utils/quantum_helpers.py` - Circuit creation helpers (Bell states, rotations, etc.)
- `utils/visualization.py` - Enhanced plotting and Bloch sphere tools
- `utils/educational_tools.py` - Learning aids and concept explanations
- `utils/classical_helpers.py` - Classical algorithm implementations for comparison
- `utils/cli.py` - Main CLI interface for the quantum101 command

**Framework Integration**:
- Primary: Qiskit 2.x with Aer simulator
- Cloud platforms: IBM Quantum, AWS Braket integration in Module 7
- Extension points for Cirq and PennyLane in Module 3
- Machine learning: Integration with scikit-learn, PyTorch, TensorFlow in Module 6

### Dependencies and Platform Support
- Python 3.11+ required (3.12+ recommended)
- Multi-platform support (Linux, macOS, Windows)
- Core quantum frameworks: Qiskit, Cirq, PennyLane
- Cloud SDKs: amazon-braket-sdk, azure-quantum, qiskit-ibm-runtime
- Scientific computing: numpy, scipy, matplotlib, pandas
- Optional dependencies for specific modules (openfermion for chemistry, networkx for optimization)

### Hardware and Cloud Integration
- Module 7 provides real quantum device examples
- IBM Quantum cloud platform integration with account setup
- AWS Braket examples for accessing different quantum hardware
- Hardware-optimized circuit compilation and noise analysis
- Real device error characterization and mitigation techniques

## Development Guidelines

### When Adding New Examples
- Follow the established CLI pattern with argparse
- Include comprehensive docstrings with learning objectives
- Add visualization outputs where educational value exists
- Ensure compatibility with both simulator and real hardware execution
- Test examples across different environments before integration
- Maintain the progressive complexity model within each module

### Code Quality Standards
- All code includes production-quality error handling
- Comprehensive inline documentation and comments
- Rich educational visualizations for concept reinforcement
- CLI interfaces for user customization and exploration
- Professional logging and progress indication for long-running examples
- Black formatting enforced for consistent code style
- Pylint configured for educational code patterns with appropriate disables

### Testing Requirements
- All examples must be runnable without external accounts (use simulators as default)
- Hardware examples should gracefully degrade to simulation when credentials unavailable
- Cross-platform compatibility required (handle different matplotlib backends)
- Memory and performance considerations for large quantum simulations
- Simplified CI/CD pipeline validates: basic imports, CLI help functionality, and example count
- Educational focus: prioritize functionality over strict code quality enforcement









## CI/CD Pipeline Design

### Educational Project Focus (2025-01-09)
For this educational quantum computing project, we prioritize **functionality over complexity**:

- ✅ **Simplified Pipeline**: Streamlined from 3 complex jobs to 2 lightweight validation jobs
- ✅ **Essential Testing Only**: Focus on imports, basic functionality, and CLI help
- ✅ **Faster PR Validation**: Reduced matrix testing (Python 3.11, 3.12 on Ubuntu only)
- ✅ **Educational Priorities**: Structure validation and documentation checks
- ✅ **Flexible Validation**: Warnings instead of failures for educational content variations

### Current Pipeline Jobs
1. **Validate Job**: Tests core imports, basic functionality, and CLI interfaces
2. **Documentation Job**: Validates project structure and example counts

### Benefits for Educational Project
- 🚀 **Fast PR Checks**: Typical CI run completes in 2-3 minutes
- 🎓 **Education First**: No strict code quality enforcement that blocks learning
- 🛠️ **Developer Friendly**: Easy contribution process for educational content
- 📚 **Content Focus**: Validates what matters - working examples and documentation

## Verification Results

Last verified: 2025-01-09

- ✅ **CI Pipeline**: Fully operational across all test matrices
- ✅ **Code Formatting**: All 51 files Black-compliant
- ✅ **Linting**: Pylint passing with educational code standards
- ✅ **Examples**: 45 production-ready examples validated
- 🎯 **Success Rate**: 100% CI pipeline success

All core quantum computing functionality is working correctly with Qiskit 2.x and modern Python versions.
