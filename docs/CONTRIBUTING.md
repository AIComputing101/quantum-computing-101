# Contributing to Quantum Computing 101

Thank you for your interest in contributing to Quantum Computing 101! This project aims to provide the most comprehensive and accessible quantum computing education platform available.

## 🎯 How You Can Contribute

### 🐛 Bug Reports
Found a bug? Help us fix it!

**Before submitting:**
- Check if the issue already exists in [GitHub Issues](https://github.com/AIComputing101/quantum-computing-101/issues)
- Test with the latest version
- Try to reproduce with minimal code

**When reporting:**
- Use a clear, descriptive title
- Provide detailed steps to reproduce
- Include your environment details (OS, Python version, Qiskit version)
- Add relevant code snippets or error messages
- Suggest a potential solution if you have one

### ✨ Feature Requests
Have an idea for improvement?

**Good feature requests include:**
- Clear description of the feature
- Explanation of why it would be useful
- Examples of how it would work
- Consideration of potential drawbacks

### 📚 Documentation Improvements
Help make our documentation even better!

**Documentation contributions can include:**
- Fixing typos or grammar errors
- Clarifying confusing explanations
- Adding more examples or use cases
- Improving code comments
- Translating content to other languages

### 🧪 Code Contributions
Ready to contribute code? Awesome!

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/AIComputing101/quantum-computing-101.git
cd quantum-computing-101

# Add the original repository as upstream
git remote add upstream https://github.com/original-username/quantum-computing-101.git
```

### 2. Set Up Development Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r examples/requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Create a Branch
```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

## 📝 Development Guidelines

### Code Style

We follow Python best practices and maintain consistency across the project:

**General Principles:**
- Use descriptive variable and function names
- Write comprehensive docstrings for all functions and classes
- Include type hints where appropriate
- Keep functions focused and modular
- Use consistent formatting (we use `black`)

**Example of good code style:**
```python
def create_quantum_circuit(num_qubits: int, add_barriers: bool = True) -> QuantumCircuit:
    """
    Create a quantum circuit with specified number of qubits.
    
    Args:
        num_qubits: Number of qubits in the circuit
        add_barriers: Whether to add barriers between gate operations
        
    Returns:
        QuantumCircuit: The created quantum circuit
        
    Raises:
        ValueError: If num_qubits is not positive
    """
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
        
    circuit = QuantumCircuit(num_qubits)
    
    if add_barriers:
        circuit.barrier()
        
    return circuit
```

### Example Structure

All examples should follow this structure:

```python
#!/usr/bin/env python3
"""
Quantum Computing 101 - Module X, Example Y
Title of the Example

Brief description of what this example demonstrates.

Learning objectives:
- Objective 1
- Objective 2
- Objective 3

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
# ... other imports

def main_function():
    """Main demonstration function."""
    pass

def visualization_function():
    """Create visualizations for the example."""
    pass

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Example Description")
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--parameter', type=int, default=5,
                       help='Example parameter (default: 5)')
    
    args = parser.parse_args()
    
    try:
        main_function()
        visualization_function()
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
```

### Testing

**For new examples:**
- Ensure the example runs without errors
- Test with different parameter values
- Verify visualizations are generated correctly
- Check that help text is informative

**For bug fixes:**
- Include a test that reproduces the bug
- Verify the fix resolves the issue
- Ensure no regressions in existing functionality

### Documentation

**For new examples:**
- Add clear docstrings to all functions
- Include learning objectives at the top
- Add comments explaining complex quantum concepts
- Update relevant README files

**For module additions:**
- Create a module README following existing patterns
- Add the module to the main project README
- Include theoretical background in the `modules/` directory

## 🔍 Code Review Process

### Before Submitting
Run these checks locally:

```bash
# Format code
black examples/

# Check style
pylint examples/

# Run type checking
mypy examples/

# Test your changes
python examples/your_new_example.py --help
python examples/your_new_example.py
```

### Pull Request Guidelines

**Good Pull Requests:**
- Have a clear, descriptive title
- Include a detailed description of changes
- Reference related issues (e.g., "Fixes #123")
- Include screenshots for UI/visualization changes
- Are focused on a single feature or fix
- Include tests when appropriate

**Pull Request Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have tested these changes locally
- [ ] The examples run without errors
- [ ] Visualizations display correctly
- [ ] Help text is informative and accurate

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## 🎯 Specific Contribution Areas

### High-Priority Areas
1. **Performance Optimization**: Make simulations faster and more memory-efficient
2. **Platform Compatibility**: Ensure examples work on Windows, macOS, and Linux
3. **Hardware Integration**: Add support for new quantum devices and cloud platforms
4. **Accessibility**: Improve documentation and error messages for beginners

### Medium-Priority Areas
1. **Additional Visualizations**: Create new ways to visualize quantum concepts
2. **Advanced Examples**: Add more sophisticated algorithm implementations
3. **Multi-Language Support**: Documentation translations
4. **Testing Infrastructure**: Comprehensive automated testing

### Ideas for New Contributors
1. **Fix typos and improve documentation**
2. **Add more visualization options to existing examples**
3. **Create Jupyter notebook versions of examples**
4. **Improve error messages and help text**
5. **Add parameter validation and better error handling**

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

### Communication
- Use clear, professional language
- Be patient with questions from beginners
- Provide helpful feedback in code reviews
- Acknowledge others' contributions
- Ask for clarification when needed

## 🏅 Recognition

Contributors will be recognized in the following ways:
- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Invited to join the project's core contributor team for sustained contributions

## 📚 Resources

### Learning Resources
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Quantum Computing: An Applied Approach](https://link.springer.com/book/10.1007/978-3-030-23922-0)
- [IBM Qiskit Textbook](https://qiskit.org/textbook/)

### Development Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pylint](https://pylint.org/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [Pre-commit Hooks](https://pre-commit.com/)

## ❓ Questions?

- **General Questions**: Use [GitHub Discussions](https://github.com/AIComputing101/quantum-computing-101/discussions)
- **Bug Reports**: Create a [GitHub Issue](https://github.com/AIComputing101/quantum-computing-101/issues)
- **Security Issues**: Email aicomputing101@gmail.com
- **Other**: Email aicomputing101@gmail.com

Thank you for contributing to Quantum Computing 101! Together, we're making quantum computing education accessible to everyone. 🚀⚛️
