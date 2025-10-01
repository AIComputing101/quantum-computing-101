#!/bin/bash
# Entrypoint script for Quantum Computing 101 containers

set -e

# Detect hardware and display information
echo "🐳 Quantum Computing 101 Container"
echo "===================================="

# Detect variant
if command -v nvidia-smi &> /dev/null; then
    echo "Platform: NVIDIA CUDA GPU"
    echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- || echo 'N/A')"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info not available"
    
    # Check PyTorch CUDA
    python -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || true
elif command -v rocm-smi &> /dev/null; then
    echo "Platform: AMD ROCm GPU"
    rocm-smi --showproductname 2>/dev/null || echo "ROCm GPU detected"
    
    # Check PyTorch ROCm
    python -c "import torch; print(f'PyTorch ROCm Available: {torch.cuda.is_available()}')" 2>/dev/null || true
else
    echo "Platform: CPU Only"
fi

# Display Python and package versions
echo ""
echo "Python: $(python --version 2>&1)"
python -c "import qiskit; print(f'Qiskit: {qiskit.__version__}')" 2>/dev/null || echo "Qiskit: Not installed"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: Not installed"

echo ""
echo "===================================="
echo "Working directory: $(pwd)"
echo ""
echo "Quick start:"
echo "  python module1_fundamentals/01_classical_vs_quantum_bits.py"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""

# Execute the command passed to docker run
exec "$@"
