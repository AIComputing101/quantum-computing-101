#!/bin/bash
# Run script for Quantum Computing 101 Docker containers v2.0
# Now supports CPU, NVIDIA CUDA, and AMD ROCm variants

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_gpu() {
    echo -e "${PURPLE}[GPU]${NC} $1"
}

# Default values
VARIANT="cpu"
INTERACTIVE=true
JUPYTER=false
EXAMPLE=""
MODULE=""
OUTPUTS_DIR="./outputs"

# Function to check if image exists
check_image() {
    local image=$1
    if docker images "$image" --format "{{.Repository}}:{{.Tag}}" | grep -q "$image"; then
        return 0
    else
        return 1
    fi
}

# Function to ensure outputs directory exists
ensure_outputs_dir() {
    if [ ! -d "$OUTPUTS_DIR" ]; then
        mkdir -p "$OUTPUTS_DIR"
        print_status "Created outputs directory: $OUTPUTS_DIR"
    fi
}

# Function to get GPU arguments based on variant
get_gpu_args() {
    local variant=$1
    case $variant in
        gpu-nvidia)
            echo "--gpus all --runtime=nvidia"
            ;;
        gpu-amd)
            echo "--device=/dev/kfd --device=/dev/dri --group-add video"
            ;;
        cpu|*)
            echo ""
            ;;
    esac
}

# Function to get port based on variant for Jupyter
get_jupyter_port() {
    local variant=$1
    case $variant in
        cpu)
            echo "8888"
            ;;
        gpu-nvidia)
            echo "8889"
            ;;
        gpu-amd)
            echo "8890"
            ;;
        *)
            echo "8888"
            ;;
    esac
}

# Function to run interactive container
run_interactive() {
    local image=$1
    local gpu_args=$(get_gpu_args "$VARIANT")
    
    print_status "Starting interactive $VARIANT container..."
    
    ensure_outputs_dir
    
    # shellcheck disable=SC2086
    docker run -it --rm \
        $gpu_args \
        -v "$(pwd)/examples:/home/qc101/quantum-computing-101/examples" \
        -v "$(pwd)/$OUTPUTS_DIR:/home/qc101/quantum-computing-101/outputs" \
        -v "$(pwd)/modules:/home/qc101/quantum-computing-101/modules:ro" \
        -w /home/qc101/quantum-computing-101/examples \
        -e PYTHONPATH=/home/qc101/quantum-computing-101 \
        --name "qc101-interactive-$VARIANT" \
        "$image" \
        /bin/bash
}

# Function to run specific example
run_example() {
    local image=$1
    local example_path=$2
    shift 2  # Remove image and example_path from arguments
    local gpu_args=$(get_gpu_args "$VARIANT")
    
    print_status "Running example: $example_path"
    print_status "Variant: $VARIANT"
    
    ensure_outputs_dir
    
    # Check if example file exists
    if [ ! -f "examples/$example_path" ]; then
        print_error "Example file not found: examples/$example_path"
        exit 1
    fi
    
    # shellcheck disable=SC2086
    docker run --rm \
        $gpu_args \
        -v "$(pwd)/examples:/home/qc101/quantum-computing-101/examples" \
        -v "$(pwd)/$OUTPUTS_DIR:/home/qc101/quantum-computing-101/outputs" \
        -v "$(pwd)/modules:/home/qc101/quantum-computing-101/modules:ro" \
        -w /home/qc101/quantum-computing-101/examples \
        -e PYTHONPATH=/home/qc101/quantum-computing-101 \
        --name "qc101-example-$VARIANT-$$" \
        "$image" \
        python "$example_path" "$@"
}

# Function to run Jupyter Lab
run_jupyter() {
    local image=$1
    local gpu_args=$(get_gpu_args "$VARIANT")
    local port=$(get_jupyter_port "$VARIANT")
    
    print_status "Starting Jupyter Lab on port $port (variant: $VARIANT)..."
    ensure_outputs_dir
    
    # Check if port is already in use
    if netstat -an 2>/dev/null | grep -q ":$port.*LISTEN" || lsof -i :$port &>/dev/null; then
        print_warning "Port $port is already in use. Trying to find alternative..."
        port=$((port + 10))
        print_status "Using alternative port: $port"
    fi
    
    # Create workspace directory if not exists
    local workspace_dir="workspace-$VARIANT"
    mkdir -p "$workspace_dir"
    
    # Install Jupyter and run
    # shellcheck disable=SC2086
    docker run -it --rm \
        $gpu_args \
        -p "$port:8888" \
        -v "$(pwd)/examples:/home/qc101/quantum-computing-101/examples" \
        -v "$(pwd)/$OUTPUTS_DIR:/home/qc101/quantum-computing-101/outputs" \
        -v "$(pwd)/modules:/home/qc101/quantum-computing-101/modules:ro" \
        -v "$(pwd)/$workspace_dir:/home/qc101/workspace" \
        -w /home/qc101/quantum-computing-101/examples \
        -e PYTHONPATH=/home/qc101/quantum-computing-101 \
        -e JUPYTER_ENABLE_LAB=yes \
        --name "qc101-jupyter-$VARIANT" \
        "$image" \
        bash -c "
            echo '🚀 Setting up Jupyter Lab environment...'
            pip install --quiet jupyter jupyterlab ipywidgets
            echo '📊 Jupyter Lab will be available at: http://localhost:$port'
            if [ '$VARIANT' != 'cpu' ]; then
                echo '🔥 GPU acceleration enabled for variant: $VARIANT'
            fi
            echo '📁 Examples: /home/qc101/quantum-computing-101/examples'
            echo '💾 Outputs: /home/qc101/quantum-computing-101/outputs'
            echo '📝 Workspace: /home/qc101/workspace'
            echo ''
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
        "
}

# Function to list available examples
list_examples() {
    local image=$1
    
    print_status "Available examples:"
    
    docker run --rm \
        -v "$(pwd)/examples:/home/qc101/quantum-computing-101/examples:ro" \
        -w /home/qc101/quantum-computing-101/examples \
        "$image" \
        bash -c "
            echo 'Quantum Computing 101 - Available Examples:'
            echo '==========================================='
            find . -name '*.py' -type f | grep -E 'module[0-9]_' | sort | while read -r file; do
                module=\$(echo \$file | cut -d'/' -f2)
                example=\$(basename \$file)
                echo \"  \$file\"
            done
            echo ''
            echo 'Usage examples:'
            echo '  ./run.sh -v $VARIANT -e module1_fundamentals/01_classical_vs_quantum_bits.py'
            echo '  ./run.sh -v $VARIANT -e module6_machine_learning/01_quantum_neural_network.py'
        "
}

# Function to show hardware info
show_hardware_info() {
    local image=$1
    local gpu_args=$(get_gpu_args "$VARIANT")
    
    print_status "Hardware information for variant: $VARIANT"
    
    # shellcheck disable=SC2086
    docker run --rm \
        $gpu_args \
        "$image" \
        bash -c "
            echo 'Container Hardware Information:'
            echo '==============================='
            echo 'Variant: $VARIANT'
            echo 'CPUs: '\$(nproc)
            echo 'Memory: '\$(free -h | awk '/^Mem:/ {print \$2}')
            
            case '$VARIANT' in
                gpu-nvidia)
                    if command -v nvidia-smi &>/dev/null; then
                        echo 'GPU Info:'
                        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
                        echo 'CUDA Available: '\$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')
                    else
                        echo 'nvidia-smi not available'
                    fi
                    ;;
                gpu-amd)
                    if command -v rocm-smi &>/dev/null; then
                        echo 'AMD GPU Info:'
                        rocm-smi --showproductname --showmeminfo | head -10
                    else
                        echo 'rocm-smi not available, but ROCm devices should be accessible'
                    fi
                    echo 'PyTorch ROCm Available: '\$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')
                    ;;
                cpu)
                    echo 'CPU-only variant - no GPU acceleration'
                    ;;
            esac
            
            echo 'Python Version: '\$(python --version)
            echo 'Qiskit Version: '\$(python -c 'import qiskit; print(qiskit.__version__)' 2>/dev/null || echo 'Qiskit not available')
        "
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run Quantum Computing 101 Docker containers with GPU support"
    echo ""
    echo "Options:"
    echo "  -v, --variant VARIANT    Container variant: cpu, gpu-nvidia, gpu-amd (default: cpu)"
    echo "  -e, --example PATH       Run specific example (e.g., module1_fundamentals/01_classical_vs_quantum_bits.py)"
    echo "  -m, --module MODULE      List examples in specific module (e.g., module1_fundamentals)"
    echo "  -j, --jupyter           Start Jupyter Lab environment"
    echo "  -i, --interactive       Start interactive shell (default)"
    echo "  -o, --outputs DIR       Output directory (default: ./outputs)"
    echo "  -l, --list             List available examples"
    echo "  --info                 Show hardware information"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Container Variants:"
    echo "  cpu           Lightweight CPU-only (1.2GB, Python 3.12, OpenBLAS)"
    echo "  gpu-nvidia    NVIDIA CUDA acceleration (3.5GB, CUDA 12.2, PyTorch GPU)"
    echo "  gpu-amd       AMD ROCm acceleration (3.2GB, ROCm 5.6, PyTorch ROCm)"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Interactive CPU container"
    echo "  $0 -v gpu-nvidia -i                       # Interactive NVIDIA GPU container" 
    echo "  $0 -v gpu-amd -i                          # Interactive AMD ROCm container"
    echo "  $0 -e module1_fundamentals/01_classical_vs_quantum_bits.py  # Run specific example"
    echo "  $0 -v gpu-nvidia -e module6_machine_learning/01_quantum_neural_network.py  # GPU ML example"
    echo "  $0 -j                                      # Start Jupyter Lab (CPU)"
    echo "  $0 -v gpu-nvidia -j                       # Start Jupyter Lab (NVIDIA GPU)"
    echo "  $0 -v gpu-amd -j                          # Start Jupyter Lab (AMD ROCm)"
    echo "  $0 -l                                      # List all examples"
    echo "  $0 --info                                  # Show hardware info"
    echo ""
    echo "Port Assignments:"
    echo "  CPU Jupyter:      http://localhost:8888"
    echo "  NVIDIA Jupyter:   http://localhost:8889"
    echo "  AMD ROCm Jupyter: http://localhost:8890"
    echo ""
    echo "Hardware Requirements:"
    echo "  cpu variant:        Any x86_64 system with Docker"
    echo "  gpu-nvidia variant: NVIDIA GPU + CUDA drivers + nvidia-docker"
    echo "  gpu-amd variant:    AMD GPU + ROCm drivers + /dev/kfd,/dev/dri access"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--variant)
            VARIANT="$2"
            shift 2
            ;;
        -e|--example)
            EXAMPLE="$2"
            INTERACTIVE=false
            shift 2
            ;;
        -m|--module)
            MODULE="$2"
            shift 2
            ;;
        -j|--jupyter)
            JUPYTER=true
            INTERACTIVE=false
            shift
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -o|--outputs)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        -l|--list)
            INTERACTIVE=false
            JUPYTER=false
            EXAMPLE="list"
            shift
            ;;
        --info)
            INTERACTIVE=false
            JUPYTER=false
            EXAMPLE="info"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate variant
if [ "$VARIANT" != "cpu" ] && [ "$VARIANT" != "gpu-nvidia" ] && [ "$VARIANT" != "gpu-amd" ]; then
    print_error "Invalid variant: $VARIANT. Must be 'cpu', 'gpu-nvidia', or 'gpu-amd'"
    exit 1
fi

# Set image name
IMAGE="quantum101:$VARIANT"

# Check if image exists
if ! check_image "$IMAGE"; then
    print_error "Image $IMAGE not found. Please build it first:"
    echo "  cd docker && ./build.sh $VARIANT"
    exit 1
fi

# Check hardware requirements for GPU variants
if [ "$VARIANT" = "gpu-nvidia" ]; then
    if ! (command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia); then
        print_error "NVIDIA Docker runtime not found. GPU variant requires NVIDIA Docker."
        print_status "Install NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    print_gpu "NVIDIA Docker runtime detected"
    
elif [ "$VARIANT" = "gpu-amd" ]; then
    if [ ! -e "/dev/kfd" ] || [ ! -e "/dev/dri" ]; then
        print_error "AMD GPU devices not found (/dev/kfd or /dev/dri missing)."
        print_status "Ensure ROCm drivers are installed and GPU devices are accessible."
        exit 1
    fi
    
    print_gpu "AMD ROCm devices detected"
fi

# Header
echo "🐳 Quantum Computing 101 Runner v2.0"
echo "====================================="
echo "Variant: $VARIANT"
echo "Image: $IMAGE"
echo ""

# Main execution logic
if [ "$JUPYTER" = true ]; then
    run_jupyter "$IMAGE"
elif [ -n "$EXAMPLE" ]; then
    if [ "$EXAMPLE" = "list" ]; then
        list_examples "$IMAGE"
    elif [ "$EXAMPLE" = "info" ]; then
        show_hardware_info "$IMAGE"
    else
        # Pass remaining arguments to the example
        run_example "$IMAGE" "$EXAMPLE" "$@"
    fi
elif [ -n "$MODULE" ]; then
    print_status "Examples in $MODULE:"
    docker run --rm \
        -v "$(pwd)/examples:/home/qc101/quantum-computing-101/examples:ro" \
        "$IMAGE" \
        find "./examples/$MODULE" -name '*.py' -type f 2>/dev/null | sort || \
        print_error "Module $MODULE not found"
elif [ "$INTERACTIVE" = true ]; then
    run_interactive "$IMAGE"
else
    print_error "No action specified"
    show_usage
    exit 1
fi