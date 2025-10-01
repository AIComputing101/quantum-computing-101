#!/bin/bash
# Quantum Computing 101 - Unified Build Script
# Supports CPU, NVIDIA GPU, and AMD GPU variants with best practices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
VARIANT="${1:-cpu}"
NO_CACHE="${2:-false}"
PYTORCH_VERSION="2.8.0"
CUDA_VERSION="12.9"
CUDNN_VERSION="9"

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "🐳 Quantum Computing 101 - Docker Build System v3.0"
    echo "====================================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "Docker is available and running"
}

detect_gpu() {
    local gpu_type="none"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_success "NVIDIA GPU detected (${gpu_count} GPU(s))"
        gpu_type="nvidia"
    # Check for AMD GPU
    elif [ -d "/opt/rocm" ] || command -v rocm-smi &> /dev/null; then
        print_success "AMD ROCm GPU detected"
        gpu_type="amd"
    else
        print_info "No GPU detected, CPU-only mode available"
        gpu_type="cpu"
    fi
    
    echo "$gpu_type"
}

show_usage() {
    cat << EOF
Usage: $0 [VARIANT] [OPTIONS]

Build Docker images for Quantum Computing 101

VARIANTS:
  cpu          Build CPU-only image (default)
  nvidia       Build NVIDIA CUDA GPU image
  amd          Build AMD ROCm GPU image
  all          Build all variants

OPTIONS:
  --no-cache   Build without using cache
  --help       Show this help message

EXAMPLES:
  $0 cpu                 # Build CPU variant
  $0 nvidia              # Build NVIDIA GPU variant
  $0 amd                 # Build AMD GPU variant
  $0 nvidia --no-cache   # Build NVIDIA variant without cache
  $0 all                 # Build all variants

EOF
    exit 0
}

build_image() {
    local variant=$1
    local no_cache_flag=""
    
    if [ "$NO_CACHE" = "true" ] || [ "$NO_CACHE" = "--no-cache" ]; then
        no_cache_flag="--no-cache"
        print_info "Building without cache"
    fi
    
    print_info "Building ${variant} variant..."
    
    local build_args="--build-arg VARIANT=${variant}"
    
    if [ "$variant" = "nvidia" ]; then
        build_args="${build_args} --build-arg PYTORCH_VERSION=${PYTORCH_VERSION}"
        build_args="${build_args} --build-arg CUDA_VERSION=${CUDA_VERSION}"
        build_args="${build_args} --build-arg CUDNN_VERSION=${CUDNN_VERSION}"
    fi
    
    local start_time=$(date +%s)
    
    if docker build \
        -f "${SCRIPT_DIR}/Dockerfile" \
        -t "quantum-computing-101:${variant}" \
        --target runtime \
        ${build_args} \
        ${no_cache_flag} \
        "${PROJECT_ROOT}"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_success "${variant} variant build completed in ${duration}s"
        
        # Show image info
        local size=$(docker images "quantum-computing-101:${variant}" --format "{{.Size}}")
        echo -e "  ${BLUE}Image:${NC} quantum-computing-101:${variant}"
        echo -e "  ${BLUE}Size:${NC} $size"
        echo ""
        
        return 0
    else
        print_error "${variant} variant build failed"
        return 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

print_header

# Parse arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
fi

# Check for --no-cache flag
if [ "$2" = "--no-cache" ] || [ "$1" = "--no-cache" ]; then
    NO_CACHE="true"
fi

# Check Docker
check_docker

# Detect GPU
GPU_TYPE=$(detect_gpu)

# Validate variant selection
case "$VARIANT" in
    cpu)
        print_info "Building CPU-only variant"
        ;;
    nvidia)
        if [ "$GPU_TYPE" != "nvidia" ]; then
            print_warning "NVIDIA GPU not detected, but building NVIDIA variant anyway"
        fi
        print_info "Building NVIDIA CUDA GPU variant"
        ;;
    amd)
        if [ "$GPU_TYPE" != "amd" ]; then
            print_warning "AMD GPU not detected, but building AMD variant anyway"
        fi
        print_info "Building AMD ROCm GPU variant"
        ;;
    all)
        print_info "Building all variants"
        build_image "cpu" && \
        build_image "nvidia" && \
        build_image "amd"
        
        print_success "All variants built successfully"
        exit 0
        ;;
    *)
        print_error "Unknown variant: $VARIANT"
        echo "Valid variants: cpu, nvidia, amd, all"
        echo "Use --help for more information"
        exit 1
        ;;
esac

# Build the selected variant
build_image "$VARIANT"

print_success "Build complete!"
echo ""
echo "To run the container:"
echo "  docker run -it --rm quantum-computing-101:${VARIANT}"
echo ""
echo "Or use Docker Compose:"
echo "  docker-compose up qc101-${VARIANT}"
