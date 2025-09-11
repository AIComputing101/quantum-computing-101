#!/bin/bash
# Build script for Quantum Computing 101 Docker containers v2.0
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

# Function to check Docker availability
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "Docker is available and running"
}

# Function to check NVIDIA Docker support
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        # Additional check for nvidia-smi
        if nvidia-smi &> /dev/null; then
            local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
            print_gpu "NVIDIA Docker support detected with $gpu_count GPU(s)"
            return 0
        else
            print_warning "NVIDIA Docker runtime detected but nvidia-smi failed"
            return 1
        fi
    else
        print_warning "NVIDIA Docker support not detected"
        return 1
    fi
}

# Function to check AMD ROCm support
check_rocm_support() {
    if [ -d "/opt/rocm" ] && command -v rocm-smi &> /dev/null; then
        print_gpu "AMD ROCm installation detected"
        return 0
    elif [ -e "/dev/kfd" ] && [ -e "/dev/dri" ]; then
        print_gpu "AMD GPU devices detected (/dev/kfd and /dev/dri present)"
        return 0
    else
        print_warning "AMD ROCm support not detected (no /dev/kfd or /dev/dri devices)"
        return 1
    fi
}

# Function to build a specific variant
build_variant() {
    local variant=$1
    local dockerfile=$2
    local tag=$3
    local description=$4
    
    print_status "Building $description..."
    
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    # Check if Dockerfile exists
    if [ ! -f "docker/$dockerfile" ]; then
        print_error "Dockerfile not found: docker/$dockerfile"
        return 1
    fi
    
    # Build the image with build context optimization
    local start_time=$(date +%s)
    
    if docker build \
        -f "docker/$dockerfile" \
        -t "$tag" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_success "$description build completed in ${duration}s"
        
        # Show image info
        local size=$(docker images "$tag" --format "{{.Size}}")
        local created=$(docker images "$tag" --format "{{.CreatedAt}}")
        echo -e "  ${BLUE}Tag:${NC} $tag"
        echo -e "  ${BLUE}Size:${NC} $size"
        echo -e "  ${BLUE}Created:${NC} $created"
        
        return 0
    else
        print_error "$description build failed"
        return 1
    fi
}

# Function to build all variants
build_all() {
    print_status "Building all available Quantum Computing 101 Docker variants..."
    echo ""
    
    local success_count=0
    local total_count=0
    local variants_built=()
    local variants_failed=()
    
    # Build CPU variant (always available)
    ((total_count++))
    if build_variant "cpu" "Dockerfile.cpu" "quantum101:cpu" "CPU-only variant"; then
        ((success_count++))
        variants_built+=("quantum101:cpu (CPU-only)")
    else
        variants_failed+=("quantum101:cpu (CPU-only)")
    fi
    echo ""
    
    # Build NVIDIA GPU variant (if available)
    if check_nvidia_docker; then
        ((total_count++))
        if build_variant "gpu-nvidia" "Dockerfile.gpu-nvidia" "quantum101:gpu-nvidia" "NVIDIA CUDA GPU variant"; then
            ((success_count++))
            variants_built+=("quantum101:gpu-nvidia (NVIDIA CUDA)")
        else
            variants_failed+=("quantum101:gpu-nvidia (NVIDIA CUDA)")
        fi
        echo ""
    fi
    
    # Build AMD ROCm variant (if available)
    if check_rocm_support; then
        ((total_count++))
        if build_variant "gpu-amd" "Dockerfile.gpu-amd" "quantum101:gpu-amd" "AMD ROCm GPU variant"; then
            ((success_count++))
            variants_built+=("quantum101:gpu-amd (AMD ROCm)")
        else
            variants_failed+=("quantum101:gpu-amd (AMD ROCm)")
        fi
        echo ""
    fi
    
    # Build base variant
    ((total_count++))
    if build_variant "base" "Dockerfile.base" "quantum101:base" "Base development variant"; then
        ((success_count++))
        variants_built+=("quantum101:base (Development)")
    else
        variants_failed+=("quantum101:base (Development)")
    fi
    
    # Summary
    echo ""
    echo "=========================================="
    print_status "Build Summary: $success_count/$total_count variants built successfully"
    echo ""
    
    if [ ${#variants_built[@]} -gt 0 ]; then
        print_success "Successfully built variants:"
        for variant in "${variants_built[@]}"; do
            echo -e "  ${GREEN}✓${NC} $variant"
        done
    fi
    
    if [ ${#variants_failed[@]} -gt 0 ]; then
        echo ""
        print_error "Failed variants:"
        for variant in "${variants_failed[@]}"; do
            echo -e "  ${RED}✗${NC} $variant"
        done
    fi
    
    if [ $success_count -eq $total_count ]; then
        echo ""
        print_success "🎉 All available variants built successfully!"
        echo ""
        print_status "Available images:"
        docker images quantum101 --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        echo ""
        print_status "Quick start commands:"
        echo -e "  ${BLUE}CPU:${NC}          ./run.sh -v cpu -i"
        if check_nvidia_docker; then
            echo -e "  ${BLUE}NVIDIA GPU:${NC}   ./run.sh -v gpu-nvidia -i"
        fi
        if check_rocm_support; then
            echo -e "  ${BLUE}AMD ROCm:${NC}     ./run.sh -v gpu-amd -i"
        fi
    else
        echo ""
        print_warning "Some builds failed. Check the output above for details."
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Build Quantum Computing 101 Docker containers with GPU support"
    echo ""
    echo "Options:"
    echo "  cpu           Build CPU-only variant (lightweight, always available)"
    echo "  gpu-nvidia    Build NVIDIA CUDA GPU variant (requires NVIDIA Docker)"
    echo "  gpu-amd       Build AMD ROCm GPU variant (requires AMD GPU devices)"
    echo "  base          Build base development variant"
    echo "  all           Build all available variants (default)"
    echo "  clean         Remove all quantum101 images and containers"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Build all available variants"
    echo "  $0 cpu            # Build only CPU variant"
    echo "  $0 gpu-nvidia     # Build only NVIDIA GPU variant"
    echo "  $0 gpu-amd        # Build only AMD ROCm variant"
    echo "  $0 clean          # Clean up all images"
    echo ""
    echo "Hardware Requirements:"
    echo "  CPU variant:      Any x86_64 system with Docker"
    echo "  NVIDIA variant:   NVIDIA GPU + CUDA drivers + nvidia-docker"
    echo "  AMD variant:      AMD GPU + ROCm drivers + /dev/kfd,/dev/dri access"
}

# Function to clean up images
clean_images() {
    print_status "Cleaning up Quantum Computing 101 Docker images and containers..."
    
    # Stop and remove containers
    local containers=$(docker ps -a --filter "ancestor=quantum101" --format "{{.ID}}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "$containers" | xargs docker rm -f
        print_success "Removed containers"
    fi
    
    # Remove images
    local images=$(docker images quantum101 -q 2>/dev/null || true)
    if [ -n "$images" ]; then
        echo "$images" | xargs docker rmi -f
        print_success "Removed images"
    else
        print_warning "No quantum101 images found to remove"
    fi
    
    # Clean up build cache
    docker builder prune -f > /dev/null 2>&1 || true
    print_success "Cleaned build cache"
}

# Main execution
main() {
    local action=${1:-all}
    
    echo "🐳 Quantum Computing 101 Docker Builder v2.0"
    echo "============================================="
    echo ""
    
    case $action in
        cpu)
            check_docker
            build_variant "cpu" "Dockerfile.cpu" "quantum101:cpu" "CPU-only variant"
            ;;
        gpu-nvidia)
            check_docker
            if check_nvidia_docker; then
                build_variant "gpu-nvidia" "Dockerfile.gpu-nvidia" "quantum101:gpu-nvidia" "NVIDIA CUDA GPU variant"
            else
                print_error "NVIDIA Docker support required for GPU builds"
                echo "Install: https://github.com/NVIDIA/nvidia-docker"
                exit 1
            fi
            ;;
        gpu-amd)
            check_docker
            if check_rocm_support; then
                build_variant "gpu-amd" "Dockerfile.gpu-amd" "quantum101:gpu-amd" "AMD ROCm GPU variant"
            else
                print_error "AMD ROCm support required for AMD GPU builds"
                echo "Ensure ROCm is installed and /dev/kfd, /dev/dri devices are available"
                exit 1
            fi
            ;;
        base)
            check_docker
            build_variant "base" "Dockerfile.base" "quantum101:base" "Base development variant"
            ;;
        all)
            check_docker
            build_all
            ;;
        clean)
            check_docker
            clean_images
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $action"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"