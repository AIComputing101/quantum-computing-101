# Quantum Computing 101 - Docker Setup Guide

## Overview

This Docker setup provides a unified, flexible solution for running Quantum Computing 101 across different hardware configurations:

- **CPU Only**: Lightweight container for systems without GPU
- **NVIDIA GPU**: CUDA-accelerated quantum computing with PyTorch 2.8.0 + CUDA 12.9
- **AMD GPU**: ROCm-accelerated quantum computing with latest ROCm PyTorch

## Architecture

### Unified Dockerfile Approach

The new `Dockerfile` uses multi-stage builds with build arguments to create different variants from a single source:

```
Base Images:
├── CPU:    python:3.12-slim
├── NVIDIA: pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
└── AMD:    rocm/pytorch:latest

Build Stages:
1. base           → Select base image variant
2. system-deps    → Install common system dependencies
3. python-setup   → Setup Python environment
4. deps-{variant} → Install variant-specific dependencies
5. app-setup      → Copy application code
6. runtime        → Final optimized image
```

### Benefits

✅ **Single Source of Truth**: One Dockerfile for all variants
✅ **Reduced Duplication**: Shared stages minimize maintenance
✅ **Build Arguments**: Flexible version control
✅ **Layer Caching**: Optimized build times
✅ **Best Practices**: Multi-stage, minimal layers, non-root user

## Quick Start

### Building Images

```bash
# Build CPU variant
./build-unified.sh cpu

# Build NVIDIA GPU variant
./build-unified.sh nvidia

# Build AMD GPU variant
./build-unified.sh amd

# Build all variants
./build-unified.sh all

# Build without cache
./build-unified.sh nvidia --no-cache
```

### Using Docker Compose

```bash
# Start CPU container
docker-compose up -d qc101-cpu

# Start NVIDIA GPU container
docker-compose up -d qc101-nvidia

# Start AMD GPU container
docker-compose up -d qc101-amd

# View logs
docker-compose logs -f qc101-nvidia

# Stop containers
docker-compose down
```

### Manual Docker Run

```bash
# CPU variant
docker run -it --rm \
  -v $(pwd)/../examples:/home/qc101/quantum-computing-101/examples \
  quantum-computing-101:cpu

# NVIDIA GPU variant
docker run -it --rm \
  --gpus all \
  -v $(pwd)/../examples:/home/qc101/quantum-computing-101/examples \
  quantum-computing-101:nvidia

# AMD GPU variant
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v $(pwd)/../examples:/home/qc101/quantum-computing-101/examples \
  quantum-computing-101:amd
```

## Directory Structure

```
docker/
├── Dockerfile              # Unified multi-variant Dockerfile
├── docker-compose.yml      # Multi-service compose configuration
├── build-unified.sh        # Unified build script
├── entrypoint.sh          # Container entry point
├── requirements/
│   ├── base.txt           # Common dependencies
│   ├── cpu.txt            # CPU-specific dependencies
│   ├── gpu-nvidia.txt     # NVIDIA GPU dependencies
│   └── gpu-amd.txt        # AMD GPU dependencies
├── build.sh               # [DEPRECATED] Old build script
├── Dockerfile.base        # [DEPRECATED] Old base Dockerfile
├── Dockerfile.cpu         # [DEPRECATED] Old CPU Dockerfile
├── Dockerfile.gpu-nvidia  # [DEPRECATED] Old NVIDIA Dockerfile
└── Dockerfile.gpu-amd     # [DEPRECATED] Old AMD Dockerfile
```

## Requirements Files Structure

### base.txt
Common dependencies for all variants:
- Qiskit core frameworks
- Scientific computing (NumPy, SciPy, Matplotlib)
- Machine learning (scikit-learn)
- Jupyter Lab

### cpu.txt
CPU-optimized packages:
- PyTorch CPU version
- TensorFlow CPU version

### gpu-nvidia.txt
NVIDIA GPU-optimized packages:
- CuPy (CUDA arrays)
- qiskit-aer-gpu (GPU-accelerated quantum simulation)
- PyTorch CUDA support
- CUDA tools and monitoring

### gpu-amd.txt
AMD GPU-optimized packages:
- ROCm-specific tools
- qiskit-aer (with ROCm support)

## Environment Variables

### Common
- `QC101_VARIANT`: cpu|gpu-nvidia|gpu-amd
- `PYTHONUNBUFFERED=1`: Real-time Python output
- `MPLBACKEND=Agg`: Non-interactive matplotlib backend

### NVIDIA-Specific
- `NVIDIA_VISIBLE_DEVICES=all`: Expose all GPUs
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: GPU capabilities
- `CUDA_VISIBLE_DEVICES=all`: CUDA device visibility

### AMD-Specific
- `ROCM_VISIBLE_DEVICES=all`: Expose all ROCm devices
- `HIP_VISIBLE_DEVICES=all`: HIP device visibility

## Jupyter Lab Access

Each variant runs Jupyter Lab on different ports to avoid conflicts:

- CPU: http://localhost:8888
- NVIDIA: http://localhost:8889
- AMD: http://localhost:8890

Start Jupyter Lab inside container:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

## Advanced Configuration

### Custom PyTorch/CUDA Versions

Edit `build-unified.sh` or pass build args:

```bash
docker build \
  --build-arg VARIANT=nvidia \
  --build-arg PYTORCH_VERSION=2.9.0 \
  --build-arg CUDA_VERSION=12.9 \
  --build-arg CUDNN_VERSION=9 \
  -t quantum-computing-101:nvidia-custom \
  -f docker/Dockerfile .
```

### Multi-GPU Configuration

```yaml
# docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']  # Specific GPUs
          capabilities: [gpu]
```

## Troubleshooting

### NVIDIA GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.9.1-base nvidia-smi

# Install nvidia-container-toolkit if missing
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### AMD GPU issues

```bash
# Check ROCm installation
rocm-smi

# Verify device permissions
ls -l /dev/kfd /dev/dri

# Add user to video/render groups
sudo usermod -a -G video,render $USER
```

### Build failures

```bash
# Clean Docker cache
docker builder prune -a

# Rebuild without cache
./build-unified.sh nvidia --no-cache

# Check disk space
df -h
```

## Migration Guide

### From Old Structure to New

1. **Backup old files** (already done as `.old` files)
2. **Use new build script**: `./build-unified.sh` instead of `./build.sh`
3. **Update compose commands**: Use service names `qc101-cpu`, `qc101-nvidia`, `qc101-amd`
4. **Check requirements**: Verify your dependencies in consolidated `requirements/` files

### Key Changes

- ✅ Single `Dockerfile` instead of 4 separate files
- ✅ Unified `build-unified.sh` script
- ✅ Updated `docker-compose.yml` with 3 services
- ✅ Centralized `entrypoint.sh` for all variants
- ✅ Improved layer caching and build times

## Performance Tips

1. **Use BuildKit**: `DOCKER_BUILDKIT=1 ./build-unified.sh nvidia`
2. **Parallel builds**: Build multiple variants simultaneously
3. **Volume mounts**: Use volumes for `examples/` and `outputs/` for hot-reloading
4. **Cache volumes**: Persistent pip cache across rebuilds

## Contributing

When adding new dependencies:

1. Add to appropriate `requirements/*.txt` file
2. Test build: `./build-unified.sh <variant> --no-cache`
3. Verify in container: `docker run -it quantum-computing-101:<variant>`
4. Update this README if needed

## License

This Docker setup is part of the Quantum Computing 101 project.
See main project LICENSE for details.
