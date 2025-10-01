# Quantum Computing 101 - Docker Setup Guide

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Volume Mounting & Development Workflow](#volume-mounting--development-workflow)
- [Directory Structure](#directory-structure)
- [Environment Variables](#environment-variables)
- [Jupyter Lab Access](#jupyter-lab-access)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)

## Overview

This Docker setup provides a unified, flexible solution for running Quantum Computing 101 across different hardware configurations:

- **CPU Only**: Lightweight container for systems without GPU
- **NVIDIA GPU**: CUDA-accelerated quantum computing with PyTorch 2.8.0 + CUDA 12.9
- **AMD GPU**: ROCm-accelerated quantum computing with latest ROCm PyTorch

### ✅ Headless Environment Ready

All examples are fully compatible with headless Docker environments:
- **Non-interactive matplotlib backend** (Agg) configured automatically
- **No display server required** - runs perfectly on remote servers
- **All visualizations save to files** - outputs available in mounted volumes
- **No blocking on plt.show()** - scripts complete without manual intervention

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

## Volume Mounting & Development Workflow

### Current Setup

The `docker-compose.yml` configures volume mounts for seamless host-container development:

```yaml
volumes:
  - ../examples:/home/qc101/quantum-computing-101/examples
  - ../outputs:/home/qc101/quantum-computing-101/outputs
  - ../modules:/home/qc101/quantum-computing-101/modules:ro
```

**What this means:**
- ✅ Edit files in `examples/` on host → Changes appear instantly in container
- ✅ Container outputs save to `outputs/` → Visible on host immediately
- ✅ `modules/` mounted read-only (`:ro`) → Protected from container modifications

### Development Workflows

#### 1. Using Docker Compose (Recommended)

```bash
# Start container with volume mounts
cd docker
docker compose up qc101-nvidia

# In another terminal, edit files on host
cd ../examples/module1_fundamentals
vim 01_classical_vs_quantum_bits.py

# Changes are instantly available in container!
```

#### 2. Interactive Development Workflow

```bash
# Start container in background
docker compose up -d qc101-nvidia

# Exec into container
docker exec -it qc101-nvidia bash

# Inside container - your edits from host are visible
cd /home/qc101/quantum-computing-101/examples
python module1_fundamentals/01_classical_vs_quantum_bits.py
```

#### 3. Jupyter Notebook with Volume Mounts

```bash
# Start with Jupyter
docker compose up qc101-nvidia

# Access Jupyter at http://localhost:8889
# All notebooks saved in container → Synced to host examples/
```

### Advanced Volume Configurations

#### Option A: Mount Entire Project (Full Development Mode)

Edit `docker-compose.yml`:

```yaml
volumes:
  # Mount entire project for maximum flexibility
  - ..:/home/qc101/quantum-computing-101
  # But exclude certain directories
  - /home/qc101/quantum-computing-101/.git
  - /home/qc101/quantum-computing-101/__pycache__
```

**Use when:** Developing library code, not just examples

#### Option B: Selective File Mounting

```yaml
volumes:
  # Mount specific files/directories only
  - ../examples:/home/qc101/quantum-computing-101/examples
  - ../outputs:/home/qc101/quantum-computing-101/outputs
  - ../src:/home/qc101/quantum-computing-101/src
  - ../tests:/home/qc101/quantum-computing-101/tests
```

**Use when:** You want precise control over what's mounted

#### Option C: Using .dockerignore

Create `docker/.dockerignore`:

```
.git
__pycache__
*.pyc
.pytest_cache
.venv
node_modules
*.log
```

Then mount entire project:
```yaml
volumes:
  - ..:/home/qc101/quantum-computing-101
```

### Docker Run Command with Volume Mounts

If not using docker-compose:

```bash
# NVIDIA GPU variant
docker run -it --rm \
  --gpus all \
  -v "$(pwd)/../examples:/home/qc101/quantum-computing-101/examples" \
  -v "$(pwd)/../outputs:/home/qc101/quantum-computing-101/outputs" \
  -p 8889:8888 \
  quantum-computing-101:nvidia

# CPU variant
docker run -it --rm \
  -v "$(pwd)/../examples:/home/qc101/quantum-computing-101/examples" \
  -v "$(pwd)/../outputs:/home/qc101/quantum-computing-101/outputs" \
  -p 8888:8888 \
  quantum-computing-101:cpu
```

### VS Code Dev Container Integration

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "Quantum Computing 101",
  "dockerComposeFile": "../docker/docker-compose.yml",
  "service": "qc101-nvidia",
  "workspaceFolder": "/home/qc101/quantum-computing-101",
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-python.vscode-pylance"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/opt/conda/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true
      }
    }
  },
  
  "mounts": [
    "source=${localWorkspaceFolder},target=/home/qc101/quantum-computing-101,type=bind,consistency=cached"
  ],
  
  "remoteUser": "qc101"
}
```

Then in VS Code: `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

### Performance Considerations

#### Linux/Mac (Native Docker)
- Volume mounts are fast (native filesystem)
- Use `consistency=cached` for better performance (macOS)

#### Windows (WSL2)
- Store project in WSL2 filesystem (`/home/user/project`)
- Avoid mounting from `/mnt/c/` (slow)

#### Example for Mac (Performance Tuning)
```yaml
volumes:
  - ../examples:/home/qc101/quantum-computing-101/examples:cached
  - ../outputs:/home/qc101/quantum-computing-101/outputs:delegated
```

- `:cached` - Host writes, container reads (good for code)
- `:delegated` - Container writes, host reads (good for outputs)

### Volume Mounting Best Practices

1. **Use Docker Compose** for consistent volume configuration
2. **Mount only what you need** to avoid clutter
3. **Use read-only (`:ro`)** for reference materials
4. **Separate data volumes** for outputs and cache
5. **Add .dockerignore** to exclude unnecessary files
6. **Test both directions**: Host→Container and Container→Host writes

### Quick Reference Table

| Scenario | Volume Mount | Description |
|----------|--------------|-------------|
| Edit examples | `../examples:/container/examples` | Live code editing |
| Save outputs | `../outputs:/container/outputs` | Persist results |
| Read-only docs | `../docs:/container/docs:ro` | Reference only |
| Full project | `..:/container/project` | Complete access |
| Named volume | `cache:/home/user/.cache` | Persistent cache |

### Testing Your Volume Setup

```bash
# Test host → container
echo "test from host" > ../examples/test.txt
docker exec qc101-nvidia cat /home/qc101/quantum-computing-101/examples/test.txt

# Test container → host
docker exec qc101-nvidia bash -c "echo 'test from container' > /home/qc101/quantum-computing-101/outputs/test.txt"
cat ../outputs/test.txt

# Cleanup
rm ../examples/test.txt ../outputs/test.txt
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

### Volume Mount Issues

#### Permission Problems
```bash
# Fix ownership if needed
docker exec -it qc101-nvidia chown -R qc101:qc101 /home/qc101/quantum-computing-101
```

#### Files Not Syncing
```bash
# Restart container to remount volumes
docker compose restart qc101-nvidia

# Or recreate container
docker compose down
docker compose up qc101-nvidia
```

#### Check Mount Points
```bash
# Inside container, verify mounts
docker exec -it qc101-nvidia df -h
docker exec -it qc101-nvidia mount | grep qc101
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
