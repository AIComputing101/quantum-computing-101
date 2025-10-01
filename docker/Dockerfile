# Quantum Computing 101 - Unified Multi-Architecture Dockerfile
# Supports CPU, NVIDIA GPU (CUDA), and AMD GPU (ROCm)
# Best practices: Multi-stage build, build args, minimal layers

# Build arguments for variant selection
ARG VARIANT=cpu
ARG PYTORCH_VERSION=2.8.0
ARG CUDA_VERSION=12.9
ARG CUDNN_VERSION=9

# =============================================================================
# Stage 1: Base Image Selection
# =============================================================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel AS base-nvidia
FROM rocm/pytorch:latest AS base-amd
FROM python:3.12-slim AS base-cpu

# Select appropriate base based on variant
FROM base-${VARIANT} AS base

# =============================================================================
# Stage 2: Common Setup - System Dependencies
# =============================================================================
FROM base AS system-deps

# Metadata
LABEL maintainer="Quantum Computing 101"
LABEL version="3.0"
LABEL description="Quantum computing educational platform with multi-GPU support"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    QC101_CONTAINER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install common system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    vim \
    htop \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 3: Python Environment Setup
# =============================================================================
FROM system-deps AS python-setup

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 qc101

WORKDIR /home/qc101/quantum-computing-101

# =============================================================================
# Stage 4: Dependencies Installation (CPU Variant)
# =============================================================================
FROM python-setup AS deps-cpu

# Copy requirements
COPY docker/requirements/base.txt /tmp/requirements/base.txt
COPY docker/requirements/cpu.txt /tmp/requirements/cpu.txt

# Install base dependencies
RUN pip install -r /tmp/requirements/base.txt

# Install CPU-specific dependencies
RUN pip install -r /tmp/requirements/cpu.txt

# =============================================================================
# Stage 5: Dependencies Installation (NVIDIA GPU Variant)
# =============================================================================
FROM python-setup AS deps-nvidia

# NVIDIA-specific environment variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=all \
    QISKIT_IN_PARALLEL=TRUE

# Copy requirements
COPY docker/requirements/base.txt /tmp/requirements/base.txt
COPY docker/requirements/gpu-nvidia.txt /tmp/requirements/gpu-nvidia.txt

# Install base dependencies (excluding qiskit-aer to avoid conflicts)
RUN sed '/^qiskit-aer/d' /tmp/requirements/base.txt > /tmp/requirements/base-no-aer.txt && \
    pip install -r /tmp/requirements/base-no-aer.txt

# Install NVIDIA-specific dependencies (includes qiskit-aer-gpu)
RUN pip install -r /tmp/requirements/gpu-nvidia.txt

# =============================================================================
# Stage 6: Dependencies Installation (AMD GPU Variant)
# =============================================================================
FROM python-setup AS deps-amd

# ROCm-specific environment variables
ENV ROCM_VISIBLE_DEVICES=all \
    HIP_VISIBLE_DEVICES=all

# Copy requirements
COPY docker/requirements/base.txt /tmp/requirements/base.txt
COPY docker/requirements/gpu-amd.txt /tmp/requirements/gpu-amd.txt

# Install base dependencies
RUN pip install -r /tmp/requirements/base.txt

# Install AMD-specific dependencies
RUN pip install -r /tmp/requirements/gpu-amd.txt

# =============================================================================
# Stage 7: Application Setup (Unified)
# =============================================================================
FROM deps-${VARIANT} AS app-setup

# Copy application code
COPY --chown=qc101:qc101 . /home/qc101/quantum-computing-101/

# Install the package in development mode
RUN pip install -e .

# Create output directory
RUN mkdir -p /home/qc101/quantum-computing-101/outputs && \
    chown -R qc101:qc101 /home/qc101/quantum-computing-101/outputs

# =============================================================================
# Stage 8: Runtime Image (Final)
# =============================================================================
FROM app-setup AS runtime

# Switch to non-root user
USER qc101

# Set working directory
WORKDIR /home/qc101/quantum-computing-101/examples

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import qiskit; print('✅ Container healthy')" || exit 1

# Resource optimizations
ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# Entrypoint script
COPY --chown=qc101:qc101 docker/entrypoint.sh /home/qc101/entrypoint.sh
RUN chmod +x /home/qc101/entrypoint.sh

ENTRYPOINT ["/home/qc101/entrypoint.sh"]
CMD ["bash"]
