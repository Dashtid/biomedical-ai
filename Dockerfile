# Medical Image Segmentation - Production Docker Image
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /build

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install dependencies and build package
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 medimg && \
    mkdir -p /app /data /models /results && \
    chown -R medimg:medimg /app /data /models /results

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder --chown=medimg:medimg /build/.venv /app/.venv
COPY --from=builder --chown=medimg:medimg /build/src /app/src
COPY --from=builder --chown=medimg:medimg /build/configs /app/configs
COPY --from=builder --chown=medimg:medimg /build/scripts /app/scripts

# Copy additional files
COPY --chown=medimg:medimg README.md QUICKSTART.md ./

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Switch to non-root user
USER medimg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; import med_seg; print('OK')" || exit 1

# Default command
CMD ["python", "--version"]

# Labels
LABEL maintainer="David Dashti <david.dashti@hermesmedical.com>"
LABEL description="Medical Image Segmentation with U-Net architectures"
LABEL version="1.0.0"

# Volume mounts
VOLUME ["/data", "/models", "/results"]

# Expose port for potential web interface
EXPOSE 8080
