# ---------------------------------------------------------
# Build Stage
# ---------------------------------------------------------
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Runtime Stage
# ---------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Ensure the virtual environment is used
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
RUN useradd -m -U appuser

# Copy application code
COPY . .

# Set up data directories with proper permissions
RUN mkdir -p /app/data /app/reports && \
    chown -R appuser:appuser /app/data /app/reports

# Switch to the non-root user
USER appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PROMPTEVO_DATA_DIR=/app/data

# Expose the API port
EXPOSE 8000

# Health check to ensure the API is running (uses Python built-in urllib to avoid curl dependency in slim images)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Start the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
