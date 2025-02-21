# Build stage
FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry with verbose output
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry

# Verify poetry installation
RUN poetry --version

# Configure poetry
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml poetry.lock ./

# Debug: List files and show pyproject.toml content
RUN ls -la && cat pyproject.toml

# Clean install dependencies
RUN poetry install

# Copy application code
COPY brutus ./brutus

# Install the application
RUN poetry install --only main --no-interaction --verbose

# Final stage
FROM python:3.10-slim

# Copy virtual environment and application from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/brutus /app/brutus

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
