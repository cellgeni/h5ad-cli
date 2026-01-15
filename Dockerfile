# Base image: Python 3.12 + uv preinstalled (Debian slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Environment variables
ENV UV_NO_DEV=1
ENV VENV=/env

# Work directory inside the container
WORKDIR /cli

# Install git so we can clone the repo
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*


# Copy the project files (from GitHub Actions checkout context)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/


# Install the project according to pyproject.toml + uv.lock
# --locked asserts that uv.lock is in sync with pyproject.toml
RUN uv sync --locked

# Create separate venv for csvkit to avoid dependency conflicts
RUN uv venv $VENV --python 3.12 && \
    uv pip install --python $VENV/bin/python csvkit

# Put the project venv on PATH so `h5ad` is directly runnable
ENV PATH="/cli/.venv/bin:${VENV}/bin:${PATH}"

# Default entrypoint: run the CLI
ENTRYPOINT ["h5ad"]
CMD ["--help"]

