# Base image: Python 3.12 + uv preinstalled (Debian slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Work directory inside the container
WORKDIR /cli

# Install git so we can clone the repo
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Use only production deps (no dev extras if you add them later)
ENV UV_NO_DEV=1

# Clone the repo into /app
RUN git clone --branch 0.1.0 https://github.com/cellgeni/h5ad-cli.git .


# Install the project according to pyproject.toml + uv.lock
# --locked asserts that uv.lock is in sync with pyproject.toml
RUN uv sync --locked

# Put the project venv on PATH so `h5ad` is directly runnable
ENV PATH="/cli/.venv/bin:${PATH}"

# Default entrypoint: run the CLI
ENTRYPOINT ["h5ad"]
CMD ["--help"]

