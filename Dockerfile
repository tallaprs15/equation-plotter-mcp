# Equation Plotter MCP server (Streamable HTTP) for Cloud Run or any container host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PLOTTER_WORKER_MAX_MEMORY_MB=512

WORKDIR /app
# Pinned, hash-checked dependencies first (cached layer), then the project itself.
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . && useradd --system --uid 10001 --no-create-home plotter

USER plotter
# Cloud Run sets $PORT; the server reads it. Bind all interfaces inside the container.
CMD ["equation-plotter-mcp", "--transport", "streamable-http", "--host", "0.0.0.0"]
