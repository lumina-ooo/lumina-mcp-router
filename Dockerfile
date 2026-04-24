# syntax=docker/dockerfile:1.7
# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# System deps for numpy wheels (usually none needed on slim + binary wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LISTEN_HOST=0.0.0.0 \
    LISTEN_PORT=8080

RUN groupadd -r -g 10001 lumina \
 && useradd  -r -u 10001 -g lumina -d /home/lumina -s /sbin/nologin -m lumina

COPY --from=builder /opt/venv /opt/venv

USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["python", "-m", "lumina_mcp_router"]
