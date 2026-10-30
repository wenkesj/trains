FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" UV_LINK_MODE=copy
WORKDIR /app
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN uv pip install --python /opt/venv/bin/python --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
COPY train /app/train
RUN uv pip install --no-sources --python /opt/venv/bin/python -e "/app/train[accelerated]"
ENTRYPOINT ["train", "serve"]
