FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 git ca-certificates bash ripgrep && rm -rf /var/lib/apt/lists/*
RUN npm install -g opencode-ai@1.19.5
WORKDIR /workspace
COPY run_harness.py /opt/run_harness.py
