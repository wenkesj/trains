FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates bash ripgrep && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex@0.165.0
RUN python3 -m venv /opt/arize-runtime && /opt/arize-runtime/bin/pip install --no-cache-dir 'coding-harness-tracing @ git+https://github.com/Arize-ai/coding-harness-tracing.git@e43113ff8420aaac09fc0caf8ce1dc93c9d9221d'
ENV PATH="/opt/arize-runtime/bin:${PATH}"
# Every invocation gets an isolated default CLI profile; no host profile is mounted.
WORKDIR /workspace
# Upstream installer resolves hook executables relative to this shared runtime path.
RUN mkdir -p /root/.arize/harness && ln -s /opt/arize-runtime /root/.arize/harness/venv
COPY run_harness.py /opt/run_harness.py
COPY codex-models.json /opt/codex-models.json
