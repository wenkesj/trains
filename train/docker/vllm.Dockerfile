FROM vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89
COPY train/vllm_plugin /opt/trains-vllm
RUN uv pip install --python /usr/bin/python3 --break-system-packages --no-deps /opt/trains-vllm
