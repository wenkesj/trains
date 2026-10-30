FROM node:24-bookworm-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/
ENV UV_PYTHON=3.12 UV_LINK_MODE=copy UV_TOOL_BIN_DIR=/usr/local/bin
WORKDIR /app
COPY pyproject.toml uv.lock Taskfile.yml ./
COPY trains/ trains/
COPY train/pyproject.toml train/README.md train/
COPY pygate/pyproject.toml pygate/README.md pygate/
COPY examples/recipes/pyproject.toml examples/recipes/
COPY docs/ docs/
RUN uv tool install go-task-bin==3.54.0 \
    && task docs:install docs:compile
FROM python:3.12-slim
COPY --from=build /app/docs/dist /site
EXPOSE 8383
CMD ["python", "-m", "http.server", "8383", "--bind", "0.0.0.0", "--directory", "/site"]
