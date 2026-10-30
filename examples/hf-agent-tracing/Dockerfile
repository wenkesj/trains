FROM node:24-bookworm-slim AS agent
WORKDIR /opt/pi
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=agent /usr/local/bin/node /usr/local/bin/node
COPY --from=agent /opt/pi /opt/pi
WORKDIR /workspace
ENTRYPOINT ["/opt/pi/node_modules/.bin/pi"]
