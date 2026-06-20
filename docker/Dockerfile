ARG DRESSAGE_BASE_PLATFORM=linux/amd64
FROM --platform=${DRESSAGE_BASE_PLATFORM} slimerl/slime:nightly-dev-20260430b

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OPENCODE_VERSION=1.17.4
ARG OPENCLAW_VERSION=2026.6.6

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    PATH=/root/.opencode/bin:/root/.openclaw/bin:${PATH} \
    DRESSAGE_BLACKBOX_RUNNER_MODE=bwrap \
    DRESSAGE_BLACKBOX_BWRAP_BIN=bwrap \
    OPENCODE_BIN=/usr/local/bin/opencode \
    OPENCLAW_BIN=/usr/local/bin/openclaw

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      bubblewrap \
      ca-certificates \
      curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://opencode.ai/install | bash -s -- --version "${OPENCODE_VERSION}" && \
    ln -sf "${HOME}/.opencode/bin/opencode" /usr/local/bin/opencode && \
    /usr/local/bin/opencode --version

RUN curl -fsSL https://openclaw.ai/install.sh | bash -s -- --version "${OPENCLAW_VERSION}" --no-onboard && \
    openclaw_resolved="$(command -v openclaw)" && \
    ln -sf "${openclaw_resolved}" /usr/local/bin/openclaw && \
    /usr/local/bin/openclaw --version

WORKDIR /root/Dressage
COPY . /root/Dressage

RUN python -m pip install --no-cache-dir --no-build-isolation -e "." && \
    python -m pip install --no-cache-dir --no-build-isolation ./blackbox_server

CMD ["bash"]
