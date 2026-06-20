#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
VERSION="${VERSION:-v0.1.0}"
IMAGE_NAME="${IMAGE_NAME:-dressage:${VERSION}}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

docker build "$@" \
  --build-arg "DRESSAGE_BASE_PLATFORM=${DOCKER_PLATFORM}" \
  --platform "${DOCKER_PLATFORM}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"
