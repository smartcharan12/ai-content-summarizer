#!/usr/bin/env bash
# Deploy (or update) the app on the EC2 host. Run by GitHub Actions through
# AWS Systems Manager (no SSH needed), or by hand:
#
#   sudo -u ec2-user /opt/ai-content-summarizer/deploy/deploy.sh <ecr-image-uri> <aws-region>
#
# Steps: pull latest compose/nginx config -> log in to ECR -> pull the new image
# -> restart containers -> wait until /health is 200 -> clean up old images.
set -euo pipefail

IMAGE="${1:?usage: deploy.sh <image-uri> <aws-region>}"
export AWS_REGION="${2:?usage: deploy.sh <image-uri> <aws-region>}"
export IMAGE
APP_DIR="${APP_DIR:-/opt/ai-content-summarizer}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"   # first start downloads the model

cd "$APP_DIR"
echo "==> Updating deployment files"
git pull --ff-only

echo "==> Logging in to ECR"
REGISTRY="${IMAGE%%/*}"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

COMPOSE=(docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml --profile proxy)

echo "==> Pulling $IMAGE"
"${COMPOSE[@]}" pull api

echo "==> Restarting containers"
"${COMPOSE[@]}" up -d --no-build --remove-orphans

echo "==> Waiting for /health (up to ${HEALTH_TIMEOUT_SECONDS}s)"
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
until curl -fsS http://localhost/health > /dev/null; do
  if (( SECONDS > deadline )); then
    echo "!! Health check timed out. Recent API logs:"
    "${COMPOSE[@]}" logs --tail 50 api || true
    exit 1
  fi
  sleep 10
done
echo "==> Healthy: $(curl -fsS http://localhost/health)"

docker image prune -f > /dev/null
echo "==> Deployed $IMAGE"
