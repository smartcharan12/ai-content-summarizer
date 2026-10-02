#!/usr/bin/env bash
# EC2 "user data" for Amazon Linux 2023: runs once, as root, at first boot.
# Installs Docker + Compose + git, clones the repo and writes a production .env.
# Paste it into "Advanced details -> User data" when launching the instance
# (after editing REPO_URL and MODEL_NAME_OR_PATH). Progress: /var/log/cloud-init-output.log
set -euxo pipefail

REPO_URL="https://github.com/<your-username>/ai-content-summarizer.git"   # <-- edit
MODEL_NAME_OR_PATH="<your-hf-username>/pegasus-samsum"                    # <-- edit (or google/pegasus-xsum)
APP_DIR=/opt/ai-content-summarizer
COMPOSE_VERSION=v5.5.1

# --- Docker, git, Compose plugin --------------------------------------------
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# --- 2 GB swap: headroom while the model loads (it is not a substitute for RAM)
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo "/swapfile swap swap defaults 0 0" >> /etc/fstab
fi

# --- App checkout + production settings --------------------------------------
git clone "$REPO_URL" "$APP_DIR"
cat > "$APP_DIR/.env" <<EOF
APP_ENV=production
LOG_FORMAT=json
LOG_TO_FILE=false
MODEL_SOURCE=path
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
DEVICE=cpu
GENERATION_PROFILE=optimized
MAX_INPUT_CHARS=20000
MAX_BATCH_SIZE=4
CORS_ORIGINS=*
EOF
mkdir -p "$APP_DIR/artifacts/models"
chmod +x "$APP_DIR/deploy/deploy.sh"
chown -R ec2-user:ec2-user "$APP_DIR"
echo "Bootstrap complete. Next: run deploy/deploy.sh (or push to main to let CI deploy)."
