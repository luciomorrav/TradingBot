#!/bin/bash
# Trading Bot — VPS Setup Script
# Run as root on a fresh Ubuntu 22.04/24.04 server
#
# Usage (recommended — verify the script before running):
#   scp deploy/setup.sh root@YOUR_VPS_IP:~ && ssh root@YOUR_VPS_IP 'bash setup.sh'

set -euo pipefail

APP_USER="botuser"
APP_DIR="/opt/tradingbot"
REPO_URL="https://github.com/luciomorrav/TradingBot.git"
PYTHON_VERSION="3.12"  # Ubuntu 24.04 ships 3.12

echo "=== Trading Bot VPS Setup ==="

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip git curl > /dev/null

# 2. Create app user (no login shell needed)
echo "[2/6] Creating app user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash "$APP_USER"
fi

# 3. Clone repo
echo "[3/6] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$APP_DIR" && sudo -u "$APP_USER" git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
fi

# 4. Python venv + dependencies
echo "[4/6] Setting up Python environment..."
cd "$APP_DIR"
sudo -u "$APP_USER" python${PYTHON_VERSION} -m venv venv
sudo -u "$APP_USER" ./venv/bin/pip install --upgrade pip -q
sudo -u "$APP_USER" ./venv/bin/pip install -r requirements.txt -q

# 5. Create directories
echo "[5/6] Creating directories..."
sudo -u "$APP_USER" mkdir -p "$APP_DIR/logs" "$APP_DIR/data"

# 6. Install systemd service
echo "[6/6] Installing systemd service..."
cp "$APP_DIR/deploy/tradingbot.service" /etc/systemd/system/tradingbot.service
systemctl daemon-reload
systemctl enable tradingbot

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Copy your secrets file:"
echo "     scp config/secrets.env root@YOUR_VPS_IP:${APP_DIR}/config/secrets.env"
echo "     chown ${APP_USER}:${APP_USER} ${APP_DIR}/config/secrets.env"
echo "     chmod 600 ${APP_DIR}/config/secrets.env"
echo ""
echo "  2. Start the bot:"
echo "     systemctl start tradingbot"
echo ""
echo "  3. Check status:"
echo "     systemctl status tradingbot"
echo "     journalctl -u tradingbot -f"
echo ""
echo "  4. View logs:"
echo "     tail -f ${APP_DIR}/logs/bot.log"
