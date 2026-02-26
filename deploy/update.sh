#!/bin/bash
# Trading Bot — Update script
# Run on the VPS to pull latest code and restart
#
# Usage: sudo bash /opt/tradingbot/deploy/update.sh

set -euo pipefail

APP_DIR="/opt/tradingbot"
APP_USER="botuser"

echo "=== Updating Trading Bot ==="

echo "[1/4] Stopping bot..."
systemctl stop tradingbot

echo "[2/4] Pulling latest code..."
cd "$APP_DIR"
sudo -u "$APP_USER" git pull

echo "[3/4] Updating dependencies..."
sudo -u "$APP_USER" ./venv/bin/pip install -r requirements.txt -q

echo "[4/4] Starting bot..."
systemctl start tradingbot

echo ""
echo "=== Update complete ==="
echo "Status:"
systemctl status tradingbot --no-pager
