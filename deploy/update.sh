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

echo "[3b/4] Running security audit on dependencies..."
if sudo -u "$APP_USER" ./venv/bin/pip install pip-audit -q 2>/dev/null; then
    sudo -u "$APP_USER" ./venv/bin/pip-audit --skip-editable 2>&1 || {
        echo "WARNING: pip-audit found vulnerabilities. Review before starting."
        read -rp "Continue anyway? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    }
else
    echo "  (pip-audit not available, skipping)"
fi

echo "[4/4] Starting bot..."
systemctl start tradingbot

echo ""
echo "=== Update complete ==="
echo "Status:"
systemctl status tradingbot --no-pager
