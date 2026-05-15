#!/bin/bash
set -e

echo "============================================================"
echo " ACS ECG Detector — Server Setup"
echo "============================================================"
echo ""

# 1. System packages
echo "[1/6] Installing system packages..."
apt update -qq && apt install -y -qq \
    python3.10 python3.10-venv python3.10-dev \
    git unzip wget screen htop curl \
    2>&1 | tail -1
echo "  OK"

# 2. CUDA check
echo "[2/6] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  WARNING: NVIDIA driver not found. Install CUDA:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    echo "  Or use: python run.py --stage cnn --cpu-only"
fi

# 3. Python virtual environment
echo "[3/6] Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
echo "  OK"

# 4. Python packages
echo "[4/6] Installing Python packages..."
pip install -r requirements.txt -q
echo "  OK"

# 5. Download data from Yandex.Disk
echo "[5/6] Downloading processed data (5.7 GB from Yandex.Disk)..."
PUBLIC_KEY="https://disk.yandex.ru/d/03GsU4NnX2RpEQ"
API_URL="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=$PUBLIC_KEY"
python3 -c "
import urllib.request, json
with urllib.request.urlopen('$API_URL') as r:
    url = json.loads(r.read())['href']
with open('/tmp/dl_url.txt', 'w') as f:
    f.write(url)
"
wget -O processed.zip -q --show-progress "$(cat /tmp/dl_url.txt)"
unzip -o processed.zip
rm -f /tmp/dl_url.txt
echo "  OK"

# 6. Verify
echo "[6/6] Verifying installation..."
python -c "
import torch, numpy as np, sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name}, Memory: {p.total_memory/1e9:.1f} GB')
print(f'  NumPy: {np.__version__}')
import os
for f in ['X_train_batch0.npy', 'X_val.npy', 'X_test.npy', 'y_train.npy']:
    p = f'data/processed/{f}'
    if os.path.exists(p):
        sz = os.path.getsize(p) / 1e6
        print(f'  Data: {f} ({sz:.0f} MB)')
"
echo "  OK"

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo "Run training:"
echo "  screen -S ecg_train"
echo "  source venv/bin/activate"
echo "  python run.py --stage all --tune"
echo ""
echo "Detach: Ctrl+A  D"
echo "Reattach: screen -r ecg_train"
echo ""
