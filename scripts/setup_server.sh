#!/bin/bash
set -e

echo "============================================================"
echo " ACS ECG Detector — Server Setup"
echo "============================================================"
echo ""

# 1. Проверка OS
if [ ! -f /etc/os-release ]; then
    echo "ERROR: Only Linux (Ubuntu/Debian) is supported"
    exit 1
fi

# 2. Системные пакеты
echo "[1/5] Installing system packages..."
apt update -qq && apt install -y -qq \
    python3.10 python3.10-venv python3.10-dev \
    git unzip wget screen htop nvtop curl \
    2>&1 | tail -1
echo "  OK"

# 3. CUDA check
echo "[2/5] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found."
    echo "  Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads"
    echo "  Or use CPU mode: python run.py --stage cnn --cpu-only"
fi

# 4. Python virtual environment
echo "[3/5] Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
echo "  OK"

# 5. Python packages
echo "[4/5] Installing Python packages..."
pip install -r requirements.txt -q
echo "  OK"

# 6. Verify
echo "[5/5] Verifying installation..."
python -c "
import torch, numpy as np, sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name}, Memory: {p.total_memory/1e9:.1f} GB')
print(f'  NumPy: {np.__version__}')
"
echo "  OK"

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Upload processed.zip to server:"
echo "     scp processed.zip root@YOUR_SERVER_IP:~/acs-risk-ai/"
echo ""
echo "  2. Extract data:"
echo "     cd ~/acs-risk-ai && unzip -o processed.zip"
echo ""
echo "  3. Run training:"
echo "     source venv/bin/activate"
echo "     screen -S ecg_train"
echo "     python run.py --stage all --tune"
echo ""
echo "  4. Detach from screen: Ctrl+A then D"
echo "  5. Reattach: screen -r ecg_train"
echo ""
