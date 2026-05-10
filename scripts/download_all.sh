#!/bin/bash
# ============================================================
# ACS ECG Detector — автоматическая загрузка датасетов (Linux/macOS)
# ============================================================
set -e

echo "=== 1/2: Скачивание PTB-XL (~15 ГБ) ==="
mkdir -p data/raw/ptb-xl/
wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/ptb-xl/ \
  https://physionet.org/files/ptb-xl/1.0.3/

echo ""
echo "=== 2/2: Скачивание MIT-BIH ST-T (~50 МБ) ==="
mkdir -p data/external/mit-bih-stt/
wget -r -N -c -np -nH --cut-dirs=3 -P data/external/mit-bih-stt/ \
  https://physionet.org/files/stdb/1.0.0/

echo ""
echo "=== ЗАГРУЗКА ЗАВЕРШЕНА ==="
echo "Проверьте: python run.py --stage check"
