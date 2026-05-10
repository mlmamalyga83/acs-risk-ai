# ============================================================
# ACS ECG Detector — загрузка датасетов (Windows PowerShell)
# ============================================================
# PowerShell не умеет рекурсивный wget.
# Скачайте ZIP-архивы вручную и распакуйте:

Write-Host "=== PTB-XL (15 ГБ) ==="
Write-Host "Скачайте ZIP:"
Write-Host "  https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
Write-Host "Распакуйте в: data/raw/ptb-xl/"
Write-Host "(содержимое папки ptb-xl-...-1.0.3/ должно быть в data/raw/ptb-xl/)"
Write-Host ""

Write-Host "=== MIT-BIH ST-T (50 МБ) ==="
Write-Host "Скачайте ZIP:"
Write-Host "  https://physionet.org/static/published-projects/stdb/mit-bih-st-t-change-database-1.0.0.zip"
Write-Host "Распакуйте в: data/external/mit-bih-stt/"
Write-Host "(содержимое папки mit-bih-.../ должно быть в data/external/mit-bih-stt/)"
Write-Host ""

Write-Host "После распаковки запустите: python run.py --stage check"
