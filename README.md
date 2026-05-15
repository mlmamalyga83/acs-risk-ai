# ACS ECG Detector

Детекция ЭКГ-признаков Острого Коронарного Синдрома с интерпретацией.
Мультимодальная модель (ResNet1D + клинические данные), Grad-CAM, Streamlit UI.

⚠️ Исследовательский прототип. Не медицинское изделие.

## Установка на сервер с GPU (A5000 и выше)

### 1. SSH на сервер
```bash
ssh root@IP_АДРЕС
```

### 2. Скачать код
```bash
git clone https://github.com/mlmamalyga83/acs-risk-ai.git
cd acs-risk-ai
```

### 3. Запустить установку
```bash
bash scripts/setup_server.sh
```

### 4. Загрузить данные с ПК
На вашем ПК (PowerShell):
```powershell
scp D:\ML_ECG\acs-risk-ai\processed.zip root@IP_АДРЕС:/root/acs-risk-ai/
```

### 5. Распаковать и запустить
```bash
unzip -o processed.zip
source venv/bin/activate
screen -S ecg_train
python run.py --stage all --tune
```

`Ctrl+A` затем `D` — выйти из screen (обучение останется в фоне).
`screen -r ecg_train` — вернуться.

### 6. Скачать модели на ПК
```powershell
scp root@IP_АДРЕС:/root/acs-risk-ai/models/*.pt D:\ML_ECG\acs-risk-ai\models\
```

## Локальный запуск (без GPU)

```bash
pip install -r requirements.txt
python run.py --stage check
python run.py --stage cnn --cpu-only
```

## Запуск Streamlit UI

```bash
streamlit run src/app/main.py
# → http://localhost:8501
```

## Данные

- PTB-XL: 21,837 записей — обучение
- MIT-BIH ST-T: 28 записей — внешняя валидация

## Полное ТЗ

См. `TOR_ACS_Risk_Scoring.md`.
