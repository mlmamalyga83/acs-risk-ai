# ACS ECG Detector

Детекция ЭКГ-признаков Острого Коронарного Синдрома с интерпретацией.
ResNet1D + Grad-CAM + Streamlit UI.

⚠️ Исследовательский прототип. Не медицинское изделие.

## Результаты

| Модель | AUC-ROC | Статус |
|--------|---------|--------|
| CNN (ResNet1D) | **0.876** [0.860-0.893] | ✅ Лучшая |
| Multimodal (ECG + clinical) | 0.685 | ⚠️ Экспериментальная |

## Быстрый старт

### На ПК (Windows)

```powershell
cd D:\ML_ECG\acs-risk-ai
streamlit run src/app/main.py
# → http://localhost:8501
```

### Загрузка моделей с сервера

```powershell
scp -i "$env:USERPROFILE\.ssh\server_temp_key" root@185.182.110.96:/root/acs-risk-ai/models/resnet1d_encoder.pt D:\ML_ECG\acs-risk-ai\models\
scp -i "$env:USERPROFILE\.ssh\server_temp_key" root@185.182.110.96:/root/acs-risk-ai/models/resnet1d_full.pt D:\ML_ECG\acs-risk-ai\models\
```

## Метрики (тестовая выборка, 2831 пациент)

| Метрика | Значение | Цель |
|---------|----------|------|
| AUC-ROC | **0.876** [0.860, 0.893] | ≥ 0.80 ✅ |
| AUC-PR | **0.565** | ≥ 0.50 ✅ |
| Sens @ Spec 90% | **0.571** | ≥ 0.70 ⚠️ |
| NPV | **0.930** | ≥ 0.90 ✅ |
| Fairness (пол) | EO diff = 0.000 | < 0.10 ✅ |
| Fairness (возраст) | EO diff = 0.003 | < 0.10 ✅ |

## Данные

- PTB-XL: 21 799 записей, 18 885 пациентов (Германия)

## Структура проекта

```
acs-risk-ai/
├── run.py                    # CLI-оркестратор
├── config/config.yaml        # Конфигурация
├── src/
│   ├── data/loader.py        # Загрузка PTB-XL
│   ├── preprocessing/        # Фильтры, R-пики, сегментация
│   ├── models/
│   │   ├── cnn_model.py      # ResNet1D, Simple1DCNN
│   │   └── multimodal.py     # MultimodalECGNet (4 heads)
│   ├── train/
│   │   ├── trainer.py        # Циклы обучения
│   │   └── metrics.py        # AUC, DCA, DeLong, Brier
│   ├── interpret/
│   │   ├── grad_cam.py       # Grad-CAM для ЭКГ
│   │   └── visualization.py  # 12-канальные графики
│   └── app/
│       ├── main.py           # Streamlit интерфейс
│       ├── inference.py      # Пайплайн инференса
│       ├── report_generator.py  # Автозаключение
│       ├── red_flags.py      # S1Q3T3, гиперкалиемия
│       └── reference_ranges.py # Референсные значения
├── models/                   # .pt файлы (2.1 MB каждый)
├── scripts/
│   ├── setup_server.sh       # Автоустановка на сервер
│   └── build_multi_labels.py # Multi-label per split
├── TOR_ACS_Risk_Scoring.md   # Техническое задание
└── requirements.txt
```
