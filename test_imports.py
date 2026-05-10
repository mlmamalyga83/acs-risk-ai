# ============================================================
# ACS ECG Detector — проверка импорта всех библиотек
# ============================================================

import sys

def check_imports():
    """Проверяет импорт всех библиотек. Возвращает True если всё ОК."""
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scipy": "scipy",
        "wfdb": "wfdb",
        "neurokit2": "neurokit2",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "shap": "shap",
        "streamlit": "streamlit",
        "omegaconf": "omegaconf",
        "yaml": "pyyaml",
        "optuna": "optuna",
    }
    
    errors = []
    for import_name, display_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} — НЕ УСТАНОВЛЕН")
            errors.append(display_name)
    
    if errors:
        print(f"\n❌ Не установлены: {', '.join(errors)}")
        print("  Выполните: pip install -r requirements.txt")
        return False
    
    print(f"\n✅ Все {len(required)} библиотек загружены")
    return True


if __name__ == "__main__":
    print(f"Python {sys.version}")
    print(f"Проверка библиотек ({len({k:v for k,v in {
        'numpy':'','pandas':'','matplotlib':'','seaborn':'',
        'scipy':'','wfdb':'','neurokit2':'','sklearn':'',
        'torch':'','shap':'','streamlit':'','omegaconf':'',
        'yaml':'','optuna':''
    }.items()})} шт.):")
    success = check_imports()
    sys.exit(0 if success else 1)
