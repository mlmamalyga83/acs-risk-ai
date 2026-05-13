#!/usr/bin/env python
"""
Упаковывает данные и конфиги для обучения на Google Colab (GPU T4).
Создаёт processed.zip в корне проекта.

Использование:
  python scripts/pack_for_colab.py
  python scripts/pack_for_colab.py --output "D:\processed.zip"

После упаковки:
  1. Загрузить processed.zip на Google Drive в папку ecg-project/
  2. Открыть notebooks/colab_train.ipynb в Colab
  3. Runtime → Change runtime type → T4 GPU
  4. Запустить все ячейки
"""

import argparse
import zipfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES_TO_PACK = [
    # Train batches (6 files, ~6.8 GB)
    "data/processed/X_train_batch0.npy",
    "data/processed/X_train_batch1.npy",
    "data/processed/X_train_batch2.npy",
    "data/processed/X_train_batch3.npy",
    "data/processed/X_train_batch4.npy",
    "data/processed/X_train_batch5.npy",
    "data/processed/X_train_manifest.txt",

    # Val and test
    "data/processed/X_val.npy",
    "data/processed/X_test.npy",

    # Labels
    "data/processed/y_train.npy",
    "data/processed/y_val.npy",
    "data/processed/y_test.npy",

    # Clinical
    "data/processed/clinical_train.npy",
    "data/processed/clinical_val.npy",
    "data/processed/clinical_test.npy",

    # Patient IDs
    "data/processed/patient_ids_train.npy",
    "data/processed/patient_ids_val.npy",
    "data/processed/patient_ids_test.npy",

    # Metadata
    "data/processed/metadata_enriched.csv",

    # Configs
    "config/config.yaml",
    "config/params.yaml",
]


def pack_for_colab(output_path=None):
    if output_path is None:
        output_path = ROOT / "processed.zip"
    output_path = Path(output_path)

    # Verify all files exist
    missing = []
    for f in FILES_TO_PACK:
        if not (ROOT / f).exists():
            missing.append(f)

    if missing:
        print("MISSING files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun preprocessing first: python run.py --stage preprocess")
        return False

    # Calculate total size
    total_bytes = sum((ROOT / f).stat().st_size for f in FILES_TO_PACK)
    print(f"Packing {len(FILES_TO_PACK)} files ({total_bytes / 1e9:.1f} GB)...")
    print(f"Output: {output_path}")
    print()

    t0 = time.time()
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, f in enumerate(FILES_TO_PACK):
            file_path = ROOT / f
            arcname = str(f.replace('\\', '/'))
            file_size = file_path.stat().st_size

            zf.write(file_path, arcname)
            elapsed = time.time() - t0
            speed = (total_bytes / max(elapsed, 1)) / 1e6
            ratio = zf.getinfo(arcname).compress_size / max(file_size, 1)
            pct = (i + 1) / len(FILES_TO_PACK) * 100
            print(f"  [{pct:5.1f}%] {f:<45s} {file_size/1e6:6.1f} MB -> {zf.getinfo(arcname).compress_size/1e6:6.1f} MB (x{ratio:.2f})")

    elapsed = time.time() - t0
    zip_size = output_path.stat().st_size

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Compressed: {total_bytes/1e9:.1f} GB -> {zip_size/1e9:.1f} GB (x{total_bytes/max(zip_size,1):.2f})")
    print(f"File: {output_path}")

    print("\nNext steps:")
    print(f"  1. Upload {output_path.name} to Google Drive -> ecg-project/")
    print("  2. Open notebooks/colab_train.ipynb in Colab")
    print("  3. Runtime -> Change runtime type -> T4 GPU")
    print("  4. Run all cells")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack data for Colab training")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for processed.zip")
    args = parser.parse_args()
    pack_for_colab(output_path=args.output)
