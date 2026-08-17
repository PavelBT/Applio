#!/usr/bin/env python3
"""
Script Asistente de Migración Liviana de Datos Locales a Google Drive.
Filtra inteligentemente solo los modelos exportados finales (.pth), índices (.index),
datasets de audio y pre-entrenados custom para evitar llenar el disco local.
Uso: python3 scripts/migrate_to_drive.py
"""

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main():
    print("=" * 65)
    print("🚀  ASISTENTE DE MIGRACIÓN LIVIANA A GOOGLE DRIVE")
    print("=" * 65)

    # Detect Google Drive local folder if mounted
    cloud_storage = Path.home() / "Library" / "CloudStorage"
    gdrive_target = None
    if cloud_storage.exists():
        for item in cloud_storage.iterdir():
            if "GoogleDrive" in item.name or "google" in item.name.lower():
                my_drive = item / "My Drive"
                if my_drive.exists():
                    gdrive_target = my_drive / "ApplioBackup"
                    break

    if not gdrive_target:
        gdrive_target = BASE / "ApplioBackup_Migration_Output"

    gdrive_target.mkdir(parents=True, exist_ok=True)
    print(f"📂 Carpeta destino de migración: {gdrive_target}\n")

    weights_dir = gdrive_target / "weights"
    indexes_dir = gdrive_target / "indexes"
    datasets_dir = gdrive_target / "datasets"
    pretrained_dir = gdrive_target / "pretrained"

    for d in [weights_dir, indexes_dir, datasets_dir, pretrained_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copiar SOLO modelos .pth exportados e índices desde logs/
    logs_local = BASE / "logs"
    copied_models = 0
    copied_indexes = 0
    if logs_local.exists():
        for model_folder in logs_local.iterdir():
            if (
                model_folder.is_dir()
                and not model_folder.name.startswith("mute")
                and model_folder.name != "reference"
            ):
                # Copy export .pth and .index
                for f in model_folder.glob("*.pth"):
                    if "_best_epoch" in f.name or (
                        "e_" in f.name and "s.pth" in f.name
                    ):
                        dest_pth = weights_dir / f.name
                        if not dest_pth.exists():
                            shutil.copy2(f, dest_pth)
                            print(f"  ➡️ Modelo .pth exportado a weights/: {f.name}")
                            copied_models += 1

                for idx in model_folder.glob("*.index"):
                    dest_idx = indexes_dir / idx.name
                    if not dest_idx.exists():
                        shutil.copy2(idx, dest_idx)
                        print(f"  ➡️ Índice .index copiado a indexes/: {idx.name}")
                        copied_indexes += 1

    # 2. Copiar datasets de audios
    datasets_local = BASE / "assets" / "datasets"
    copied_datasets = 0
    if datasets_local.exists():
        for dataset_folder in datasets_local.iterdir():
            if dataset_folder.is_dir() and not dataset_folder.name.startswith("."):
                dest_ds = datasets_dir / dataset_folder.name
                if not dest_ds.exists():
                    print(f"  🎵 Migrando dataset de audio: {dataset_folder.name}")
                    shutil.copytree(dataset_folder, dest_ds, dirs_exist_ok=True)
                    copied_datasets += 1

    # 3. Copiar pre-entrenados custom (TITAN Medium)
    custom_pretrained_local = BASE / "rvc" / "models" / "pretraineds" / "custom"
    copied_pretraineds = 0
    if custom_pretrained_local.exists():
        for pt in custom_pretrained_local.glob("*.pth"):
            dest_pt = pretrained_dir / pt.name
            if not dest_pt.exists():
                shutil.copy2(pt, dest_pt)
                print(f"  🧠 Pre-entrenado custom copiado a pretrained/: {pt.name}")
                copied_pretraineds += 1

    print("\n" + "=" * 65)
    print("🎉 RESUMEN DE MIGRACIÓN LIVIANA PREPARADA:")
    print(f"   • Modelos (.pth): {copied_models}")
    print(f"   • Índices (.index): {copied_indexes}")
    print(f"   • Datasets de audio: {copied_datasets}")
    print(f"   • Pre-entrenados custom: {copied_pretraineds}")
    print("=" * 65)
    print(f"✅ Carpeta de migración limpia generada en: {gdrive_target}")


if __name__ == "__main__":
    main()
