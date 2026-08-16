#!/usr/bin/env python3
"""
Script Asistente de Migración de Datos Locales a Google Drive.
Uso: python3 scripts/migrate_to_drive.py
"""

import os
import shutil
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def main():
    print("=" * 65)
    print("🚀  ASISTENTE DE MIGRACIÓN DE DATOS LOCALES A GOOGLE DRIVE")
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
    logs_dir = gdrive_target / "logs"
    pretrained_dir = gdrive_target / "pretrained"

    for d in [weights_dir, indexes_dir, datasets_dir, logs_dir, pretrained_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copiar modelos .pth exportados e índices de logs
    logs_local = BASE / "logs"
    copied_models = 0
    copied_indexes = 0
    if logs_local.exists():
        for model_folder in logs_local.iterdir():
            if model_folder.is_dir() and not model_folder.name.startswith("mute") and model_folder.name != "reference":
                # Copy folder to logs/
                dest_log = logs_dir / model_folder.name
                if not dest_log.exists():
                    print(f"  📦 Migrando carpeta de log: {model_folder.name}")
                    shutil.copytree(model_folder, dest_log, dirs_exist_ok=True)
                
                # Copy .pth and .index to weights/ and indexes/
                for f in model_folder.glob("*.pth"):
                    if "_best_epoch" in f.name or ("e_" in f.name and "s.pth" in f.name):
                        dest_pth = weights_dir / f.name
                        if not dest_pth.exists():
                            shutil.copy2(f, dest_pth)
                            print(f"    ➡️ Modelo .pth exportado a weights/: {f.name}")
                            copied_models += 1

                for idx in model_folder.glob("*.index"):
                    dest_idx = indexes_dir / idx.name
                    if not dest_idx.exists():
                        shutil.copy2(idx, dest_idx)
                        print(f"    ➡️ Índice .index copiado a indexes/: {idx.name}")
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
    print("🎉 RESUMEN DE MIGRACIÓN PREPARADA:")
    print(f"   • Modelos (.pth): {copied_models}")
    print(f"   • Índices (.index): {copied_indexes}")
    print(f"   • Datasets de audio: {copied_datasets}")
    print(f"   • Pre-entrenados custom: {copied_pretraineds}")
    print("=" * 65)

    # Zip output if not direct CloudStorage
    if "CloudStorage" not in str(gdrive_target):
        zip_path = BASE / "Applio_Drive_Migration.zip"
        print(f"\n📦 Creando archivo comprimido listo para subir a Google Drive: {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(gdrive_target):
                for file in files:
                    file_p = Path(root) / file
                    rel_p = file_p.relative_to(gdrive_target)
                    zipf.write(file_p, Path("ApplioBackup") / rel_p)
        print(f"✅ ¡Archivo {zip_path.name} creado exitosamente!")
        print(f"👉 Simplemente sube 'Applio_Drive_Migration.zip' a tu Google Drive y descompresiónala o arrastra la carpeta ApplioBackup_Migration_Output a tu Google Drive.")

if __name__ == "__main__":
    main()
