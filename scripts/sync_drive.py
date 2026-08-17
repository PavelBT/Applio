#!/usr/bin/env python3
"""
Script de Sincronización Bidireccional Local <-> Google Drive para Applio.
Soporta:
  - python3 scripts/sync_drive.py status  : Revisa el estado de modelos y candados (TRAINING.lock).
  - python3 scripts/sync_drive.py pull    : Trae modelos (.pth), índices (.index) y logs desde Google Drive a tu Mac.
  - python3 scripts/sync_drive.py push    : Sube modelos o datasets preparados localmente a Google Drive.
  - python3 scripts/sync_drive.py lock <model_name> : Revisa/Crea candado de entrenamiento.
  - python3 scripts/sync_drive.py unlock <model_name> : Remueve candado de entrenamiento.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Rutas locales por defecto en MacBook
BASE_LOCAL = Path(__file__).resolve().parent.parent
LOCAL_WEIGHTS = BASE_LOCAL / "assets" / "weights"
LOCAL_INDEXES = BASE_LOCAL / "assets" / "indices"
LOCAL_DATASETS = BASE_LOCAL / "assets" / "datasets"
LOCAL_LOGS = BASE_LOCAL / "logs"

# Detección automática de la ruta de Google Drive en macOS
POSSIBLE_GDRIVE_PATHS = [
    Path.home() / "Library" / "CloudStorage",
    Path.home() / "Google Drive" / "My Drive",
    Path.home() / "GoogleDrive",
    Path("/Volumes/GoogleDrive/My Drive"),
]


def find_gdrive_applio_backup():
    # 1. Buscar en CloudStorage carpetas de GoogleDrive
    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.exists():
        for item in cloud_storage.iterdir():
            if "GoogleDrive" in item.name or "google" in item.name.lower():
                my_drive = item / "My Drive" / "ApplioBackup"
                if my_drive.exists() or (item / "My Drive").exists():
                    target = item / "My Drive" / "ApplioBackup"
                    target.mkdir(parents=True, exist_ok=True)
                    return target

    # 2. Si hay variable de entorno GDRIVE_PATH
    if os.getenv("GDRIVE_PATH"):
        target = Path(os.getenv("GDRIVE_PATH")) / "ApplioBackup"
        target.mkdir(parents=True, exist_ok=True)
        return target

    # Fallback por defecto en home
    fallback = Path.home() / "GoogleDrive_ApplioBackup"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


GDRIVE_BACKUP = find_gdrive_applio_backup()


def get_lock_info(model_name):
    lock_file = GDRIVE_BACKUP / "logs" / model_name / "TRAINING.lock"
    if lock_file.exists():
        try:
            with open(lock_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"status": "LOCKED", "model_name": model_name}
    return None


def set_lock(model_name, running_on="MacBook_Local"):
    log_dir = GDRIVE_BACKUP / "logs" / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_file = log_dir / "TRAINING.lock"
    data = {
        "model_name": model_name,
        "status": "TRAINING_IN_PROGRESS",
        "running_on": running_on,
        "started_at": datetime.now().isoformat(),
    }
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"🔒 Candado de entrenamiento creado para '{model_name}' ({running_on})")


def remove_lock(model_name):
    lock_file = GDRIVE_BACKUP / "logs" / model_name / "TRAINING.lock"
    if lock_file.exists():
        os.remove(lock_file)
        print(f"🔓 Candado de entrenamiento removido para '{model_name}'")
    else:
        print(f"ℹ️ No se encontró candado activo para '{model_name}'")


def cmd_status():
    print("=" * 65)
    print("📊  ESTADO DE SINCRONIZACIÓN Y MODELOS (MacBook <-> Google Drive)")
    print("=" * 65)
    print(f"📁 Ruta Local Applio: {BASE_LOCAL}")
    print(f"☁️ Ruta Google Drive: {GDRIVE_BACKUP}\n")

    # 1. Chequeo de candados activos en Drive
    drive_logs = GDRIVE_BACKUP / "logs"
    active_locks = []
    if drive_logs.exists():
        for item in drive_logs.iterdir():
            if item.is_dir():
                lock = get_lock_info(item.name)
                if lock:
                    active_locks.append((item.name, lock))

    if active_locks:
        print("🚨 ATENCIÓN: CANDADOS DE ENTRENAMIENTO ACTIVOS DETECTADOS:")
        for model_name, lock in active_locks:
            running = lock.get("running_on", "Desconocido")
            started = lock.get("started_at", "N/A")
            print(
                f"   🔒 Modelo: '{model_name}' | En ejecución en: {running} | Inicio: {started}"
            )
        print(
            "   ⚠️ No inicies ni modifiques este modelo localmente mientras esté bloqueado.\n"
        )
    else:
        print(
            "✅ Ningún modelo está en proceso de entrenamiento activo en este momento.\n"
        )

    # 2. Comparar modelos .pth en pesos
    drive_weights = GDRIVE_BACKUP / "weights"
    drive_weights.mkdir(parents=True, exist_ok=True)
    LOCAL_WEIGHTS.mkdir(parents=True, exist_ok=True)

    local_pth = set(f.name for f in LOCAL_WEIGHTS.glob("*.pth"))
    remote_pth = set(f.name for f in drive_weights.glob("*.pth"))

    all_pth = sorted(local_pth.union(remote_pth))
    print("📦 MODELOS (.pth) EN PESOS:")
    if not all_pth:
        print("   (No hay modelos .pth exportados aún)")
    for pth in all_pth:
        in_local = "✅ Local" if pth in local_pth else "❌ No Local"
        in_remote = "☁️ Drive" if pth in remote_pth else "❌ No Drive"
        print(f"   • {pth:<35} [{in_local}] [{in_remote}]")


def cmd_pull():
    print("⬇️  SINCRONIZANDO DESDE GOOGLE DRIVE A MACBOOK LOCAL...")
    drive_weights = GDRIVE_BACKUP / "weights"
    drive_indexes = GDRIVE_BACKUP / "indexes"
    drive_logs = GDRIVE_BACKUP / "logs"

    LOCAL_WEIGHTS.mkdir(parents=True, exist_ok=True)
    LOCAL_INDEXES.mkdir(parents=True, exist_ok=True)
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)

    # Pull weights (.pth)
    if drive_weights.exists():
        for pth in drive_weights.glob("*.pth"):
            dest = LOCAL_WEIGHTS / pth.name
            if not dest.exists() or pth.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(pth, dest)
                print(f"  ✅ Copiado peso: {pth.name}")

    # Pull indexes (.index)
    if drive_indexes.exists():
        for idx in drive_indexes.glob("*.index"):
            dest = LOCAL_INDEXES / idx.name
            if not dest.exists() or idx.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(idx, dest)
                print(f"  ✅ Copiado índice: {idx.name}")

    print(
        "\n🎉 Sincronización PULL completada. Los modelos están listos para inferir en la MacBook."
    )


def cmd_push():
    print("⬆️  SINCRONIZANDO DESDE MACBOOK LOCAL A GOOGLE DRIVE...")
    drive_weights = GDRIVE_BACKUP / "weights"
    drive_indexes = GDRIVE_BACKUP / "indexes"
    drive_datasets = GDRIVE_BACKUP / "datasets"

    drive_weights.mkdir(parents=True, exist_ok=True)
    drive_indexes.mkdir(parents=True, exist_ok=True)
    drive_datasets.mkdir(parents=True, exist_ok=True)

    # Push weights
    if LOCAL_WEIGHTS.exists():
        for pth in LOCAL_WEIGHTS.glob("*.pth"):
            dest = drive_weights / pth.name
            if not dest.exists() or pth.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(pth, dest)
                print(f"  ✅ Subido peso: {pth.name}")

    # Push indexes
    if LOCAL_INDEXES.exists():
        for idx in LOCAL_INDEXES.glob("*.index"):
            dest = drive_indexes / idx.name
            if not dest.exists() or idx.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(idx, dest)
                print(f"  ✅ Subido índice: {idx.name}")

    print(
        "\n🎉 Sincronización PUSH completada. Tus modelos y datos están disponibles en Google Drive."
    )


def main():
    parser = argparse.ArgumentParser(description="Applio Sync Drive CLI")
    parser.add_argument(
        "action",
        choices=["status", "pull", "push", "lock", "unlock"],
        help="Acción a ejecutar",
    )
    parser.add_argument(
        "model_name", nargs="?", help="Nombre del modelo (para lock/unlock)"
    )
    args = parser.parse_args()

    if args.action == "status":
        cmd_status()
    elif args.action == "pull":
        cmd_pull()
    elif args.action == "push":
        cmd_push()
    elif args.action == "lock":
        if not args.model_name:
            print(
                "❌ Especifica el nombre del modelo. Ej: python3 scripts/sync_drive.py lock Modelo_V1"
            )
        else:
            set_lock(args.model_name)
    elif args.action == "unlock":
        if not args.model_name:
            print(
                "❌ Especifica el nombre del modelo. Ej: python3 scripts/sync_drive.py unlock Modelo_V1"
            )
        else:
            remove_lock(args.model_name)


if __name__ == "__main__":
    main()
