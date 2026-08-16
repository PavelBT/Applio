#!/usr/bin/env python3
"""
Script de Validación y Chequeo de Sincronización Local para Applio (macOS -> Colab).
Uso: python3 scripts/sync_check.py
"""

import sys
import os
import subprocess
import py_compile

def print_header(title):
    print("\n" + "=" * 60)
    print(f" 🔍  {title}")
    print("=" * 60)

def check_git_status():
    print_header("Verificando Estado de Git (Local vs PavelBT/Applio)")
    try:
        # Check uncommitted changes
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if status:
            print("⚠️ Hay cambios locales sin commit:")
            for line in status.split("\n"):
                print(f"   - {line}")
            print("\n👉 Se recomienda hacer commit y push antes de ejecutar en Colab.")
            return False
        else:
            print("✅ El directorio de trabajo está limpio (sin cambios pendientes).")

        # Check unpushed commits
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        remote_check = subprocess.check_output(["git", "status", "-sb"], text=True).strip()
        if "behind" in remote_check:
            print(f"⚠️ Tu rama local esta por detrás de origin/{branch}. Ejecuta: git pull origin {branch}")
            return False
        elif "ahead" in remote_check:
            print(f"⚠️ Tienes commits locales sin subir a GitHub. Ejecuta: git push origin {branch}")
            return False
        else:
            print(f"✅ Tu repositorio local está 100% al día con origin/{branch} (PavelBT/Applio).")
            return True

    except Exception as e:
        print(f"❌ Error al consultar Git: {e}")
        return False

def check_python_syntax():
    print_header("Verificando Sintaxis de Archivos Python Principales")
    files_to_check = ["app.py", "core.py"]
    all_ok = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                py_compile.compile(file_path, doraise=True)
                print(f"✅ Sintaxis correcta: {file_path}")
            except py_compile.PyCompileError as err:
                print(f"❌ Error de sintaxis en {file_path}: {err}")
                all_ok = False
        else:
            print(f"⚠️ Archivo no encontrado: {file_path}")
    return all_ok

def check_environment():
    print_header("Verificando Entorno Local y Dispositivo PyTorch")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ Aceleración Apple Silicon (MPS) DISPONIBLE en PyTorch.")
        elif torch.cuda.is_available():
            print(f"✅ CUDA Disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ PyTorch se ejecutará en modo CPU local.")
    except ImportError:
        print("⚠️ PyTorch no está instalado en el entorno Python actual.")

def main():
    print("🚀 Applio - Script de Verificación Previa a Sincronización")
    git_ok = check_git_status()
    syntax_ok = check_python_syntax()
    check_environment()
    
    print_header("Resultado General")
    if git_ok and syntax_ok:
        print("🎉 ¡TODO LISTO! Tu repositorio local está preparado y sincronizado.")
        print("   Puedes abrir Colab e iniciar entrenamiento con la garantía de tener la última versión.")
    else:
        print("⚠️ Revisa los avisos anteriores antes de iniciar tus celdas en Google Colab.")

if __name__ == "__main__":
    main()
