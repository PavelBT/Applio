# AGENTS.md - Reglas, Políticas y Arquitectura del Proyecto Applio Híbrido

> **Propósito:** Este documento define la arquitectura, normas operativas, políticas de sincronización y directivas de desarrollo para el entorno híbrido de **Applio (RVC)** entre ejecución **Local (MacBook macOS)** y **Nube (Google Colab GPU)**. Todos los desarrolladores y Agentes de IA deben seguir strictly estas reglas.

---

## 1. Visión General del Proyecto

Este proyecto está diseñado para el entrenamiento e inferencia de modelos de conversión de voz por inteligencia artificial (RVC - Retrieval-based Voice Conversion) usando la plataforma **Applio**.

### Objetivo Principal
Lograr una integración fluida y automatizada entre:
1. **Desarrollo e Inferencia Local (MacBook):** Edición de código, pruebas de componentes, preparación de datos e inferencias livianas en macOS usando aceleración MPS/CPU.
2. **Entrenamiento e Inferencia de Alto Rendimiento (Google Colab):** Ejecución intensiva de entrenamiento de épocas RVC, extracción de características con GPU (Nvidia T4/V100/A100) y procesamiento masivo.
3. **Sincronización Automática:** Garantizar que cualquier cambio en el código fuente local esté disponible instantáneamente en Google Colab sin perder persistencia de datos (modelos `.pth`, índices FAISS `.index`, logs y datasets).

---

## 2. Arquitectura de Entornos y Matriz de Roles

| Componente | Entorno Local (MacBook) | Entorno Nube (Google Colab) |
| :--- | :--- | :--- |
| **Sistema Operativo** | macOS (Darwin) | Linux (Ubuntu Colab Runtime) |
| **Dispositivo PyTorch** | `mps` (Apple Silicon) / `cpu` | `cuda` (Nvidia GPU) |
| **Rol Principal** | Desarrollo, refactorización, UI testing, curación de audios | Entrenamiento pesado, extracción de f0/embeds, FAISS index build |
| **Gestión de Código** | Repositorio Git Local | Enlace/Clonación automática vía Git desde GitHub/Drive |
| **Almacenamiento Pesado** | Disco Local (`assets/`, `logs/`) | Google Drive Montado (`/content/drive/MyDrive/Applio_Data`) |

---

## 3. Estrategia de Sincronización Automática (Local <-> Colab)

Para lograr una automatización limpia sin saturar el control de versiones con gigabytes de modelos binarios, se aplica la **Estrategia Híbrida Cero-Fricción**:

### A. Sincronización de Código (Ligera & Automatizada)
- Todo el código fuente (`app.py`, `core.py`, `rvc/`, `tabs/`, notebooks `.ipynb`) se gestiona mediante el repositorio personal **Git**: `https://github.com/PavelBT/Applio.git`.
- **Notebook Recomendado:** Se prioriza el uso de **`Applio_NoUI.ipynb`** en Google Colab por su máxima estabilidad, menor consumo de recursos y compatibilidad total con dispositivos móviles (iPadOS/Safari).
- **Flujo de trabajo:**
  1. El cambio se realiza y prueba localmente en el MacBook.
  2. Se realiza `git commit` y `git push` al repositorio `PavelBT/Applio`.
  3. En Google Colab, las celdas iniciales ejecutan automáticamente un `git pull origin main` antes de invocar los scripts de entrenamiento o inferencia.

### B. Sincronización de Datos Pesados (Estructura en Google Drive)
- **Modelos (`.pth`), Índices (`.index`), Datasets (`.wav`) y Logs de TensorBoard** NUNCA se suben al repositorio Git.
- En Google Colab, se conecta Google Drive en `/content/drive/MyDrive/ApplioBackup` con la siguiente **Estructura Recomendada**:
  ```
  ApplioBackup/
  ├── datasets/         # Carpetas de audios originales (ej. datasets/artista1/audio1.wav)
  ├── weights/          # Modelos finales exportados .pth
  ├── indexes/          # Índices de recuperación FAISS .index
  ├── logs/             # Checkpoints de entrenamiento activos (G_xxx.pth, D_xxx.pth)
  ├── pretrained/       # Modelos base RVC (f0G40k.pth, hubert_base.pt, rmvpe.pt)
  └── outputs/          # Audios resultantes de inferencias
  ```
- Los directorios pesados en Colab se enlazan mediante **symlinks** hacia estas carpetas persistentes en Google Drive:
  ```bash
  ln -s /content/drive/MyDrive/ApplioBackup/logs /content/Applio/logs
  ln -s /content/drive/MyDrive/ApplioBackup/weights /content/Applio/assets/weights
  ```
- **Resultado:** Tras cualquier desconexión de Colab, ningún progreso de entrenamiento se pierde y los datos quedan perfectamente organizados.

---

## 4. Normas de Desarrollo y Calidad de Código

### 4.1. Compatibilidad Multiplataforma
1. **Detección Dinámica de Hardware:** NUNCA asumir que existe CUDA disponible. Usar siempre comprobaciones dinámicas:
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
   ```
2. **Gestión de Rutas:** Prohibido usar separadores de ruta rígidos (`/` o `\`). Usar siempre `os.path.join()` o `pathlib.Path` para garantizar funcionamiento uniforme entre macOS y Linux.
3. **Manejo de Dependencias:** Modificaciones en `requirements.txt` deben especificar plataformas si aplica (ej. `sys_platform == 'darwin'` vs `sys_platform == 'linux'`).

### 4.2. Estructura de Directorios Clave
- `app.py`: Punto de entrada principal para la interfaz Web Gradio.
- `core.py`: Lógica central y backend de integración de Applio.
- `rvc/`: Módulos principales de RVC (entrenamiento, inferencia, modelos, utilidades lib).
- `tabs/`: Interfaces de la UI organizadas por pestaña (entrenamiento, inferencia, TTS, etc.).
- `assets/`: Configuración, presets, temas e i18n.
- `logs/`: Logs de entrenamiento y checkpoints.

---

## 5. Políticas de Protección de Datos de Modelos y Seguridad

1. **Intangibilidad de Pesos y Checkpoints (`.pth` / `.index`):**
   - **Prohibida la modificación destructiva:** Ningún script de automatización ni Agente de IA puede modificar, recortar o sobreescribir un archivo de pesos (`.pth`) o índice (`.index`) existente sin crear un respaldo previo o confirmar versionado explícito.
   - **Tratamiento como Solo Lectura:** Durante las tareas de inferencia, evaluación o exportación, los archivos de modelos deben abrirse estrictamente en modo de solo lectura.
2. **Protección de Datasets Originales:**
   - Los audios originales cargados en `assets/datasets/` son sagrados. Procesos de re-muestreo (resampling), normalización o corte de silencios (slicing) deben generar archivos derivados en carpetas de trabajo o temporales (`/tmp/` o subcarpetas procesadas), sin alterar el dataset fuente.
3. **Validación Anticorrupción en Sincronización:**
   - Todo script de sincronización con Google Drive (`ApplioBackup`) debe comparar tamaños o fechas de modificación antes de mover o reemplazar archivos, evitando que un fallo de red trunque un checkpoint de entrenamiento.
4. **Seguridad de Credenciales y Secretos:**
   - **Cero secretos en Git:** Queda strictly prohibido incluir tokens de HuggingFace, claves API o llaves privadas en código rastreado (`.py`, `.ipynb`).
   - Las variables sensibles deben consumirse a través de variables de entorno o mediante lectura de archivos `.env` ignorados por Git.
5. **Protocolo Anti-Colisión de Entrenamiento (`TRAINING.lock`):**
   - Al iniciar entrenamiento en Colab o Local, se genera automáticamente `ApplioBackup/logs/{model_name}/TRAINING.lock`.
   - Queda estrictamente prohibido reanudar, modificar o sobreescribir un modelo que posea un candado activo desde otro entorno para evitar corrupción de checkpoints (`G_*.pth`/`D_*.pth`) o TensorBoard. El candado se libera automáticamente al terminar las épocas.

---

## 6. Reglas de Operación para Agentes de IA

Cualquier Agente de IA que trabaje en esta base de código debe cumplir estrictamente:

1. **Verificación Empírica:** No declarar una tarea finalizada ni un error corregido sin haber probado la sintaxis o ejecutado la verificación correspondiente.
2. **Preservación de Contratos:** No modificar firmas de funciones públicas en `core.py` o módulos de `rvc/` sin actualizar todos los sitios donde son llamadas en `tabs/` o `app.py`.
3. **Protección de `.gitignore`:** Asegurar que archivos temporales, entornos virtuales (`.venv/`), audios generados y checkpoints de modelos no queden expuestos a Git.
4. **Respuestas Concisas y Estructuradas:** Informar de los cambios de forma clara, con enlaces a los archivos modificados.

---

## 7. Checklist para Nuevas Funcionalidades o Parches

Antes de entregar cualquier actualización:
- [ ] ¿El código funciona sin errores en macOS (Apple Silicon / CPU)?
- [ ] ¿Es compatible con la ejecución en Google Colab con GPU Linux?
- [ ] ¿Se preservó la integridad de los datos de modelos, pesos y datasets sin modificaciones destructivas?
- [ ] ¿Se actualizaron las rutas relativas o configuraciones necesarias?
- [ ] ¿Se verificó que los notebooks `.ipynb` (especialmente `Applio_NoUI.ipynb`) reflejen la automatización requerida?
- [ ] ¿Se mantuvieron los docstrings y comentarios existentes?
