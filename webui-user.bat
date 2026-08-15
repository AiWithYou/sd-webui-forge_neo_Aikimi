@echo off
cd /d "%~dp0"

:: set PYTHON=
:: set GIT=
set "VENV_DIR=%~dp0venv"

set "COMMANDLINE_ARGS=--uv --bnb --api --port 7861 --theme dark --tiled-conv2d 128"
if exist "%~dp0forge_neo_model_paths.yaml" (
  set "COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --forge-ref-comfy-yaml forge_neo_model_paths.yaml"
) else (
  echo [Forge NeoW] forge_neo_model_paths.yaml was not found; using the standard model folders.
)

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
