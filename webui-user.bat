@echo off
cd /d "%~dp0"

:: set PYTHON=
:: set GIT=
set "VENV_DIR=%~dp0venv"

set "COMMANDLINE_ARGS=--uv --bnb --api --port 7861 --theme dark --tiled-conv2d 128"

:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
