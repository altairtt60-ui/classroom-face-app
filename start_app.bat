@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Birinshi iske qosu: ornatu zhurip zhatyr, 10-20 minut alady...
  py -3.11 -m venv .venv || (echo Python 3.11 ornatylmagan: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r backend\requirements.txt
)
if not exist ".env" copy ".env.example" ".env" >nul
if not exist "data\cadets" mkdir "data\cadets"
if not exist "data\embeddings" mkdir "data\embeddings"
.venv\Scripts\python.exe launcher.py
pause
