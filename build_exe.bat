@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --onedir --name ClassroomFaceApp ^
  --add-data "frontend;frontend" --add-data ".env.example;." ^
  --collect-all ultralytics --collect-all insightface --collect-all onnxruntime ^
  --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto ^
  launcher.py
echo Dayyn: dist\ClassroomFaceApp\ClassroomFaceApp.exe
pause
