@echo off
rem ---------------------------------------------------------------------------------------
rem Classroom Face App - Windows жинағы.
rem
rem ЕСКЕРТУ (2026-10-01 түзетуі): PyInstaller өз бетінше жинамайтын екі файл бар -
rem   * torchvision-ның туған кеңейтімі (_C_stable.pyd, image_stable.pyd, DLL-дері)
rem   * backend/app/trackers/classroom_bytetrack.yaml (дәл осы атаумен)
rem Оларсыз қосымша іске қосылады, камера көрсетеді, бірақ беттерді танымайды және
rem есімдерді шығармайды. Сондықтан жинақтан кейін tools\fix_bundle.py міндетті түрде
rem орындалады, ол жетіспегенін қосып, нәтижесін тексереді.
rem ---------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [0/3] Виртуалды орта жасалып, тәуелділіктер орнатылып жатыр...
  python -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r backend\requirements.txt || goto :fail
)

.venv\Scripts\python.exe -m pip install pyinstaller || goto :fail

echo.
echo [1/3] PyInstaller жинағы (бірнеше минут алады)...
.venv\Scripts\python.exe -m PyInstaller --noconfirm --onedir --name ClassroomFaceApp ^
  --add-data "frontend;frontend" ^
  --add-data "backend/app/trackers;backend/app/trackers" ^
  --add-data ".env.example;." ^
  --collect-all ultralytics --collect-all insightface --collect-all onnxruntime ^
  --collect-all torchvision ^
  --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto ^
  --hidden-import uvicorn.protocols.http.auto ^
  launcher.py || goto :fail

echo.
echo [2/3] Жетіспеген файлдарды қосу...
.venv\Scripts\python.exe tools\fix_bundle.py dist\ClassroomFaceApp || goto :fail

echo.
echo [3/3] Дайын: dist\ClassroomFaceApp\ClassroomFaceApp.exe
echo      Буманы тексеру: .venv\Scripts\python.exe tools\check_bundle.py dist\ClassroomFaceApp
pause
exit /b 0

:fail
echo.
echo ЖИНАУ СӘТСІЗ АЯҚТАЛДЫ - жоғарыдағы қатені қараңыз.
pause
exit /b 1
