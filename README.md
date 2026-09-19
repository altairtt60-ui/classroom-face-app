# Classroom Face App

Бір камера арқылы аудиториядағы курсанттарды тіркеу, тану және tracking етуге арналған локалды Windows/Linux приложение.

## Негізгі артықшылық

Қосымша нақты бір камераға байланбайды. Бірдей pipeline мына көздермен жұмыс істейді:

```text
0, 1, 2       → моноблок немесе USB камера индексі
rtsp://...    → IP камера
http://...    → HTTP камера stream
video.mp4     → тестілік видео файл
```

Камера көзін кейін settings арқылы ауыстыруға болады. Detection, tracking және recognition кодын өзгерту қажет емес.

## Қазіргі компьютер профилі

- OS: Windows 11 Pro 64-bit
- CPU: Intel Core i7-13700T
- RAM: 32 GB
- GPU: белгісіз; алғашқы режим CPU
- Камера: моноблок камерасы, мақсатты capture 2560×1440

## Архитектура

```text
CameraSource → OpenCV → YOLO detection → tracking → InsightFace recognition → identity cache → FastAPI → browser dashboard
```

`CameraSource` USB, моноблок, RTSP/IP және видео файлды бір интерфейске біріктіреді. Қалған модульдер камераның нақты түрін білмейді.

## Windows орнату

```powershell
git clone https://github.com/altairtt60-ui/classroom-face-app.git
cd classroom-face-app
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Браузерде ашыңыз: `http://127.0.0.1:8000`

API документациясы: `http://127.0.0.1:8000/docs`

## Камераны тексеру

Моноблок/USB камера:

```powershell
python -m backend.app.camera_cli --source 0 --backend dshow
```

Екінші камера:

```powershell
python -m backend.app.camera_cli --source 1 --backend dshow
```

IP камера:

```powershell
python -m backend.app.camera_cli --source "rtsp://user:password@192.168.1.20:554/stream"
```

Видео файл:

```powershell
python -m backend.app.camera_cli --source "C:\\Videos\\lesson.mp4"
```

## Жоба каталогы

- `backend/app/camera_sources.py` — камераға тәуелсіз source қабаты
- `backend/app/camera_cli.py` — USB/IP/video test құралы
- `backend/app/pipeline.py` — YOLO detection және persistent tracking
- `backend/app/recognition.py` — InsightFace embedding және matching
- `backend/app/worker.py` — realtime frame worker және MJPEG stream
- `backend/app/enrollment.py` — фото және embedding тіркеу
- `backend/app/database.py` — SQLite сақтау қабаты
- `frontend/index.html` — бастапқы оператор dashboard
- `data/` — локалды профильдер, embedding және база; Git-ке кірмейді

## Camera API

```text
GET  /api/camera/config
POST /api/camera/config?source=0
POST /api/camera/start
POST /api/camera/stop
GET  /api/camera/status
GET  /api/camera/stream
```

`source` мәні:

- `0` — әдепкі камера;
- `1` немесе `2` — басқа USB камера;
- `rtsp://...` — IP камера;
- `.mp4`, `.avi`, `.mkv` — видео файл.

## Ескерту

20–30 метрден тану камера оптикасына, бет пиксельдеріне және жарыққа тәуелді. Бет сапасы төмен болса, жүйе `unknown` немесе `review` күйін беруі тиіс. Киім мен шаш бірдей болса, оларды негізгі identity белгісі ретінде қолдануға болмайды.
