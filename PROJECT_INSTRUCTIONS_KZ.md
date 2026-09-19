# Classroom Face App — Project Instructions

## Рөл

Сен осы жобада senior Python computer-vision және full-stack engineer ретінде жұмыс істейсің. Мақсат — Windows 11-де локалды жұмыс істейтін, бір уақытта бір камера ағынын өңдейтін курсанттарды тіркеу, бет арқылы тану және tracking application жасау.

## Негізгі мақсат

Пайдаланушы application ішінде курсанттың аты-жөнін, student code, тобын және бірнеше фотосын қосады. Жүйе фотолардан face embedding жасайды. Камера іске қосылғанда адамды анықтайды, оған tracking ID береді, сапалы бет кадры болғанда курсантты таниды және аты-жөнін bounding box үстіне шығарады. Танылған identity сол track-ке бекітіледі және адам кадрда жүргенше tracking арқылы көрсетіледі.

Бір камера ғана қолданылады. Cross-camera tracking жасама.

## Камераға тәуелсіздік

Application нақты бір камераға байланбауы керек. Бір `CameraSource` интерфейсі арқылы мыналарды қолда:

- `0`, `1`, `2` — моноблок немесе USB камера индекстері;
- `rtsp://...` — IP камера;
- `http://...` — HTTP camera stream;
- `.mp4`, `.avi`, `.mkv` — тест видео файл.

Қалған pipeline камераның нақты түрін білмеуі керек. Камера source өзгергенде detection, tracking және recognition кодын өзгертуге болмайды.

`CameraSource` мына әдістерді ұсынсын:

```python
open() -> None
read() -> tuple[bool, frame | None]
reconnect() -> None
actual_properties() -> dict
release() -> None
```

Windows-та `CAP_DSHOW`, `CAP_MSMF` және `CAP_ANY` backend-терін қолда. Камераның сұралған resolution-ы нақты қолдау таппаса, нақты алынған resolution мен FPS-ті status арқылы көрсет.

## Міндетті стек

- Python 3.11+
- FastAPI
- OpenCV
- Ultralytics YOLO
- BoxMOT немесе tracker adapter
- InsightFace
- ONNX Runtime
- SQLite MVP үшін
- PostgreSQL-ге кейін ауысуға дайын repository қабаты
- HTML/CSS/JavaScript немесе React/Vite dashboard
- pytest
- Windows PowerShell scripts

Негізгі pipeline:

```text
CameraSource
→ Frame Scheduler
→ YOLO person detection
→ Tracker
→ Face Quality Gate
→ InsightFace embedding
→ Cosine similarity
→ Temporal voting
→ TrackIdentityCache
→ FastAPI/WebSocket/MJPEG
→ Operator Dashboard
```

## Project structure

```text
backend/app/
├── config.py
├── camera_sources.py
├── camera_cli.py
├── database.py
├── enrollment.py
├── recognition.py
├── detector.py
├── tracker.py
├── identity_cache.py
├── pipeline.py
├── worker.py
└── main.py
frontend/
data/
models/
tests/
README.md
requirements.txt
.env.example
install_windows.ps1
run_windows.ps1
```

## Курсанттарды тіркеу

UI арқылы курсант қосу формасын жаса:

```text
student_code — міндетті және unique
full_name — міндетті
group_name — optional
photos — бірнеше JPG/PNG/WEBP фото
```

Тіркеу кезінде:

1. MIME type және файл көлемін тексер.
2. Суреттен дәл бір бет тап.
3. Бет өлшемін, blur, жарық және pose сапасын тексер.
4. InsightFace embedding жаса.
5. Embedding-ті нормала.
6. Әр курсантқа бірнеше face template сақта.
7. Фото мен embedding-ті project-relative private storage-та сақта.
8. Фото, embedding, database және `.env` Git-ке түспесін.

## Recognition ережелері

Бір кадр нәтижесіне сүйеніп identity бекітпе. Соңғы бірнеше сапалы recognition нәтижесіне temporal voting қолдан.

Статустар:

```text
unknown
candidate
recognized
ambiguous
review
```

Similarity және quality threshold `.env` арқылы өзгеретін болсын:

```text
RECOGNITION_MIN_SIMILARITY
RECOGNITION_MIN_QUALITY
RECOGNITION_MIN_VOTES
RECOGNITION_WINDOW_SECONDS
TRACK_IDENTITY_TTL_SECONDS
```

Егер екі курсанттың similarity мәні жақын болса, ең жақынын күштеп таңдама. `ambiguous` немесе `review` қайтар.

Face quality төмен болса, толық аты-жөнін бекітпе.

Бет уақытша көрінбесе:

- tracking жалғаса алады;
- identity қысқа TTL ішінде сақталсын;
- confidence decay жасалсын;
- track толық жоғалса, identity cache тазалансын;
- ескі identity жаңа адамға автоматты берілмесін.

Киім, шаш және дене пішінін негізгі identity белгісі ретінде қолданба. Бірдей формадағы курсанттарды киімге қарап ажыратуға болмайды.

## API

Мына endpoint-терді жаса:

```text
GET    /api/health
GET    /api/camera/config
POST   /api/camera/config
POST   /api/camera/start
POST   /api/camera/stop
GET    /api/camera/status
GET    /api/camera/stream
GET    /api/cadets
POST   /api/cadets
GET    /api/cadets/{id}
PATCH  /api/cadets/{id}
DELETE /api/cadets/{id}
POST   /api/cadets/{id}/photos
POST   /api/sessions
POST   /api/sessions/{id}/start
POST   /api/sessions/{id}/stop
GET    /api/sessions/{id}/events
POST   /api/events/{id}/confirm
POST   /api/events/{id}/reject
WS     /ws/sessions/{id}
```

Inference-ті HTTP request ішінде орындама. Camera worker background thread немесе process ретінде жұмыс істесін. API тек CRUD, status және event/stream тарату қызметін атқарсын.

## Dashboard

Dashboard қазақша интерфейспен жасалсын. Міндетті элементтер:

- live camera stream;
- bounding box;
- аты-жөн;
- track ID;
- recognition confidence;
- recognized/unknown/candidate/ambiguous/review тізімі;
- camera source input;
- actual resolution және FPS;
- camera start/stop;
- курсант қосу және бірнеше фото қосу;
- курсанттар тізімі;
- сабақ сессиясын бастау/аяқтау;
- manual confirm/reject;
- camera/model error хабарламалары.

## Дерекқор

MVP үшін SQLite қолдан. Кемінде мына кестелер болсын:

```text
cadets
face_templates
class_sessions
tracking_events
recognition_events
access_audit_log
```

Кодта дерекқор query логикасын жеке repository/helper қабатына бөл. Кейін PostgreSQL-ге ауыстыру мүмкін болсын.

## Қауіпсіздік

- `.env`, камера password, фотолар, embedding және database Git-ке түспесін;
- upload MIME және size тексерілсін;
- path traversal-ден қорған;
- user filename-ін storage filename ретінде қолданба;
- enrollment және delete тек authorized local operator үшін ашық болсын;
- embedding private local storage-та сақталсын;
- unknown адамның суретін ұзақ сақтама;
- recognition нәтижесін ресми тәртіптік шешімнің жалғыз негізі етпе;
- төмен confidence кезінде `review` көрсет.

## Windows

`install_windows.ps1` және `run_windows.ps1` жаса.

Install script:

1. Python 3.11 барын тексер.
2. `.venv` жаса.
3. pip жаңарт.
4. requirements орнат.
5. `data/cadets`, `data/embeddings`, `models` каталогтарын жаса.
6. `.env.example`-ден `.env` жаса.

Run script:

1. virtual environment іске қос.
2. FastAPI серверін `127.0.0.1:8000` портында іске қос.
3. Browser URL көрсет.
4. Қате жағдайында нақты түсіндірме бер.

## Development workflow

Жұмысты міндетті түрде фазаларға бөліп жаса:

1. Project scaffold және config.
2. CameraSource және CLI.
3. SQLite және cadet CRUD.
4. Enrollment және InsightFace template сақтау.
5. YOLO detector және tracker adapter.
6. Recognition, quality gate және identity cache.
7. Camera worker және stream.
8. FastAPI API.
9. Dashboard.
10. Sessions және attendance events.
11. Tests.
12. Windows scripts және README.

Әр фазадан кейін:

- толық файлдарды сақта;
- `python -m compileall backend` іске қос;
- `pytest -q` іске қос;
- қателерді түзет;
- өзгерген файлдар тізімін көрсет;
- іске қосу командасын көрсет;
- келесі фазаға тек алдыңғысы тексерілгенде өт.

## Міндетті тесттер

- device index parser;
- RTSP parser;
- video file parser;
- camera reconnect;
- cadet CRUD;
- duplicate student code;
- invalid upload MIME;
- one-face enrollment;
- empty templates;
- similarity matching;
- temporal voting;
- unknown/ambiguous decision;
- identity TTL and decay;
- API health;
- API camera config;
- API cadet enrollment;
- mocked frame/video test, real camera dependency жоқ.

## Definition of Done

Проект тек мына жағдайда дайын деп есептеледі:

- `python -m compileall backend` өтеді;
- `pytest -q` өтеді;
- `/api/health` 200 қайтарады;
- dashboard браузерде ашылады;
- device source және video-file source жұмыс істейді;
- курсант UI арқылы қосылады;
- бірнеше фото және embedding сақталады;
- YOLO detection және tracking жұмыс істейді;
- recognition нәтижесі track ID-ге бекітіледі;
- source ауысқанда pipeline коды өзгермейді;
- жеке деректер Git-ке түспейді;
- Windows install/run құжатталған;
- 20–30 метрден «қатесіз таниды» деген кепілдік берілмейді;
- төмен сапа `unknown/review` ретінде көрсетіледі.

## Жауап форматы

Әр жұмыс кезеңінің соңында мынаны бер:

```text
1. Өзгерген файлдар
2. Іске қосу командалары
3. Тест нәтижелері
4. Белгілі шектеулер
5. Келесі кезең
```

Placeholder success response қалдырма. Жұмыс істемейтін функцияны TODO деп нақты белгіле. Кодты бір үлкен файлға жинама. Барлық component-ті жеке модульдерге бөл.
