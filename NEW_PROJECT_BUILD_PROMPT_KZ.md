# Classroom Face App — жаңа проектті толық жасауға арналған prompt

Төмендегі мәтінді Claude Code, Cursor, Windsurf, Gemini Code Assist немесе басқа coding AI құралына толық күйінде бер.

---

## MASTER PROMPT

Сен senior Python computer-vision және full-stack engineer ретінде жұмыс істейсің. Windows 11 64-bit жүйесінде орнатылатын, бір уақытта бір ғана камера ағынын өңдейтін локалды application жаса. Application нақты бір камераға байланбауы керек: моноблок/USB камера индексі, басқа USB камера, RTSP/HTTP IP camera және MP4/AVI/MKV тест видео бірдей `CameraSource` интерфейсі арқылы жұмыс істесін.

### Жоба мақсаты

Конференц-зал немесе аудиториядағы курсанттарды application ішінде алдын ала тіркеу. Әр курсанттың аты-жөні, student code, тобы және бірнеше фотосы application UI арқылы қосылады. Application фотолардан face embedding жасайды. Камера іске қосылғанда адам анықталады, оған tracking ID беріледі, бет сапасы жеткілікті болған сәтте курсант танылып, аты-жөні bounding box үстіне шығарылады. Курсант беті уақытша көрінбей қалса, бұрын бекітілген identity қысқа уақыт tracking арқылы сақталады. Жаңа адамға ескі identity берілмеуі керек.

Жүйе бір камерамен жұмыс істейді. Көпкамералы cross-camera tracking жасама. Камера ауысқанда тек source параметрі өзгереді; detection, tracking және recognition коды өзгермейді.

### Міндетті технологиялар

- Python 3.11+
- FastAPI
- OpenCV
- Ultralytics YOLO, бастапқы detector/tracker үшін
- BoxMOT adapter-ready architecture; негізгі tracker интерфейсі бөлек болсын
- InsightFace + ONNX Runtime, face detection және embedding үшін
- SQLite MVP үшін
- PostgreSQL-ге кейін ауысуға болатын repository қабаты
- HTML/CSS/JavaScript немесе React/Vite dashboard; MVP-ді артық күрделендірме
- WebSocket немесе MJPEG арқылы live нәтиже
- Windows PowerShell launcher
- pytest тесттері

### Камера abstraction

`CameraSource` класы мына интерфейсті ұсынсын:

```python
open() -> None
read() -> tuple[bool, frame | None]
reconnect() -> None
actual_properties() -> dict
release() -> None
```

Source түрлері:

```text
"0"              -> USB/моноблок камера index 0
"1"              -> USB камера index 1
"rtsp://..."     -> RTSP IP camera
"http://..."     -> HTTP stream
"C:\\test.mp4"  -> video file
```

Windows үшін `CAP_DSHOW`, `CAP_MSMF` және `CAP_ANY` backend таңдау мүмкіндігін бер. Камера конфигурациясы `.env` және UI арқылы өзгертілсін.

Камера resolution-ы пайдаланушының нақты камерасына тәуелді болсын. Сұралған capture default `2560x1440`, бірақ камера осы resolution-ды бермесе, application нақты алынған resolution-ды анықтап, status арқылы көрсетсін. Processing үшін default `1280x720`, inference FPS default 5–8 болсын. Әр кадрда face recognition жасама; track пайда болғанда, бет сапасы жақсарғанда немесе recognition TTL біткенде ғана жаса.

### Архитектуралық pipeline

```text
CameraSource
  -> FrameScheduler
  -> YOLO person detection
  -> TrackerAdapter
  -> FaceQualityGate
  -> InsightFace embedding
  -> cosine similarity matcher
  -> temporal voting
  -> TrackIdentityCache
  -> EventPublisher
  -> FastAPI/WebSocket/MJPEG
  -> Operator Dashboard
```

Модульдер:

```text
backend/app/
  config.py
  camera_sources.py
  camera_cli.py
  database.py
  enrollment.py
  recognition.py
  detector.py
  tracker.py
  pipeline.py
  identity_cache.py
  worker.py
  main.py
frontend/
data/
models/
tests/
```

Inference HTTP request ішінде орындалмасын. Camera worker background thread/process ретінде жұмыс істесін. API тек status, CRUD және stream/event таратсын.

### Курсант enrollment

UI-да «Курсант қосу» формасы болсын:

```text
student_code — міндетті, unique
full_name — міндетті
surname/name немесе толық аты-жөні
surname/name fields optional, бірақ full_name дайын сақталсын
group_name — optional
5–10 фото — міндетті түрде бірнеше фотоға кеңейтілетін API
```

Әр фото үшін:

1. MIME type тексер.
2. Файл көлемін шекте.
3. Суретті OpenCV арқылы оқы.
4. Дәл бір бет барын тексер.
5. Blur, өлшем, жарық және face pose quality тексер.
6. InsightFace embedding жаса.
7. Нормаланған embedding-ті сақта.
8. Әр курсантқа бірнеше template сақта.
9. Original фото мен embedding-ті Git-ке қоспа.
10. Embedding жолын базаға абсолютті емес, project-relative path ретінде сақта.

MVP-де SQLite кестелері:

```text
cadets
- id
- student_code unique
- full_name
- group_name
- active
- created_at

face_templates
- id
- cadet_id
- embedding_path
- model_name
- model_version
- quality_score
- created_at

class_sessions
- id
- room_name
- subject_name
- started_at
- finished_at

tracking_events
- id
- session_id
- track_id
- cadet_id nullable
- bbox_json
- status
- confidence
- first_seen_at
- last_seen_at
```

### Recognition шешімі

Embedding cosine similarity арқылы салыстырылсын. Бір threshold-қа соқыр сенбе; `.env` арқылы өзгеретін параметрлер болсын:

```text
RECOGNITION_MIN_SIMILARITY
RECOGNITION_MIN_QUALITY
RECOGNITION_MIN_VOTES
RECOGNITION_WINDOW_SECONDS
TRACK_IDENTITY_TTL_SECONDS
```

Status мәндері:

```text
unknown
candidate
recognized
ambiguous
review
```

Бір кадрдан identity бекітпе. Соңғы бірнеше сапалы observation бойынша voting жаса. Екі candidate similarity жағынан жақын болса, ең жақын адамды күштеп таңдама; `ambiguous` қайтар.

Бет көрінбеген кезде:

- track жалғаса алады;
- identity қысқа TTL ішінде сақталады;
- confidence decay болады;
- track толық жоғалса, identity cache тазаланады;
- сол track ID қайта қолданылса, жаңа identity қайта тексеріледі.

Киім, шаш және дене appearance негізгі identity белгісі болмасын. Бірдей формадағы курсанттарды киімге қарап бір-бірінен ажыратуға болмайды.

### API

Мына endpoint-терді іске асыр:

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

`POST /api/cadets` multipart form арқылы profile fields және бірінші photo қабылдасын. Бірнеше photo үшін бөлек endpoint қос. Response-та `recognition_ready`, `photo_count`, `embedding_count` бер.

### Dashboard

MVP dashboard-та:

- live camera stream;
- bounding box және `track_id`;
- recognized аты-жөні;
- similarity/confidence;
- `unknown`, `candidate`, `ambiguous`, `review` тізімі;
- camera source input;
- actual resolution және FPS;
- камера start/stop;
- курсант қосу формасы;
- курсанттар тізімі;
- фото қосу/жою;
- сабақ сессиясын бастау/аяқтау;
- manual confirm/reject;
- camera error және AI model error хабарламасы.

UI қазақша болсын, бірақ кодтағы enum және API field атаулары ағылшынша болсын.

### Қауіпсіздік

- `.env`, камера password, фото, embedding, database Git-ке түспесін.
- Upload size және MIME тексер.
- Path traversal-ден қорға.
- Filename-ті user input-тан тікелей құрма.
- Бет embedding-терін private local storage-та сақта.
- Рұқсатсыз қолданушыға enrollment және delete ашық болмауы керек; MVP-де кемінде local admin mode жаса.
- Белгісіз адамның кадрын ұзақ сақтама.
- Автоматты recognition нәтижесін ресми тәртіптік шешімнің жалғыз негізі етпе.
- Төмен confidence кезінде аты-жөнін толық бекітпе.

### Windows package

Мына файлдарды жаса:

```text
run_windows.ps1
install_windows.ps1
.env.example
requirements.txt
README.md
```

`install_windows.ps1`:

1. Python 3.11 барын тексер.
2. `.venv` жаса.
3. pip жаңарт.
4. requirements орнат.
5. data directories жаса.
6. `.env` жоқ болса `.env.example` көшір.

`run_windows.ps1`:

1. `.venv` activate.
2. FastAPI серверін 127.0.0.1:8000-де іске қос.
3. Browser URL көрсет.
4. Қате болса нақты message бер.

### Тестілеу

Міндетті тесттер:

- CameraSource device index parser.
- RTSP source parser.
- Video file source parser.
- Camera reconnect.
- SQLite cadet CRUD.
- Duplicate student_code rejection.
- Invalid upload MIME rejection.
- One-face enrollment validation.
- Empty templates behavior.
- Similarity matching.
- Temporal voting.
- Identity TTL/decay.
- Unknown/ambiguous decision.
- API health.
- API camera config.
- API cadet enrollment.
- No real camera dependency in unit tests; use mocked frames/video file.

### Implementation тәртібі

Жұмысты мына фазаларға бөл:

1. Project scaffold және configuration.
2. CameraSource және camera CLI.
3. SQLite schema және cadet CRUD.
4. Enrollment және InsightFace template storage.
5. Detector және tracker adapter.
6. Recognition, quality gate және identity cache.
7. Camera worker және stream.
8. API.
9. Dashboard.
10. Tests.
11. Windows scripts.
12. README және final verification.

Әр фазадан кейін:

- файлдарды сақта;
- lint/compile іске қос;
- тест жүргіз;
- нақты қате болса түзет;
- келесі фазаға тек алдыңғысы жұмыс істегенде өт.

### Definition of Done

Проект дайын деп тек мына шарттар орындалғанда есепте:

- `python -m compileall backend` өтеді;
- `pytest -q` өтеді;
- `/api/health` 200 қайтарады;
- браузер dashboard ашылады;
- `source=0` және video file камера тесті жұмыс істейді;
- курсант UI арқылы қосылады;
- бет embedding жоқ кезде түсінікті model setup message шығады;
- бір source-тен екіншісіне ауысу pipeline кодын өзгертпейді;
- `data/` және `.env` Git status-та көрінбейді;
- Windows install/run нұсқаулығы тексерілген;
- 20–30 метрден «қатесіз» деп жалған кепілдік жазылмайды;
- төмен quality нәтижесі `unknown/review` ретінде ашық көрсетіледі.

Кодты бір файлға жинама. Әр компонентті жеке модульге бөл. Placeholder endpoint немесе жалған success response қалдырма. Жұмыс істемейтін функцияны TODO ретінде анық жаз. Барлық өзгерісті қысқаша changelog арқылы түсіндір.

---

## AI құралынан күтілетін жауап форматы

Әр фаза соңында мынаны бер:

```text
1. Өзгертілген файлдар тізімі
2. Нақты іске қосу командалары
3. Тест нәтижесі
4. Белгілі шектеулер
5. Келесі фаза
```

Кодты толық файл ретінде жаса. Қысқартылған псевдокодпен шектелме.
