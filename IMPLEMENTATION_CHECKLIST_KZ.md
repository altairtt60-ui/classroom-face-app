# Classroom Face App — іске асыру және тестілеу чек-листі

## Фаза 0 — Құралдар

- [ ] Windows 11 64-bit
- [ ] Python 3.11
- [ ] Git
- [ ] VS Code
- [ ] Node.js LTS, егер React frontend қолданылса
- [ ] Windows Camera privacy permission қосылған
- [ ] Кемінде 10 GB бос диск
- [ ] NVIDIA GPU бар болса, моделі анықталған

## Фаза 1 — Орнату

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

- [ ] `python --version` 3.11 көрсетеді
- [ ] `python -c "import cv2, fastapi, numpy"` өтеді
- [ ] `python -c "import ultralytics"` өтеді
- [ ] `python -c "import insightface"` өтеді
- [ ] `python -c "import onnxruntime"` өтеді

## Фаза 2 — Камера

```powershell
python -m backend.app.camera_cli --source 0 --backend dshow
```

- [ ] Моноблок камерасы ашылады
- [ ] Нақты resolution жазылады
- [ ] Q арқылы дұрыс жабылады
- [ ] `--source 1` басқа USB камераны тексереді
- [ ] RTSP URL тексеріледі
- [ ] MP4 тест видео тексеріледі
- [ ] Камера ажыратылса, reconnect қатесі түсінікті көрінеді

## Фаза 3 — Курсант тіркеу

- [ ] `student_code` unique
- [ ] full_name бос қабылданбайды
- [ ] JPG/PNG/WEBP қабылданады
- [ ] Басқа MIME type қабылданбайды
- [ ] Суретте дәл бір бет тексеріледі
- [ ] Төмен сапалы фото `review/error` береді
- [ ] Бір курсантқа 5–10 фото қосуға болады
- [ ] Embedding сақталады
- [ ] Фото және embedding Git status-та көрінбейді
- [ ] Курсанттар тізімі UI-да көрінеді
- [ ] Delete soft delete ретінде орындалады

## Фаза 4 — Detection және tracking

- [ ] YOLO модельі жүктеледі
- [ ] Адам bounding box-ы көрінеді
- [ ] Әр адамға track ID беріледі
- [ ] Екі адам қиылысқанда ID мүмкіндігінше сақталады
- [ ] Адам уақытша жабылғанда track жалғасады
- [ ] Track жоғалса, state TTL-ден кейін тазаланады
- [ ] Бейне файлмен камерасыз тест жасалады

## Фаза 5 — Recognition

- [ ] InsightFace model бірінші іске қосылғанда жүктеледі
- [ ] Бір курсанттың бірнеше template-і сақталады
- [ ] Cosine similarity есептеледі
- [ ] Threshold `.env` арқылы өзгереді
- [ ] Бір кадр identity-ді бірден бекітпейді
- [ ] Temporal voting жұмыс істейді
- [ ] Екі candidate жақын болса `ambiguous`
- [ ] Төмен quality болса `unknown/review`
- [ ] Бет жоғалса confidence decay болады
- [ ] Ескі identity жаңа адамға берілмейді

## Фаза 6 — Dashboard

- [ ] `http://127.0.0.1:8000` ашылады
- [ ] Live stream көрінеді
- [ ] Bounding box және аты-жөн көрсетіледі
- [ ] Track ID көрінеді
- [ ] Confidence көрінеді
- [ ] Camera source ауыстыруға болады
- [ ] Камера start/stop жұмыс істейді
- [ ] Курсант қосу формасы жұмыс істейді
- [ ] Курстар тізімі жаңарады
- [ ] Unknown/review оқиғалары көрінеді
- [ ] Қате хабарламалары түсінікті

## Фаза 7 — Сессия және attendance

- [ ] Сабақ сессиясы басталады
- [ ] Бірінші көрінген уақыт сақталады
- [ ] Соңғы көрінген уақыт сақталады
- [ ] Қатысу автоматты final decision ретінде белгіленбейді
- [ ] Оқытушы confirm/reject жасай алады
- [ ] Сессия жабылады
- [ ] CSV export қажет болса кейін қосылады

## Фаза 8 — Windows package

- [ ] `install_windows.ps1` жұмыс істейді
- [ ] `.env` автоматты жасалады
- [ ] `run_windows.ps1` серверді іске қосады
- [ ] Python path қатесі түсінікті көрсетіледі
- [ ] Camera permission қатесі түсінікті көрсетіледі
- [ ] CPU режимі GPU жоқ кезде жұмыс істейді
- [ ] Басқа Windows 11 компьютерінде clean install тексеріледі

## Қабылдау өлшемдері

- [ ] `python -m compileall backend` өтеді
- [ ] `pytest -q` өтеді
- [ ] `/api/health` HTTP 200 қайтарады
- [ ] Бір USB/device source жұмыс істейді
- [ ] Бір RTSP немесе video-file source жұмыс істейді
- [ ] Курсант application ішінен тіркеледі
- [ ] Recognition және tracking бір pipeline-де жұмыс істейді
- [ ] Camera source ауысқанда AI pipeline өзгермейді
- [ ] Жеке фото/embedding GitHub-қа кетпейді
- [ ] Бірдей киім/шаш жағдайында төмен confidence нәтижесі күштеп бекітілмейді
- [ ] 20–30 метрлік нәтиже нақты камера calibration арқылы ғана бағаланады
