# Classroom Face App

Бір камера арқылы аудиториядағы курсанттарды тіркеу, тану және tracking етуге арналған локалды приложение.

## Қазіргі компьютер профилі

- OS: Windows 11 Pro 64-bit
- CPU: Intel Core i7-13700T
- RAM: 32 GB
- GPU: белгісіз, алғашқы MVP CPU режимінде жұмыс істейді
- Камера: моноблок камерасы, мақсатты capture 2560×1440

## Архитектура

```text
Camera → OpenCV → YOLO detection → BoxMOT tracking → InsightFace recognition → temporal voting → FastAPI/WebSocket → dashboard
```

## Жоба каталогы

- `backend/` — камера, detection, tracking және API
- `data/` — локалды курсант деректері; Git-ке қосылмайды
- `models/` — модель файлдары; Git-ке қосылмайды
- `frontend/` — оператор интерфейсі
- `tests/` — unit және integration tests

## Орнату жоспары

Windows 11-де Python 3.11, Git және Node.js LTS орнатылады. Python virtual environment ішінде тәуелділіктер орнатылады. Алғашқы іске қосу CPU режимінде болады. GPU моделі анықталса, кейін inference backend бөлек оңтайландырылады.

## Жауапкершілік шегі

Жүйе төмен сапалы бетке ат қоймайды. Ол `unknown`, `candidate` және `review` күйлерін қолдайды. Автоматты recognition нәтижесі ресми attendance шешімінің жалғыз негізі болмауы керек.

## Келесі кезең

1. Windows камера тесті.
2. Курсант enrollment API.
3. Бір суреттен InsightFace recognition.
4. YOLO detection.
5. BoxMOT tracking.
6. Recognition нәтижесін `track_id`-ге байланыстыру.
7. Web dashboard және сабақ сессиясы.
8. Windows installer.
