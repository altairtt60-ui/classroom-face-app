"""PyInstaller жинаған бумаға жетіспеген жұмыс файлдарын қосады, содан кейін тексереді.

PyInstaller таба алмайтын екі нәрсе бар:

1. `torchvision`-ның туған кеңейтімі. Ол `torch.ops.load_library()` арқылы жүктеледі,
   сондықтан статикалық талдау оны көрмейді. Кеңейтімсіз `import torchvision` қате
   береді, ал ultralytics оны әр inference алдында шақырады -> бір де бір детекция
   жасалмайды (камера көрінісінде тек "шикі" видео қалады).
2. `backend/app/trackers/classroom_bytetrack.yaml` — дұрыс атаумен салынуы керек.
   Бұрын ол `trackers` деген атаумен салынып, `model.track()` әр кадрда FileNotFoundError
   беріп, қате үнсіз жұтылған.

Сонымен қатар, егер жобада `yolo11n.pt` (адам детекторының салмақтары) болса, ол да
бумаға көшіріледі: онсыз қосымша оны бірінші іске қосылғанда интернеттен жүктейді, ал
интернеті жоқ компьютерде детекция мүлдем істемейді.

Қолданылуы:
    python tools/fix_bundle.py dist\\ClassroomFaceApp
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import check_bundle  # noqa: E402  (қатар тұрған тексеру құралы)

TORCHVISION_PATTERNS = ("_C*.pyd", "image*.pyd", "*.dll")
TRACKER_SOURCE = PROJECT_ROOT / "backend" / "app" / "trackers" / "classroom_bytetrack.yaml"
TRACKER_DEST = Path("_internal/backend/app/trackers/classroom_bytetrack.yaml")
WEIGHT_SOURCES = (
    PROJECT_ROOT / "yolo11n.pt",
    PROJECT_ROOT / "models" / "yolo11n.pt",
    PROJECT_ROOT / "backend" / "yolo11n.pt",
)


def copy_torchvision_natives(internal: Path) -> int:
    """Қазіргі ортадағы torchvision кеңейтімін бума ішіне көшіреді."""
    try:
        import torchvision
    except ImportError:
        print("ЕСКЕРТУ: torchvision орнатылмаған — кеңейтімді көшіру мүмкін емес.")
        return 0

    source_dir = Path(torchvision.__file__).parent
    dest_dir = internal / "torchvision"
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pattern in TORCHVISION_PATTERNS:
        for native in sorted(source_dir.glob(pattern)):
            target = dest_dir / native.name
            if target.is_file() and target.stat().st_size == native.stat().st_size:
                print(f"  бар: torchvision/{native.name}")
                continue
            shutil.copy2(native, target)
            print(f"  қосылды: torchvision/{native.name} ({native.stat().st_size:,} байт)")
            copied += 1
    if copied == 0:
        print("  torchvision кеңейтімі толық орнында.")
    return copied


def ensure_tracker_config(internal: Path) -> int:
    target = internal / TRACKER_DEST.relative_to("_internal")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not TRACKER_SOURCE.is_file():
        print(f"ЕСКЕРТУ: {TRACKER_SOURCE} табылмады — трекер баптауын көшіру мүмкін емес.")
        return 0
    if target.is_file() and target.read_bytes() == TRACKER_SOURCE.read_bytes():
        print(f"  бар: {TRACKER_DEST}")
        return 0
    shutil.copy2(TRACKER_SOURCE, target)
    print(f"  қосылды: {TRACKER_DEST}")

    legacy = target.parent / "trackers"
    if legacy.is_file():
        legacy.unlink()
        print("  өшірілді: _internal/backend/app/trackers/trackers (ескі, қате атау)")
    return 1


def copy_detector_weights(bundle: Path) -> int:
    """`yolo11n.pt` жобада болса, бума түбіріне көшіреді (интернетсіз де істеуі үшін)."""
    target = bundle / "yolo11n.pt"
    if target.is_file():
        print(f"  бар: yolo11n.pt ({target.stat().st_size:,} байт)")
        return 0
    for candidate in WEIGHT_SOURCES:
        if candidate.is_file():
            shutil.copy2(candidate, target)
            print(f"  қосылды: yolo11n.pt ({candidate.stat().st_size:,} байт, {candidate.name})")
            return 1
    print(
        "  ескерту: yolo11n.pt жобада табылмады — қосымша оны бірінші іске қосылғанда\n"
        "           интернеттен жүктейді. Интернетсіз керек болса, осы файлды жоба\n"
        "           түбіріне қойып, fix_bundle.py-ді қайта іске қосыңыз."
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    bundle = Path(argv[1]).expanduser().resolve()
    if not bundle.is_dir():
        print(f"ҚАТЕ: бума табылмады: {bundle}")
        return 2

    internal = bundle / "_internal"
    print("1) torchvision кеңейтімі:")
    copy_torchvision_natives(internal)
    print("2) ByteTrack баптауы:")
    ensure_tracker_config(internal)
    print("3) Детектор салмақтары:")
    copy_detector_weights(bundle)
    print("4) Тексеру:")
    return check_bundle.check(bundle)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
