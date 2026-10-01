"""Жиналған буманы тексереді: жұмыс кезінде ғана білінетін жетіспеген файлдарды ұстайды.

Мұның себебі нақты: PyInstaller `torchvision`-ның туған кеңейтімін (`_C_stable.pyd`,
`image_stable.pyd`, оның DLL-дері) жинамай кеткен, ал `classroom_bytetrack.yaml` қате
атаумен (`trackers`) салынған. Екеуі де қатені үнсіз жұтып, камера көрінісінде бір де бір
тіктөртбұрыш пен есім шықпай қалған.

Қолданылуы:
    python tools/check_bundle.py dist\\ClassroomFaceApp
    python tools/check_bundle.py "C:\\Users\\User\\Downloads\\ClassroomFaceApp-windows\\ClassroomFaceApp"
"""
from __future__ import annotations

import sys
from pathlib import Path


def _use_utf8_output() -> None:
    """Шығысты UTF-8 етеді.

    GitHub Actions-тың Windows жүгірткішінде консоль кодтауы cp1252, ал нақты
    компьютерде cp1251/cp866 болуы мүмкін: онда осы құралдың қазақша хабарлары
    `UnicodeEncodeError` беріп, жинау қадамы құлайды (дәл осылай болған).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_use_utf8_output()

# Бума ішіндегі міндетті жолдар -> не үшін керек
REQUIRED_FILES = {
    "ClassroomFaceApp.exe": "қосымшаның іске қосылатын файлы",
    "_internal/frontend/index.html": "басқару панелі",
    "_internal/cv2/cv2.pyd": "OpenCV кеңейтімі",
    "_internal/backend/app/trackers/classroom_bytetrack.yaml": "ByteTrack баптаулары (дәл осы атаумен)",
}

REQUIRED_GLOBS = {
    "_internal/torchvision/_C*.pyd": "torchvision кеңейтімі (_C_stable.pyd)",
    "_internal/torchvision/image_stable.pyd": "torchvision image кеңейтімі",
    "_internal/torchvision/*.dll": "torchvision DLL-дері (libwebp, libpng16, zlib, jpeg8, libsharpyuv)",
}

# Ескерту ғана: бұл файлдар болмаса да қосымша істейді
ADVISORY_FILES = {
    "yolo11n.pt": (
        "адам детекторының салмақтары. Онсыз қосымша оны бірінші іске қосылғанда "
        "интернеттен жүктейді — интернеті жоқ компьютерде детекция істемейді. "
        "Жинаққа қосу үшін yolo11n.pt файлын жоба түбіріне (немесе models/ ішіне) қойып, "
        "tools/fix_bundle.py-ді қайта іске қосыңыз."
    ),
}

ADVISORY_GLOBS = {
    "_internal/backend/app/trackers/trackers": "ескі, қате атаумен салынған трекер файлы (өшірген дұрыс)",
    "_internal/cv2/_unicode_text.py": "OpenCV-ге қазақша мәтін қосатын патч (тек ескі жинақтарда керек)",
}


def check(bundle: Path) -> int:
    print(f"Тексеріліп жатқан бума: {bundle}")
    if not bundle.is_dir():
        print(f"ҚАТЕ: бума табылмады: {bundle}")
        return 2

    problems: list[str] = []
    for relative, why in REQUIRED_FILES.items():
        if not (bundle / relative).is_file():
            problems.append(f"жоқ: {relative}  ({why})")

    for pattern, why in REQUIRED_GLOBS.items():
        if not list(bundle.glob(pattern)):
            problems.append(f"жоқ: {pattern}  ({why})")

    for relative, why in ADVISORY_FILES.items():
        if not (bundle / relative).is_file():
            print(f"ескерту: {relative} жоқ — {why}")

    for pattern, why in ADVISORY_GLOBS.items():
        for stray in bundle.glob(pattern):
            print(f"ескерту: {stray.relative_to(bundle)} бар — {why}")

    if problems:
        print("\nЖИНАҚ ТОЛЫҚ ЕМЕС:")
        for problem in problems:
            print("  -", problem)
        print(
            "\nТүзету: python tools/fix_bundle.py <бума жолы>\n"
            "Немесе build_exe.bat арқылы қайта жинаңыз."
        )
        return 1

    print("Жинақ толық: барлық міндетті файлдар орнында.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    return check(Path(argv[1]).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
