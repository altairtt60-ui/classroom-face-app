import argparse

import cv2

from .camera_sources import CameraSource, config_from_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Test USB, IP/RTSP or video-file camera source")
    parser.add_argument("--source", default="0", help="0, 1, RTSP URL or video file path")
    parser.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"])
    args = parser.parse_args()

    source = CameraSource(config_from_source(args.source, backend=args.backend))
    source.open()
    print(f"Opened {source.config.kind.value}: {source.config.source}")
    print(f"Actual properties: {source.actual_properties()}")
    print("Press Q to exit")

    try:
        for frame in source.frames():
            cv2.imshow("Classroom Face App - Camera Test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
