import cv2


def main() -> None:
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened. Check Windows camera permissions.")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_width}x{actual_height}")
    print("Press Q to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Frame read failed")
            break
        cv2.imshow("Classroom Face App - Camera Test", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
