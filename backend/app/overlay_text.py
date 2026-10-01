"""Overlay text with Kazakh/Cyrillic support.

``cv2.putText`` only knows the ASCII Hershey fonts, so Kazakh names were drawn as
"??????????" on the live camera view even after the face had been recognized (this is
exactly what the "атым көрінмейді" complaint was about, next to the bundled-bundle
problems).

ASCII-only labels still go through OpenCV (fast, and pixel-identical to before). Anything
with non-ASCII characters is rendered with Pillow, which ships full Cyrillic coverage and
is already a dependency of the application.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Optional override: CLASSROOM_FACE_FONT=C:\path\to\font.ttf
FONT_ENV = "CLASSROOM_FACE_FONT"
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

_font_cache: dict[int, object] = {}


def _font_files():
    override = os.environ.get(FONT_ENV)
    if override:
        yield override
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            yield path
    try:  # always present: matplotlib bundles DejaVuSans with the app
        import matplotlib

        yield os.path.join(
            os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"
        )
    except Exception:
        pass


def _font(pixel_size: int):
    """A TrueType font of the requested size, or None when nothing usable is installed."""
    cached = _font_cache.get(pixel_size)
    if cached is not None:
        return cached or None
    from PIL import ImageFont

    for path in _font_files():
        try:
            font = ImageFont.truetype(path, pixel_size)
            _font_cache[pixel_size] = font
            return font
        except Exception:
            continue
    _font_cache[pixel_size] = False  # do not retry for every frame
    return None


def is_ascii(text) -> bool:
    return all(ord(char) < 128 for char in str(text))


def pixel_size(font_scale: float) -> int:
    """Hershey text at scale 0.6 is ~16 px tall; TrueType size ~ scale * 30 looks the same."""
    try:
        return max(10, int(round(float(font_scale) * 30)))
    except Exception:
        return 16


def _rgb(color) -> tuple[int, int, int]:
    if isinstance(color, (int, float)):
        value = int(color)
        return (value, value, value)
    channels = [int(channel) for channel in color]
    if len(channels) >= 3:  # OpenCV is BGR, Pillow wants RGB
        return (channels[2], channels[1], channels[0])
    value = channels[0]
    return (value, value, value)


def label_size(text, font_scale: float = 0.6, thickness: int = 2) -> tuple[int, int]:
    """(width, height) of `text` as it will be drawn by draw_label()."""
    if is_ascii(text):
        (width, height), _baseline = cv2.getTextSize(
            str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        return int(width), int(height)
    font = _font(pixel_size(font_scale))
    if font is None:
        (width, height), _baseline = cv2.getTextSize(
            str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        return int(width), int(height)
    left, top, right, bottom = font.getbbox(str(text), stroke_width=max(0, int(thickness) - 1))
    return int(right - left), int(bottom - top)


def draw_label(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color=(255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Draw `text` on `frame` (in place) with `org` at its bottom-left, like cv2.putText."""
    text = str(text)
    if is_ascii(text):
        return cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )

    font = _font(pixel_size(font_scale))
    if font is None:  # no TrueType font on this machine: better "?" than nothing at all
        return cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )

    try:
        from PIL import Image, ImageDraw

        image = np.asarray(frame)
        if image.ndim == 2:
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
        draw = ImageDraw.Draw(pil_image)
        fill = _rgb(color)
        draw.text(
            (int(org[0]), int(org[1])),
            text,
            font=font,
            fill=fill,
            anchor="ls",  # Pillow: left + baseline, which is what cv2.putText uses
            stroke_width=max(0, int(thickness) - 1),
            stroke_fill=fill,
        )
        frame[...] = np.asarray(pil_image)[:, :, ::-1]
    except Exception:  # rendering must never take the camera loop down
        return cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )
    return frame
