"""Screen capture via mss, plus one-time play-region calibration."""

import mss
import numpy as np

from bootbreaker import config, detect


def _to_bgr(shot) -> np.ndarray:
    arr = np.asarray(shot)  # BGRA
    return arr[:, :, :3]  # drop alpha -> BGR (channel order matches OpenCV)


def grab(region: dict, sct=None) -> np.ndarray:
    box = {
        "left": region["left"],
        "top": region["top"],
        "width": region["width"],
        "height": region["height"],
    }
    if sct is not None:
        return _to_bgr(sct.grab(box))
    with mss.mss() as s:
        return _to_bgr(s.grab(box))


def grab_fullscreen(sct=None) -> np.ndarray:
    if sct is not None:
        return _to_bgr(sct.grab(sct.monitors[1]))
    with mss.mss() as s:
        return _to_bgr(s.grab(s.monitors[1]))


def display_scale(sct) -> tuple[float, int, int]:
    """Empirically measure the primary monitor's pixels-per-point and origin.

    On a Retina Mac, mss reports monitor geometry in logical *points* but the
    grabbed image comes back in physical *pixels* (2x), so a region detected in
    image-pixel space must be divided by this factor before it can be handed
    back to `grab()` as a screen box. On a non-Retina display the factor is
    1.0 and this is a harmless no-op — and because we derive it from the actual
    grab (image width / monitor width), it stays correct regardless of which
    coordinate convention a given mss version uses.
    """
    mon = sct.monitors[1]
    shot = sct.grab(mon)
    scale = shot.width / mon["width"] if mon["width"] else 1.0
    return scale, mon["left"], mon["top"]


def calibrate(
    config_path: str,
    grabber=grab_fullscreen,
    scale: float = 1.0,
    offset: tuple[int, int] = (0, 0),
) -> dict:
    image = grabber()
    region = detect.detect_play_region(image)
    if region is None:
        raise RuntimeError(
            "Could not find the Bootbreaker play area. Make sure Dota is "
            "visible and windowed, then press F8 again — or edit config.json. "
            "If the whole screen looks black to the bot, grant this terminal "
            "Screen Recording permission (System Settings -> Privacy & Security)."
        )
    if scale != 1.0 or offset != (0, 0):
        ox, oy = offset
        region = {
            "left": ox + round(region["left"] / scale),
            "top": oy + round(region["top"] / scale),
            "width": round(region["width"] / scale),
            "height": round(region["height"] / scale),
        }
    config.save_config(region, config_path)
    return region
