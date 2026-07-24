"""Screen capture via mss, plus one-time play-region calibration."""

import time

import mss
import numpy as np

from bootbreaker import config, detect

# Calibration samples a few frames per monitor and keeps the widest detected
# panel, so a transitional frame (gold ornament momentarily fragmented) can't
# produce a cropped region.
_CALIB_SAMPLES = 6
_CALIB_GAP = 0.08  # seconds between calibration samples


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


def calibrate_auto(config_path: str, sct) -> dict:
    """Scan every physical monitor and calibrate on the first one showing the
    Bootbreaker popup, storing the region in absolute (virtual-desktop) screen
    coordinates so `grab()` reads the correct display on multi-monitor setups.

    mss `monitors[0]` is the union of all displays; `monitors[1:]` are the real
    ones. We grab each, run the gold-panel detector, and take the first hit.
    """
    saw_black = False
    for i, mon in enumerate(sct.monitors[1:], start=1):
        # Sample several frames and keep the WIDEST detected panel. The gold
        # ornament can fragment on a transitional frame, yielding a too-narrow
        # (cropped) region; the full panel gives the widest contour, so max-width
        # across a few frames rejects those fragments.
        scale = shot_scale = None
        region = None
        black_here = False
        for _ in range(_CALIB_SAMPLES):
            shot = sct.grab(mon)
            image = _to_bgr(shot)
            if not image.any():
                black_here = True
                break
            r = detect.detect_play_region(image)
            if r is not None and (region is None or r["width"] > region["width"]):
                region = r
                scale = shot.width / mon["width"] if mon["width"] else 1.0
            time.sleep(_CALIB_GAP)
        if black_here:
            saw_black = True
            continue
        if region is not None:
            left = mon["left"] + round(region["left"] / scale)
            top = mon["top"] + round(region["top"] / scale)
            # Clamp to the monitor so the region never spills into the
            # off-screen black void beyond the display edges - the play field's
            # height ratio can extend past the bottom of the screen, which would
            # otherwise push detect_cart's bottom strip into that void.
            right = min(left + round(region["width"] / scale), mon["left"] + mon["width"])
            bottom = min(top + round(region["height"] / scale), mon["top"] + mon["height"])
            abs_region = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
            print(f"[bootbreaker] found play area on monitor {i} {mon} "
                  f"(scale {scale:.2f})")
            config.save_config(abs_region, config_path)
            return abs_region
    hint = (
        "If the bot only saw black screens, grant this terminal Screen Recording "
        "permission (System Settings -> Privacy & Security -> Screen Recording), "
        "then quit and reopen the terminal."
        if saw_black else
        "Make sure the Bootbreaker minigame popup is actually visible on one of "
        "your screens (windowed/borderless, not exclusive fullscreen), then press "
        "F8 again."
    )
    raise RuntimeError("Could not find the Bootbreaker play area on any monitor. " + hint)


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
