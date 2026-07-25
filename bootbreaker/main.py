"""CLI, F8 pause/resume hotkey, and the capture->decide->act loop."""

import argparse
import time

from pynput import keyboard as _kb

from bootbreaker import capture, config, detect
from bootbreaker.bot import Bot
from bootbreaker.input import Controller

_MOVE_KEY = {"left": "a", "right": "d"}
_DEBUG_WINDOW = "bootbreaker"
_debug_win_ready = False
# The aim is steered with A/D (NOT an auto-sweep we wait on), so _launch runs a
# closed loop: pulse A/D to rotate the aim toward vertical, reading the angle
# back as feedback, and throw once within the deadband. We don't know which key
# rotates which way, so the loop self-corrects: if a pulse makes the angle worse,
# it flips direction. Deadband can't be 0 (the loop would oscillate forever a
# fraction of a degree off) - a few degrees off vertical is plenty catchable.
_AIM_TOLERANCE = 6  # degrees from vertical: close enough to straight-up to throw
# Proportional pulse length: hold A/D for a time proportional to how far off
# vertical we are, so far shots take a big rotation but near-vertical shots take
# a tiny nudge and don't fly past. A fixed pulse overshot on re-throws that
# started far off, swinging the aim wildly across vertical (never settling).
_AIM_STEP_GAIN = 0.0007  # seconds of hold per degree off vertical
_AIM_STEP_MIN = 0.006    # never pulse shorter than this (must register at all)
_AIM_STEP_MAX = 0.05     # never pulse longer than this (cap the big far-shot swing)
_AIM_SETTLE = 0.06  # seconds after releasing before re-reading (let the aim update)
_AIM_MAX_ADJUST = 60  # max correction pulses before throwing anyway
_AIM_SAMPLE_DELAY = 0.02  # seconds between re-reads when the aim isn't visible yet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bootbreaker", description=__doc__)
    parser.add_argument("--debug", action="store_true", help="show detection window")
    parser.add_argument(
        "--recalibrate", action="store_true", help="force play-region re-detection"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="seconds to count down after F8 before reading the screen, so you "
             "can focus/un-pause Dota first (default 3)",
    )
    return parser


def _countdown(seconds: float) -> None:
    """Give the user a moment to bring Dota to the front and un-pause before the
    bot starts capturing."""
    whole = int(seconds)
    for n in range(whole, 0, -1):
        print(f"[bootbreaker] reading in {n}...")
        time.sleep(1)
    frac = seconds - whole
    if frac > 0:
        time.sleep(frac)


class _Toggle:
    """Tracks running/paused, flipped by F8."""

    def __init__(self):
        self.running = False

    def flip(self):
        self.running = not self.running
        print("[bootbreaker] " + ("RUNNING" if self.running else "PAUSED"))


def _ensure_region(recalibrate: bool) -> dict:
    region = None if recalibrate else config.load_config()
    if region is None:
        print("[bootbreaker] calibrating play region (scanning all monitors)...")
        import mss

        with mss.mss() as sct:
            region = capture.calibrate_auto(config.DEFAULT_CONFIG_PATH, sct)
        print(f"[bootbreaker] region: {region}")
    return region


def _launch(controller: Controller, region: dict, debug: bool = False) -> None:
    # Lock the cart, then steer the aim to vertical with A/D (closed loop) before
    # throwing, so the ball goes straight up and is easiest to catch.
    controller.release_all()
    controller.tap_space()  # lock cart position
    time.sleep(0.3)  # let the aim phase begin

    key = "a"  # initial guess; the loop flips it if it makes the angle worse
    prev_abs = None
    samples: list[float] = []
    angle = None
    for _ in range(_AIM_MAX_ADJUST):
        frame = capture.grab(region)
        angle, pts, unit = detect.aim_fit(frame)
        if debug:
            _draw_aim_debug(frame, detect.detect_cart(frame), angle, pts, unit)
        if angle is None:
            time.sleep(_AIM_SAMPLE_DELAY)  # aim not visible yet; retry
            continue
        samples.append(round(angle, 1))
        if abs(angle) <= _AIM_TOLERANCE:
            break  # close enough to straight up
        # If the last pulse didn't reduce the tilt, we're pushing the wrong way
        # (or overshot) - flip the key.
        if prev_abs is not None and abs(angle) >= prev_abs:
            key = "d" if key == "a" else "a"
        prev_abs = abs(angle)
        # Proportional pulse: big when far off, tiny when nearly vertical, so we
        # ease onto vertical instead of overshooting and oscillating across it.
        step = min(_AIM_STEP_MAX, max(_AIM_STEP_MIN, abs(angle) * _AIM_STEP_GAIN))
        controller.hold(key)  # pulse A/D to rotate the aim
        time.sleep(step)
        controller.release_all()
        time.sleep(_AIM_SETTLE)

    controller.release_all()
    print(f"[bootbreaker] throwing (aim angle: {angle}, samples: {samples[-12:]})")
    controller.tap_space()  # throw
    time.sleep(0.4)  # let the ball leave the cart before we track it


def _render_aim_debug(frame, cart, angle, pts, unit):
    """Overlay for the aim phase: the detected aim dots (yellow), the fitted aim
    line (magenta), the aim-dot search band (grey), and a TRUE-vertical reference
    at the cart (green). If magenta isn't parallel to green, the throw isn't
    actually straight up - that's the discrepancy this is built to expose."""
    import cv2

    vis = frame.copy()
    h, w = vis.shape[:2]

    # Aim-dot search band (grey lines) - shows where we look for dots.
    y1, y2 = int(detect._AIM_BAND[0] * h), int(detect._AIM_BAND[1] * h)
    cv2.line(vis, (0, y1), (w, y1), (90, 90, 90), 1)
    cv2.line(vis, (0, y2), (w, y2), (90, 90, 90), 1)

    # True vertical at the cart (green) - what "straight up" should look like.
    if cart:
        cv2.line(vis, (cart[0], 0), (cart[0], h), (0, 220, 0), 1)

    # Detected aim dots (yellow).
    for px, py in pts:
        cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255), -1)

    # Fitted aim line (magenta), drawn full-length through the dots' centroid.
    if unit is not None and pts:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        vx, vy = unit
        p1 = (int(mx - vx * h), int(my - vy * h))
        p2 = (int(mx + vx * h), int(my + vy * h))
        cv2.line(vis, p1, p2, (255, 0, 255), 2)

    cv2.rectangle(vis, (0, 0), (360, 70), (0, 0, 0), -1)
    cv2.putText(vis, f"AIM angle={angle}  dots={len(pts)}", (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(vis, "dots=yellow  fit=magenta  vertical=green", (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis


def _draw_aim_debug(frame, cart, angle, pts, unit):
    import cv2

    vis = _render_aim_debug(frame, cart, angle, pts, unit)
    _ensure_debug_window(cv2, vis)
    cv2.imshow(_DEBUG_WINDOW, vis)
    cv2.waitKey(1)


def _ensure_debug_window(cv2, frame) -> None:
    """Create the debug window once: resizable, always-on-top, small, top-left,
    so it can sit beside the (borderless/windowed) game."""
    global _debug_win_ready
    if _debug_win_ready:
        return
    cv2.namedWindow(_DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    cv2.resizeWindow(_DEBUG_WINDOW, 360, max(1, round(360 * h / w)))
    cv2.moveWindow(_DEBUG_WINDOW, 0, 0)
    try:  # keep it above the game (needs a recent OpenCV; harmless if missing)
        cv2.setWindowProperty(_DEBUG_WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        print("[bootbreaker] note: could not pin debug window on top")
    _debug_win_ready = True


def _render_debug(frame, bot: Bot, action: str):
    """Draw the debug overlay onto a copy of the frame and return it."""
    import cv2

    vis = frame.copy()
    h = vis.shape[0]

    # Cart (red): full-height line at its x, a marker at its position, and a
    # faint deadzone band [cart-deadzone, cart+deadzone] - inside the band the
    # bot holds, so the band vs the target line shows exactly why it moves/holds.
    if bot.last_cart:
        cx, cy = bot.last_cart
        dz = int(bot.deadzone)
        band = vis.copy()
        cv2.rectangle(band, (cx - dz, 0), (cx + dz, h), (0, 0, 255), -1)
        cv2.addWeighted(band, 0.15, vis, 0.85, 0, vis)
        cv2.line(vis, (cx, 0), (cx, h), (0, 0, 255), 1)
        cv2.circle(vis, (cx, cy), 16, (0, 0, 255), 3)

    # Target (green): where the cart is steering to.
    if bot._target_x is not None:
        tx = int(bot._target_x)
        cv2.line(vis, (tx, 0), (tx, h), (0, 255, 0), 2)

    # Ball (cyan): outline + centre dot.
    if bot.last_ball:
        bx, by = bot.last_ball
        cv2.circle(vis, (bx, by), 12, (255, 255, 0), 2)
        cv2.circle(vis, (bx, by), 3, (255, 255, 0), -1)

    tgt = None if bot._target_x is None else int(bot._target_x)
    hud = [
        f"{bot.state}  action: {action}",
        f"prethrow: {bot.last_prethrow}",
        f"ball:   {bot.last_ball}",
        f"cart:   {bot.last_cart}",
        f"target: {tgt}",
    ]
    cv2.rectangle(vis, (0, 0), (330, 24 + 30 * len(hud)), (0, 0, 0), -1)
    for i, txt in enumerate(hud):
        cv2.putText(vis, txt, (8, 34 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # legend
    cv2.putText(vis, "ball=cyan cart=red target=green", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return vis


def _draw_debug(frame, bot: Bot, action: str):
    import cv2

    vis = _render_debug(frame, bot, action)
    _ensure_debug_window(cv2, vis)
    cv2.imshow(_DEBUG_WINDOW, vis)
    cv2.waitKey(1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    controller = Controller()
    bot = Bot()
    region = None
    toggle = _Toggle()

    # pynput global hotkey listener. Fires toggle.flip() on F8 from any window
    # (needs Accessibility permission, same as sending keys). Runs on its own
    # thread; we stop it in the finally below.
    def _on_press(key):
        if key == _kb.Key.f8:
            toggle.flip()

    listener = _kb.Listener(on_press=_on_press)
    listener.start()
    print("[bootbreaker] started PAUSED. Open Dota Bootbreaker, then press F8.")
    print("[bootbreaker] press F8 to pause/resume, Ctrl+C to quit.")

    frame_i = 0
    was_running = False
    last_t = time.perf_counter()
    try:
        while True:
            if not toggle.running:
                was_running = False
                controller.release_all()
                time.sleep(0.05)
                continue

            if not was_running:  # just resumed via F8 - give the user a moment
                _countdown(args.delay)
                was_running = True
                last_t = time.perf_counter()

            if region is None:
                region = _ensure_region(args.recalibrate)

            frame = capture.grab(region)
            action = bot.step(frame)

            if action == "launch":
                _launch(controller, region, debug=args.debug)
                bot.prime_playing()  # give the ball time to appear; don't churn
                last_t = time.perf_counter()  # don't count launch in fps
            elif action == "hold":
                controller.release_all()
            else:  # "left" / "right"
                controller.hold(_MOVE_KEY[action])

            if args.debug:
                now = time.perf_counter()
                fps = 1.0 / (now - last_t) if now > last_t else 0.0
                last_t = now
                print(
                    f"[dbg] fps={fps:4.1f} action={action} "
                    f"ball={bot.last_ball} cart={bot.last_cart} target={bot._target_x}"
                )
                if frame_i % 3 == 0:  # the window refresh is the costly part
                    _draw_debug(frame, bot, action)
            frame_i += 1
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        controller.release_all()
        print("\n[bootbreaker] stopped.")
