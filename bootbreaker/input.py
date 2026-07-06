"""Keyboard output for the game via pynput (Quartz CGEvent on macOS).

The Windows original used pydirectinput (scancode SendInput); that library is
Windows-only and won't even import on macOS. pynput synthesizes key events
through Quartz `CGEventPost`, which Dota 2 on macOS reads like real hardware
input — *provided the terminal running this bot has Accessibility permission*
(System Settings -> Privacy & Security -> Accessibility). Without it, pynput's
key events are silently dropped and the cart never moves.
"""

import time

from pynput.keyboard import Controller as _Keyboard, Key

# How long tap_space holds space down. The Windows notes still apply: an
# instantaneous down+up tap lasts microseconds, which Dota's per-frame (~16ms)
# input polling misses entirely - the cart never locked/threw. So we hold space
# explicitly for a few frames. This only runs in _launch (which already sleeps
# hundreds of ms), never in the tracking loop.
_TAP_HOLD = 0.06

# The rest of the app speaks in these string key names; map them to the pynput
# key objects the real backend needs. Letters pass through as-is; "space" is a
# special key.
_KEYS = {"a": "a", "d": "d", "space": Key.space}


class _PynputBackend:
    """Adapts pynput's press/release to the keyDown/keyUp interface Controller
    expects. Kept as a separate object so tests can inject a fake backend and
    never touch the real keyboard (or need Accessibility permission)."""

    def __init__(self):
        self._kb = _Keyboard()

    def keyDown(self, key: str) -> None:
        self._kb.press(_KEYS[key])

    def keyUp(self, key: str) -> None:
        self._kb.release(_KEYS[key])


class Controller:
    def __init__(self, backend=None):
        # Build the real backend lazily so importing this module (and the test
        # suite) never constructs a live keyboard controller.
        self._backend = backend if backend is not None else _PynputBackend()
        self._held: set[str] = set()

    def hold(self, key: str) -> None:
        if key in self._held:
            return
        for other in list(self._held):
            self._backend.keyUp(other)
            self._held.discard(other)
        self._backend.keyDown(key)
        self._held.add(key)

    def release_all(self) -> None:
        for key in list(self._held):
            self._backend.keyUp(key)
            self._held.discard(key)

    def tap_space(self) -> None:
        # Hold space long enough for the game to register it (see _TAP_HOLD).
        self._backend.keyDown("space")
        time.sleep(_TAP_HOLD)
        self._backend.keyUp("space")
