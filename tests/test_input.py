from bootbreaker.input import Controller, _KEYS


def test_key_names_map_to_pynput_keys():
    # The rest of the app steers with these three string keys; the backend must
    # know how to translate every one of them into a pynput key object.
    from pynput.keyboard import Key

    assert set(_KEYS) == {"a", "d", "space"}
    assert _KEYS["space"] == Key.space


class FakeBackend:
    def __init__(self):
        self.events = []

    def keyDown(self, key):
        self.events.append(("down", key))

    def keyUp(self, key):
        self.events.append(("up", key))

    def press(self, key):
        self.events.append(("press", key))


def test_hold_presses_key_once():
    fb = FakeBackend()
    c = Controller(backend=fb)
    c.hold("a")
    c.hold("a")  # already held -> no new event
    assert fb.events == [("down", "a")]


def test_hold_switches_direction_releases_previous():
    fb = FakeBackend()
    c = Controller(backend=fb)
    c.hold("a")
    c.hold("d")
    assert fb.events == [("down", "a"), ("up", "a"), ("down", "d")]


def test_release_all_releases_held_keys():
    fb = FakeBackend()
    c = Controller(backend=fb)
    c.hold("d")
    c.release_all()
    assert fb.events == [("down", "d"), ("up", "d")]
    c.release_all()  # nothing held -> no new events
    assert fb.events == [("down", "d"), ("up", "d")]


def test_tap_space_holds_then_releases():
    # tap_space must send a real keyDown+keyUp (holding briefly), NOT the
    # instantaneous press() - with PAUSE=0 a press() tap is too short for Dota's
    # per-frame input polling to register, so the cart never locked/threw.
    fb = FakeBackend()
    c = Controller(backend=fb)
    c.tap_space()
    assert fb.events == [("down", "space"), ("up", "space")]
