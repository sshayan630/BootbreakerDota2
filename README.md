# Bootbreaker Autoplayer (macOS)

A Python bot that auto-plays the Dota 2 **Bootbreaker** arcade minigame — it
launches, aims, and tracks the boot to keep the ball alive as long as possible.

**macOS port** of [jackblk/dota-bootbreaker-auto](https://github.com/jackblk/dota-bootbreaker-auto)
(originally Windows-only). The computer vision and game strategy are unchanged;
only the OS-specific input and hotkey layers were swapped:

| Concern | Windows original | macOS port |
| --- | --- | --- |
| Send key presses | `pydirectinput` (SendInput) | `pynput` (Quartz `CGEventPost`) |
| F8 pause/resume hotkey | `keyboard` | `pynput.keyboard.Listener` |
| Screen capture | `mss` | `mss` (Retina scale auto-detected) |

## Permissions (required)

macOS blocks screen reading and synthetic input until you grant them, per app.
Grant these to **the terminal you run the bot from** (Terminal, iTerm, VS Code…):

**System Settings → Privacy & Security →**

- **Screen Recording** — so the bot can see the game. Without it, capture is all
  black and calibration fails.
- **Accessibility** — so the bot can send keystrokes and listen for F8. Without
  it, key events are silently dropped and the cart never moves.

Quit and reopen the terminal after granting (macOS only applies the change to
newly launched processes).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync
```

## Run

```sh
uv run python -m bootbreaker            # play
uv run python -m bootbreaker --debug    # play with live overlay window
```

Open Dota 2 and press **F8** to start/pause the bot (it starts paused).
On first run it auto-detects the play region and saves it to `config.json`.
Run with `--recalibrate` to force re-detection (e.g. after moving the game
window or changing resolution).

## Test

```sh
uv run pytest
```

## Notes & caveats

- Run Dota 2 **windowed** or **borderless windowed** so the bot can capture it
  reliably; exclusive fullscreen can interfere with screen capture.
- The CV thresholds (HSV colour ranges, play-field ratios) are inherited from
  the upstream project and were tuned against its capture setup. If detection
  misfires on your display, run `--debug` to see the overlay and adjust the
  constants in `bootbreaker/detect.py`.
- Use responsibly and at your own risk — automating gameplay may violate the
  game's Terms of Service.
