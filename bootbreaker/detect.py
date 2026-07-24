"""Color-based detection of the play region, ball, and cart (OpenCV/HSV)."""

import os

import cv2
import numpy as np

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Larger kernel to merge the ball's fragmented glow ring into a single blob.
_BALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# The play field is anchored on the ornate top panel (BOOTS/LEVEL/SCORE), which
# is the widest gold structure at the very top of the arcade popup. These ratios
# (relative to the panel's width) were measured from the reference screenshots
# and are stable because the popup scales uniformly.
_PLAY_LEFT_OFF = 0.037
_PLAY_TOP_OFF = 0.166
_PLAY_WIDTH = 0.920
_PLAY_HEIGHT = 1.197


def _largest_contour(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def _centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def detect_play_region(image) -> dict | None:
    """Find the inner play field by anchoring on the gold top panel."""
    h = image.shape[0]
    strip = image[0:int(0.17 * h), :]  # panel lives in the very top strip
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    # Gold panel: yellow/orange hue, high saturation and value.
    mask = cv2.inRange(hsv, (15, 90, 120), (40, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # The panel is the widest gold structure in the top strip.
    px, py, pw, ph = max(
        (cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3]
    )
    if pw < 100:  # nothing panel-like found
        return None
    return {
        "left": px + round(_PLAY_LEFT_OFF * pw),
        "top": py + round(_PLAY_TOP_OFF * pw),
        "width": round(_PLAY_WIDTH * pw),
        "height": round(_PLAY_HEIGHT * pw),
    }


_MIN_BALL_AREA = 60  # px; ball glow blobs are ~230+, debris fragments are tiny
_MAX_BALL_WIDTH = 75  # px; wider cyan blobs are steel-blue bricks, not the ball
_MIN_BOOT_BROWN = 400  # px of brown near the glow to confirm it's the boot (ball)
_BOOT_PAD = 30  # px window around the glow to look for the boot
# The cart's teal glass is cyan like the ball glow, and its red body reads as
# "brown" next to it, so we blank the cart before searching. Keep this blank as
# small as possible: a wide/tall blank swallows the ball on a tight-angle
# descent onto the cart (it disappears in the blank's shadow, then re-emerges
# too low to catch). Measured on the reference frames (tmp/probe_cart_blank.py):
# the glass only leaks as a false ball at half-width < ~70 (it sits ~70px off
# centre, below it), and never leaks above the centroid at any height. So the
# width must stay >= 80, but the top can be shallow, keeping the descending ball
# visible right down to just above the cart.
_CART_BLANK_UP = 45  # px above the cart centroid
_CART_BLANK_HALF_W = 80  # px each side of the cart centroid
_MIN_CART_AREA = 40  # px; smallest red blob considered part of the cart
_CART_PART_FRAC = 0.4  # keep red blocks >= this fraction of the largest (awning
#                        halves); rejects small decorations and the move arrows

# The boot's cyan glow is a bright, saturated cyan. The steel-blue bricks share
# its hue but are a flat, dull colour (measured S=110, V=162 vs the glow's
# S~214, V>=180), so a high saturation+value floor rejects the bricks while
# keeping the glow. Upper bound stays wide to survive motion blur.
_CYAN_LO = (80, 130, 170)
_CYAN_HI = (110, 255, 255)


def detect_ball(image, cart: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Locate the ball (the flying boot). The boot has a cyan glow ring plus a
    brown boot body; we find cyan blobs and keep the one with a brown boot
    beside it. That rejects the pastel steel-blue bricks (same hue as the glow
    but no adjacent boot) and the cart's own teal glass.
    """
    h, w = image.shape[:2]
    roi = image.copy()
    if cart is not None:
        cx, cy = cart
        # Blank just the cart (its teal glass), not the whole bottom strip, so
        # a low ball off to the side is still visible.
        roi[max(0, cy - _CART_BLANK_UP):,
            max(0, cx - _CART_BLANK_HALF_W):min(w, cx + _CART_BLANK_HALF_W)] = 0
    else:
        roi[int(h * 0.82):, :] = 0  # fallback: blank the bottom strip
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Cyan glow: saturated + bright to reject dull steel-blue bricks of the same
    # hue. Closed into one blob.
    cyan = cv2.inRange(hsv, _CYAN_LO, _CYAN_HI)
    cyan = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, _BALL_KERNEL)
    # Brown boot body: warm hue, darker than the bright gold bricks.
    brown = cv2.inRange(hsv, (3, 90, 40), (18, 255, 175))

    contours, _ = cv2.findContours(cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_brown = -1
    best_win = None
    for c in contours:
        if cv2.contourArea(c) < _MIN_BALL_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > _MAX_BALL_WIDTH:
            continue
        x1, y1 = max(0, x - _BOOT_PAD), max(0, y - _BOOT_PAD)
        x2, y2 = min(w, x + bw + _BOOT_PAD), min(h, y + bh + _BOOT_PAD)
        brown_near = int(brown[y1:y2, x1:x2].sum() / 255)
        if brown_near > best_brown:
            best_brown = brown_near
            best = c
            best_win = (x1, y1, x2, y2)
    if best is None or best_brown < _MIN_BOOT_BROWN:
        return None
    # Center on the brown boot body, not the cyan glow (the glow sits to one
    # side of the boot, so its centroid is off-centre).
    x1, y1, x2, y2 = best_win
    m = cv2.moments(brown[y1:y2, x1:x2], binaryImage=True)
    if m["m00"] > 0:
        return (int(x1 + m["m10"] / m["m00"]), int(y1 + m["m01"] / m["m00"]))
    return _centroid(best)  # fallback (shouldn't happen: best_brown >= min)


# Aim-dot search band (fractions of region height): below the bricks, above the
# cart body. Exposed so the debug overlay can draw the same band.
_AIM_BAND = (0.52, 0.83)


def aim_fit(image):
    """Fit the launch-trajectory aim line. Returns (angle, pts, unit):
      angle - degrees from vertical (0 = straight up, + right, - left), or None
              if fewer than 3 aim dots were found;
      pts   - the detected aim-dot centroids [(x, y), ...] (region-local);
      unit  - the fitted direction unit vector (vx, vy) oriented up, or None.
    The dots are small warm blobs in the empty band between the lowest bricks
    and the cart. Only meaningful during the THROW BOOT phase.
    """
    h = image.shape[0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 150), (30, 255, 255))
    band = np.zeros(mask.shape, dtype=mask.dtype)
    band[int(_AIM_BAND[0] * h):int(_AIM_BAND[1] * h), :] = 255
    mask = cv2.bitwise_and(mask, band)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        _, _, bw, bh = cv2.boundingRect(c)
        if bw > 30 or bh > 30:  # dots are small; skip bricks/blobs
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    if len(pts) < 3:
        return None, pts, None
    vx, vy = cv2.fitLine(
        np.array(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()[:2]
    if vy > 0:  # orient the direction vector to point up (decreasing y)
        vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, -vy)))
    return angle, pts, (float(vx), float(vy))


def detect_aim_angle(image) -> float | None:
    """Angle (degrees) of the launch trajectory: 0 = straight up, + right,
    - left. None if no aim line found. Thin wrapper over aim_fit."""
    return aim_fit(image)[0]


def detect_cart(image, strip_frac: float = 0.25) -> tuple[int, int] | None:
    h = image.shape[0]
    y0 = int(h * (1 - strip_frac))
    strip = image[y0:, :]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    # Red wraps the hue circle -> two ranges.
    mask = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 120, 90), (179, 255, 255)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CART_AREA]
    if not contours:
        return None
    # The cart's red top is split into two blocks by the central boot ornament.
    # Keep every block comparable in size to the largest (both awning halves) and
    # drop small decorations / the move arrows, then take the midpoint of their
    # combined span so the centre is the true middle of the bounce surface - not
    # the centre of just one half.
    largest = max(cv2.contourArea(c) for c in contours)
    parts = [c for c in contours if cv2.contourArea(c) >= _CART_PART_FRAC * largest]
    boxes = [cv2.boundingRect(c) for c in parts]
    cx = (min(x for x, _, _, _ in boxes) + max(x + bw for x, _, bw, _ in boxes)) // 2
    cy = int(sum(y + bh / 2 for _, y, _, bh in boxes) / len(boxes)) + y0
    return (cx, cy)


# The pre-throw states ("LOCK CART POSITION" and "THROW BOOT") both show the same
# grey space-bar key icon in the centre prompt band. It's a fixed sprite, so we
# template-match it rather than fight its dull-grey colour. Matched in the band
# y[0.5,0.72]; scores ~1.0 in pre-throw frames vs ~0.6 mid-play, so 0.8 is safe.
_KEY_ICON_PATH = os.path.join(os.path.dirname(__file__), "key_icon.png")
_KEY_TEMPLATE = None
_PRETHROW_THRESHOLD = 0.8
_PRETHROW_BAND = (0.5, 0.72)


# The key icon is a fixed sprite, but its on-screen pixel size scales with the
# game's resolution - and template matching is NOT scale-invariant. The bundled
# template was cut at one resolution, so on a different display it would never
# score high enough. Match it at a spread of scales and take the best; 1.0 is in
# the set so this is a superset of the old single-scale behaviour.
_PRETHROW_SCALES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)


def prethrow_score(image) -> float:
    """Best multi-scale template-match score for the space-bar key icon in the
    prompt band (0..1). Exposed so tooling can read the raw score."""
    global _KEY_TEMPLATE
    if _KEY_TEMPLATE is None:
        tmpl = cv2.imread(_KEY_ICON_PATH, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            return -1.0  # template missing -> never claim pre-throw
        _KEY_TEMPLATE = tmpl
    h = image.shape[0]
    band = image[int(_PRETHROW_BAND[0] * h):int(_PRETHROW_BAND[1] * h), :]
    if band.shape[0] < 2 or band.shape[1] < 2:
        return -1.0
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    best = -1.0
    for s in _PRETHROW_SCALES:
        t = _KEY_TEMPLATE if s == 1.0 else cv2.resize(
            _KEY_TEMPLATE, None, fx=s, fy=s,
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if t.shape[0] > gray.shape[0] or t.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best


def detect_prethrow(image, threshold: float = _PRETHROW_THRESHOLD) -> bool:
    """True when the pre-throw prompt (the space-bar key icon) is on screen -
    i.e. the game is waiting for us to lock/throw the boot."""
    return prethrow_score(image) >= threshold
