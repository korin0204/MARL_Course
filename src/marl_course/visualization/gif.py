"""Convert ANSI grid render frames into GIF images."""
from __future__ import annotations

from pathlib import Path


PALETTE = {
    "#": (45, 45, 55),
    "+": (150, 95, 50),
    "X": (45, 45, 55),
    " ": (235, 235, 220),
    "O": (200, 235, 130),
    "D": (120, 180, 255),
    "P": (220, 120, 70),
    "S": (120, 220, 160),
    "C": (190, 170, 140),
    "*": (255, 190, 40),
    "0": (90, 150, 255),
    "1": (255, 100, 120),
    "2": (80, 190, 120),
    "3": (230, 190, 60),
    "4": (110, 110, 120),
    "R": (255, 150, 90),
    "B": (140, 120, 255),
    "A": (90, 150, 255),
    "E": (80, 190, 120),
    "G": (230, 190, 60),
    "H": (180, 120, 255),
    "U": (140, 120, 255),
    "F": (255, 120, 50),
    "o": (120, 200, 80),
    "d": (80, 140, 230),
    "s": (230, 120, 80),
}

TEXT_COLOR = (35, 35, 42)
LEGEND_BG = (250, 250, 242)
LEGEND_BORDER = (185, 185, 175)

FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}

BOMBER_LEGEND = [
    ("#", "WALL"),
    ("+", "WOOD"),
    (" ", "FLOOR"),
    ("A", "AGENT"),
    ("4", "BOMB COUNT"),
    ("*", "FLAME"),
    ("U", "BOMB UP"),
    ("F", "FIRE UP"),
]

COOP_LEGEND = [
    ("X", "WALL"),
    (" ", "FLOOR"),
    ("O", "ONION"),
    ("D", "DISH"),
    ("P", "POT"),
    ("R", "READY POT"),
    ("S", "DELIVERY"),
    ("C", "COUNTER"),
    ("A", "AGENT"),
    ("2", "POT PROGRESS"),
    ("o", "HOLD ONION"),
    ("d", "HOLD DISH"),
    ("s", "HOLD SOUP"),
]


def ansi_frames_to_gif(
    ansi_frames: list[str],
    path: Path,
    tile_size: int = 16,
    fps: int = 8,
    agent_labels: list[tuple[str, str]] | None = None,
) -> Path | None:
    """Convert text renders into a GIF.

    The implementation uses imageio when available. If a student runs the core
    package without visualization dependencies, training still succeeds and the
    caller can simply skip W&B media upload.
    """

    if not ansi_frames:
        return None
    try:
        import imageio.v2 as imageio
        import numpy as np
    except Exception as exc:
        print(f"GIF disabled: install imageio and numpy to render episode GIFs ({exc})")
        return None

    images = [ansi_to_rgb_array(frame, tile_size=tile_size, np=np, agent_labels=agent_labels) for frame in ansi_frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1.0 / max(1, fps))
    return path


def ansi_to_rgb_array(ansi: str, tile_size: int = 16, np=None, agent_labels: list[tuple[str, str]] | None = None):  # type: ignore[no-untyped-def]
    """Render one ANSI frame to an RGB numpy array.

    GIF output and pygame live output both call this function, so classroom
    live display and saved media stay visually consistent.
    """

    if np is None:
        import numpy as np  # type: ignore[no-redef]

    return _ansi_to_image(ansi, tile_size, np, agent_labels=agent_labels or [])


def _ansi_to_image(ansi: str, tile_size: int, np, agent_labels: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    """Rasterize one ANSI text frame using the fixed visualization palette."""
    raw_lines = ansi.splitlines()
    header = raw_lines[0] if raw_lines else ""
    lines = [line for line in raw_lines[1:] if line]
    height = len(lines)
    width = max((len(line) for line in lines), default=1)
    legend_items = _legend_for_header(header)
    legend_width = _legend_width(tile_size, legend_items, agent_labels)
    canvas_height = max(height * tile_size, _legend_height(tile_size, len(legend_items), len(agent_labels)))
    image = np.zeros((canvas_height, width * tile_size + legend_width, 3), dtype=np.uint8)
    image[:, :, :] = LEGEND_BG
    for r, line in enumerate(lines):
        for c in range(width):
            char = line[c] if c < len(line) else " "
            _draw_tile(image, r * tile_size, c * tile_size, tile_size, char, _env_kind(header), np)
    if legend_items:
        _draw_legend(image, width * tile_size, tile_size, legend_items, agent_labels, _env_kind(header), np)
    return image


def _legend_for_header(header: str) -> list[tuple[str, str]]:
    """Choose legend entries based on the environment name in render header."""
    if header.startswith("BomberArena"):
        return BOMBER_LEGEND
    if header.startswith("CoopKitchen"):
        return COOP_LEGEND
    return []


def _env_kind(header: str) -> str:
    """Return renderer mode used to disambiguate shared symbols."""
    if header.startswith("BomberArena"):
        return "bomber"
    if header.startswith("CoopKitchen"):
        return "coop"
    return "generic"


def _legend_width(tile_size: int, items: list[tuple[str, str]], agent_labels: list[tuple[str, str]]) -> int:
    """Legend width grows when model names are shown."""
    if not items and not agent_labels:
        return 0
    longest = max([len(label) for _char, label in items + agent_labels] + [10])
    scale = max(1, tile_size // 12)
    return max(220, tile_size * 12, 36 + min(longest, 30) * 6 * scale)


def _legend_height(tile_size: int, item_count: int, agent_count: int) -> int:
    """Minimum image height needed to show entity rows and agent rows."""
    if item_count == 0 and agent_count == 0:
        return 0
    icon = max(10, min(tile_size, 18))
    scale = max(1, tile_size // 12)
    entity_height = 8 + max(14, 9 * scale) + item_count * (icon + 6)
    agent_height = 0 if agent_count == 0 else max(14, 9 * scale) + 4 + agent_count * (icon + 6)
    return entity_height + agent_height + 8


def _draw_tile(image, top: int, left: int, size: int, char: str, env_kind: str, np) -> None:  # type: ignore[no-untyped-def]
    """Draw one grid cell as a simple pixel-art symbol."""
    color = PALETTE.get(char, (245, 245, 235))
    image[top : top + size, left : left + size, :] = PALETTE.get(" ", (235, 235, 220))
    if env_kind == "bomber":
        _draw_bomber_tile(image, top, left, size, char, color, np)
    elif env_kind == "coop":
        _draw_coop_tile(image, top, left, size, char, color, np)
    else:
        _draw_basic_tile(image, top, left, size, char, color, np)
    _draw_rect_border(image, top, left, size, size, (210, 210, 200))


def _draw_basic_tile(image, top: int, left: int, size: int, char: str, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    margin = max(1, size // 8)
    if char.isdigit():
        _draw_circle(image, top, left, size, color, np)
        _draw_center_text(image, char, top, left, size)
    elif char == "*":
        _draw_diamond(image, top, left, size, color, np)
    elif char in {"A", "B", "C", "D", "E", "F", "G", "H", "R", "U", "o", "d", "s"}:
        _draw_diamond(image, top, left, size, color, np)
        _draw_center_text(image, char.upper(), top, left, size)
    else:
        image[top + margin : top + size - margin, left + margin : left + size - margin, :] = color


def _draw_bomber_tile(image, top: int, left: int, size: int, char: str, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    if char in {"A", "B", "C", "D"}:
        _draw_circle(image, top, left, size, color, np)
        _draw_center_text(image, char, top, left, size)
    elif char.isdigit():
        timer = max(0, min(4, int(char)))
        _draw_bomb(image, top, left, size, timer, np)
    elif char == "*":
        _draw_flame(image, top, left, size, np)
    elif char == "+":
        _draw_crate(image, top, left, size)
    elif char == "#":
        _draw_wall(image, top, left, size)
    elif char in {"U", "F"}:
        _draw_powerup(image, top, left, size, char, color, np)
    else:
        _draw_basic_tile(image, top, left, size, char, color, np)


def _draw_coop_tile(image, top: int, left: int, size: int, char: str, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    if char in {"A", "B", "E", "G"}:
        _draw_circle(image, top, left, size, color, np)
        _draw_center_text(image, char, top, left, size)
    elif char.isdigit():
        _draw_pot_progress(image, top, left, size, max(0, min(3, int(char))))
    elif char == "R":
        _draw_pot_progress(image, top, left, size, 3, ready=True)
    elif char == "O":
        _draw_onion(image, top, left, size, color, np)
    elif char == "D":
        _draw_plate(image, top, left, size, color, np)
    elif char == "P":
        _draw_pot_progress(image, top, left, size, 0)
    elif char == "S":
        _draw_delivery(image, top, left, size, color)
    elif char == "C":
        _draw_counter(image, top, left, size, color)
    elif char == "X":
        _draw_wall(image, top, left, size)
    elif char in {"o", "d", "s"}:
        _draw_carrier(image, top, left, size, char, color, np)
    else:
        _draw_basic_tile(image, top, left, size, char, color, np)


def _draw_legend(
    image,
    left: int,
    tile_size: int,
    items: list[tuple[str, str]],
    agent_labels: list[tuple[str, str]],
    env_kind: str,
    np,
) -> None:  # type: ignore[no-untyped-def]
    """Draw a compact legend panel on the right side of each GIF frame."""
    h, w, _channels = image.shape
    image[:, left:, :] = LEGEND_BG
    image[:, left : left + 1, :] = LEGEND_BORDER
    scale = max(1, tile_size // 12)
    text_x = left + tile_size + 12
    y = 8
    _draw_text(image, "LEGEND", left + 10, y, scale, TEXT_COLOR)
    y += max(14, 9 * scale)
    icon = max(10, min(tile_size, 18))
    for char, label in items:
        if y + icon + 2 >= h:
            break
        _draw_tile(image, y, left + 10, icon, char, env_kind, np)
        _draw_text(image, label, text_x, y + max(0, (icon - 7 * scale) // 2), scale, TEXT_COLOR)
        y += icon + 6
    if agent_labels and y + max(14, 9 * scale) < h:
        y += 4
        _draw_text(image, "AGENTS", left + 10, y, scale, TEXT_COLOR)
        y += max(14, 9 * scale)
        for char, label in agent_labels:
            if y + icon + 2 >= h:
                break
            _draw_tile(image, y, left + 10, icon, char, env_kind, np)
            _draw_text(image, _legend_label(label), text_x, y + max(0, (icon - 7 * scale) // 2), scale, TEXT_COLOR)
            y += icon + 6


def _draw_text(image, text: str, x: int, y: int, scale: int, color: tuple[int, int, int]) -> None:  # type: ignore[no-untyped-def]
    """Draw small 5x7 bitmap text; avoids adding a font dependency."""
    cursor = x
    for char in text.upper():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    top = y + row_idx * scale
                    left = cursor + col_idx * scale
                    image[top : top + scale, left : left + scale, :] = color
        cursor += 6 * scale


def _legend_label(label: str) -> str:
    """Normalize long model labels for the compact pixel legend."""
    value = label.upper().replace("_", "-")
    return value[:30]


def _draw_center_text(image, text: str, top: int, left: int, size: int, color: tuple[int, int, int] = TEXT_COLOR) -> None:  # type: ignore[no-untyped-def]
    """Draw bitmap text near the center of a tile."""
    scale = max(1, size // 14)
    text_width = max(1, len(text)) * 6 * scale
    x = left + (size - text_width) // 2
    y = top + (size - 7 * scale) // 2
    _draw_text(image, text, x, y, scale, color)


def _draw_rect_border(image, top: int, left: int, height: int, width: int, color: tuple[int, int, int]) -> None:  # type: ignore[no-untyped-def]
    image[top, left : left + width, :] = color
    image[top + height - 1, left : left + width, :] = color
    image[top : top + height, left, :] = color
    image[top : top + height, left + width - 1, :] = color


def _draw_circle(image, top: int, left: int, size: int, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    radius = max(2.0, size * 0.38)
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    image[top : top + size, left : left + size, :][mask] = color


def _draw_diamond(image, top: int, left: int, size: int, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    radius = max(2.0, size * 0.42)
    mask = abs(yy - center) + abs(xx - center) <= radius
    image[top : top + size, left : left + size, :][mask] = color


def _draw_wall(image, top: int, left: int, size: int) -> None:  # type: ignore[no-untyped-def]
    """Draw a block wall with a simple brick pattern."""
    image[top : top + size, left : left + size, :] = (48, 48, 58)
    line = (72, 72, 82)
    step = max(3, size // 3)
    for y in range(top + step, top + size, step):
        image[y : y + 1, left : left + size, :] = line
    for x in range(left + step, left + size, step):
        image[top : top + size, x : x + 1, :] = line


def _draw_crate(image, top: int, left: int, size: int) -> None:  # type: ignore[no-untyped-def]
    """Draw a destructible wooden crate."""
    margin = max(1, size // 8)
    wood = (150, 95, 50)
    dark = (95, 58, 32)
    image[top + margin : top + size - margin, left + margin : left + size - margin, :] = wood
    image[top + margin : top + size - margin, left + margin : left + margin + 1, :] = dark
    image[top + margin : top + size - margin, left + size - margin - 1 : left + size - margin, :] = dark
    for i in range(size - 2 * margin):
        r = top + margin + i
        c1 = left + margin + i
        c2 = left + size - margin - 1 - i
        if left + margin <= c1 < left + size - margin:
            image[r : r + 1, c1 : c1 + 1, :] = dark
        if left + margin <= c2 < left + size - margin:
            image[r : r + 1, c2 : c2 + 1, :] = dark


def _draw_bomb(image, top: int, left: int, size: int, timer: int, np) -> None:  # type: ignore[no-untyped-def]
    """Draw a bomb with countdown number and urgency bar."""
    body = (38, 38, 45)
    fuse = (245, 190, 75)
    body_size = max(6, int(size * 0.78))
    body_top = top + max(1, size // 7)
    body_left = left + (size - body_size) // 2
    _draw_circle(image, body_top, body_left, body_size, body, np)
    stem_top = top + max(1, size // 9)
    stem_left = left + size // 2
    image[stem_top : stem_top + max(1, size // 4), stem_left : stem_left + max(1, size // 8), :] = fuse
    urgency = {
        4: (80, 190, 120),
        3: (230, 190, 60),
        2: (255, 150, 90),
        1: (255, 80, 70),
        0: (255, 80, 70),
    }.get(timer, (255, 80, 70))
    bar_h = max(2, size // 7)
    bar_w = max(2, int((size - 4) * max(1, timer) / 4))
    image[top + size - bar_h - 2 : top + size - 2, left + 2 : left + 2 + bar_w, :] = urgency
    _draw_center_text(image, str(timer), top, left, size, (245, 245, 235))


def _draw_flame(image, top: int, left: int, size: int, np) -> None:  # type: ignore[no-untyped-def]
    """Draw flame as layered diamonds."""
    _draw_diamond(image, top, left, size, (255, 95, 45), np)
    inner = max(4, int(size * 0.62))
    offset = (size - inner) // 2
    _draw_diamond(image, top + offset, left + offset, inner, (255, 210, 55), np)


def _draw_powerup(image, top: int, left: int, size: int, char: str, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    """Draw Bomber power-up: U for extra bomb, F for flame range."""
    _draw_diamond(image, top, left, size, color, np)
    _draw_center_text(image, char, top, left, size)


def _draw_onion(image, top: int, left: int, size: int, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    _draw_circle(image, top, left, size, color, np)
    leaf = (70, 150, 70)
    image[top + 2 : top + max(3, size // 4), left + size // 2 : left + size // 2 + max(1, size // 6), :] = leaf


def _draw_plate(image, top: int, left: int, size: int, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    _draw_circle(image, top, left, size, (235, 245, 255), np)
    inner = max(4, int(size * 0.55))
    offset = (size - inner) // 2
    _draw_circle(image, top + offset, left + offset, inner, color, np)


def _draw_pot_progress(image, top: int, left: int, size: int, onions: int, ready: bool = False) -> None:  # type: ignore[no-untyped-def]
    """Draw pot with a three-step fill gauge for cooking progress."""
    margin = max(1, size // 8)
    metal = (92, 92, 102)
    soup = (255, 160, 70) if ready else (120, 200, 80)
    image[top + margin : top + size - margin, left + margin : left + size - margin, :] = metal
    fill_max = size - 2 * margin - 2
    fill_h = int(fill_max * (3 if ready else onions) / 3)
    image[top + size - margin - 1 - fill_h : top + size - margin - 1, left + margin + 1 : left + size - margin - 1, :] = soup
    _draw_center_text(image, "R" if ready else str(onions), top, left, size, (245, 245, 235))


def _draw_delivery(image, top: int, left: int, size: int, color: tuple[int, int, int]) -> None:  # type: ignore[no-untyped-def]
    margin = max(1, size // 7)
    image[top + margin : top + size - margin, left + margin : left + size - margin, :] = color
    _draw_center_text(image, "S", top, left, size)


def _draw_counter(image, top: int, left: int, size: int, color: tuple[int, int, int]) -> None:  # type: ignore[no-untyped-def]
    margin = max(1, size // 9)
    image[top + margin : top + size - margin, left + margin : left + size - margin, :] = color
    image[top + margin : top + margin + max(1, size // 6), left + margin : left + size - margin, :] = (160, 145, 115)


def _draw_carrier(image, top: int, left: int, size: int, char: str, color: tuple[int, int, int], np) -> None:  # type: ignore[no-untyped-def]
    """Draw a carrying agent with the held item shown as a small badge."""
    _draw_circle(image, top, left, size, (125, 125, 135), np)
    badge = max(4, size // 3)
    badge_top = top + size - badge - 2
    badge_left = left + size - badge - 2
    if char == "o":
        _draw_onion(image, badge_top, badge_left, badge, color, np)
    elif char == "d":
        _draw_plate(image, badge_top, badge_left, badge, color, np)
    else:
        _draw_diamond(image, badge_top, badge_left, badge, color, np)
    _draw_center_text(image, char.upper(), top, left, size)
