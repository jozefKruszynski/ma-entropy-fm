"""
The look of the Music Assistant covers: the style presets, the prompt and the title overlay.

The presets, the base prompt and the overlay come from the AI generated covers of the Playlist
Metadata provider of the Music Assistant server (`music_assistant.providers.playlist_metadata`,
Apache-2.0, Copyright (c) the Music Assistant authors). When that module can be imported its
presets and its overlay are used, so the covers follow upstream. When it cannot (a stable server
that does not have it, or a refactor) the copy below is used; it was taken from the dev branch of
2026-11-18 and keeps its Apache-2.0 notice. Pillow is a dependency of the server; when it is not
there the overlay is left out and the cover is served as the engine made it.

The module imports nothing else from Music Assistant.
"""

from __future__ import annotations

import contextlib
import importlib
import string
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont

CANVAS_SIZE: Final = 1024
MAX_TITLE_LINES: Final = 3
MIN_TITLE_SCALE: Final = 0.4
MAX_SOURCE_PIXELS: Final = 4096 * 4096
UPSTREAM_MODULE: Final = "music_assistant.providers.playlist_metadata"


class Style(StrEnum):
    """The artwork style presets."""

    MUSIC_ASSISTANT = "music_assistant"
    NEON = "neon"
    BOLD = "bold"
    SOFT_GRADIENT = "soft_gradient"
    RISOGRAPH = "risograph"
    GOUACHE = "gouache"
    CUSTOM = "custom"


class TitleLayout(StrEnum):
    """Where and how the title overlay is drawn."""

    TOP_LEFT_LARGE = "top_left_large"
    CENTER = "center"
    BOTTOM_LEFT_SMALL = "bottom_left_small"


@dataclass(frozen=True, kw_only=True)
class StylePreset:
    """The prompt fragments and title overlay layout of a style."""

    art: str
    # how the image model should set the title, when it draws the title itself
    typography: str
    # the area to keep clear when the title is overlaid afterwards
    title_area: str
    layout: TitleLayout


@dataclass(frozen=True, kw_only=True)
class CoverLook:
    """The settings that shape how a cover looks."""

    # the value of a `Style`, or "custom"
    style: str
    custom_prompt: str
    title_overlay: bool


BASE_PROMPT: Final = (
    'Create square album cover art for a music playlist called "{name}". {context}{concept} {style}'
)

DEFAULT_CONCEPT: Final = (
    "Capture the mood of the playlist name as a striking visual scene or concept "
    "(do not depict real artists)."
)
CONCEPT_FROM_ENGINE: Final = "Visual concept: {concept} (do not depict real artists)."

TITLE_RULES: Final = (
    "Spell the title exactly. No other text, subtitles, logos, watermarks or artist names."
)
NO_TEXT_RULES: Final = (
    "No text, letters, numbers, logos or watermarks anywhere in the image. Keep the {area} "
    "calm and uncluttered, a title will be set there afterwards."
)
FULL_BLEED: Final = (
    "Full-bleed: the art fills the entire square edge to edge, no border, frame, mockup or margins."
)

STYLE_PRESETS: Final[dict[Style, StylePreset]] = {
    Style.BOLD: StylePreset(
        art=(
            "Style: a bold editorial playlist cover. One striking full-bleed image (stylized "
            "photo or bold illustration) with a vivid duotone or saturated color wash that sets "
            "the mood, simple uncluttered composition, cohesive with a series of playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, set large in a bold '
            "modern geometric sans-serif (like Circular or Gotham Black), tight letter "
            "spacing, clean and crisp, high contrast against the image, placed confidently in the "
            "upper-left or lower-left area."
        ),
        title_area="upper-left area",
        layout=TitleLayout.TOP_LEFT_LARGE,
    ),
    Style.MUSIC_ASSISTANT: StylePreset(
        art=(
            "Style: a Music Assistant playlist cover. Modern, clean and full of vibe: a rich, "
            "atmospheric gradient of two to three colors chosen specifically for this playlist's "
            "mood and genre, so covers in a series each get their own distinct palette (do not "
            "default to purple and orange), smooth and luminous with a soft light bloom and fine "
            "film grain. The hero is a detailed white line-art spot illustration that symbolizes "
            "the playlist's theme, like the icons on Music Assistant's genre tiles: one main, "
            "clearly recognizable object (an instrument, vehicle, object or scene) with a few "
            "related details such as sparkles, sound lines, stars or motion lines, drawn with "
            "thick and thin rounded white strokes, with a few accents filled in one bright color "
            "from the palette. It is drawn as an extreme close-up, so large that it does not fit "
            "on the cover: it fills the upper and right part of the square and runs off the top "
            "and right edges, cropped by them like a zoomed-in detail. The illustration stays "
            "within the upper and right two thirds of the square; the lower-left area, where the "
            "title goes, is plain gradient with no lines or details at all, gently darkening "
            "there. Cohesive with a series of playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, in a bold, clean modern '
            "sans-serif (like Inter Bold), white, crisp and perfectly legible, large, in the "
            "clear lower-left corner, with nothing drawn behind or across it."
        ),
        title_area="lower-left area",
        layout=TitleLayout.BOTTOM_LEFT_SMALL,
    ),
    Style.NEON: StylePreset(
        art=(
            "Style: a Music Assistant playlist cover. A night scene that captures the playlist, "
            "drawn as glowing neon light lines and long-exposure light trails over a nearly black "
            "background, like a neon sign or light painting: the main subject outlined in sky "
            "blue #03A9F4 neon with one second neon color from the mood, soft bloom, reflections "
            "and haze. Electric, nocturnal, minimal, cohesive with a series of playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, as a crisp white '
            "neon-tube script or clean sans-serif sign, centered near the bottom."
        ),
        title_area="lower area",
        layout=TitleLayout.BOTTOM_LEFT_SMALL,
    ),
    Style.SOFT_GRADIENT: StylePreset(
        art=(
            "Style: a soft gradient editorial playlist cover. A soft, blurred, vivid mesh gradient "
            "of two to four rich colors that flow into each other, with an optional subtle, "
            "out-of-focus hint of the scene, smooth and luminous with fine grain, minimal and "
            "elegant, cohesive with a series of playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, small, in white, set in '
            "a clean bold sans-serif (like Inter or Helvetica Neue), in the bottom-left corner over a "
            "subtle dark vignette."
        ),
        title_area="bottom-left corner",
        layout=TitleLayout.BOTTOM_LEFT_SMALL,
    ),
    Style.RISOGRAPH: StylePreset(
        art=(
            "Style: a Music Assistant playlist cover. A two-color risograph print: a bold "
            "illustration of a scene that captures the playlist, printed in sky blue #03A9F4 and "
            "one fluorescent warm ink (pink, orange or yellow, matching the mood) on off-white "
            "paper, with visible halftone dots, slight ink misregistration, overprint where the "
            "colors overlap and paper grain. Indie, handmade, joyful, cohesive with a series of "
            "playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, large, in a chunky '
            "condensed sans-serif printed in the same ink, at the top."
        ),
        title_area="top band",
        layout=TitleLayout.TOP_LEFT_LARGE,
    ),
    Style.GOUACHE: StylePreset(
        art=(
            "Style: a Music Assistant playlist cover. A warm, painterly editorial illustration in "
            "gouache: a scene that captures the playlist with simplified shapes, visible brush "
            "texture and a rich, harmonious palette led by the mood of the playlist, with a touch "
            "of sky blue #03A9F4 somewhere in the scene (the sky, water, a light, clothing). "
            "Feels like a magazine cover illustration, nostalgic and human, cohesive with a "
            "series of playlist covers."
        ),
        typography=(
            'Typography: the playlist title "{title}" as the only text, in a clean bold '
            "sans-serif, white or cream, in a calm area at the top."
        ),
        title_area="top area",
        layout=TitleLayout.TOP_LEFT_LARGE,
    ),
}

# custom templates have no typography of their own, so an overlay uses the Bold layout
CUSTOM_LAYOUT: Final = TitleLayout.TOP_LEFT_LARGE

DEFAULT_CUSTOM_PROMPT: Final = (
    'Create square album cover art for a music playlist called "{name}". {context}{concept} '
    "Style: a retro 1970s screen-printed poster with bold flat shapes and a limited warm palette. "
    "No text. Full-bleed, no border or frame."
)


def format_template(template: str, values: dict[str, str]) -> str:
    """Fill the placeholders of a user template, leaving anything unknown as typed."""
    result = ""
    try:
        for literal, field_name, _spec, _conversion in string.Formatter().parse(template):
            result += literal
            if field_name is None:
                continue
            result += values.get(field_name, f"{{{field_name}}}")
    except ValueError:
        # unbalanced braces: the template is meant literally
        return template
    return result


def render_cover(image: bytes, layout: TitleLayout | None, title: str) -> bytes:
    """Return the generated image as a square JPEG cover, with the title drawn when asked."""
    from PIL import Image, ImageOps  # noqa: PLC0415

    with Image.open(BytesIO(image)) as source:
        # the size is known before the pixels are decoded
        if source.width * source.height > MAX_SOURCE_PIXELS:
            msg = f"the engine returned an image of {source.width}x{source.height} pixels"
            raise ValueError(msg)
        canvas = ImageOps.fit(source.convert("RGB"), (CANVAS_SIZE, CANVAS_SIZE))
    if layout is not None:
        canvas = draw_title(canvas, layout, title)
    buffer = BytesIO()
    canvas.save(buffer, "JPEG", optimize=True, quality=90)
    return buffer.getvalue()


def draw_title(canvas: Image.Image, layout: TitleLayout, title: str) -> Image.Image:
    """Draw the title over the cover, with a shade behind it for legibility."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    size = canvas.width
    margin = int(size * 0.07)
    if layout == TitleLayout.BOTTOM_LEFT_SMALL:
        font_size = int(size * 0.065)
    elif layout == TitleLayout.CENTER:
        font_size = int(size * 0.09)
    else:
        font_size = int(size * 0.12)
    font, font_size, lines = fit_title(title, font_size, size - 2 * margin)
    line_height = int(font_size * 1.08)
    block_height = line_height * len(lines)

    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_shade = ImageDraw.Draw(shade)
    if layout == TitleLayout.CENTER:
        # an even dim keeps the centered title legible on any background
        draw_shade.rectangle((0, 0, size, size), fill=(0, 0, 0, 70))
        top = (size - block_height) // 2
    elif layout == TitleLayout.BOTTOM_LEFT_SMALL:
        vertical_gradient(draw_shade, size, from_top=False, height=size // 2)
        top = size - margin - block_height
    else:
        vertical_gradient(draw_shade, size, from_top=True, height=size // 2)
        top = margin
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shade)

    draw = ImageDraw.Draw(canvas)
    # thickens the strokes into a heavy weight, also when only a regular font is installed
    weight = 0 if layout == TitleLayout.BOTTOM_LEFT_SMALL else max(1, font_size // 40)
    for index, line in enumerate(lines):
        y = top + index * line_height
        if layout == TitleLayout.CENTER:
            x = (size - int(draw.textlength(line, font=font))) // 2
        else:
            x = margin
        draw.text(
            (x + 3, y + 3),
            line,
            fill=(0, 0, 0, 120),
            font=font,
            stroke_width=weight,
            stroke_fill=(0, 0, 0, 120),
        )
        draw.text(
            (x, y),
            line,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=weight,
            stroke_fill=(255, 255, 255, 255),
        )
    return canvas.convert("RGB")


def fit_title(
    title: str, font_size: int, max_width: int
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, list[str]]:
    """
    Return the font, its size and the lines that fit the whole title within the width.

    The font shrinks until the title takes at most a few lines; nothing of it is ever dropped.

    :param title: The title to fit.
    :param font_size: The size to start from.
    :param max_width: The width each line has to fit in, in pixels.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    min_size = max(1, int(font_size * MIN_TITLE_SCALE))
    while True:
        font = get_font(font_size)
        lines = wrap_title(title, font, max_width, measure)
        if len(lines) <= MAX_TITLE_LINES or font_size <= min_size:
            return font, font_size, lines
        font_size = max(min_size, int(font_size * 0.9))


def wrap_title(
    title: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    measure: ImageDraw.ImageDraw,
) -> list[str]:
    """Return the title wrapped into lines that fit the width, breaking only too long words."""
    lines: list[str] = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}" if current else word
        if measure.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        # a word wider than the line on its own is split over as many lines as it needs
        current = ""
        for char in word:
            if current and measure.textlength(current + char, font=font) > max_width:
                lines.append(current)
                current = ""
            current += char
    if current:
        lines.append(current)
    return lines or [title]


def vertical_gradient(draw: ImageDraw.ImageDraw, size: int, *, from_top: bool, height: int) -> None:
    """Draw a dark gradient fading out from the top or bottom edge."""
    for offset in range(height):
        alpha = int(170 * (1 - offset / height) ** 1.5)
        y = offset if from_top else size - 1 - offset
        draw.line([(0, y), (size, y)], fill=(0, 0, 0, alpha))


FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Windows/Fonts/arialbd.ttf",
)


_FONTS: dict[str, Any] = {}


def _upstream_fonts() -> Any | None:
    """Return the font module of the server, or None; the answer is kept."""
    if "module" not in _FONTS:
        try:
            _FONTS["module"] = importlib.import_module(f"{UPSTREAM_MODULE}.fonts")
        except Exception:  # noqa: BLE001
            _FONTS["module"] = None
    return _FONTS["module"]


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Return a bold TrueType font at the given size, falling back to Pillow's built-in.

    :param size: The font size, in pixels.
    """
    from PIL import ImageFont  # noqa: PLC0415

    fonts = _upstream_fonts()
    if fonts is not None:
        with contextlib.suppress(Exception):
            return fonts.get_font(size)  # type: ignore[no-any-return]
    for path in FONT_PATHS:
        if Path(path).exists():
            with contextlib.suppress(OSError):
                return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# -- what is used: upstream when it can be imported, the copy above when not ---------------------

_UPSTREAM: dict[str, SimpleNamespace | None] = {}
_NEEDED = (
    "Style",
    "STYLE_PRESETS",
    "BASE_PROMPT",
    "NO_TEXT_RULES",
    "TITLE_RULES",
    "FULL_BLEED",
    "DEFAULT_CONCEPT",
    "DEFAULT_CUSTOM_PROMPT",
    "CUSTOM_LAYOUT",
)


def upstream() -> SimpleNamespace | None:
    """
    Return the presets and the overlay of the server when it has them, or None.

    Importing it can fail in many ways (a stable server, a refactor, a missing Pillow); every one of
    them means that the copy in this module is used. The result is kept.
    """
    if "found" in _UPSTREAM:
        return _UPSTREAM["found"]
    found: SimpleNamespace | None = None
    try:
        constants = importlib.import_module(f"{UPSTREAM_MODULE}.ai_constants")
        values = {name: getattr(constants, name) for name in _NEEDED}
        render: Any = None
        with contextlib.suppress(Exception):
            render = importlib.import_module(f"{UPSTREAM_MODULE}.ai")._render_cover  # noqa: SLF001
        found = SimpleNamespace(**values, render_cover=render)
    except Exception:  # noqa: BLE001 - whatever goes wrong, the copy is used
        found = None
    _UPSTREAM["found"] = found
    return found


def _source() -> SimpleNamespace:
    """Return the presets in use: those of the server, or the copy."""
    found = upstream()
    if found is not None:
        return found
    return SimpleNamespace(
        Style=Style,
        STYLE_PRESETS=STYLE_PRESETS,
        BASE_PROMPT=BASE_PROMPT,
        NO_TEXT_RULES=NO_TEXT_RULES,
        TITLE_RULES=TITLE_RULES,
        FULL_BLEED=FULL_BLEED,
        DEFAULT_CONCEPT=DEFAULT_CONCEPT,
        DEFAULT_CUSTOM_PROMPT=DEFAULT_CUSTOM_PROMPT,
        CUSTOM_LAYOUT=CUSTOM_LAYOUT,
        render_cover=None,
    )


def style_options() -> list[tuple[str, str]]:
    """Return the styles to choose from as (value, title), the Music Assistant look first."""
    source = _source()
    return [
        (str(style.value), str(style.value).replace("_", " ").title()) for style in source.Style
    ]


def build_prompt(name: str, look: CoverLook, context: str, concept: str = "") -> str:
    """
    Return the image prompt of a cover, as the Playlist Metadata provider builds it.

    :param name: The title of the cover.
    :param look: The look of the cover.
    :param context: The mood description of the day (see `covers.describe_day`).
    :param concept: The visual concept; the default one of the preset when empty.
    """
    source = _source()
    concept = concept or source.DEFAULT_CONCEPT
    values = {
        "name": name,
        "title": name,
        "context": context,
        "concept": concept,
        "description": concept,
    }
    if look.style == "custom":
        template = look.custom_prompt or source.DEFAULT_CUSTOM_PROMPT
        return format_template(template, values).strip()
    preset = source.STYLE_PRESETS[source.Style(look.style)]
    if look.title_overlay:
        text_rules = source.NO_TEXT_RULES.format(area=preset.title_area)
    else:
        text_rules = f"{preset.typography.format(title=name)} {source.TITLE_RULES}"
    values["style"] = f"{preset.art} {text_rules} {source.FULL_BLEED}"
    return str(source.BASE_PROMPT.format(**values))


def layout_of(look: CoverLook) -> Any | None:
    """Return how the title is drawn over the image, None when the image carries it."""
    if not look.title_overlay:
        return None
    source = _source()
    if look.style == "custom":
        return source.CUSTOM_LAYOUT
    return source.STYLE_PRESETS[source.Style(look.style)].layout


def render_with_title(image: bytes, look: CoverLook, title: str) -> bytes:
    """
    Return the image of the engine as a square JPEG with the title drawn over it.

    The overlay of the server is used when there is one, the copy when not, and when Pillow is
    missing the image is returned as the engine made it.

    :param image: The image of the engine.
    :param look: The look, which says whether and where the title goes.
    :param title: The title.
    """
    layout = layout_of(look)
    source = _source()
    try:
        if source.render_cover is not None:
            return bytes(source.render_cover(image, layout, title))
        return render_cover(
            image, TitleLayout(getattr(layout, "value", layout)) if layout else None, title
        )
    except ImportError:
        return image


def draw_title_on(canvas: Image.Image, title: str) -> Image.Image:
    """
    Draw the title in the lower-left corner of a cover, the way the covers of the server do.

    The overlay of the server is used when it has one, else the copy above.

    :param canvas: A square RGB image of 1024 pixels.
    :param title: The title.
    """
    found = upstream()
    if found is not None:
        with contextlib.suppress(Exception):
            ai = importlib.import_module(f"{UPSTREAM_MODULE}.ai")
            layout = found.STYLE_PRESETS[found.Style("music_assistant")].layout
            return ai._draw_title(canvas, layout, title)  # type: ignore[no-any-return]  # noqa: SLF001
    return draw_title(canvas, TitleLayout.BOTTOM_LEFT_SMALL, title)
