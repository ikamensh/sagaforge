"""Re-render schematic sprites with an image model.

A game packs the frames of one subject into a :class:`Sheet`: a grid of equal
cells on a flat chroma background, every frame anchored at the same point of
its cell, thin cell borders and an empty margin around the grid so the model
keeps the layout.  A provider (Codex's built-in image tool through
:func:`render_with_codex`, or an OpenRouter image model through
:func:`render_with_openrouter`) repaints the sheet from a prompt.  :func:`cut`
keys the background out (a ground shadow the model painted in a darker key
colour becomes translucent black), registers the result against the original
with one uniform scale and shift for the whole sheet (per-cell fitting would
make frames jitter), and reports per-cell checks so a bad sheet is rejected
rather than shipped.  :func:`recolor` swaps a team colour by hue, so one sheet
serves every player.
"""

from __future__ import annotations

import base64
import colorsys
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

RGB = tuple[int, int, int]
MAGENTA: RGB = (255, 0, 255)
GRID_LINE = (40, 40, 40, 255)


@dataclass(frozen=True)
class Cell:
    key: str
    col: int
    row: int
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sheet:
    """The layout of one sprite sheet.

    *cell* is the pixel size of every cell, *origin* the pixel inside a cell
    where the sprite's anchor (its feet) lands, *scale* the pixels per logical
    unit the frames were rendered at, *pad* the empty margin around the grid.
    """

    cell: tuple[int, int]
    cols: int
    rows: int
    origin: tuple[float, float]
    scale: float
    cells: tuple[Cell, ...]
    pad: tuple[int, int]
    chroma: RGB = MAGENTA
    grid: bool = True

    @classmethod
    def layout(cls, keys: list[tuple[str, dict[str, Any]]], *, cols: int, cell: tuple[int, int],
               origin: tuple[float, float], scale: float, chroma: RGB = MAGENTA, grid: bool = True) -> Sheet:
        """Lay *keys* (key, tags) out row by row, *cols* per row, with a half-cell margin."""
        rows = math.ceil(len(keys) / cols)
        cells = tuple(Cell(key, i % cols, i // cols, tags) for i, (key, tags) in enumerate(keys))
        return cls(cell, cols, rows, origin, scale, cells, (cell[0] // 2, cell[1] // 2), chroma, grid)

    @property
    def size(self) -> tuple[int, int]:
        return (self.cols * self.cell[0] + 2 * self.pad[0], self.rows * self.cell[1] + 2 * self.pad[1])

    @property
    def drop(self) -> float:
        """How far the cell's bottom edge lies below the anchor, in logical units."""
        return (self.cell[1] - self.origin[1]) / self.scale

    @property
    def logical_size(self) -> tuple[float, float]:
        return (self.cell[0] / self.scale, self.cell[1] / self.scale)

    def box(self, cell: Cell) -> tuple[int, int, int, int]:
        cw, ch = self.cell
        x, y = self.pad[0] + cell.col * cw, self.pad[1] + cell.row * ch
        return (x, y, x + cw, y + ch)

    def find(self, **tags: Any) -> Cell:
        matches = [c for c in self.cells if all(c.tags.get(k) == v for k, v in tags.items())]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} cells match {tags}")
        return matches[0]

    def compose(self, images: dict[str, Image.Image]) -> Image.Image:
        """The sheet image: every cell's RGBA frame (already cell-sized) on the chroma ground."""
        sheet = Image.new("RGBA", self.size, (*self.chroma, 255))
        draw = ImageDraw.Draw(sheet)
        for c in self.cells:
            frame = images[c.key]
            if frame.size != self.cell:
                raise ValueError(f"{c.key}: frame is {frame.size}, the cell is {self.cell}")
            x0, y0, x1, y1 = self.box(c)
            sheet.alpha_composite(frame.convert("RGBA"), (x0, y0))
            if self.grid:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=GRID_LINE)
        return sheet

    def to_json(self) -> dict[str, Any]:
        return {"cell": list(self.cell), "cols": self.cols, "rows": self.rows, "origin": list(self.origin), "scale": self.scale,
                "pad": list(self.pad), "chroma": list(self.chroma), "grid": self.grid,
                "cells": [{"key": c.key, "col": c.col, "row": c.row, "tags": c.tags} for c in self.cells]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Sheet:
        cells = tuple(Cell(c["key"], c["col"], c["row"], dict(c.get("tags", {}))) for c in data["cells"])
        return cls(tuple(data["cell"]), data["cols"], data["rows"], tuple(data["origin"]), data["scale"], cells,
                   tuple(data["pad"]), tuple(data["chroma"]), data["grid"])

    def save(self, path: Path, images: dict[str, Image.Image]) -> None:
        """Write ``<path>.png`` (flattened, for the model) and ``<path>.json``; *path* is a
        stem that may itself contain dots."""
        self.compose(images).convert("RGB").save(file(path, "png"))
        file(path, "json").write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> Sheet:
        return cls.from_json(json.loads(file(path, "json").read_text()))


def file(stem: Path, extension: str) -> Path:
    """``<stem>.<extension>`` without :meth:`Path.with_suffix`'s idea that ``a.b`` has suffix ``.b``."""
    return stem.parent / f"{stem.name}.{extension}"


# -- Keying ---------------------------------------------------------------------------


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = np.maximum(delta, 1e-6)
    rc, gc, bc = (mx - r) / d, (mx - g) / d, (mx - b) / d
    hue = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc)) / 6 % 1
    return hue, sat, mx


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [np.dstack([v, t, p]), np.dstack([q, v, p]), np.dstack([p, v, t]), np.dstack([p, q, v]), np.dstack([t, p, v]), np.dstack([v, p, q])]
    return np.choose(i[..., None], choices)


def key_out(image: Image.Image, chroma: RGB = MAGENTA, *, hue_tol: float = 0.08, shadow: float = 0.55) -> Image.Image:
    """Alpha from hue: pixels of the key's hue turn transparent, and darker pixels of that
    hue (the model's ground shadow) become black with an alpha of their darkness times
    *shadow*.  Edge pixels lose their chroma spill by fading towards black."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255
    h, s, v = _hsv(rgb)
    kh, _, kv = _hsv(np.array(chroma, dtype=np.float32).reshape(1, 1, 3) / 255)
    kh, kv = float(kh[0, 0]), float(kv[0, 0])
    hue_dist = np.abs((h - kh + 0.5) % 1 - 0.5)
    keyness = np.clip(1 - hue_dist / hue_tol, 0, 1) * np.clip((s - 0.25) / 0.35, 0, 1)
    darkness = np.clip((kv - v) / kv, 0, 1)
    alpha = np.clip(1 - keyness + keyness * darkness * shadow, 0, 1)
    out = np.dstack([rgb * (1 - keyness)[..., None], alpha[..., None]])
    return despill(Image.fromarray((out * 255).round().astype(np.uint8), "RGBA"), chroma)


def despill(image: Image.Image, chroma: RGB = MAGENTA) -> Image.Image:
    """Take the key's cast off the edge pixels.  The model anti-aliases a figure into the key
    colour, so a partly transparent pixel carries some of it: the amount by which the key's
    strong channels exceed its weak one moves over to the weak one, and the pixel keeps its
    brightness but loses the tint.  Opaque pixels are the painting and stay as they are."""
    arr = np.asarray(image.convert("RGBA")).astype(np.float32)
    strong = [i for i in range(3) if chroma[i] > 127]
    weak = [i for i in range(3) if chroma[i] <= 127]
    if not strong or not weak:
        raise ValueError(f"a key colour needs strong and weak channels, not {chroma}")
    edge = (arr[..., 3] > 0) & (arr[..., 3] < 255)
    spill = np.clip(arr[..., strong].min(axis=-1) - arr[..., weak].max(axis=-1), 0, None) * edge
    # The strong channels come down and the weak ones go up by amounts that meet in the middle
    # and leave the channels' sum unchanged.
    for i in strong:
        arr[..., i] -= spill * len(weak) / 3
    for i in weak:
        arr[..., i] += spill * len(strong) / 3
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), "RGBA")


def recolor(image: Image.Image, source: RGB, target: RGB, *, hue_tol: float = 0.09, min_sat: float = 0.25) -> Image.Image:
    """Move pixels near *source*'s hue to *target*'s hue, saturation and brightness ratio;
    the rest of the image is untouched.  Grey metal and skin keep their colours."""
    arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255
    h, s, v = _hsv(arr[..., :3])
    sh, ss, sv = colorsys.rgb_to_hsv(*(c / 255 for c in source))
    th, ts, tv = colorsys.rgb_to_hsv(*(c / 255 for c in target))
    dist = np.abs((h - sh + 0.5) % 1 - 0.5)
    weight = np.clip(1 - dist / hue_tol, 0, 1) * (s >= min_sat)
    moved = _hsv_to_rgb((h + th - sh) % 1, np.clip(s * ts / max(ss, 1e-6), 0, 1), np.clip(v * tv / max(sv, 1e-6), 0, 1))
    arr[..., :3] = arr[..., :3] * (1 - weight)[..., None] + moved * weight[..., None]
    return Image.fromarray((arr * 255).round().astype(np.uint8), "RGBA")


# -- Cutting --------------------------------------------------------------------------


@dataclass(frozen=True)
class Registration:
    scale: float
    dx: float
    dy: float


@dataclass(frozen=True)
class CellReport:
    key: str
    coverage: float
    original_coverage: float
    drift: float  # centroid distance to the original, px
    feet_drift: float  # bottom edge offset to the original, px
    touches_edge: bool

    @property
    def ok(self) -> bool:
        ratio = self.coverage / max(self.original_coverage, 1e-6)
        return self.coverage > 0 and not self.touches_edge and 0.5 < ratio < 2.2


@dataclass(frozen=True)
class Cut:
    frames: dict[str, Image.Image]  # cell-sized RGBA, registered to the original layout
    registration: Registration
    report: tuple[CellReport, ...]

    @property
    def flagged(self) -> tuple[CellReport, ...]:
        return tuple(r for r in self.report if not r.ok)


def _clear_border(frame: Image.Image, px: int) -> Image.Image:
    a = np.asarray(frame).copy()
    a[:px, :, 3] = 0
    a[-px:, :, 3] = 0
    a[:, :px, 3] = 0
    a[:, -px:, 3] = 0
    return Image.fromarray(a, "RGBA")


def _shape(frame: Image.Image, threshold: int = 160) -> dict[str, Any] | None:
    mask = np.asarray(frame)[..., 3] > threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return {"coverage": float(mask.mean()), "cx": float(xs.mean()), "cy": float(ys.mean()),
            "bottom": int(ys.max()), "height": int(ys.max() - ys.min() + 1),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}


def _components(mask: np.ndarray) -> np.ndarray:
    """Label the 8-connected True regions of *mask*; 0 is the background."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    previous: list[tuple[int, int, int]] = []  # runs of the row above: start, end (exclusive), label
    for y in range(height):
        row = mask[y]
        if not row.any():
            previous = []
            continue
        padded = np.concatenate(([False], row, [False]))
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        current = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = 0
            for above_start, above_end, above in previous:
                if above_start <= end and above_end >= start:  # overlapping, or touching at a corner
                    if label == 0:
                        label = above
                    else:
                        root, other = find(label), find(above)
                        if root != other:
                            parent[max(root, other)] = min(root, other)
            if label == 0:
                parent.append(len(parent))
                label = len(parent) - 1
            labels[y, start:end] = label
            current.append((start, end, label))
        previous = current
    roots = np.fromiter((find(i) for i in range(len(parent))), dtype=np.int32, count=len(parent))
    return roots[labels]


def _near(mask: np.ndarray, radius: int) -> np.ndarray:
    """*mask* grown by *radius* px in every direction (a square neighbourhood)."""
    grown = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            shifted[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = mask[max(-dy, 0):height - max(dy, 0), max(-dx, 0):width - max(dx, 0)]
            grown |= shifted
    return grown


def strays(frame: Image.Image, *, keep: int = 60, gap: int = 3, band: float = 0.12, threshold: int = 8) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of what a cut brought in from beyond the figure: the sheet's own cell borders
    and guide lines, the neighbours' spill, specks.  A stray is a cluster of visible pixels (alpha
    above *threshold*, low enough to catch a faint line) separated from the figure (the largest
    cluster) by more than *gap* px that is a thin line of any length, a speck of six pixels or
    fewer, or a blob of at most *keep* px lying in the outer *band* of the cell.  Anything within
    *gap* of the figure is a piece of it, however the threshold cut it; a larger detached shape
    further in stays too (a thrown effect)."""
    alpha = np.asarray(frame.convert("RGBA"))[..., 3]
    mask = alpha > threshold
    if not mask.any():
        return []
    labels = _components(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    figure = int(sizes.argmax())
    near_figure = _near(labels == figure, gap)
    height, width = mask.shape
    margin = max(6, round(min(height, width) * band))
    boxes = []
    for label in np.flatnonzero(sizes > 0).tolist():
        if label == figure:
            continue
        cluster = labels == label
        if (cluster & near_figure).any():
            continue
        ly, lx = np.nonzero(cluster)
        y0, y1, x0, x1 = int(ly.min()), int(ly.max()), int(lx.min()), int(lx.max())
        h, w = y1 - y0 + 1, x1 - x0 + 1
        thin = (h <= 2 and w >= 8) or (w <= 2 and h >= 8)
        small = int(sizes[label]) <= keep
        outer = y0 < margin or x0 < margin or y1 >= height - margin or x1 >= width - margin
        if thin or int(sizes[label]) <= 6 or (small and outer):
            boxes.append((x0, y0, x1, y1))
    return boxes


def declutter(frame: Image.Image, **rule: int) -> Image.Image:
    """*frame* without its :func:`strays`; every other pixel is untouched."""
    boxes = strays(frame, **rule)
    if not boxes:
        return frame
    a = np.asarray(frame.convert("RGBA")).copy()
    threshold = rule.get("threshold", 8)
    labels = _components(a[..., 3] > threshold)
    for x0, y0, x1, y1 in boxes:
        window = labels[y0:y1 + 1, x0:x1 + 1]
        for label in np.unique(window[window > 0]).tolist():
            ly, lx = np.nonzero(labels == label)
            if ly.min() >= y0 and ly.max() <= y1 and lx.min() >= x0 and lx.max() <= x1:  # the stray itself, not a neighbour crossing its box
                a[..., 3][labels == label] = 0
    return Image.fromarray(a, "RGBA")


def _place(frame: Image.Image, size: tuple[int, int], dx: int, dy: int) -> Image.Image:
    placed = Image.new("RGBA", size, (0, 0, 0, 0))
    placed.alpha_composite(frame.crop((max(0, -dx), max(0, -dy), frame.size[0], frame.size[1])), (max(0, dx), max(0, dy)))
    return placed


def _line_runs(fraction: np.ndarray) -> list[float]:
    """Centres of the thin runs of consecutive indices whose *fraction* stands out: above
    0.15 and above 60% of the strongest, so a sheet one cell tall still finds its lines.
    Wide runs are figures, not lines, and are dropped."""
    smooth = np.maximum(fraction, np.maximum(np.roll(fraction, 1), np.roll(fraction, -1)))
    hits = np.nonzero(smooth > max(0.15, 0.6 * smooth.max()))[0]
    runs: list[list[int]] = []
    for i in hits:
        if runs and i <= runs[-1][1] + 2:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return [(a + b) / 2 for a, b in runs if b - a <= max(6, len(fraction) * 0.01)]


def _evenly_spaced(centres: list[float], count: int) -> list[float] | None:
    """The widest *count* of *centres* that lie on an even grid (within 15% of a step), or None."""
    if len(centres) < count:
        return None
    pairs = sorted(((i, j) for i in range(len(centres)) for j in range(i + count - 1, len(centres))),
                   key=lambda p: centres[p[0]] - centres[p[1]])
    for i, j in pairs:
        expected = np.linspace(centres[i], centres[j], count)
        step = (centres[j] - centres[i]) / (count - 1)
        chosen = [min(centres, key=lambda c: abs(c - e)) for e in expected]
        if all(abs(c - e) <= step * 0.15 for c, e in zip(chosen, expected)) and len(set(chosen)) == count:
            return chosen
    return None


def locate_grid(rendered: Image.Image, sheet: Sheet) -> tuple[float, float, float, float] | None:
    """The outer frame of the cell borders in a model's output, or None when the expected
    ``cols + 1`` vertical and ``rows + 1`` horizontal lines cannot be found.

    Models repaint the thin grey borders as faint lines a shade darker than the key colour,
    sometimes spread over two pixel columns; a border is a thin column (row) whose pixels
    darker than the key span far more of the image than any figure does, and the borders
    are evenly spaced, which tells them from a dark figure or its base."""
    rgb = np.asarray(rendered.convert("RGB")).astype(np.float32) / 255
    _, _, v = _hsv(rgb)
    dark = v < max(sheet.chroma) / 255 * 0.78
    cols = _evenly_spaced(_line_runs(dark.mean(axis=0)), sheet.cols + 1)
    rows = _evenly_spaced(_line_runs(dark.mean(axis=1)), sheet.rows + 1)
    if cols is None or rows is None:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def align(sheet: Sheet, rendered: Image.Image) -> Image.Image:
    """*rendered* resampled into the sheet's own pixel grid.  With cell borders the frame
    found by :func:`locate_grid` is mapped onto the sheet's frame, so a model that changed
    the margins or the aspect ratio still lands on the cells; otherwise the whole image
    is assumed to be the whole sheet."""
    frame = locate_grid(rendered, sheet) if sheet.grid else None
    if frame is None:
        return rendered.resize(sheet.size, Image.LANCZOS)
    x0, y0, x1, y1 = frame
    sx, sy = (sheet.cols * sheet.cell[0]) / (x1 - x0), (sheet.rows * sheet.cell[1]) / (y1 - y0)
    region = (round(x0 - sheet.pad[0] / sx), round(y0 - sheet.pad[1] / sy), round(x1 + sheet.pad[0] / sx), round(y1 + sheet.pad[1] / sy))
    canvas = Image.new("RGB", (region[2] - region[0], region[3] - region[1]), sheet.chroma)
    canvas.paste(rendered.convert("RGB"), (-region[0], -region[1]))
    return canvas.resize(sheet.size, Image.LANCZOS)


def cut(sheet: Sheet, rendered: Image.Image, original: Image.Image, *, rescale: bool = True, max_drift: float = 0.12) -> Cut:
    """Key the model's *rendered* sheet, register it to *original* and split it into frames.

    The output is first aligned to the sheet's pixel grid (see :func:`align`).
    Registration is one similarity transform for the whole sheet: the median of the
    per-cell height ratios (when *rescale*) and the median centre and feet offsets.
    A cell whose centroid still drifts more than *max_drift* of the cell width, that
    touches its border or whose coverage is far from the original's is flagged.
    """
    rendered = align(sheet, rendered)
    if original.size != sheet.size:
        raise ValueError(f"original sheet is {original.size}, expected {sheet.size}")
    keyed = key_out(rendered.convert("RGB"), sheet.chroma)
    margin = np.asarray(keyed)[..., 3].copy()
    margin[sheet.pad[1]:sheet.size[1] - sheet.pad[1], sheet.pad[0]:sheet.size[0] - sheet.pad[0]] = 0
    if sheet.pad != (0, 0) and margin.mean() > 255 * 0.05:
        raise ValueError("the model did not keep the chroma background: the margin is not the key colour")
    reference = key_out(original.convert("RGB"), sheet.chroma, shadow=0.0)
    border = 4 if sheet.grid else 2
    cw, ch = sheet.cell
    ai = {c.key: _shape(_clear_border(keyed.crop(sheet.box(c)), border)) for c in sheet.cells}
    og = {c.key: _shape(_clear_border(reference.crop(sheet.box(c)), border)) for c in sheet.cells}
    pairs = [(ai[k], og[k]) for k in ai if ai[k] and og[k]]
    if not pairs:
        raise ValueError("no cell has a figure both in the original and in the model's output")
    scale = float(np.median([o["height"] / a["height"] for a, o in pairs])) if rescale else 1.0
    dx = float(np.median([o["cx"] - a["cx"] * scale for a, o in pairs]))
    dy = float(np.median([o["bottom"] - a["bottom"] * scale for a, o in pairs]))
    registration = Registration(scale, dx, dy)
    frames: dict[str, Image.Image] = {}
    report = []
    for c in sheet.cells:
        frame = _clear_border(keyed.crop(sheet.box(c)), border)
        if scale != 1.0:
            frame = frame.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.LANCZOS)
        frame = declutter(_place(frame, sheet.cell, round(dx), round(dy)))
        frames[c.key] = frame
        shape, ref = _shape(frame), og[c.key]
        if shape is None or ref is None:
            report.append(CellReport(c.key, shape["coverage"] if shape else 0.0, ref["coverage"] if ref else 0.0, math.inf, math.inf, False))
            continue
        x0, y0, x1, y1 = shape["bbox"]
        drift = math.hypot(shape["cx"] - ref["cx"], shape["cy"] - ref["cy"])
        report.append(CellReport(c.key, shape["coverage"], ref["coverage"], drift, shape["bottom"] - ref["bottom"],
                                 x0 <= 0 or y0 <= 0 or x1 >= cw - 1 or y1 >= ch - 1 or drift > cw * max_drift))
    return Cut(frames, registration, tuple(report))


def save_frames(cutting: Cut, sheet: Sheet, path: Path) -> None:
    """Write the registered frames as one RGBA sheet without margins or grid, plus the layout."""
    plain = Sheet(sheet.cell, sheet.cols, sheet.rows, sheet.origin, sheet.scale, sheet.cells, (0, 0), sheet.chroma, False)
    image = Image.new("RGBA", plain.size, (0, 0, 0, 0))
    for c in plain.cells:
        image.alpha_composite(cutting.frames[c.key], plain.box(c)[:2])
    image.save(file(path, "png"))
    file(path, "json").write_text(json.dumps(plain.to_json(), indent=1))


def load_frames(path: Path) -> tuple[Sheet, dict[str, Image.Image]]:
    sheet = Sheet.load(path)
    image = Image.open(file(path, "png")).convert("RGBA")
    return sheet, {c.key: image.crop(sheet.box(c)) for c in sheet.cells}


# -- Providers ------------------------------------------------------------------------


def render_with_codex(input_png: Path, prompt: str, output_png: Path, *, timeout: float = 900) -> str:
    """Repaint *input_png* with Codex's built-in image tool (the user's ChatGPT plan; no API key).

    Runs ``codex exec`` non-interactively in the output's directory; the agent is asked
    to edit the attached image and copy the result to *output_png*.  Returns the log."""
    workdir = output_png.parent
    workdir.mkdir(parents=True, exist_ok=True)
    task = (f"{prompt}\n\nUse the built-in image_gen tool in edit mode on the attached image (also at {input_png}) "
            f"with the specification above, then copy the generated PNG to {output_png} (it is saved under "
            f"$CODEX_HOME/generated_images). Do not modify anything else; finish by printing the path you copied from.")
    result = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "-C", str(workdir), "-i", str(input_png), "-"],
        input=task, capture_output=True, text=True, timeout=timeout,
    )
    log = result.stdout + result.stderr
    if not output_png.exists():
        raise RuntimeError(f"codex did not write {output_png}:\n{log[-2000:]}")
    return log


def openrouter_api_key() -> str:
    """``OPENROUTER_API_KEY`` from the environment, else from the secrets index the stack keeps."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    secrets = Path("~/secrets/llm-providers.md").expanduser()
    for line in secrets.read_text().splitlines() if secrets.exists() else ():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"OPENROUTER_API_KEY is neither in the environment nor in {secrets}")


def render_with_openrouter(input_png: Path, prompt: str, output_png: Path, *, model: str, api_key: str,
                           aspect_ratio: str = "3:2", image_size: str = "1K", timeout: float = 600) -> dict[str, Any]:
    """Repaint *input_png* with an OpenRouter image model; returns the usage record."""
    encoded = base64.b64encode(input_png.read_bytes()).decode()
    body = {"model": model, "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}]}],
            "image_config": {"aspect_ratio": aspect_ratio, "image_size": image_size}}
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    started = time.time()
    response = json.load(urllib.request.urlopen(request, timeout=timeout))
    message = response["choices"][0]["message"]
    images = message.get("images") or []
    if not images:
        raise RuntimeError(f"{model} returned no image: {json.dumps(response)[:1000]}")
    match = re.match(r"data:image/\w+;base64,(.*)", images[0]["image_url"]["url"], re.S)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_png.write_bytes(base64.b64decode(match.group(1)))
    return {"elapsed": time.time() - started, "usage": response.get("usage"), "text": message.get("content")}


# -- Previews -------------------------------------------------------------------------


def strip(frames: dict[str, Image.Image], keys: list[str], *, scale: float = 1.0, background: RGB = (60, 90, 50)) -> Image.Image:
    """Cell-sized *frames* for *keys* side by side, on a flat ground."""
    images = [frames[k] for k in keys]
    w, h = images[0].size
    w, h = round(w * scale), round(h * scale)
    out = Image.new("RGBA", (w * len(images), h), (*background, 255))
    for i, image in enumerate(images):
        out.alpha_composite(image.resize((w, h), Image.LANCZOS) if scale != 1.0 else image, (i * w, 0))
    return out


def gif(frames: list[Image.Image], path: Path, *, ms: int = 200) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, disposal=2)


# -- Judging --------------------------------------------------------------------------


def review_image(sheet: Sheet, originals: dict[str, Image.Image], painted: dict[str, Image.Image], *,
                 rows: list[int] | None = None, cols: list[int] | None = None, row_names: list[str] | None = None,
                 background: RGB = (60, 90, 50)) -> Image.Image:
    """The sheet (or its *rows* and *cols*) laid out for a reviewer: for every row of cells the
    stand-ins above the painted frames, with the row's name and column numbers written in, so
    a judge can name a cell as "row 3, column 5".  Keep a review image around two rows by four
    columns: a judge sees a big image scaled down and misses a duplicated hilt at 100 px."""
    cw, ch = sheet.cell
    label_h, gutter = 22, 8
    rows = list(range(sheet.rows)) if rows is None else rows
    cols = list(range(sheet.cols)) if cols is None else cols
    width = len(cols) * cw + 40
    height = len(rows) * (2 * ch + label_h + gutter)
    image = Image.new("RGBA", (width, height), (*background, 255))
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows):
        top = index * (2 * ch + label_h + gutter)
        name = row_names[row] if row_names else f"row {row}"
        draw.text((4, top + 4), f"row {row}: {name}   (stand-ins above, painted below)", fill=(255, 255, 255, 255))
        for cell in sheet.cells:
            if cell.row != row or cell.col not in cols:
                continue
            x = 40 + cols.index(cell.col) * cw
            image.alpha_composite(originals[cell.key], (x, top + label_h))
            image.alpha_composite(painted[cell.key], (x, top + label_h + ch))
            draw.text((x + 4, top + label_h + 4), f"col {cell.col}", fill=(255, 255, 255, 255))
        draw.line((0, top + label_h + 2 * ch + gutter // 2, width, top + label_h + 2 * ch + gutter // 2), fill=(30, 40, 30, 255), width=2)
    return image


JUDGE_INSTRUCTIONS = """You are checking a repainted sprite sheet against its stand-ins. The image shows, for each row, the
low-poly stand-in frames above and the painted frames below, labelled "row N: name" and "col N".

Work cell by cell, painted row only, and count what the painted figure carries: weapons (count each hilt, blade,
bow, staff, club or lance separately), shields, heads, mounts. Then compare with the stand-in directly above and
with the expected inventory. A cell is wrong if:
- any count differs from the inventory (two hilts or two blades where one sword is expected is the common error:
  look for a second gold crossguard near the shield or the hip);
- it faces a different direction than the stand-in;
- its pose disagrees with the stand-in directly above it (judge the pose against the stand-in, not against the row's
  name: the name says what the row is for, the stand-in shows what the painter was asked to keep);
- a limb, the weapon or the head is missing or merged into the body;
- it is a different subject (another unit type, mount or race).
Style, proportion and detail may differ freely; the painter is allowed to make the figure prettier.

Reply with one JSON object and nothing else:
{"cells": [{"row": 0, "col": 0, "weapons": 1, "shields": 1, "ok": true, "issue": ""}, ...]}
List every cell of the rows shown. Keep issues short and concrete, like "two hilts" or "faces left, stand-in faces right"."""


FACING_NAMES = ("right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right")


def judge_with_codex(review_png: Path, subject: str, sheet: Sheet, *, rows: list[int] | None = None, cols: list[int] | None = None,
                     inventory: str = "", instructions: str = JUDGE_INSTRUCTIONS, timeout: float = 900) -> list[dict[str, Any]]:
    """Ask Codex (which can look at images) to check *review_png*, which shows *rows* and *cols*
    of the sheet (all by default), against the subject's *inventory*; returns the per-cell
    verdicts with the judge's counts.  *instructions* say what to count and what makes a cell
    wrong (the default is written for figures; a game judging buildings or tokens passes its own,
    ending in the same JSON contract)."""
    rows = list(range(sheet.rows)) if rows is None else rows
    cols = list(range(sheet.cols)) if cols is None else cols
    facings = ", ".join(f"col {c} = {FACING_NAMES[c]}" for c in cols) if sheet.cols == 8 else "as labelled"
    prompt = (f"{instructions}\n\nThe subject is {subject}. Expected inventory in every cell: {inventory or 'as the stand-in shows'}. "
              f"The image shows rows {', '.join(map(str, rows))} and columns {', '.join(map(str, cols))} of the sheet; "
              f"the columns are facings: {facings}. Use the row and column numbers written in the image.")
    result = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(review_png.parent), "-i", str(review_png), "-"],
        input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    text = result.stdout
    start, end = text.rfind("{\"cells\""), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"the judge returned no verdicts:\n{text[-1500:]}")
    try:
        verdicts = json.loads(text[start:end + 1])["cells"]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"the judge's verdicts are not JSON ({error}):\n{text[start:end + 1][:1500]}") from error
    expected = len(rows) * len(cols)
    if len(verdicts) != expected:
        raise RuntimeError(f"the judge listed {len(verdicts)} cells, the image shows {expected}")
    return verdicts
