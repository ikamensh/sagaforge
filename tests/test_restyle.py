"""sagaforge.restyle: sheets survive a model that repaints, shifts and reframes them.

The model is simulated: the original sheet is repainted (figures recoloured), moved,
scaled and given new margins, and ``cut`` must hand back frames that sit where the
originals sat.  That is the whole contract a game relies on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from sagaforge import restyle


def figure(cell: tuple[int, int], height: int, color: tuple[int, int, int], feet: tuple[float, float]) -> Image.Image:
    """A stick figure of *height* px standing on *feet*, with a grey head so recolouring can be told apart."""
    image = Image.new("RGBA", cell, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x, y = feet
    draw.rectangle((x - height / 6, y - height, x + height / 6, y), fill=(*color, 255))
    draw.ellipse((x - height / 5, y - height * 1.3, x + height / 5, y - height * 0.95), fill=(160, 160, 160, 255))
    return image


def make_sheet(rows: int = 2, cols: int = 3) -> tuple[restyle.Sheet, dict[str, Image.Image]]:
    cell, origin = (64, 96), (32.0, 80.0)
    keys = [(f"unit.{r}.{c}", {"row": r, "col": c}) for r in range(rows) for c in range(cols)]
    sheet = restyle.Sheet.layout(keys, cols=cols, cell=cell, origin=origin, scale=2.0)
    images = {key: figure(cell, 40 + 6 * tags["row"], (70, 130, 255), origin) for key, tags in keys}
    return sheet, images


def test_layout_round_trips_through_json_and_places_cells_after_the_margin():
    sheet, _ = make_sheet()
    again = restyle.Sheet.from_json(sheet.to_json())
    assert again == sheet
    assert sheet.box(sheet.find(row=1, col=2)) == (32 + 2 * 64, 48 + 96, 32 + 3 * 64, 48 + 2 * 96)
    assert sheet.drop == (96 - 80) / 2.0


def test_key_out_keeps_figures_and_turns_a_darker_key_shadow_translucent():
    image = Image.new("RGB", (3, 1))
    image.putdata([(255, 0, 255), (70, 130, 255), (128, 0, 128)])
    alpha = np.asarray(restyle.key_out(image))[0, :, 3]
    assert alpha[0] == 0
    assert alpha[1] == 255
    assert 0 < alpha[2] < 255


def test_cut_registers_a_repainted_shifted_and_reframed_sheet():
    sheet, images = make_sheet()
    original = sheet.compose(images).convert("RGB")
    # The "model": repaints the figures redder, draws them 10% larger, and returns the sheet with
    # wider margins on a canvas of a different aspect ratio (as gpt-image-2 and Gemini both do).
    repainted = {k: figure(sheet.cell, int((40 + 6 * sheet.find(key=k).tags["row"] if False else 0) or 0), (0, 0, 0), (0, 0)) for k in []}
    bigger = {}
    for cell in sheet.cells:
        height = round((40 + 6 * cell.tags["row"]) * 1.1)
        bigger[cell.key] = figure(sheet.cell, height, (200, 60, 60), (sheet.origin[0] + 3, sheet.origin[1] + 2))
    rendered = sheet.compose(bigger).convert("RGB")
    w, h = rendered.size
    canvas = Image.new("RGB", (w + 120, h + 40), sheet.chroma)
    canvas.paste(rendered, (90, 25))
    result = restyle.cut(sheet, canvas, original)
    assert not result.flagged, result.flagged
    assert abs(result.registration.scale - 1 / 1.1) < 0.05
    for cell in sheet.cells:
        alpha = np.asarray(result.frames[cell.key])[..., 3]
        ys, xs = np.nonzero(alpha > 160)
        assert abs(ys.max() - sheet.origin[1]) <= 2, cell.key  # feet back on the anchor line
        assert abs(xs.mean() - sheet.origin[0]) <= 3, cell.key


def test_cut_rejects_an_output_without_the_key_background():
    sheet, images = make_sheet()
    original = sheet.compose(images).convert("RGB")
    try:
        restyle.cut(sheet, Image.new("RGB", sheet.size, (255, 255, 255)), original)
    except ValueError as error:
        assert "chroma" in str(error)
    else:
        raise AssertionError("a white sheet must be rejected")


def test_recolor_moves_the_team_hue_and_leaves_grey_alone():
    _, images = make_sheet(rows=1, cols=1)
    frame = next(iter(images.values()))
    red = np.asarray(restyle.recolor(frame, (70, 130, 255), (225, 70, 60)))
    body = red[70, 32]
    head = red[40, 32]
    assert body[0] > body[2] + 60, body  # blue tunic became red
    assert tuple(head[:3]) == (160, 160, 160), head


def test_save_and_load_frames_round_trip(tmp_path):
    sheet, images = make_sheet()
    result = restyle.cut(sheet, sheet.compose(images).convert("RGB"), sheet.compose(images).convert("RGB"))
    restyle.save_frames(result, sheet, tmp_path / "unit")
    loaded_sheet, frames = restyle.load_frames(tmp_path / "unit")
    assert loaded_sheet.origin == sheet.origin and loaded_sheet.pad == (0, 0)
    for key, frame in frames.items():
        assert frame.size == sheet.cell
        assert np.array_equal(np.asarray(frame)[..., 3] > 0, np.asarray(result.frames[key])[..., 3] > 0), key
