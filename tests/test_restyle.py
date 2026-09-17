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


def test_despill_takes_the_key_tint_off_edge_pixels_and_leaves_the_painting_alone():
    image = Image.new("RGBA", (3, 1))
    image.putdata([(240, 180, 235, 128), (240, 180, 235, 255), (80, 200, 90, 128)])  # a pink-tinted edge, opaque pink paint, a green edge
    out = list(restyle.despill(image).getdata())
    edge, paint, green = out
    assert abs(edge[0] - edge[1]) < 8 and abs(edge[2] - edge[1]) < 4, edge  # neutral now (red keeps its own few points over blue)
    assert abs(sum(edge[:3]) - sum((240, 180, 235))) < 4, edge  # as bright as before
    assert paint == (240, 180, 235, 255) and green == (80, 200, 90, 128)


def test_key_out_leaves_no_tint_on_the_edge_it_keys():
    image = Image.new("RGB", (1, 1))
    image.putdata([(230, 110, 230)])  # a figure pixel half blended into the key
    r, g, b, a = restyle.key_out(image).getpixel((0, 0))
    assert 0 < a < 255 and abs(r - g) < 4 and abs(b - g) < 4, (r, g, b, a)


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


def test_review_image_pairs_every_stand_in_with_its_painted_frame():
    """A reviewer sees, for each requested row and column, the stand-in directly above the
    painted frame at the same size, and nothing from other rows or columns."""
    sheet, images = make_sheet(rows=2, cols=3)
    painted = {k: restyle.recolor(v, (70, 130, 255), (225, 70, 60)) for k, v in images.items()}
    review = restyle.review_image(sheet, images, painted, rows=[1], cols=[0, 2], row_names=["a", "b"])
    cw, ch = sheet.cell
    assert review.size[0] == 2 * cw + 40 and review.size[1] == 2 * ch + 22 + 8
    top = np.asarray(review.crop((40, 22, 40 + cw, 22 + ch)))
    bottom = np.asarray(review.crop((40, 22 + ch, 40 + cw, 22 + 2 * ch)))
    body_top, body_bottom = top[70, 32], bottom[70, 32]
    assert body_top[2] > body_top[0] + 60, body_top  # the stand-in's blue body
    assert body_bottom[0] > body_bottom[2] + 60, body_bottom  # the painted red body under it
    second = np.asarray(review.crop((40 + cw, 22 + ch, 40 + 2 * cw, 22 + 2 * ch)))
    assert second[70, 32][0] > second[70, 32][2] + 60  # column 2 sits right after column 0


def test_strays_are_guide_lines_specks_and_edge_spill_but_not_what_the_figure_holds_or_throws():
    """A guide line above the figure (even under a spear tip), a neighbour's fragment at the edge, a speck:
    gone.  A spear tip the alpha threshold broke off its shaft, an arrow flying in the interior: kept."""
    from PIL import ImageDraw
    from sagaforge.restyle import declutter, strays

    frame = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((30, 30, 50, 70), fill=(90, 140, 60, 255))  # the figure
    draw.line((40, 30, 40, 6), fill=(120, 90, 60, 255), width=1)  # its spear shaft
    draw.rectangle((39, 2, 41, 4), fill=(200, 200, 210, 255))  # its tip, broken off by a 1 px gap
    draw.line((5, 8, 30, 8), fill=(40, 40, 40, 200), width=1)  # a guide line above the figure, 8 px into the cell
    draw.rectangle((76, 40, 79, 44), fill=(200, 200, 200, 255))  # the neighbour's lance tip at the right edge
    draw.point((60, 60), fill=(0, 0, 0, 255))  # a speck
    draw.rectangle((14, 40, 20, 43), fill=(200, 120, 40, 255))  # an arrow in flight, interior, detached
    draw.line((0, 78, 79, 78), fill=(30, 30, 30, 24))  # a faint guide line the whole width of the cell, under the shadow
    draw.point((51, 50), fill=(90, 140, 60, 20))  # the figure's own faint edge pixel
    draw.point((24, 20), fill=(255, 0, 255, 20))  # faint keying residue in the interior
    draw.ellipse((28, 71, 52, 75), fill=(0, 0, 0, 30))  # a soft shadow blob just below the feet
    draw.rectangle((0, 20, 5, 60), fill=(20, 20, 20, 16))  # the ghost of the sheet's border, hugging the left edge
    assert strays(frame) == [(5, 8, 30, 8), (0, 20, 5, 60), (76, 40, 79, 44), (60, 60, 60, 60), (0, 78, 79, 78)]
    cleaned = declutter(frame)
    before, after = np.asarray(frame)[..., 3], np.asarray(cleaned)[..., 3]
    gone = (before > 0) & (after == 0)
    assert gone.sum() == 26 + 20 + 1 + 80 + 6 * 41 and after[8, 5:31].max() == 0 and after[40:45, 76:80].max() == 0 and after[60, 60] == 0
    assert after[20:61, 0:6].max() == 0
    assert after[78].max() == 0
    assert (after[30:71, 30:51] == 255).all() and after[6:31, 40].min() == 255 and after[2:5, 39:42].min() == 255
    assert after[40:44, 14:21].min() == 255 and after[50, 51] == 20 and after[20, 24] == 20 and after[73, 40] == 30
    assert strays(cleaned) == []
    assert declutter(cleaned) is cleaned
