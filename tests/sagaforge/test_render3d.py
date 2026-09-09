"""The low-poly renderer: projections, culling and the shading of a box."""

import math

import pytest

from sagaforge import render3d as r3


def _opaque_rows(image):
    alpha = image.getchannel("A")
    w, h = image.size
    return [y for y in range(h) if any(alpha.getpixel((x, y)) == 255 for x in range(w))]


def test_dimetric_projection_makes_the_tile_top_a_two_to_one_diamond() -> None:
    p = r3.Projection.dimetric(128)
    assert p.tile_w == pytest.approx(128) and p.tile_h == pytest.approx(64)
    corners = [p.project(c) for c in ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0))]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    assert max(xs) - min(xs) == pytest.approx(128) and max(ys) - min(ys) == pytest.approx(64)
    assert p.view[2] == pytest.approx(0.5) and p.view[0] == p.view[1] > 0


def test_front_projection_keeps_the_footprint_square_and_shows_the_near_face() -> None:
    p = r3.Projection.front(32, elevation_deg=55)
    corners = [p.project(c) for c in ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0))]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    assert max(xs) - min(xs) == 32 and max(ys) - min(ys) == 32
    assert p.project((0, 0, 1))[1] == -32 * math.cos(math.radians(55))  # height rises on screen
    assert p.view[0] == 0 and p.view[1] > 0 and p.view[2] > 0  # the camera stands at +y, above the ground
    assert p.project((0, 1, 0))[1] > p.project((0, -1, 0))[1]  # +y is nearer the camera: lower on screen


def test_a_box_seen_from_the_front_shows_a_bright_top_over_a_darker_front() -> None:
    p = r3.Projection.front(32)
    image = r3.render(r3.box((0, 0, 0.5), (1, 1, 1), (200, 200, 200)), p, scale=1.0, canvas=(48, 64), origin=(24, 44))
    rows = _opaque_rows(image)
    assert rows, "nothing rendered"
    pixels = image.load()
    x = 24
    top_row, front_row = rows[0] + 3, rows[-1] - 3
    assert pixels[x, top_row][3] == 255 and pixels[x, front_row][3] == 255
    assert sum(pixels[x, top_row][:3]) > sum(pixels[x, front_row][:3])
    assert pixels[x, top_row][:3] == (200, 200, 200)  # an upward face shows its colour exactly


def test_facing_winds_towards_the_given_view_so_a_flat_sign_is_never_culled() -> None:
    p = r3.Projection.front(32)
    quad = [(-0.2, 0, 0.1), (0.2, 0, 0.1), (0.2, 0, 0.5), (-0.2, 0, 0.5)]
    forwards = r3.facing(quad, (255, 0, 0), p.view)
    backwards = r3.facing(quad[::-1], (255, 0, 0), p.view)
    assert forwards[0].points == backwards[0].points
    image = r3.render(forwards, p, scale=1.0, canvas=(32, 32), origin=(16, 24))
    assert _opaque_rows(image)


def test_rotating_a_mesh_about_z_turns_its_footprint() -> None:
    mesh = r3.box((0.3, 0, 0.1), (0.2, 0.2, 0.2), (255, 255, 255))
    turned = r3.rotate_z(mesh, 90)
    centre = [sum(pt[i] for f in turned for pt in f.points) / sum(len(f.points) for f in turned) for i in range(3)]
    assert abs(centre[0]) < 1e-9 and abs(centre[1] - 0.3) < 1e-9 and abs(centre[2] - 0.1) < 1e-9
