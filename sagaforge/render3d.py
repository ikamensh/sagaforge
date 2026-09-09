"""A tiny software renderer for low-poly props.

Low-poly meshes (boxes, pyramids, cones, cylinders, spheres, gable
roofs) are projected by a :class:`Projection` (a fixed camera), lit by
one directional light with flat shading, sorted back-to-front and
rasterised with Pillow at a multiple of the target size, then
downsampled for anti-aliasing.  Games pre-render their tiles, buildings
and units with it at the display's pixel density and register the images
with the asset manager.

Face colours are RGB or RGBA.  Translucent faces (a ground shadow) are
rasterised first and opaque faces overwrite them, so translucency only
shows where nothing solid covers it; Pillow does not blend polygons.

Model space: the tile is the unit square ``[-0.5, 0.5]²`` in x/y with
z up.  Screen y grows downwards, like everywhere else in saga2d.  Which
way the camera looks is the projection's business: :meth:`Projection.dimetric`
sits over the ``(+x, +y)`` corner (Tribes), :meth:`Projection.front`
looks from ``+y`` towards ``-y`` with square tile footprints (Warband).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageDraw

Vec3 = tuple[float, float, float]
RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
Color = RGB | RGBA


def _normalized(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / length, v[1] / length, v[2] / length)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


#: Direction towards the light: from the upper left-front, so top faces are
#: brightest, left (``+y``) faces medium and right (``+x``) faces darkest.
LIGHT: Vec3 = _normalized((-0.3, 0.4, 1.0))
AMBIENT = 0.55


@dataclass(frozen=True)
class Face:
    points: tuple[Vec3, ...]
    color: Color

    @property
    def opaque(self) -> bool:
        return len(self.color) == 3 or self.color[3] == 255


Mesh = list[Face]


@dataclass(frozen=True)
class Projection:
    """A fixed camera: where it looks from and how model units land on screen.

    *right* and *depth* are unit vectors in the ground plane: *right* maps
    to screen-right, *depth* to screen-down (towards the camera).  The
    camera is *elevation* radians above the ground.  Ground distance along
    *depth* is foreshortened by ``sin(elevation)`` unless *depth_scale*
    overrides it — a top-down game with square tiles keeps footprints square
    and still gets lit vertical faces.  *scale* is logical units per model
    unit along *right*.
    """

    scale: float
    right: tuple[float, float]
    depth: tuple[float, float]
    elevation: float
    depth_scale: float | None = None

    @classmethod
    def dimetric(cls, tile_w: float) -> Projection:
        """Over the ``(+x, +y)`` corner at ``asin(1/2)``: the unit tile's top
        face is a diamond *tile_w* wide and exactly half as tall."""
        s = 1 / math.sqrt(2)
        return cls(tile_w * s, (s, -s), (s, s), math.asin(0.5))

    @classmethod
    def front(cls, tile_w: float, elevation_deg: float = 55.0) -> Projection:
        """From ``+y`` looking towards ``-y``, *elevation_deg* above the ground,
        with the unit tile a *tile_w* square on screen."""
        return cls(tile_w, (1.0, 0.0), (0.0, 1.0), math.radians(elevation_deg), depth_scale=1.0)

    @property
    def view(self) -> Vec3:
        """Unit vector from the scene towards the camera."""
        c, s = math.cos(self.elevation), math.sin(self.elevation)
        return (self.depth[0] * c, self.depth[1] * c, s)

    @property
    def tile_w(self) -> float:
        """Screen width of the unit tile."""
        rx, ry = self.right
        return self.scale * (abs(rx) + abs(ry))

    @property
    def tile_h(self) -> float:
        """Screen height of the unit tile's top face."""
        dx, dy = self.depth
        return self.scale * self._depth_scale * (abs(dx) + abs(dy))

    @property
    def z_scale(self) -> float:
        """Logical units per model unit of height."""
        return self.scale * math.cos(self.elevation)

    @property
    def _depth_scale(self) -> float:
        return math.sin(self.elevation) if self.depth_scale is None else self.depth_scale

    def project(self, p: Vec3) -> tuple[float, float]:
        x, y, z = p
        sx = (x * self.right[0] + y * self.right[1]) * self.scale
        sy = (x * self.depth[0] + y * self.depth[1]) * self.scale * self._depth_scale - z * self.z_scale
        return (sx, sy)


# -- Mesh helpers ------------------------------------------------------------


def _newell_normal(points: tuple[Vec3, ...]) -> Vec3:
    nx = ny = nz = 0.0
    n = len(points)
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return _normalized((nx, ny, nz))


def _centroid(points: tuple[Vec3, ...]) -> Vec3:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, sum(p[2] for p in points) / n)


def _oriented(faces: list[tuple[list[Vec3], Color]], inside: Vec3) -> Mesh:
    """Wind every face so its normal points away from *inside* (a point inside a convex solid)."""
    mesh: Mesh = []
    for points, color in faces:
        pts = tuple(points)
        normal = _newell_normal(pts)
        c = _centroid(pts)
        outward = (c[0] - inside[0], c[1] - inside[1], c[2] - inside[2])
        if _dot(normal, outward) < 0:
            pts = pts[::-1]
        mesh.append(Face(pts, color))
    return mesh


def rotate_z(mesh: Mesh, degrees: float, about: tuple[float, float] = (0.0, 0.0)) -> Mesh:
    """Rotate about a vertical axis through *about* (counter-clockwise seen from above)."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    ax, ay = about

    def rot(p: Vec3) -> Vec3:
        x, y = p[0] - ax, p[1] - ay
        return (ax + x * c - y * s, ay + x * s + y * c, p[2])

    return [Face(tuple(rot(p) for p in f.points), f.color) for f in mesh]


def scale(mesh: Mesh, factor: float, about: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    """Scale a mesh uniformly about *about*."""
    ax, ay, az = about
    return [Face(tuple((ax + (x - ax) * factor, ay + (y - ay) * factor, az + (z - az) * factor) for x, y, z in f.points), f.color) for f in mesh]


def _ring(cx: float, cy: float, z: float, radius: float, sides: int, rotation: float) -> list[Vec3]:
    return [
        (cx + radius * math.cos(rotation + 2 * math.pi * i / sides), cy + radius * math.sin(rotation + 2 * math.pi * i / sides), z)
        for i in range(sides)
    ]


# -- Primitives ------------------------------------------------------------------


def box(center: Vec3, size: Vec3, color: RGB) -> Mesh:
    cx, cy, cz = center
    hw, hd, hh = size[0] / 2, size[1] / 2, size[2] / 2
    x0, x1, y0, y1, z0, z1 = cx - hw, cx + hw, cy - hd, cy + hd, cz - hh, cz + hh
    faces = [
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], color),  # top
        ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], color),  # bottom
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], color),  # +x
        ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], color),  # -x
        ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], color),  # +y
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], color),  # -y
    ]
    return _oriented(faces, center)


def pyramid(base_center: Vec3, size: tuple[float, float], height: float, color: RGB, *, apex_shift: tuple[float, float] = (0.0, 0.0)) -> Mesh:
    """Square-based pyramid; a negative *height* points it downwards."""
    cx, cy, z0 = base_center
    hw, hd = size[0] / 2, size[1] / 2
    corners = [(cx - hw, cy - hd, z0), (cx + hw, cy - hd, z0), (cx + hw, cy + hd, z0), (cx - hw, cy + hd, z0)]
    apex = (cx + apex_shift[0], cy + apex_shift[1], z0 + height)
    faces = [(corners, color)] + [([corners[i], corners[(i + 1) % 4], apex], color) for i in range(4)]
    return _oriented(faces, (cx, cy, z0 + height / 4))


def cone(base_center: Vec3, radius: float, height: float, color: RGB, *, sides: int = 8, rotation: float = 0.0) -> Mesh:
    cx, cy, z0 = base_center
    ring = _ring(cx, cy, z0, radius, sides, rotation)
    apex = (cx, cy, z0 + height)
    faces = [(ring, color)] + [([ring[i], ring[(i + 1) % sides], apex], color) for i in range(sides)]
    return _oriented(faces, (cx, cy, z0 + height / 4))


def cylinder(base_center: Vec3, radius: float, height: float, color: RGB, *, sides: int = 10, rotation: float = 0.0) -> Mesh:
    cx, cy, z0 = base_center
    bottom = _ring(cx, cy, z0, radius, sides, rotation)
    top = _ring(cx, cy, z0 + height, radius, sides, rotation)
    faces = [(bottom, color), (top, color)] + [
        ([bottom[i], bottom[(i + 1) % sides], top[(i + 1) % sides], top[i]], color) for i in range(sides)
    ]
    return _oriented(faces, (cx, cy, z0 + height / 2))


def sphere(center: Vec3, radius: float, color: RGB, *, rings: int = 5, sides: int = 8) -> Mesh:
    """Low-poly UV sphere."""
    cx, cy, cz = center
    bands = []
    for i in range(1, rings):
        phi = math.pi * i / rings
        bands.append(_ring(cx, cy, cz + radius * math.cos(phi), radius * math.sin(phi), sides, math.pi / sides))
    top, bottom = (cx, cy, cz + radius), (cx, cy, cz - radius)
    faces = [([top, bands[0][(i + 1) % sides], bands[0][i]], color) for i in range(sides)]
    for a, b in zip(bands, bands[1:]):
        faces.extend(([a[i], a[(i + 1) % sides], b[(i + 1) % sides], b[i]], color) for i in range(sides))
    faces.extend(([bottom, bands[-1][i], bands[-1][(i + 1) % sides]], color) for i in range(sides))
    return _oriented(faces, center)


def gable_roof(base_center: Vec3, size: tuple[float, float], height: float, color: RGB) -> Mesh:
    """Triangular prism with the ridge running along x."""
    cx, cy, z0 = base_center
    hw, hd = size[0] / 2, size[1] / 2
    x0, x1, y0, y1 = cx - hw, cx + hw, cy - hd, cy + hd
    r0, r1 = (x0, cy, z0 + height), (x1, cy, z0 + height)
    faces = [
        ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], color),  # underside
        ([(x0, y0, z0), (x1, y0, z0), r1, r0], color),  # -y slope
        ([(x0, y1, z0), (x1, y1, z0), r1, r0], color),  # +y slope
        ([(x0, y0, z0), (x0, y1, z0), r0], color),  # -x gable
        ([(x1, y0, z0), (x1, y1, z0), r1], color),  # +x gable
    ]
    return _oriented(faces, (cx, cy, z0 + height / 3))


def flat(points: list[tuple[float, float]], z: float, color: Color) -> Mesh:
    """A single horizontal face facing up (points are x/y, listed in any order)."""
    pts = tuple((x, y, z) for x, y in points)
    if _newell_normal(pts)[2] < 0:
        pts = pts[::-1]
    return [Face(pts, color)]


def facing(points: list[Vec3], color: RGB, view: Vec3) -> Mesh:
    """A single face wound towards *view* (see :attr:`Projection.view`) so it is never back-face culled."""
    pts = tuple(points)
    if _dot(_newell_normal(pts), view) < 0:
        pts = pts[::-1]
    return [Face(pts, color)]


def ribbon(path: list[Vec3], across: Vec3, width: float, color: RGB, view: Vec3) -> Mesh:
    """Quads of *width* along *path*, spread in the *across* direction (a unit vector), facing *view*."""
    hx, hy, hz = (c * width / 2 for c in across)
    mesh: Mesh = []
    for (x0, y0, z0), (x1, y1, z1) in zip(path, path[1:]):
        mesh += facing([(x0 - hx, y0 - hy, z0 - hz), (x1 - hx, y1 - hy, z1 - hz), (x1 + hx, y1 + hy, z1 + hz), (x0 + hx, y0 + hy, z0 + hz)], color, view)
    return mesh


# -- Rendering -----------------------------------------------------------------


def shade(color: Color, normal: Vec3, *, light: Vec3 = LIGHT, ambient: float = AMBIENT) -> RGBA:
    """Flat shading normalised so an upward face shows *color* exactly; alpha passes through."""
    diffuse = max(0.0, _dot(normal, light)) / light[2]
    intensity = ambient + (1 - ambient) * diffuse
    r, g, b = (min(255, round(c * intensity)) for c in color[:3])
    return (r, g, b, color[3] if len(color) == 4 else 255)


def bounds(mesh: Mesh, projection: Projection) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of the projected mesh in logical units around the model origin."""
    xs, ys = [], []
    for face in mesh:
        for p in face.points:
            sx, sy = projection.project(p)
            xs.append(sx)
            ys.append(sy)
    return min(xs), min(ys), max(xs), max(ys)


ToPixel = Callable[[Vec3], tuple[float, float]]
Decorate = Callable[[ImageDraw.ImageDraw, ToPixel, float], None]


def render(
    mesh: Mesh,
    projection: Projection,
    *,
    scale: float,
    canvas: tuple[float, float],
    origin: tuple[float, float],
    supersample: int = 3,
    decorate: Decorate | None = None,
) -> Image.Image:
    """Rasterise *mesh* into an RGBA image.

    *canvas* is the image size and *origin* the pixel where the model
    origin lands, both in logical units; the result is ``scale`` times
    that in pixels.  *decorate* may draw 2-D details on the supersampled
    canvas (it receives the draw, a model→pixel function and the pixels
    per logical unit) before downsampling.
    """
    ss = scale * supersample
    width, height = round(canvas[0] * ss), round(canvas[1] * ss)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ox, oy = origin

    def to_px(p: Vec3) -> tuple[float, float]:
        sx, sy = projection.project(p)
        return ((ox + sx) * ss, (oy + sy) * ss)

    view = projection.view
    visible = []
    for face in mesh:
        normal = _newell_normal(face.points)
        if _dot(normal, view) <= 1e-9:
            continue
        depth = sum(_dot(p, view) for p in face.points) / len(face.points)
        visible.append((face.opaque, depth, face, normal))
    visible.sort(key=lambda item: item[:2])  # translucent faces first, then back to front
    for _opaque, _depth, face, normal in visible:
        draw.polygon([to_px(p) for p in face.points], fill=shade(face.color, normal))
    if decorate is not None:
        decorate(draw, to_px, ss)
    return _spread_edge_colors(_downsample(image, (round(canvas[0] * scale), round(canvas[1] * scale))))


def _downsample(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Box-filter *image* to *size* in premultiplied alpha.

    Averaging straight RGBA drags every edge pixel towards the transparent
    black around it, leaving a dark fringe on each sprite; averaging colour
    weighted by coverage keeps the edge the colour of the surface.  Box
    filtering has no ringing (Lanczos overshoots and leaves a bright
    hairline along tile boundaries).
    """
    import numpy as np

    arr = np.asarray(image).astype(np.float32)
    coverage = arr[..., 3] / 255
    planes = [arr[..., i] * coverage for i in range(3)] + [arr[..., 3]]
    small = np.stack([np.asarray(Image.fromarray(plane, "F").resize(size, Image.Resampling.BOX)) for plane in planes], axis=-1)
    alpha = small[..., 3:4]
    rgb = np.divide(small[..., :3] * 255, alpha, out=np.zeros_like(small[..., :3]), where=alpha > 0)
    return Image.fromarray(np.concatenate([rgb, alpha], axis=-1).round().clip(0, 255).astype(np.uint8), "RGBA")


def _spread_edge_colors(image: Image.Image, passes: int = 3) -> Image.Image:
    """Copy each edge pixel's colour into the transparent pixels around it.

    Transparent pixels are black; when the GPU samples an edge with bilinear
    filtering it blends that black in, drawing a dark hairline around every
    sprite and along every tile boundary.  Filling the margin with the
    neighbouring colour (alpha stays zero) makes the interpolation neutral.
    """
    import numpy as np

    arr = np.asarray(image).astype(np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3]
    known = alpha > 0
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for _ in range(passes):
        total = np.zeros_like(rgb)
        count = np.zeros(alpha.shape, dtype=np.float32)
        for dy, dx in offsets:
            mask = np.roll(known, (dy, dx), axis=(0, 1))
            total += np.roll(rgb, (dy, dx), axis=(0, 1)) * mask[..., None]
            count += mask
        fill = ~known & (count > 0)
        rgb[fill] = total[fill] / count[fill][:, None]
        known |= fill
    return Image.fromarray(np.concatenate([rgb, alpha[..., None]], axis=-1).round().astype(np.uint8), "RGBA")
