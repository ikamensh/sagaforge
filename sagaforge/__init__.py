"""Sagaforge — procedural asset generation for Saga2D games.

Pure NumPy and Pillow, no window or audio device:

* ``synth``    — sample-based sound and music synthesis written to WAV.
* ``render3d`` — a software low-poly renderer that turns meshes into sprite images.

Games call these at build time or lazily at runtime and hand the results to
``saga2d`` as ordinary image and sound assets.
"""

__version__ = "0.1.0"
