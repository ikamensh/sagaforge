# Sagaforge

Procedural asset generation for [Saga2D](../saga2d) games, in pure NumPy and
Pillow:

* `sagaforge.synth` — sample-based sound and music synthesis written to WAV:
  oscillators, plucks, sustained tones, filters, formants, reverb, loop
  helpers and a cache. See [docs/synth.md](docs/synth.md).
* `sagaforge.render3d` — a software low-poly renderer with a configurable
  camera that turns small meshes into crisp sprites at any pixel density.

[Tribes](../tribes), [Warband](../warband) and [Shardbound](../shardbound)
build their art, effects, voices and soundtracks on these primitives; no
recorded samples or downloaded assets are involved.

The development environment uses the published `saga2d==0.2.0` release for
the audio demo; it does not require a sibling engine checkout.

For an existing environment that used the editable engine, run
`uv sync --locked --extra dev --reinstall-package saga2d` once. A plain sync
can retain an editable install of the same version. This also restores the
release after local engine testing.

```bash
uv sync --extra dev
uv run pytest -q
uv run python tools/demo_synth.py --audible
```
