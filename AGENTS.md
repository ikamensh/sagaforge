# Sagaforge — procedural assets

Pure NumPy and Pillow, no window, no audio device. Games generate their art
and sound with it, at build time or lazily at runtime, and hand the results to
`saga2d` as ordinary image and sound assets. Part of the Saga stack (`~/saga/`,
see `../AGENTS.md`); `saga2d` is only a dev dependency here (the demo plays
through it).

## Commands

```bash
uv sync --extra dev
uv run pytest -q
uv run python tools/demo_synth.py --audible   # compose a WAV and play it through saga2d
```

## Layout

- `sagaforge/synth.py` — sample-based synthesis: oscillators, `pluck`,
  `sustained`, filters, `formant`, reverb, `soft_clip`, `loop_add`, WAV
  writing and a cache. Warband's orchestra and Tribes' and Shardbound's cues
  are built on it. See `docs/synth.md`.
- `sagaforge/render3d.py` — a Pillow software renderer for low-poly meshes
  with a configurable camera (`Projection`); the games pre-render blocks,
  props and units to sprites with it.

## Rules

- Keep it a library of primitives: instruments, meshes and palettes belong to
  the game that needs them. Something is promoted here when a second game
  needs it in the same form.
- Generation must be deterministic for a given seed; Shardbound's asset
  provenance hashes `sagaforge/synth.py`, so a change here means rebuilding
  its audio manifest (`cd ../shardbound && uv run python tools/build_audio.py`).
- Clear exceptions over silent fallbacks. Delete rather than deprecate.
- Look at (and listen to) what you generate before calling it done:
  `Image.save` a rendered sprite, write a WAV, open them.
- Commit each working increment.
