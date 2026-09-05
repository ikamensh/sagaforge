# Compose audio with samples

`saga2d.synth` extracts the pure NumPy synthesis functions used in Tribes
and offered by Warband's committed sound module. Games supply compositions;
Saga2D supplies sample math and WAV encoding. Shardbound can ship generated
assets without doing synthesis or creating a second audio manager at launch.

```python
from pathlib import Path
from saga2d.synth import level, mix, noise, pan, thump, tone, write_wav

impact = level(mix(
    thump(180, 60, .2),
    noise(.08, 400, 4000, seed=3) * .3,
    (.05, pan(tone("D4", .3), -.2)),
), .65)
write_wav(Path("assets/sounds/impact.wav"), impact)

# A Game using asset_path="assets" plays the ordinary file:
game.audio.play_sound("impact")
```

Samples use 44,100 Hz. Mono clips have shape `(frames,)`; stereo clips have
shape `(frames, 2)`. `mix` sums clips, expanding mono to both channels when
needed. A `(start_seconds, clip)` places a layer later in the composition.
It does not limit the amplitude; `level(clip, peak)` normalizes a non-silent
clip to a finite peak in 0–1 before export. Silence cannot be normalized and
raises `ValueError`, as do nonfinite samples. Exporting a silent clip directly
is valid.

`tone` accepts a frequency or a note such as `"F#4"`. `SOFT`, `GLASS`, `DARK`,
`BELL`, `PAD` and `BRASS` are harmonic recipes. `envelope` shapes attack and
decay, `noise` produces seeded band-limited bursts, `thump` glides a sine from
one frequency to another, and `pan` places mono audio from −1 (left) to +1
(right). Use positive lengths, frequencies and decay times, nonnegative
attack times and start offsets, and noise bounds below the Nyquist frequency.
`hz` and `seconds` are available for custom compositions.

`write_wav` produces 16-bit PCM mono/stereo. It rejects empty, nonfinite,
out-of-range or unsupported-channel samples before opening the destination.
It is an asset-generation tool, not a durable save-file transaction. Asset
build/caching policy, names, event routing, music transitions and playback
remain the game's responsibility; no bank wrapper or runtime cache is added.

Run `uv run python tools/demo_synth.py` to compose, decode and play a stereo
asset with native Pyglet's silent driver. Add `--audible` for a brief cue at
25% master volume. Tests verify PCM accuracy, timing and playback through
the public audio interface; all 18 existing Tribes WAVs remain byte-identical
after replacing its duplicate helper code. These checks establish asset and
playback correctness, not artistic sound quality.
