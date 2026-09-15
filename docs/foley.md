# Generated sound pieces with `sagaforge.foley`

How the Saga games get sounds a synthesiser cannot make, a dying orc, a sword hitting stone,
out of a text-to-audio model, and keep them reproducible. The library is `sagaforge/foley.py`;
a game drives it from a tool of its own (Warband: `tools/deaths.py`).

## The idea

A game does not ask a model for "a death". It describes **pieces**, short sounds with one
prompt each, and composes them itself: a cry, then the weapon dropping, the body landing,
the gear settling, with gaps it controls. The model is good at texture and bad at timing,
so timing stays in code. Each piece is a `Piece(name, prompt, seconds, seed, steps, shape,
peak)`; `build()` generates what is missing, cuts each clip to its sound, band-limits it,
checks it, writes a mono WAV and a manifest with prompt, seed, hash and length. The
committed WAVs are the asset; the manifest and the tool's spec are how to remake them.

```python
from sagaforge.foley import Piece, StableAudioMLX, build, sampler

STYLE = "medieval fantasy battlefield, close, dry, no music, no reverb"
pieces = [
    Piece("orc_cry_0", f"a large man with a deep gravelly voice cries out in pain and dies, {STYLE}", seconds=2.0, seed=5000, peak=0.72),
    Piece("orc_weapon_0", "a heavy iron axe dropped onto packed dirt, one thud with a metallic clink, one short sound then silence, no voice",
          seconds=1.6, seed=4010, shape="impact"),
]
build(pieces, "warband/assets/deaths", StableAudioMLX("medium"), license="Stability AI Community License")
sampler("warband/assets/deaths", [p.name for p in pieces], "/tmp/deaths.wav")   # one file to listen to
```

## Setup: the model runs in its own checkout

Stable Audio 3 is not a dependency of sagaforge (it needs MLX and its own Python). Install
Stability's runtime once, next to the stack:

```bash
git clone https://github.com/Stability-AI/stable-audio-3 ~/stable-audio-3
cd ~/stable-audio-3/optimized/mlx && ./install.sh -y --download medium     # or sm-sfx; about 6 GB for both
export STABLE_AUDIO_MLX=~/stable-audio-3/optimized/mlx                     # or pass runtime= to StableAudioMLX
```

The bundles come from the ungated `stabilityai/stable-audio-3-optimized` repository, so no
Hugging Face login is needed. `StableAudioMLX` runs `sagaforge/foley_worker.py` under the
runtime's virtualenv with the whole batch, loading each model once. Apple Silicon only.

Numbers from the pilot on an M4 MacBook Air, 8 steps:

| Model | Memory | One 2 s clip | Notes |
|---|---|---|---|
| `small-sfx` | 1.6 GB | 0.8 s | Diverged at 1.6 s clips and at 16 steps; fine at 2 s and 8 steps |
| `medium` | 3.3 GB | 2.3 s | Stable at every setting tried; cleaner harmonics, more varied takes |

Large is API-only; Medium is the largest downloadable weight in this family. Both licences
are the Stability AI Community License: outputs are owned and may be sold, training data is
licensed. Say so in the manifest's `license` and in the game's provenance note.

## Shapes: how a clip is cut

- `voice`: keep everything from the first to the last moment louder than −42 dB below the
  peak, plus 60 ms; band 90 Hz–9 kHz. For cries, words, animal calls.
- `impact`: keep the first event: from the onset until the envelope has stayed under 3.5 % of
  the peak for 250 ms, between 0.25 and 1 s; band 30 Hz–9 kHz. For hits, drops, clatters.
  Models keep rattling for the whole requested length, so this cut is what makes a stage.
- `collapse`: an impact allowed to rumble on: the same cut with 400 ms of quiet to stop and 0.5–2.5 s
  kept; band 30 Hz–9 kHz. For masonry coming down, timber crashing, debris settling.

Two failures are rejected, after everything else in the batch is written: a **silent** clip
(peak under 0.05) and a **click**, a clip whose energy sits mostly above 9 kHz (the low-passed
peak under 60 % of the raw peak). `build()` raises `BuildError` naming them; change their seed
or prompt and build again. Everything is then faded 20 ms at both ends and levelled to `peak`.

## Prompting, learned the hard way

- **Seeds do not give variants.** The same prompt with four seeds gave four takes with
  spectral similarity 0.95–0.98: same length, same contour. Write a different sentence per
  take ("a pained shout cut off", "a choked groan, falling", "a last gasp, breath knocked
  out"); that dropped similarity to 0.6–0.8 and the takes differ audibly.
- **Say what is dying, not what it is.** "Orc death roar, monstrous" gives a beast; "a large
  man with a deep gravelly voice cries out in pain and dies" gives a dying warrior.
- **One event per piece.** End impact prompts with "one short sound then silence, no voice";
  ask for the fall as stages, not as "a body collapses with armour and a sword".
- **A shared style suffix** ("medieval fantasy battlefield, close, dry, no music, no
  reverb") keeps a game's pieces in one room. Keep it in the game's tool, not in the library.
- **Guidance above 1 did not help** and pushed peaks past 1.0; audio-to-audio from a synth
  stand-in works (`--init-audio` in the runtime's CLI) but is not wired into `build()` yet.
- **Listen once.** `sampler()` writes every piece back to back; `spectrogram()` gives a
  picture to compare takes before listening. Neither replaces ears.

## Composing in the game

Warband's `deaths.py` is the reference: it reads the pieces, places the weapon `FALL_START`
seconds from the end of the cry, the body and the settle at fixed gaps, rotates the body take
against the cry take so no two cues share a whole fall, and levels the mix. Its tests hold
the fall after the cry (the sub-70 Hz peak lands late), the level, the length and the
absence of clicks. Keep that split: pieces from `foley`, timeline and mix in the game.
