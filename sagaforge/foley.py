"""Generated sound pieces: prompt a text-to-audio model, cut, check, cache and record provenance.

A game describes each piece it wants (name, prompt, length, seed, shape) and calls :func:`build`.
Pieces already in the destination with the same spec are kept; the rest are generated in one batch,
cut to their sound, band-limited, checked, written as mono 16-bit WAVs and listed in a manifest with
prompt, seed, hash and length, so any piece can be regenerated or challenged.  The generator is
Stable Audio 3 through Stability's MLX runtime (:class:`StableAudioMLX`) or any callable with the
same contract, so tests and other models plug in.  ``docs/foley.md`` is the guide.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image

from sagaforge.synth import SAMPLE_RATE, highpass, level, lowpass, mix, write_wav

#: How a piece is cut out of the model's clip, and the band it is limited to afterwards.
SHAPES = ("voice", "impact", "collapse")
BANDS = {"voice": (90.0, 9000.0), "impact": (30.0, 9000.0), "collapse": (30.0, 9000.0)}
SILENCE = 0.05          # a raw clip peaking below this is a failed generation
CLICK_RATIO = 0.6       # a cut whose 9 kHz low-passed peak is below this share of its peak is a click, not a sound
CUT_VERSION = 2         # part of every spec hash: a change to how clips are cut regenerates every piece


@dataclass(frozen=True)
class Piece:
    """One sound to generate: the file stem, the whole prompt, and how to cut the clip.

    *shape* ``voice`` keeps everything between the first and last loud moment (a cry, a word);
    ``impact`` keeps the first event only (a hit, a drop) and stops when the clip goes quiet;
    ``collapse`` is an impact allowed to rumble on, up to 2.5 s (masonry coming down, debris).
    """

    name: str
    prompt: str
    seconds: float = 2.0
    seed: int = 0
    steps: int = 8
    shape: str = "voice"
    peak: float = 0.8

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"{self.name}: shape must be one of {SHAPES}, not {self.shape!r}")
        if self.seconds <= 0 or self.steps < 1 or not 0 < self.peak <= 1:
            raise ValueError(f"{self.name}: seconds and steps must be positive and peak within (0, 1]")

    def spec(self, generator: str) -> str:
        """A hash of everything that decides the sound; a different hash means regenerate."""
        return hashlib.sha256(json.dumps({**asdict(self), "generator": generator, "cut": CUT_VERSION}, sort_keys=True).encode()).hexdigest()[:16]


#: Takes the pieces to generate and returns each name's raw clip, mono ``(frames,)`` or stereo
#: ``(frames, 2)`` at :data:`SAMPLE_RATE`; ``name`` labels it in the manifest.
class Generator:
    name: str

    def __call__(self, pieces: list[Piece]) -> dict[str, np.ndarray]:
        raise NotImplementedError


class StableAudioMLX(Generator):
    """Stable Audio 3 through the ``optimized/mlx`` runtime of Stability's ``stable-audio-3`` checkout.

    *runtime* is that directory (default: ``$STABLE_AUDIO_MLX``), installed with its ``install.sh``
    and the bundle for *model* downloaded.  Generation runs in the runtime's own virtualenv through
    :mod:`sagaforge.foley_worker`, models loaded once per batch.
    """

    MODELS = {"small-sfx": ("sm-sfx", "same-s"), "medium": ("medium", "same-l")}

    def __init__(self, model: str = "medium", runtime: Path | str | None = None) -> None:
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {tuple(self.MODELS)}, not {model!r}")
        location = runtime if runtime is not None else os.environ.get("STABLE_AUDIO_MLX")
        if not location:
            raise EnvironmentError("Set STABLE_AUDIO_MLX to the optimized/mlx directory of a stable-audio-3 checkout, or pass runtime=")
        self.runtime = Path(location).expanduser().resolve()
        self.python = self.runtime / ".venv" / "bin" / "python"
        if not (self.runtime / "scripts" / "sa3_mlx.py").exists() or not self.python.exists():
            raise FileNotFoundError(f"{self.runtime} is not an installed stable-audio-3 MLX runtime (scripts/sa3_mlx.py and .venv expected)")
        self.model = model
        self.name = f"stable-audio-3 {model} (MLX)"

    def __call__(self, pieces: list[Piece]) -> dict[str, np.ndarray]:
        dit, decoder = self.MODELS[self.model]
        with tempfile.TemporaryDirectory(prefix="foley-") as scratch:
            jobs = Path(scratch) / "jobs.json"
            jobs.write_text(json.dumps({
                "runtime": str(self.runtime), "dit": dit, "decoder": decoder, "out": scratch,
                "jobs": [{"name": p.name, "prompt": p.prompt, "seconds": p.seconds, "seed": p.seed, "steps": p.steps} for p in pieces],
            }))
            worker = Path(__file__).with_name("foley_worker.py")
            run = subprocess.run([str(self.python), str(worker), str(jobs)], capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"foley worker failed:\n{run.stderr[-3000:]}")
            print(run.stdout, end="", file=sys.stderr)
            return {p.name: read_wav(Path(scratch) / f"{p.name}.wav") for p in pieces}


def read_wav(path: Path | str) -> np.ndarray:
    """A 16-bit WAV as floats in −1..1: ``(frames,)`` when mono, else ``(frames, channels)``."""
    with wave.open(str(path), "rb") as src:
        channels, width, rate, frames = src.getparams()[:4]
        if rate != SAMPLE_RATE or width != 2:
            raise ValueError(f"{path}: expected 16-bit {SAMPLE_RATE} Hz, got {width * 8}-bit {rate} Hz")
        data = np.frombuffer(src.readframes(frames), dtype="<i2").astype(float) / 32767
    return data if channels == 1 else data.reshape(-1, channels)


def mono(clip: np.ndarray) -> np.ndarray:
    return clip if clip.ndim == 1 else clip.mean(axis=1)


class Rejected(ValueError):
    """A generation that is not a usable sound; change the seed or the prompt."""


def cut_voice(clip: np.ndarray, *, floor_db: float = -42.0, tail: float = 0.06) -> np.ndarray:
    """Everything between the first and last moment louder than *floor_db* below the peak, plus *tail*."""
    env = np.abs(clip)
    loud = np.flatnonzero(env > env.max() * 10 ** (floor_db / 20))
    start = max(0, loud[0] - int(0.01 * SAMPLE_RATE))
    end = min(len(clip), loud[-1] + int(tail * SAMPLE_RATE))
    return clip[start:end]


def cut_impact(clip: np.ndarray, *, quiet: float = 0.035, hold: float = 0.25, shortest: float = 0.25, longest: float = 1.0) -> np.ndarray:
    """From the onset until the 30 ms envelope has stayed below *quiet* of its own peak for *hold* seconds.

    Models keep rattling after the hit they were asked for; this keeps the hit and its tail.  Quiet is
    judged against the smoothed envelope, not the loudest sample: one overshooting transient must not
    make the rumble that follows it count as silence."""
    env = np.abs(clip)
    onset = max(0, np.flatnonzero(env > env.max() * 0.03)[0] - int(0.008 * SAMPLE_RATE))
    window = int(0.03 * SAMPLE_RATE)
    smooth = np.convolve(env[onset:], np.ones(window) / window, mode="same")
    below = smooth < smooth.max() * quiet
    end, run = len(smooth), 0
    for i in range(int(0.08 * SAMPLE_RATE), len(smooth)):
        run = run + 1 if below[i] else 0
        if run >= int(hold * SAMPLE_RATE):
            end = i - run + int(0.04 * SAMPLE_RATE)
            break
    end = min(max(end, int(shortest * SAMPLE_RATE)), int(longest * SAMPLE_RATE), len(smooth))
    return clip[onset:onset + end]


CUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "voice": cut_voice,
    "impact": cut_impact,
    "collapse": partial(cut_impact, hold=0.4, shortest=0.5, longest=2.5),
}


def finish(clip: np.ndarray, piece: Piece) -> np.ndarray:
    """Cut *clip* to the piece's shape, reject what is not a sound, band-limit, fade the ends and level it."""
    clip = mono(clip)
    if np.abs(clip).max() < SILENCE:
        raise Rejected("silent")
    cut = CUTS[piece.shape](clip)
    low, high = BANDS[piece.shape]
    smooth = lowpass(cut, high)
    if np.abs(smooth).max() < CLICK_RATIO * np.abs(cut).max():
        raise Rejected(f"mostly above {high / 1000:g} kHz, a click rather than a sound")
    out = highpass(smooth, low)
    fade = min(len(out) // 2, int(0.02 * SAMPLE_RATE))
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return level(out, piece.peak)


class BuildError(RuntimeError):
    """Some pieces were rejected; the others were written."""


def build(pieces: Iterable[Piece], dest: Path | str, generator: Generator, *, license: str) -> dict:
    """Bring *dest* to exactly these pieces, generating what is missing or whose spec changed.

    Returns the manifest (also written as ``dest/manifest.json``).  Pieces the generator produced but
    that failed a check are left out and reported together in :class:`BuildError` after everything
    else is written, so a long batch is not lost to one bad seed."""
    pieces = list(pieces)
    names = [p.name for p in pieces]
    if len(set(names)) != len(names):
        raise ValueError("piece names must be unique")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    entries: dict[str, dict] = manifest.get("pieces", {})
    for stale in set(entries) - set(names):
        (dest / f"{stale}.wav").unlink(missing_ok=True)
        del entries[stale]
    wanted = [p for p in pieces if entries.get(p.name, {}).get("spec") != p.spec(generator.name) or not (dest / f"{p.name}.wav").exists()]
    rejected: dict[str, str] = {}
    if wanted:
        raw = generator(wanted)
        for piece in wanted:
            file = dest / f"{piece.name}.wav"
            try:
                clip = finish(raw[piece.name], piece)
            except Rejected as why:
                rejected[piece.name] = str(why)
                file.unlink(missing_ok=True)
                entries.pop(piece.name, None)
                continue
            write_wav(file, clip)
            entries[piece.name] = {**asdict(piece), "spec": piece.spec(generator.name),
                                   "length": round(len(clip) / SAMPLE_RATE, 3), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest = {"generator": generator.name, "license": license, "encoding": f"16-bit PCM WAV, {SAMPLE_RATE} Hz, mono",
                "pieces": {name: entries[name] for name in names if name in entries}}
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    if rejected:
        raise BuildError("rejected " + ", ".join(f"{name} ({why})" for name, why in rejected.items()) + "; change their seed or prompt and build again")
    return manifest


def sampler(dest: Path | str, names: Iterable[str], path: Path | str, *, gap: float = 0.4) -> list[tuple[float, str]]:
    """Write every named piece back to back with *gap* seconds between, for one listen; returns (start, name) labels."""
    dest = Path(dest)
    layers, labels, cursor = [], [], 0.25
    for name in names:
        clip = read_wav(dest / f"{name}.wav")
        layers.append((cursor, clip))
        labels.append((round(cursor, 3), name))
        cursor += len(clip) / SAMPLE_RATE + gap
    write_wav(Path(path), mix(*layers, (cursor, np.zeros(SAMPLE_RATE // 4))))
    return labels


def spectrogram(clip: np.ndarray, *, height: int = 160, hop: int = 256, window: int = 1024) -> Image.Image:
    """A log-frequency spectrogram to look at a piece: bright is loud, time runs left to right."""
    clip = mono(clip)
    frames = max(1, (len(clip) - window) // hop)
    taper = np.hanning(window)
    spectrum = np.stack([np.abs(np.fft.rfft(clip[i * hop:i * hop + window] * taper)) for i in range(frames)], axis=1)
    db = 20 * np.log10(spectrum + 1e-6)
    db = np.clip((db - db.max() + 70) / 70, 0, 1)
    rows = np.geomspace(1, db.shape[0] - 1, height).astype(int)
    return Image.fromarray((db[rows][::-1] * 255).astype(np.uint8))
