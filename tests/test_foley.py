"""sagaforge.foley: pieces come out cut to their sound, checked, cached by spec and listed with provenance.

The model is simulated: a generator that returns a burst in silence for a voice, two hits for an
impact, silence and a pure 14 kHz tone for the failures a real model produces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sagaforge import foley
from sagaforge.synth import SAMPLE_RATE, noise, tone

LICENSE = "test license"


def burst(start: float, length: float, seconds: float, *, seed: int = 1, low: float = 200, high: float = 3000, tau: float | None = None) -> np.ndarray:
    """A band-limited noise burst *length* seconds long at *start* inside *seconds* of silence, stereo."""
    clip = np.zeros(int(seconds * SAMPLE_RATE))
    sound = noise(length, low, high, attack=0.02, tau=tau or length, seed=seed) * 0.6
    clip[int(start * SAMPLE_RATE):int(start * SAMPLE_RATE) + len(sound)] = sound
    return np.stack([clip, clip], axis=1)


class Synthetic(foley.Generator):
    name = "synthetic"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, pieces: list[foley.Piece]) -> dict[str, np.ndarray]:
        self.calls.append([p.name for p in pieces])
        clips = {}
        for p in pieces:
            if p.name.startswith("cry"):
                clips[p.name] = burst(0.4, 0.5, p.seconds, seed=p.seed)
            elif p.name.startswith("hit"):  # a hit, then a second one 0.6 s later the impact cut must not keep
                clips[p.name] = burst(0.1, 0.12, p.seconds, seed=p.seed) + burst(0.82, 0.12, p.seconds, seed=p.seed + 1)
            elif p.name.startswith("rumble"):  # three hits with 0.35 s pauses, then a straggler after a second of silence
                clips[p.name] = sum(burst(0.1 + 0.45 * k, 0.15, p.seconds, seed=p.seed + k, tau=0.04) for k in range(3)) + burst(2.3, 0.15, p.seconds, seed=p.seed + 9, tau=0.04)
            elif p.name == "quiet":
                clips[p.name] = np.zeros(int(p.seconds * SAMPLE_RATE))
            elif p.name == "click":
                clips[p.name] = tone(14000.0, p.seconds, attack=0.001, tau=p.seconds) * 0.5
        return clips


def length(path: Path) -> float:
    return len(foley.read_wav(path)) / SAMPLE_RATE


def test_pieces_are_cut_to_their_sound_levelled_and_listed_with_provenance(tmp_path: Path) -> None:
    pieces = [foley.Piece("cry_0", "a short cry", seconds=2.0, seed=3, shape="voice", peak=0.72),
              foley.Piece("hit_0", "a sword dropped on stone", seconds=1.6, seed=4, shape="impact")]
    manifest = foley.build(pieces, tmp_path, Synthetic(), license=LICENSE)
    assert 0.45 <= length(tmp_path / "cry_0.wav") <= 0.7          # the burst and its tail, not the 0.4 s of silence before it
    assert 0.25 <= length(tmp_path / "hit_0.wav") <= 0.6           # the first hit only; the second starts at 0.72 s after the onset
    for piece in pieces:
        clip = foley.read_wav(tmp_path / f"{piece.name}.wav")
        assert clip.ndim == 1 and abs(np.abs(clip).max() - piece.peak) < 0.01 and abs(clip[0]) < 0.01 and abs(clip[-1]) < 0.01
        entry = manifest["pieces"][piece.name]
        assert entry["prompt"] == piece.prompt and entry["seed"] == piece.seed and entry["shape"] == piece.shape
        assert entry["sha256"] == hashlib.sha256((tmp_path / f"{piece.name}.wav").read_bytes()).hexdigest()
        assert entry["length"] == pytest.approx(length(tmp_path / f"{piece.name}.wav"), abs=0.001)
    assert manifest["generator"] == "synthetic" and manifest["license"] == LICENSE
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_a_collapse_keeps_a_long_event_that_an_impact_would_cut_short(tmp_path: Path) -> None:
    """The same rumble cut both ways: the impact shape stops at the first quiet, the collapse rides through the pauses."""
    pieces = [foley.Piece("rumble_impact", "a wall comes down", seconds=3.0, shape="impact"),
              foley.Piece("rumble_collapse", "a wall comes down", seconds=3.0, shape="collapse")]
    foley.build(pieces, tmp_path, Synthetic(), license=LICENSE)
    assert length(tmp_path / "rumble_impact.wav") <= 0.5             # the first hit; a 0.35 s pause ends an impact
    assert 0.9 <= length(tmp_path / "rumble_collapse.wav") <= 1.4    # all three hits, not the straggler after a second of quiet


def test_unchanged_pieces_are_kept_changed_ones_regenerated_and_removed_ones_deleted(tmp_path: Path) -> None:
    generator = Synthetic()
    pieces = [foley.Piece("cry_0", "a cry", seed=1), foley.Piece("cry_1", "a cry", seed=2), foley.Piece("hit_0", "a hit", shape="impact")]
    foley.build(pieces, tmp_path, generator, license=LICENSE)
    foley.build(pieces, tmp_path, generator, license=LICENSE)
    assert generator.calls == [["cry_0", "cry_1", "hit_0"]]  # nothing to do the second time
    before = (tmp_path / "cry_0.wav").read_bytes()
    reseeded = [foley.Piece("cry_0", "a cry", seed=1), foley.Piece("cry_1", "a cry", seed=9)]
    manifest = foley.build(reseeded, tmp_path, generator, license=LICENSE)
    assert generator.calls[-1] == ["cry_1"]
    assert (tmp_path / "cry_0.wav").read_bytes() == before
    assert not (tmp_path / "hit_0.wav").exists() and set(manifest["pieces"]) == {"cry_0", "cry_1"}
    (tmp_path / "cry_0.wav").unlink()
    foley.build(reseeded, tmp_path, generator, license=LICENSE)
    assert generator.calls[-1] == ["cry_0"]  # a missing file is regenerated even with its spec on record


def test_failed_generations_are_reported_together_after_the_rest_is_written(tmp_path: Path) -> None:
    pieces = [foley.Piece("cry_0", "a cry"), foley.Piece("quiet", "nothing"), foley.Piece("click", "a spike", shape="impact")]
    with pytest.raises(foley.BuildError) as failure:
        foley.build(pieces, tmp_path, Synthetic(), license=LICENSE)
    assert "quiet (silent)" in str(failure.value) and "click (mostly above 9 kHz" in str(failure.value)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["pieces"]) == {"cry_0"} and (tmp_path / "cry_0.wav").exists()
    assert not (tmp_path / "quiet.wav").exists() and not (tmp_path / "click.wav").exists()


def test_the_sampler_plays_every_piece_in_order_with_gaps(tmp_path: Path) -> None:
    pieces = [foley.Piece(f"cry_{i}", "a cry", seed=i) for i in range(3)]
    foley.build(pieces, tmp_path, Synthetic(), license=LICENSE)
    labels = foley.sampler(tmp_path, [p.name for p in pieces], tmp_path / "sampler.wav", gap=0.5)
    starts = [start for start, _ in labels]
    assert [name for _, name in labels] == ["cry_0", "cry_1", "cry_2"] and starts == sorted(starts)
    total = sum(length(tmp_path / f"{p.name}.wav") for p in pieces)
    assert length(tmp_path / "sampler.wav") == pytest.approx(0.25 + total + 3 * 0.5 + 0.25, abs=0.01)
    assert foley.spectrogram(foley.read_wav(tmp_path / "sampler.wav")).size[1] == 160


def test_piece_specs_reject_nonsense() -> None:
    with pytest.raises(ValueError):
        foley.Piece("x", "p", shape="loop")
    with pytest.raises(ValueError):
        foley.Piece("x", "p", seconds=0)
    with pytest.raises(ValueError):
        foley.build([foley.Piece("x", "p"), foley.Piece("x", "q")], "/tmp/unused", Synthetic(), license=LICENSE)
