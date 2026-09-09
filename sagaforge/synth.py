"""Compose original audio as NumPy samples and write ordinary WAV assets.

Pure synthesis helpers shared by Tribes, Warband and Shardbound. Sound names,
compositions, cache/build policy and playback belong to callers. See
``tools/demo_synth.py`` for a composition playable through ``game.audio``.

Short percussive material comes from ``tone``, ``noise`` and ``thump``; held
voices from ``sustained`` (vibrato, unison detune, an ADSR shape) and strings
from ``pluck``.  ``lowpass``, ``highpass`` and ``formant`` colour a clip,
``reverb`` puts it in a room, ``soft_clip`` tames a dense mix and
``loop_add`` places a clip on a seamless loop, wrapping its tail to the start.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 44_100
_TABLE = 4096  # samples in one period of a held voice's waveform

_NOTE_INDEX = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

# Partials as (harmonic multiple, relative amplitude): the timbre of a tone.
SOFT = ((1, 1.0), (2, 0.3), (3, 0.1))                                # flute-like
GLASS = ((1, 1.0), (2, 0.5), (3, 0.28), (4, 0.14), (5, 0.07))        # electric-piano pluck
DARK = ((1, 1.0), (2, 0.15))                                         # muted, almost a sine
BELL = ((1, 1.0), (2, 0.45), (3, 0.25), (4.16, 0.12), (5.43, 0.05))  # a little inharmonic shimmer
PAD = ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25))                     # saw-like
BRASS = ((1, 1.0), (2, 0.7), (3, 0.55), (4, 0.4), (5, 0.3), (6, 0.2))  # bright, for horns


def hz(note: str) -> float:
    """``"A4"`` → 440.0; sharps as ``"F#5"``."""
    midi = 12 * (int(note[-1]) + 1) + _NOTE_INDEX[note[:-1]]
    return 440.0 * 2 ** ((midi - 69) / 12)


def seconds(count: float) -> np.ndarray:
    """Sample times for *count* seconds."""
    return np.arange(int(round(count * SAMPLE_RATE))) / SAMPLE_RATE


def envelope(length: float, attack: float, tau: float) -> np.ndarray:
    """Raised-cosine attack, exponential decay with time constant *tau*,
    and a 5 ms fade at the very end so no clip ends mid-cycle."""
    t = seconds(length)
    env = np.ones_like(t)
    rising = t < attack
    env[rising] = 0.5 - 0.5 * np.cos(np.pi * t[rising] / attack)
    env[~rising] = np.exp(-(t[~rising] - attack) / tau)
    tail = min(len(t), int(0.005 * SAMPLE_RATE))
    env[-tail:] *= np.linspace(1.0, 0.0, tail)
    return env


def tone(note: str | float, length: float, *, attack: float = 0.005, tau: float = 0.1, partials=SOFT) -> np.ndarray:
    """A decaying note (a name or a frequency); higher partials die faster, as on a plucked string."""
    freq = hz(note) if isinstance(note, str) else float(note)
    t = seconds(length)
    out = np.zeros_like(t)
    for k, amp in partials:
        out += amp * envelope(length, attack, tau / (1 + 0.6 * (k - 1))) * np.sin(2 * np.pi * freq * k * t)
    return out / sum(amp for _, amp in partials)


def _fft_size(n: int) -> int:
    """The power of two at or above *n*: pocketfft is many times slower on awkward lengths."""
    return 1 << max(0, n - 1).bit_length()


def noise(length: float, low: float, high: float, *, attack: float = 0.002, tau: float = 0.03, seed: int = 0) -> np.ndarray:
    """Band-limited noise burst between *low* and *high* Hz (soft 8th-order edges)."""
    n = int(round(length * SAMPLE_RATE))
    size = _fft_size(n)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n), size)
    freqs = np.fft.rfftfreq(size, 1 / SAMPLE_RATE)
    mask = np.zeros_like(freqs)
    f = freqs[1:]
    mask[1:] = 1 / (1 + (f / high) ** 8) / (1 + (low / f) ** 8)
    burst = np.fft.irfft(spectrum * mask, size)[:n]
    return level(burst, 1) * envelope(length, attack, tau)


def thump(f0: float, f1: float, length: float, *, attack: float = 0.002, tau: float = 0.05) -> np.ndarray:
    """A sine gliding exponentially from *f0* to *f1* Hz: drums and impacts."""
    t = seconds(length)
    freq = f0 * (f1 / f0) ** (t / length)
    return np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * envelope(length, attack, tau)


def mix(*layers: np.ndarray | tuple[float, np.ndarray]) -> np.ndarray:
    """Sum clips (mono or stereo alike); a ``(start_seconds, clip)`` pair places the clip later."""
    placed = [(0.0, layer) if isinstance(layer, np.ndarray) else layer for layer in layers]
    starts = [int(round(start * SAMPLE_RATE)) for start, _ in placed]
    stereo = any(clip.ndim == 2 for _, clip in placed)
    length = max(start + len(clip) for start, (_, clip) in zip(starts, placed))
    out = np.zeros((length, 2) if stereo else length)
    for start, (_, clip) in zip(starts, placed):
        if stereo and clip.ndim == 1:
            clip = np.stack([clip, clip], axis=1)
        out[start:start + len(clip)] += clip
    return out


def level(clip: np.ndarray, peak: float) -> np.ndarray:
    """Scale so the loudest sample is *peak*, rejecting silence or nonfinite data."""
    if not np.isfinite(peak) or not 0 <= peak <= 1:
        raise ValueError("Peak must be finite and between 0 and 1")
    magnitude = np.max(np.abs(clip))
    if not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Cannot normalize a silent or nonfinite clip")
    return clip * (peak / magnitude)


def pan(clip: np.ndarray, position: float) -> np.ndarray:
    """Mono → stereo with equal-power panning; *position* −1 (left) … 1 (right)."""
    angle = (position + 1) * np.pi / 4
    return np.stack([clip * np.cos(angle), clip * np.sin(angle)], axis=1)


def _frequency(note: str | float) -> float:
    freq = hz(note) if isinstance(note, str) else float(note)
    if not (math.isfinite(freq) and 0 < freq < SAMPLE_RATE / 2):
        raise ValueError(f"Frequency must be positive and below Nyquist, got {freq!r}")
    return freq


def adsr(length: float, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """A held note's shape: raised-cosine attack to 1, exponential decay to *sustain*, and a
    raised-cosine release over the last *release* seconds.  A note too short for its stages
    shortens the attack and release to half its length each, so it still speaks."""
    if not 0 <= sustain <= 1:
        raise ValueError(f"sustain must be between 0 and 1, got {sustain!r}")
    if sustain < 1 and decay <= 0:
        raise ValueError("A decay time is needed to reach a sustain level below 1")
    attack, release = min(attack, length / 2), min(release, length / 2)
    t = seconds(length)
    env = np.ones_like(t)
    if attack > 0:
        env *= np.where(t < attack, 0.5 - 0.5 * np.cos(np.pi * t / attack), 1.0)
    if sustain < 1:
        env *= np.where(t < attack, 1.0, sustain + (1 - sustain) * np.exp(-np.maximum(t - attack, 0) / decay))
    if release > 0:
        start = length - release
        env *= np.where(t > start, 0.5 + 0.5 * np.cos(np.pi * np.minimum(t - start, release) / release), 1.0)
    tail = min(len(t), int(0.002 * SAMPLE_RATE))
    env[-tail:] *= np.linspace(1.0, 0.0, tail)
    return env


def sustained(note: str | float, length: float, *, partials=PAD, attack: float = 0.05, decay: float = 0.3,
              sustain: float = 1.0, release: float = 0.1, vibrato: tuple[float, float] = (5.0, 0.0),
              voices: int = 1, detune: float = 0.0, seed: int = 0) -> np.ndarray:
    """A held tone: strings, horns, flutes and choirs start here.

    *vibrato* is ``(rate_hz, depth)`` where depth 0.006 is about a tenth of a semitone; it
    settles in after the attack the way a player's does.  *voices* copies spread evenly across
    ±*detune* (a ratio; 0.004 is a chorus) with independent phases thicken the tone.
    """
    freq = _frequency(note)
    if voices < 1:
        raise ValueError("voices must be at least 1")
    t = seconds(length)
    rng = np.random.default_rng(seed)
    rate, depth = vibrato
    onset = np.minimum(1.0, t / max(attack, 0.25))
    cycle = np.arange(_TABLE) / _TABLE
    out = np.zeros_like(t)
    for voice in range(voices):
        spread = 0.0 if voices == 1 else detune * (2 * voice / (voices - 1) - 1)
        # One period of this voice's partials (each at its own phase), read by a wavering phase.
        table = np.zeros(_TABLE)
        for k, amp in partials:
            if freq * k * (1 + abs(spread)) < SAMPLE_RATE / 2:
                table += amp * np.sin(2 * np.pi * k * cycle + rng.uniform(0, 2 * np.pi))
        wobble = 1 + depth * onset * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
        position = np.cumsum(freq * (1 + spread) * wobble) * (_TABLE / SAMPLE_RATE)
        index = position.astype(np.int64)
        fraction = position - index
        index %= _TABLE
        out += table[index] * (1 - fraction) + table[(index + 1) % _TABLE] * fraction
    return out * adsr(length, attack, decay, sustain, release) / (voices * sum(amp for _, amp in partials))


def pluck(note: str | float, length: float, *, brightness: float = 0.6, tau: float = 0.4, seed: int = 0) -> np.ndarray:
    """A plucked string (Karplus–Strong): a noise burst circulating through a delay line whose
    averaging dulls the upper partials first.  *tau* is the fundamental's decay time and
    *brightness* (0–1) how much of the pick's edge survives.  Peak-normalised."""
    freq = _frequency(note)
    if not 0 <= brightness <= 1:
        raise ValueError(f"brightness must be between 0 and 1, got {brightness!r}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau!r}")
    count = int(round(length * SAMPLE_RATE))
    if count < 2:
        raise ValueError("A pluck needs at least two samples")
    burst = np.random.default_rng(seed).uniform(-1, 1, max(2, int(round(SAMPLE_RATE / freq))))
    burst -= burst.mean()
    size = _fft_size(2 * count)
    w = 2 * np.pi * np.arange(size // 2 + 1) / size
    hertz = w * SAMPLE_RATE / (2 * np.pi)
    delay = SAMPLE_RATE / freq - 0.5  # the two-tap average adds half a sample
    rho = math.exp(-1 / (tau * freq))
    comb = 1 / (1 - rho * 0.5 * (np.exp(-1j * w * delay) + np.exp(-1j * w * (delay + 1))))
    comb /= np.sqrt(1 + (hertz / (freq * 2 ** (1 + 4 * brightness))) ** 2)  # the pick's edge: 2× to 32× the pitch
    comb /= np.sqrt(1 + (freq / 2 / np.maximum(hertz, 1e-9)) ** 8)  # nothing below the string's own pitch
    out = np.fft.irfft(np.fft.rfft(burst, size) * comb, size)[:count]
    edge = min(count // 2, int(0.0015 * SAMPLE_RATE))  # a real pluck takes a moment to speak; no click
    out[:edge] *= np.linspace(0.0, 1.0, edge)
    tail = min(count, int(0.005 * SAMPLE_RATE))
    out[-tail:] *= np.linspace(1.0, 0.0, tail)
    return level(out, 1.0)


def _shape_spectrum(clip: np.ndarray, gain: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Zero-phase filtering: scale the spectrum by *gain(frequencies)*; mono or stereo."""
    if clip.ndim not in (1, 2) or not clip.size:
        raise ValueError("Filters take a nonempty mono or stereo clip")
    n = len(clip)
    size = _fft_size(n)
    mask = gain(np.fft.rfftfreq(size, 1 / SAMPLE_RATE))
    if clip.ndim == 2:
        mask = mask[:, None]
    return np.fft.irfft(np.fft.rfft(clip, size, axis=0) * mask, size, axis=0)[:n]


def lowpass(clip: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """Butterworth-shaped low-pass at *cutoff* Hz (−3 dB), *order* × 6 dB per octave beyond."""
    if not 0 < cutoff < SAMPLE_RATE / 2:
        raise ValueError(f"cutoff must be between 0 and Nyquist, got {cutoff!r}")
    return _shape_spectrum(clip, lambda f: 1 / np.sqrt(1 + (f / cutoff) ** (2 * order)))


def highpass(clip: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """Butterworth-shaped high-pass at *cutoff* Hz; removes DC and rumble."""
    if not 0 < cutoff < SAMPLE_RATE / 2:
        raise ValueError(f"cutoff must be between 0 and Nyquist, got {cutoff!r}")

    def gain(f: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(f > 0, 1 / np.sqrt(1 + (cutoff / np.maximum(f, 1e-9)) ** (2 * order)), 0.0)

    return _shape_spectrum(clip, gain)


#: Vowel formants as (centre Hz, bandwidth Hz, gain) for ``formant``.
AH = ((700, 130, 1.0), (1220, 150, 0.5), (2600, 200, 0.25))
OH = ((500, 110, 1.0), (850, 130, 0.45), (2400, 200, 0.15))
OO = ((330, 90, 1.0), (750, 120, 0.3), (2300, 200, 0.1))


def formant(clip: np.ndarray, formants=AH) -> np.ndarray:
    """Give a rich tone a vowel by emphasising resonant peaks; the strongest peak passes at unity."""

    def gain(f: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(f)
        for centre, bandwidth, weight in formants:
            mask += weight / (1 + ((f - centre) / (bandwidth / 2)) ** 2)
        return mask / mask.max()

    return _shape_spectrum(clip, gain)


def reverb(clip: np.ndarray, *, decay: float = 2.0, mix: float = 0.3, predelay: float = 0.02,
           damping: float = 5000.0, wrap: bool = False, seed: int = 0) -> np.ndarray:
    """Add a room: a decorrelated stereo tail *decay* seconds long (its −60 dB time), darkened
    above *damping* Hz, at *mix* times the dry level.  Always stereo.  With *wrap* the tail
    folds round to the start for a seamless loop; otherwise the result grows by the tail."""
    if decay <= 0 or mix < 0 or predelay < 0:
        raise ValueError("reverb needs a positive decay and nonnegative mix and predelay")
    dry = clip if clip.ndim == 2 else np.stack([clip, clip], axis=1)
    rng = np.random.default_rng(seed)
    t = seconds(decay)
    tail = rng.standard_normal((len(t), 2)) * np.exp(-6.91 * t / decay)[:, None]
    tail = highpass(lowpass(tail, damping), 120.0)
    tail /= np.sqrt(np.sum(tail ** 2, axis=0))
    impulse = np.concatenate([np.zeros((int(round(predelay * SAMPLE_RATE)), 2)), tail])
    n = len(dry)
    size = _fft_size(n + len(impulse))
    wet = np.fft.irfft(np.fft.rfft(dry, size, axis=0) * np.fft.rfft(impulse, size, axis=0), size, axis=0)[:n + len(impulse)]
    if wrap:
        out = dry + wet[:n] * mix
        for start in range(n, len(wet), n):  # a tail longer than the loop folds over more than once
            chunk = wet[start:start + n] * mix
            out[:len(chunk)] += chunk
        return out
    out = np.zeros_like(wet)
    out[:n] = dry
    return out + wet * mix


def soft_clip(clip: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Round off peaks with a tanh curve: unity gain for small signals, never beyond ±1/*drive*."""
    if not drive > 0:
        raise ValueError(f"drive must be positive, got {drive!r}")
    return np.tanh(clip * drive) / drive


def loop_add(out: np.ndarray, clip: np.ndarray, start_seconds: float) -> None:
    """Add *clip* to the loop *out* at *start_seconds*; whatever runs past the end wraps round
    to the start, so a loop's last notes ring into its first bar."""
    if len(clip) > len(out):
        raise ValueError(f"Clip of {len(clip)} samples does not fit a loop of {len(out)}")
    if out.ndim == 2 and clip.ndim == 1:
        clip = np.stack([clip, clip], axis=1)
    n = len(out)
    start = int(round(start_seconds * SAMPLE_RATE)) % n
    first = min(len(clip), n - start)
    out[start:start + first] += clip[:first]
    out[:len(clip) - first] += clip[first:]


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Write nonempty mono/stereo 16-bit PCM; invalid samples leave an existing file intact.

    Samples must be finite and in [-1, 1], shaped ``(n,)``, ``(n, 1)`` or
    ``(n, 2)``. Use ``level`` before writing if a mix exceeds that range.
    """
    if samples.ndim not in (1, 2) or not samples.size or (samples.ndim == 2 and samples.shape[1] not in (1, 2)):
        raise ValueError("WAV samples must be a nonempty mono or stereo clip")
    if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1:
        raise ValueError("WAV samples must be finite and between -1 and 1")
    pcm = np.round(samples * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
