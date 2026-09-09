"""Public synthesis helpers compose real WAV assets accepted by the audio API."""

import wave

import numpy as np
import pytest

from saga2d import Game
from sagaforge.synth import (AH, SAMPLE_RATE, adsr, formant, highpass, hz, level, loop_add, lowpass, mix, noise, pan,
                          pluck, reverb, soft_clip, sustained, thump, tone, write_wav)


def test_composed_stereo_asset_roundtrips_pcm_and_plays(tmp_path):
    """A layered mono/stereo composition keeps timing, panning and signal through file playback."""
    clip = level(mix(pan(tone("D4", .2), -.5),
                     (.1, pan(noise(.15, 300, 4000, seed=19), .5)),
                     thump(180, 60, .12)), .7)
    path = tmp_path / "sounds" / "impact.wav"
    write_wav(path, clip)
    with wave.open(str(path), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (2, 2, SAMPLE_RATE)
        restored = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2) / 32767
    assert restored.shape == (round(.25 * SAMPLE_RATE), 2)
    assert np.max(np.abs(restored - clip)) <= 1 / 32767
    assert np.max(np.abs(restored)) <= .7001
    assert np.max(np.abs(restored[-1])) == 0
    game = Game("Composed audio", backend="mock", asset_path=tmp_path)
    try:
        game.audio.play_sound("impact")
        assert game.backend.sounds_played[-1]["handle"] == game.backend.load_sound(str(path))
    finally:
        game._teardown()


def test_silent_normalization_refuses_nan_audio():
    """Normalizing silence has no defined gain and must fail instead of manufacturing NaNs."""
    with pytest.raises(ValueError, match="silent"):
        level(np.zeros(128), .6)


def test_noise_too_short_for_a_signal_refuses_nan_audio():
    """A one-sample filtered burst has no non-DC signal and cannot be normalized."""
    with pytest.raises(ValueError, match="silent"):
        noise(1 / SAMPLE_RATE, 300, 4000)


@pytest.mark.parametrize("samples", [np.array([np.nan, 0.]), np.array([0., np.inf]),
                                     np.zeros((5, 3)), np.array([0., 1.2]), np.array([])])
def test_invalid_asset_is_rejected_before_replacing_file(tmp_path, samples):
    """Bad samples must not silently clip, change channel shape or destroy a valid prior asset."""
    path = tmp_path / "cue.wav"
    write_wav(path, tone("C4", .1))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_wav(path, samples)
    assert path.read_bytes() == before


def _peak_hz(clip: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(clip))
    return float(np.fft.rfftfreq(len(clip), 1 / SAMPLE_RATE)[spectrum.argmax()])


def test_adsr_shapes_any_length_smoothly():
    """Every stage overlaps gracefully; a short note still reaches full level and ends silent."""
    env = adsr(2.0, 0.1, 0.3, 0.5, 0.4)
    t = np.arange(len(env)) / SAMPLE_RATE
    assert env[0] == 0 and env[-1] == 0 and env.max() == pytest.approx(1.0, abs=1e-3)
    assert env[(t > 1.3) & (t < 1.5)] == pytest.approx(0.5, abs=0.02)
    assert np.abs(np.diff(env)).max() < 0.01
    short = adsr(0.05, 0.1, 0.3, 1.0, 0.2)
    assert short.max() > 0.99 and short[-1] == 0
    with pytest.raises(ValueError):
        adsr(1.0, 0.1, 0.0, 0.5, 0.1)


def test_sustained_holds_pitch_with_vibrato_and_unison():
    """A held voice sits on its note, wavers by its vibrato depth and thickens with detuned copies."""
    steady = sustained("A4", 1.0, vibrato=(5.0, 0.0))
    assert _peak_hz(steady) == pytest.approx(440, abs=2)
    assert 0.5 < np.abs(steady).max() <= 1.0 and abs(steady[-1]) < 1e-3
    wobbly = sustained("A4", 1.0, vibrato=(5.0, 0.03), seed=3)
    zero_crossings = np.flatnonzero(np.diff(np.sign(wobbly[22050:])))
    periods = np.diff(zero_crossings)[::2]
    assert periods.max() - periods.min() >= 4  # the period breathes with the vibrato (±3 % of 100 samples)
    chorus = sustained("A2", 1.0, voices=3, detune=0.005, seed=1)
    solo = sustained("A2", 1.0, seed=1)
    assert np.corrcoef(chorus, solo)[0, 1] < 0.9
    with pytest.raises(ValueError):
        sustained("A4", 1.0, voices=0)
    with pytest.raises(ValueError):
        sustained(30000.0, 0.1)


def test_pluck_is_a_decaying_string_at_pitch_without_a_thump():
    """The string speaks at once, sits on its pitch, dies away and carries no DC or sub-fundamental weight."""
    string = pluck("E3", 1.5, tau=0.3, seed=4)
    assert len(string) == round(1.5 * SAMPLE_RATE) and np.abs(string).max() == pytest.approx(1.0)
    spectrum = np.abs(np.fft.rfft(string))
    freqs = np.fft.rfftfreq(len(string), 1 / SAMPLE_RATE)
    fundamental = (freqs > hz("E3") * 0.99) & (freqs < hz("E3") * 1.01)
    assert spectrum[fundamental].max() > 0.5 * spectrum.max()
    assert np.abs(string[:2205]).max() > 0.5 and np.abs(string[-4410:]).max() < 0.05
    assert spectrum[freqs < hz("E3") * 0.6].max() < 0.05 * spectrum.max()
    bright, dull = pluck("E3", 0.5, brightness=1.0), pluck("E3", 0.5, brightness=0.0)
    high = freqs[: len(np.fft.rfftfreq(len(bright), 1 / SAMPLE_RATE))] > 2000
    assert np.abs(np.fft.rfft(bright))[high].sum() > 2 * np.abs(np.fft.rfft(dull))[high].sum()
    with pytest.raises(ValueError):
        pluck("E3", 0.5, brightness=2.0)


def test_filters_and_formants_shape_the_spectrum_in_place():
    """Low-pass, high-pass and formant filters keep the clip's length and shape only its spectrum."""
    rich = sustained("A2", 0.5, partials=tuple((k, 1 / k) for k in range(1, 20)))
    freqs = np.fft.rfftfreq(len(rich), 1 / SAMPLE_RATE)
    raw = np.abs(np.fft.rfft(rich))
    low = np.abs(np.fft.rfft(lowpass(rich, 600)))
    assert low[freqs > 2000].sum() < 0.05 * raw[freqs > 2000].sum() and low[freqs < 300].sum() > 0.9 * raw[freqs < 300].sum()
    rumble = rich + 0.3 * np.sin(2 * np.pi * 20 * np.arange(len(rich)) / SAMPLE_RATE) * np.hanning(len(rich))
    cleaned = np.abs(np.fft.rfft(highpass(rumble, 60)))
    assert cleaned[freqs < 30].max() < 0.02 * np.abs(np.fft.rfft(rumble))[freqs < 30].max()
    assert cleaned[freqs > 100].sum() == pytest.approx(raw[freqs > 100].sum(), rel=0.02)
    stereo = highpass(pan(rich, 0.2), 60)
    assert stereo.shape == (len(rich), 2)
    vowel = np.abs(np.fft.rfft(formant(rich, AH)))
    band = lambda spec, lo, hi: spec[(freqs > lo) & (freqs < hi)].sum()  # noqa: E731
    assert band(vowel, 600, 800) / band(vowel, 1500, 1900) > 3 * band(raw, 600, 800) / band(raw, 1500, 1900)
    with pytest.raises(ValueError):
        lowpass(rich, 30000)


def test_reverb_adds_a_tail_that_can_wrap_a_loop():
    """The room grows a dry clip by its tail, or folds that tail round to the loop's start."""
    hit = thump(180, 60, 0.3)
    roomy = reverb(hit, decay=1.0, mix=0.5, seed=2)
    assert roomy.shape[1] == 2 and len(roomy) > len(hit) + SAMPLE_RATE
    assert np.abs(roomy[len(hit) + 4410:len(hit) + 8820]).max() > 0.01  # the tail rings after the hit ends
    assert np.abs(roomy[-2205:]).max() < 0.01
    looped = reverb(hit, decay=1.0, mix=0.5, wrap=True, seed=2)
    assert looped.shape == (len(hit), 2)
    folded = roomy[:len(hit)].copy()
    for start in range(len(hit), len(roomy), len(hit)):
        chunk = roomy[start:start + len(hit)]
        folded[:len(chunk)] += chunk
    assert np.allclose(looped, folded, atol=1e-9)
    assert np.array_equal(reverb(hit, decay=1.0, mix=0.0)[:len(hit), 0], hit)
    with pytest.raises(ValueError):
        reverb(hit, decay=0)


def test_soft_clip_bounds_a_hot_mix_and_loop_add_wraps():
    hot = np.linspace(-3, 3, 101)
    squashed = soft_clip(hot, 2.0)
    assert np.abs(squashed).max() <= 0.5 and squashed[50] == 0
    assert soft_clip(np.array([0.01]))[0] == pytest.approx(0.01, rel=1e-3) and soft_clip(np.array([3.0]))[0] < 1
    loop = np.zeros((100, 2))
    loop_add(loop, np.ones(30), 90 / SAMPLE_RATE)
    assert loop[90:, 0].sum() == 10 and loop[:20, 1].sum() == 20 and loop[20:90].sum() == 0
    with pytest.raises(ValueError):
        loop_add(loop, np.ones(101), 0.0)
