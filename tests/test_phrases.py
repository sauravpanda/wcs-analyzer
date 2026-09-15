"""Tests for the phrase-structure analysis (pure numpy core, no audio decoding)."""
import numpy as np

from wcs_analyzer.phrases import PhraseMap, analyze_structure_from_features, foote_novelty, phrase_context, song_map_svg


def _song(n_beats=160, bpm=120.0, sections=(32, 32, 16, 32, 48), drop_at=None, phase=0, intro=0):
    """Synthetic per-beat features: a new chroma/timbre profile per section, accents on the 8s."""
    rng = np.random.default_rng(0)
    beat_times = [intro * 0.5 + i * 60.0 / bpm for i in range(n_beats)]
    chroma = np.zeros((12, n_beats))
    mfcc = np.zeros((13, n_beats))
    rms = np.full(n_beats, 0.6)
    onset = np.full(n_beats, 0.2)
    bounds = [phase]
    for s in sections:
        bounds.append(bounds[-1] + s)
    for k in range(len(bounds) - 1):
        a, b = bounds[k], min(bounds[k + 1], n_beats)
        prof = rng.random(12)
        chroma[:, a:b] = prof[:, None] + 0.05 * rng.random((12, b - a))
        mfcc[:, a:b] = (rng.random(13) * 4)[:, None] + 0.1 * rng.random((13, b - a))
    if bounds[-1] < n_beats:
        chroma[:, bounds[-1]:] = rng.random(12)[:, None]
        mfcc[:, bounds[-1]:] = (rng.random(13) * 4)[:, None]
    onset[phase::8] = 1.0
    onset[phase::32] = 1.4
    if drop_at is not None:
        rms[drop_at:drop_at + 8] = 0.15
    return beat_times, chroma, mfcc, rms, onset, bounds


def test_novelty_peaks_where_sections_change():
    beat_times, chroma, mfcc, rms, onset, bounds = _song()
    X = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    nov = foote_novelty(X)
    for b in bounds[1:4]:
        assert nov[b - 2:b + 3].max() > 0.5


def test_finds_grid_phase_and_section_lengths():
    beat_times, chroma, mfcc, rms, onset, bounds = _song(phase=3)
    pm = analyze_structure_from_features(beat_times, 120.0, chroma, mfcc, rms, onset)
    assert pm.phase == 3
    found = {b.beat for b in pm.boundaries}
    # every true boundary inside the song is recovered on the exact beat
    for b in bounds[:-1]:
        assert b in found, (b, sorted(found))
    lengths = [b.counts for b in pm.boundaries[1:]]
    assert lengths[:4] == [32, 32, 16, 32]


def test_energy_drop_is_labelled():
    beat_times, chroma, mfcc, rms, onset, bounds = _song(drop_at=64)
    pm = analyze_structure_from_features(beat_times, 120.0, chroma, mfcc, rms, onset)
    at = {b.beat: b for b in pm.boundaries}
    assert 64 in at and at[64].kind == "energy drop"


def test_plain_song_defaults_to_32s():
    """No section changes at all: the grid prior alone should still produce 32-count phrases."""
    n = 130
    beat_times = [i * 0.5 for i in range(n)]
    rng = np.random.default_rng(1)
    chroma = np.tile(rng.random(12)[:, None], (1, n)) + 0.02 * rng.random((12, n))
    mfcc = np.tile(rng.random(13)[:, None], (1, n))
    rms = np.full(n, 0.5)
    onset = np.full(n, 0.2)
    onset[0::8] = 1.0
    pm = analyze_structure_from_features(beat_times, 120.0, chroma, mfcc, rms, onset)
    assert [b.counts for b in pm.boundaries[1:]] == [32, 32, 32]
    assert all(b.kind == "regular" for b in pm.boundaries[1:])
    assert all(b.confidence <= 0.6 for b in pm.boundaries[1:])


def test_short_audio_gives_empty_map():
    pm = analyze_structure_from_features([0.0, 0.5], 120.0, np.zeros((12, 2)), np.zeros((13, 2)), np.ones(2), np.ones(2))
    assert pm.boundaries == [] and pm.starts == []


def test_context_and_svg():
    beat_times, chroma, mfcc, rms, onset, bounds = _song()
    pm = analyze_structure_from_features(beat_times, 120.0, chroma, mfcc, rms, onset)
    text = phrase_context(pm)
    assert "32 counts" in text and "Count 1 of each 8" in text
    svg = song_map_svg(pm, duration=90.0, old_grid=[0.0, 16.0])
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "old fixed 32-count grid" in svg
    d = pm.to_dict()
    assert d["method"] == "structure-v2" and len(d["boundaries"]) == len(pm.boundaries)
    assert isinstance(PhraseMap(bpm=1.0, beat_times=[], phase=0).to_dict()["boundaries"], list)
