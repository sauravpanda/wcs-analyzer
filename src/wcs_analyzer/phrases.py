"""Music structure for coaching: the 8-count grid and the real phrase boundaries.

A fixed 32-count grid from the first beat is wrong for most competition songs: there are
intros, 16- and 48-count sections, breaks, and drops. This module finds where the music
actually changes and expresses it the way a dancer counts it.

Pipeline (all beat-synchronous, so results land on counts, not arbitrary seconds):

1. Features per beat: chroma (harmony), MFCC (timbre), RMS (energy), onset strength.
2. Novelty: cosine self-similarity of the harmony+timbre vectors, scored with a Foote
   checkerboard kernel. Peaks are where one section ends and another begins.
3. The 8-count grid: the beat phase (mod 8) whose recurring positions carry the most
   accent, novelty and energy change is count 1 of each 8.
4. Boundaries: a dynamic programme over the grid positions. Every candidate boundary
   earns its novelty, energy change and accent; section lengths of 32 and 16 counts get
   a bonus, odd lengths a penalty, and each boundary costs a little, so the regular
   32-count phrase wins by default and the grid is only broken for a real change.
5. Each boundary is labelled: section, energy drop, build, break, or regular.

Confidence is heuristic (0..1) and the report should say so. The song map SVG lets a
dancer check the result by ear.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

HOP = 512
KERNEL_BEATS = 16          # width of the novelty kernel, in beats
# Length bonus per section (counts -> bonus) and the flat cost of every boundary. A
# 32-count phrase nets +0.1 with no evidence at all, so a plain song still gets its
# 32s; a 16 must show a real novelty peak or energy change to beat the 32 it splits,
# and 8s only appear for strong changes (breaks, drops).
ALLOWED_LENGTHS = {8: -0.20, 16: 0.45, 24: 0.10, 32: 0.90, 40: -0.10, 48: 0.35, 64: 0.30}
INTRO_MAX = 32             # an intro may be any multiple of 4 counts up to this
BOUNDARY_COST = 0.80
NOVELTY_FLOOR = 0.3        # a song whose strongest change is below this fraction of the theoretical max is "flat"
PEAK_RADIUS = 4            # a novelty value only counts where it is the local maximum within this many beats
PEAK_MIN = 0.15
ENERGY_DEADBAND = 0.25     # log-energy change between neighbouring 8s that counts as nothing


@dataclass
class PhraseBoundary:
    time: float          # seconds
    beat: int            # index into the beat list
    counts: int          # counts in the section that ends here (0 for the first)
    kind: str            # start | section | energy drop | build | break | regular
    confidence: float    # 0..1, heuristic
    novelty: float = 0.0
    energy_change: float = 0.0   # log ratio of the next 8 counts over the previous 8


@dataclass
class PhraseMap:
    bpm: float
    beat_times: list[float]
    phase: int                              # beat index (mod 8) that is count 1
    eights: list[float] = field(default_factory=list)    # time of count 1 of every 8
    boundaries: list[PhraseBoundary] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)    # per beat, 0..1
    novelty: list[float] = field(default_factory=list)   # per beat, 0..1
    method: str = "structure-v2"

    @property
    def starts(self) -> list[float]:
        return [b.time for b in self.boundaries]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beat_times"] = [round(t, 3) for t in self.beat_times]
        d["eights"] = [round(t, 2) for t in self.eights]
        d["energy"] = [round(v, 3) for v in self.energy]
        d["novelty"] = [round(v, 3) for v in self.novelty]
        for b in d["boundaries"]:
            b["time"] = round(b["time"], 2)
            b["confidence"] = round(b["confidence"], 2)
            b["novelty"] = round(b["novelty"], 3)
            b["energy_change"] = round(b["energy_change"], 3)
        return d


# --------------------------------------------------------------------------- features

def _beat_features(y: np.ndarray, sr: int, beat_times: list[float]) -> dict[str, np.ndarray]:
    """Per-beat chroma, mfcc, rms and onset strength (column i = beat i)."""
    import librosa

    beat_frames = [int(f) for f in librosa.time_to_frames(np.asarray(beat_times), sr=sr, hop_length=HOP)]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)[None, :]

    def sync(feat: np.ndarray, agg) -> np.ndarray:
        s = librosa.util.sync(feat, beat_frames, aggregate=agg, pad=True)
        # pad=True gives one leading segment before the first beat; drop it and
        # keep exactly one column per beat.
        s = s[:, 1:1 + len(beat_times)]
        if s.shape[1] < len(beat_times):
            s = np.pad(s, ((0, 0), (0, len(beat_times) - s.shape[1])), mode="edge")
        return s

    return {
        "chroma": sync(chroma, np.median),
        "mfcc": sync(mfcc, np.mean),
        "rms": sync(rms, np.mean)[0],
        "onset": sync(onset, np.max)[0],
    }


def foote_novelty(X: np.ndarray, width: int = KERNEL_BEATS) -> np.ndarray:
    """Checkerboard-kernel novelty over the cosine self-similarity of columns of X."""
    n = X.shape[1]
    if n == 0:
        return np.zeros(0)
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    half = max(2, width // 2)
    g = np.exp(-0.5 * (np.arange(-half, half) + 0.5) ** 2 / (half / 2.0) ** 2)
    sign = np.sign(np.arange(-half, half) + 0.5)
    K = np.outer(g * sign, g * sign)
    Sp = np.pad(S, half, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = float(np.sum(K * Sp[i:i + 2 * half, i:i + 2 * half]))
    nov = np.maximum(nov, 0.0)
    # Scale against the theoretical maximum (two internally identical, mutually
    # orthogonal blocks) rather than the song's own peak, so a flat song stays flat
    # instead of having its noise promoted to a 100% boundary.
    theoretical_max = 2.0 * float(g.sum() / 2.0) ** 2
    scale = max(float(nov.max()), NOVELTY_FLOOR * theoretical_max, 1e-9)
    return np.minimum(1.0, nov / scale)


def local_peaks(values: np.ndarray, radius: int = PEAK_RADIUS, floor: float = PEAK_MIN) -> np.ndarray:
    """Keep only local maxima (within +/- radius) above floor; everything else becomes 0."""
    out = np.zeros_like(values)
    n = len(values)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        if values[i] >= floor and values[i] >= values[lo:hi].max():
            out[i] = values[i]
    return out


def _standardize(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


# --------------------------------------------------------------------------- structure

def analyze_structure_from_features(
    beat_times: list[float], bpm: float, chroma: np.ndarray, mfcc: np.ndarray, rms: np.ndarray, onset: np.ndarray,
) -> PhraseMap:
    """Pure-numpy core: grid phase, boundaries and labels from per-beat features."""
    n = len(beat_times)
    if n < 16:
        return PhraseMap(bpm=bpm, beat_times=list(beat_times), phase=0)

    X = np.vstack([chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9),
                   0.5 * _standardize(mfcc) / math.sqrt(mfcc.shape[0])])
    novelty = foote_novelty(X)
    energy = rms / (rms.max() + 1e-9)
    log_e = np.log(energy + 1e-3)
    accent = np.clip(_standardize(onset), 0, None)
    accent = accent / (accent.max() + 1e-9)

    def energy_change(i: int) -> float:
        before = log_e[max(0, i - 8):i]
        after = log_e[i:min(n, i + 8)]
        if len(before) == 0 or len(after) == 0:
            return 0.0
        return float(after.mean() - before.mean())

    ech = np.array([energy_change(i) for i in range(n)])
    nov_peak = local_peaks(novelty)
    energy_term = np.minimum(1.0, np.maximum(0.0, np.abs(ech) - ENERGY_DEADBAND))
    # Evidence that a beat is a boundary: a novelty peak, an energy change beyond the
    # dead band, and a little accent. Non-peak beats earn nothing from novelty.
    score = 1.5 * nov_peak + energy_term + 0.2 * accent

    # 8-count grid: the phase whose positions carry the most evidence (raw novelty and
    # accent both count here, since the grid is about where the emphasis recurs).
    grid_evidence = novelty + energy_term + 0.75 * accent
    phase_scores = [float(np.mean(grid_evidence[p::8])) for p in range(8)]
    phase = int(np.argmax(phase_scores))
    eights = list(range(phase, n, 8))

    # Dynamic programme over grid positions (plus an intro that may start anywhere on a 4-grid).
    grid = eights
    starts = [i for i in range(0, min(n, INTRO_MAX + 1), 4)] + grid[:1]
    starts = sorted(set(s for s in starts if s < n))
    best: dict[int, float] = {}
    back: dict[int, int | None] = {}
    for s in starts:
        # an intro longer than 8 counts costs a little; a start exactly on the grid is free
        best[s] = -0.02 * s + score[s]
        back[s] = None
    for k, g in enumerate(grid):
        if g in best and back.get(g, "x") is None and g in starts:
            pass
        cands = []
        for j in list(starts) + grid[:k]:
            if j >= g or j not in best:
                continue
            length = g - j
            if length not in ALLOWED_LENGTHS:
                continue
            cands.append((best[j] + ALLOWED_LENGTHS[length] - BOUNDARY_COST + score[g], j))
        if cands:
            val, j = max(cands)
            if g not in best or val > best[g]:
                best[g], back[g] = val, j
    # terminal: the chain may end on any grid point, with the tail (< 64 counts) unscored;
    # a boundary in the last few beats is not a phrase anyone can acknowledge, so the
    # tail must be at least 4 counts long.
    tail_ok = [g for g in grid if g in best and 4 <= n - g <= 64]
    if not tail_ok:
        tail_ok = [max(k for k in best)]
    end = max(tail_ok, key=lambda g: best[g])
    chain = []
    cur: int | None = end
    while cur is not None:
        chain.append(cur)
        cur = back.get(cur)
    chain.reverse()

    boundaries: list[PhraseBoundary] = []
    med_e = float(np.median(energy)) + 1e-9
    for idx, b in enumerate(chain):
        counts = b - chain[idx - 1] if idx else 0
        if idx == 0:
            kind = "start"
        elif energy[max(0, b - 2):b].max() < 0.15 * med_e:
            kind = "break"
        elif ech[b] < -0.6:
            kind = "energy drop"
        elif ech[b] > 0.6:
            kind = "build"
        elif nov_peak[b] > 0.45:
            kind = "section"
        else:
            kind = "regular"
        conf = float(min(1.0, 0.25 + nov_peak[b] + 0.5 * energy_term[b] + 0.25 * accent[b]))
        if kind == "regular":
            conf = min(conf, 0.6)
        boundaries.append(PhraseBoundary(time=float(beat_times[b]), beat=int(b), counts=int(counts), kind=kind,
                                         confidence=conf, novelty=float(novelty[b]), energy_change=float(ech[b])))

    return PhraseMap(bpm=bpm, beat_times=list(beat_times), phase=phase,
                     eights=[float(beat_times[i]) for i in eights], boundaries=boundaries,
                     energy=[float(v) for v in energy], novelty=[float(v) for v in novelty])


def analyze_structure(y: np.ndarray, sr: int, beat_times: list[float], bpm: float) -> PhraseMap:
    """Full analysis from decoded audio and the (octave-corrected) beat list."""
    if len(beat_times) < 16:
        return PhraseMap(bpm=bpm, beat_times=list(beat_times), phase=0)
    f = _beat_features(y, sr, beat_times)
    return analyze_structure_from_features(beat_times, bpm, f["chroma"], f["mfcc"], f["rms"], f["onset"])


# --------------------------------------------------------------------------- presentation

def _fmt(t: float) -> str:
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def phrase_context(pm: PhraseMap) -> str:
    """Prompt text: where the phrases change and what kind of change each is."""
    if not pm.boundaries:
        return ""
    parts = []
    for b in pm.boundaries:
        if b.kind == "start":
            intro = f" after a {b.beat}-count intro" if b.beat else ""
            parts.append(f"{_fmt(b.time)} (first phrase starts{intro})")
        else:
            parts.append(f"{_fmt(b.time)} ({b.counts} counts, {b.kind})")
    grid = ", ".join(_fmt(t) for t in pm.eights[:4])
    return (
        "Phrase changes found in the audio, each on count 1 of a new phrase, with the length of the "
        "section that ends there and what kind of change the music makes: " + "; ".join(parts) + ". "
        f"Count 1 of each 8 falls at {grid}, ... (every 8 beats). Judge acknowledgment within about one "
        "second of each phrase change; the timings can be off by a beat."
    )


def song_map_svg(pm: PhraseMap, duration: float, old_grid: list[float] | None = None, width: int = 1100) -> str:
    """A Step-Back-style strip: energy, novelty, the 8-count grid and the boundaries."""
    H, top, bottom, left, right = 150, 22, 34, 8, 8
    W = width
    span = max(duration, (pm.beat_times[-1] if pm.beat_times else duration)) or 1.0
    x = lambda t: left + (W - left - right) * (t / span)  # noqa: E731
    plot_h = H - top - bottom
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="Song map: energy, novelty and phrase boundaries">',
           '<style>text{font:10px system-ui,sans-serif}</style>']
    # energy area
    if pm.energy:
        pts = [f"{x(pm.beat_times[i]):.1f},{top + plot_h * (1 - pm.energy[i]):.1f}" for i in range(len(pm.energy))]
        out.append(f'<polygon points="{x(pm.beat_times[0]):.1f},{top + plot_h} {" ".join(pts)} '
                   f'{x(pm.beat_times[-1]):.1f},{top + plot_h}" fill="#b7791f" opacity="0.18"/>')
        npts = [f"{x(pm.beat_times[i]):.1f},{top + plot_h * (1 - pm.novelty[i]):.1f}" for i in range(len(pm.novelty))]
        out.append(f'<polyline points="{" ".join(npts)}" fill="none" stroke="#1565c0" stroke-width="1.2" opacity="0.8"/>')
    # 8-count ticks
    for t in pm.eights:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top + plot_h}" y2="{top + plot_h + 6}" stroke="#888" stroke-width="1"/>')
    # old 32-grid for comparison
    for t in old_grid or []:
        out.append(f'<path d="M{x(t):.1f},{top + plot_h + 14} l-4,7 l8,0 z" fill="#aaa"/>')
    # boundaries
    colors = {"start": "#555", "section": "#c62828", "energy drop": "#6a1b9a", "build": "#ef6c00", "break": "#000", "regular": "#2e7d32"}
    for b in pm.boundaries:
        c = colors.get(b.kind, "#333")
        out.append(f'<line x1="{x(b.time):.1f}" x2="{x(b.time):.1f}" y1="{top - 4}" y2="{top + plot_h}" stroke="{c}" '
                   f'stroke-width="{1.5 + 1.5 * b.confidence:.1f}" opacity="{0.45 + 0.55 * b.confidence:.2f}"/>')
        label = f"{b.counts}" if b.kind != "start" else "start"
        out.append(f'<text x="{x(b.time) + 3:.1f}" y="{top - 8}" fill="{c}">{label}</text>')
        out.append(f'<text x="{x(b.time) + 3:.1f}" y="{H - 4}" fill="#666">{_fmt(b.time)}</text>')
    # time axis every 15 s
    t = 0.0
    while t <= span:
        out.append(f'<text x="{x(t):.1f}" y="{top + plot_h + 30}" fill="#999" text-anchor="middle">{int(t)}s</text>')
        t += 15
    out.append(f'<text x="{left}" y="12" fill="#444">energy (amber), novelty (blue), 8-count ticks, phrase boundaries '
               f'labelled with the counts that end there; grey triangles = old fixed 32-count grid</text>')
    out.append("</svg>")
    return "\n".join(out)
