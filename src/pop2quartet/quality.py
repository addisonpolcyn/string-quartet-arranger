"""Rule-based checks of a finished arrangement, so settings can be compared before rendering.

Every check is a fraction (0 = clean, 1 = everything wrong), except `fidelity`, which compares
the arrangement's pitch content with the original recording's (1 = identical chroma).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .arrange import CHORDS
from .export import NAMES, detect_key
from .transcribe import Note, sung_onsets

WEIGHTS = {"rubs": 1.0, "out_of_key": 1.0, "incomplete": 0.5, "flicker": 0.3, "leaps": 1.0,
           "tempo": 0.5, "rhythm": 1.0}
POP_TEMPO = (70, 160)  # BPM; a beat grid outside this is most likely half or double time


@lru_cache(maxsize=2)
def song_chroma(mix: Path) -> tuple[np.ndarray, np.ndarray]:
    """(frame times, 12 x frames chroma) of the original recording; CQT chroma corrects tuning."""
    import librosa

    y, sr = librosa.load(str(mix), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    return librosa.times_like(chroma, sr=sr, hop_length=2048), chroma


def fidelity(parts: dict[str, list[Note]], to_sec, mix: Path, window: float = 0.5) -> float:
    """Mean cosine similarity between the recording's chroma and the arrangement's, in fixed
    windows of real time (so arrangements on different beat grids are compared fairly)."""
    times, chroma = song_chroma(mix)
    n = int(times[-1] // window) + 1
    ours = np.zeros((n, 12))
    for ns in parts.values():
        for note in ns:
            s, e = to_sec(note.start), to_sec(note.end)
            for k in range(max(0, int(s // window)), min(n, int(e // window) + 1)):
                ov = min(e, (k + 1) * window) - max(s, k * window)
                if ov > 0:
                    ours[k, note.pitch % 12] += ov
    theirs = np.zeros((n, 12))
    idx = np.minimum(n - 1, (times // window).astype(int))
    np.add.at(theirs, idx, chroma.T)
    a, b = np.linalg.norm(ours, axis=1), np.linalg.norm(theirs, axis=1)
    both = (a > 0) & (b > 0)
    if not both.any():
        return 0.0
    return float(np.mean(np.sum(ours[both] * theirs[both], axis=1) / (a[both] * b[both])))


def evaluate(parts: dict[str, list[Note]], changes, to_sec, mix: Path, bpm: float,
             vocals: Path | None = None, step: float = 0.25) -> dict[str, float]:
    """All checks plus a weighted `total` (higher is better)."""
    tonic, mode = detect_key(parts)
    scale = {(NAMES.index(tonic) + i) % 12
             for i in ((0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10))}
    end = max((n.end for ns in parts.values() for n in ns), default=0.0)
    names = list(parts)
    rubs = samples = 0
    t = 0.0
    while t < end:  # a close minor 2nd (or minor 9th) between any two parts really rubs;
        # major 7ths are left alone, since tunes sit on them over the bass all the time
        sounding = [next((n.pitch for n in parts[p] if n.start <= t < n.end), None) for p in names]
        sounding = [p for p in sounding if p is not None]
        if len(sounding) >= 2:
            samples += 1
            rubs += any(abs(a - b) in (1, 13) for i, a in enumerate(sounding) for b in sounding[i + 1:])
        t += step

    melody = parts["vln1"]
    tune_time = sum(n.end - n.start for n in melody) or 1.0
    out_of_key = sum(n.end - n.start for n in melody if n.pitch % 12 not in scale) / tune_time
    leaps = sum(abs(a.pitch - b.pitch) > 12 for a, b in zip(melody, melody[1:])) / max(1, len(melody) - 1)

    named = complete = 0  # root and 3rd heard, sampled mid-beat through each chord's span
    bounds = [b for b, _ in changes] + [end]
    for (start, chord), stop in zip(changes, bounds[1:]):
        beat = start
        while beat < stop:
            heard = {n.pitch % 12 for ns in parts.values() for n in ns if n.start <= beat + 0.5 < n.end}
            named += 1
            complete += chord[0] in heard and (chord[0] + CHORDS[chord[1]][1]) % 12 in heard
            beat += 1.0
    bars = max(1.0, end / 4)
    checks = {
        "fidelity": fidelity(parts, to_sec, mix),
        "rubs": rubs / max(1, samples),
        "out_of_key": out_of_key,
        "incomplete": 1 - complete / max(1, named),
        "flicker": max(0.0, len(changes) / bars - 2) / 2,  # pop rarely changes chord >2x a bar
        "leaps": leaps,
        "tempo": max(0.0, np.log2(bpm / POP_TEMPO[1]), np.log2(POP_TEMPO[0] / bpm)),  # octaves out
    }
    sung = sung_onsets(vocals) if vocals else np.array([])
    starts = np.array([to_sec(n.start) for n in melody])
    if len(sung) and len(starts):  # share of the singer's note starts violin I doesn't play
        checks["rhythm"] = float(np.mean([np.min(np.abs(starts - t)) >= 0.1 for t in sung]))
    else:
        checks["rhythm"] = 0.0
    checks["total"] = checks["fidelity"] - sum(w * checks[k] for k, w in WEIGHTS.items())
    return checks
