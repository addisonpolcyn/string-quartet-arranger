"""Stage 2: audio stems -> note events with Spotify's basic-pitch, plus beat tracking."""
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class Note:
    start: float  # seconds after transcription, beats after to_beats()
    end: float
    pitch: int  # MIDI note number
    velocity: float  # 0..1


# Per-stem transcription settings: frequency limits cut down on octave/harmonic errors.
SETTINGS = {
    "vocals": dict(onset_threshold=0.6, frame_threshold=0.3, minimum_note_length=90,
                   minimum_frequency=80, maximum_frequency=1200),
    "bass": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=90,
                 minimum_frequency=30, maximum_frequency=400),
    "other": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=120,
                  minimum_frequency=60, maximum_frequency=2500),
}


@lru_cache(maxsize=1)
def _model():
    try:  # basic-pitch is tiny; TF's GPU JIT often clashes with torch's CUDA libs, so use CPU
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except ImportError:
        pass
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    return Model(ICASSP_2022_MODEL_PATH)


def transcribe(path: Path, kind: str) -> list[Note]:
    cache = path.with_suffix(".notes.json")
    if cache.exists():
        return [Note(**n) for n in json.loads(cache.read_text())]
    import soundfile as sf
    from basic_pitch.inference import predict

    y, _ = sf.read(str(path))
    if np.sqrt(np.mean(y**2)) < 2e-3:  # essentially silent: transcribing it would just be noise
        print(f"[transcribe] {kind}: stem is silent, skipping")
        notes = []
    else:
        print(f"[transcribe] {kind} ...")
        _, _, events = predict(str(path), _model(), **SETTINGS[kind])
        notes = [Note(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]
    cache.write_text(json.dumps([asdict(n) for n in notes]))
    return notes


def _pitch_track(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """(frame times, MIDI pitch per frame with the recording's tuning offset removed, offset).
    pYIN f0 (NaN = unvoiced), cached next to the stem."""
    cache = path.with_suffix(".f0.npz")
    if cache.exists():
        d = np.load(cache)
        times, f0 = d["times"], d["f0"]
    else:
        import librosa

        print(f"[transcribe] {path.stem}: tracking pitch with pYIN ...")
        y, sr = librosa.load(str(path), sr=16000, mono=True)
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=1200, sr=sr, hop_length=320)  # NaN = unvoiced
        times = librosa.times_like(f0, sr=sr, hop_length=320)
        np.savez(cache, times=times, f0=f0)
    with np.errstate(invalid="ignore"):
        midi = 69 + 12 * np.log2(f0 / 440.0)
    # recordings are often tuned a little off A440; a singer 20 cents flat plus vibrato keeps
    # landing nearer the semitone below, so measure the offset (circular mean) and undo it
    voiced = midi[~np.isnan(midi)]
    tuning = float(np.angle(np.mean(np.exp(2j * np.pi * voiced))) / (2 * np.pi)) if len(voiced) else 0.0
    print(f"[transcribe] {path.stem}: tuned {tuning * 100:+.0f} cents from A440")
    return times, midi - tuning, tuning


@lru_cache(maxsize=2)
def sung_onsets(vocals: Path) -> np.ndarray:
    """Times (s) where the singer starts a note, ignoring quiet bleed in the vocal stem."""
    import librosa

    y, sr = librosa.load(str(vocals), sr=22050, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    return np.array([t for t in onsets if rms[min(len(rms) - 1, np.searchsorted(rms_times, t))] > 0.02])


def sung_line(vocals: Path, to_sec):
    """The singing on the beat grid, for `best_melody`: line(total, step) -> (sung pitch per
    step, NaN where nobody sings; whether the singer starts a syllable in that step)."""
    times, midi, _ = _pitch_track(vocals)
    onsets = sung_onsets(vocals)

    def line(total: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        n = int(np.ceil(total / step))
        edges = np.array([to_sec(k * step) for k in range(n + 1)])
        idx = np.searchsorted(times, edges)
        f0 = np.full(n, np.nan)
        for k in range(n):
            seg = midi[idx[k]:idx[k + 1]]
            sung = seg[~np.isnan(seg)]
            if len(sung) and len(sung) >= 0.5 * len(seg):
                f0[k] = np.median(sung)
        starts = np.zeros(n, bool)
        mids = (edges[1:] + edges[:-1]) / 2  # an onset belongs to the nearest grid line
        k = np.searchsorted(mids, onsets[(onsets >= edges[0]) & (onsets < edges[-1])])
        starts[np.clip(k, 0, n - 1)] = True
        return f0, starts

    return line


def refine_pitch(path: Path, notes: list[Note], max_move: int = 2) -> list[Note]:
    """Re-pitch notes from a pYIN f0 track: the median over each note ignores vibrato and
    slides. Only small corrections are taken, so pYIN's own octave slips can't leak in."""
    times, midi, _ = _pitch_track(path)
    est = []
    for n in notes:
        seg = midi[(times >= n.start) & (times < n.end)]
        seg = seg[~np.isnan(seg)]
        est.append(int(round(float(np.median(seg)))) if len(seg) >= 3 else None)
    sure = [e is not None and abs(e - n.pitch) <= max_move for n, e in zip(notes, est)]
    out, moved = [], 0
    for n, e, ok in zip(notes, est, sure):
        p = n.pitch
        if ok:
            p = e
        elif e is not None:  # the two heard different voices: keep the one that fits the tune
            near = [est[j] for j, m in enumerate(notes) if sure[j] and abs(m.start - n.start) < 3.0]
            if near:
                p = min((n.pitch, e), key=lambda c: abs(c - float(np.median(near))))
        moved += p != n.pitch
        out.append(Note(n.start, n.end, p, n.velocity))
    print(f"[transcribe] {path.stem}: pYIN corrected {moved} of {len(notes)} notes")
    return out


@lru_cache(maxsize=2)
def _chroma(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)  # also corrects tuning
    return librosa.times_like(chroma, sr=sr, hop_length=1024), chroma


def chroma_repair(vocals: Path, to_sec, ratio: float = 1.5, floor: float = 0.5):
    """A final check of the finished tune against the vocal stem itself: a note whose pitch class
    is clearly weaker in the voice than a neighbour's (+-1 or 2 semitones) moves to that neighbour.
    Catches slips from transcription and from clean-up (e.g. the louder of two notes winning)."""
    times, chroma = _chroma(vocals)

    def repair(melody: list[Note], scale: set[int]) -> tuple[list[Note], int]:
        out, fixed = [], 0
        for n in melody:
            seg = chroma[:, (times >= to_sec(n.start)) & (times < to_sec(n.end))]
            p = n.pitch
            # backing vocals and ad-libs can outweigh the tune, so never pull an in-key note
            # out of the key; out-of-key notes may move anywhere the voice clearly is
            alts = [n.pitch + d for d in (-2, -1, 1, 2)
                    if n.pitch % 12 not in scale or (n.pitch + d) % 12 in scale]
            if seg.shape[1] and alts:
                c = seg.mean(axis=1)
                alt = max(alts, key=lambda q: c[q % 12])
                if c[alt % 12] > ratio * c[n.pitch % 12] and c[alt % 12] > floor:
                    p = alt
            fixed += p != n.pitch
            out.append(Note(n.start, n.end, p, n.velocity))
        return out, fixed

    return repair


def track_beats(drums: Path, mix: Path, bpm: float | None = None) -> tuple[float, np.ndarray]:
    """Return (tempo, beat_times). Tracks on the drum stem, falling back to the full mix."""
    import librosa

    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=sr)[0]  # one value per second
    if np.mean(rms < 0.01) > 0.2:  # drums missing for much of the song (intro, breakdown, ...)
        y, sr = librosa.load(str(mix), sr=22050, mono=True)
    duration = len(y) / sr
    if bpm:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", bpm=bpm)
        first = beats[0] if len(beats) else 0.0
        return bpm, np.arange(first, duration, 60.0 / bpm)
    # trim=False: librosa otherwise drops beats in quieter endings, and extrapolating drifts
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 4:
        return tempo, np.arange(0.0, duration, 60.0 / tempo)
    steady = steady_beats(y, sr, beats)
    if steady is not None:
        return 60.0 / float(np.median(np.diff(steady))), steady
    return tempo, beats


def steady_beats(y: np.ndarray, sr: int, beats: np.ndarray, tolerance: float = 0.03):
    """A perfectly even grid fitted to the drum hits, or None if the song's tempo really moves.
    The tracker can only place beats on ~23 ms frames and lets the tempo bend, so on songs made
    to a click it drifts and then jumps to catch up; the even grid wins when it fits the hits
    at least as well (share of strong hits more than 30 ms from the nearest eighth)."""
    import librosa

    hop = 64  # ~3 ms
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    et = librosa.times_like(env, sr=sr, hop_length=hop)
    k = np.arange(len(beats))
    period, first = np.polyfit(k, beats, 1)
    best = (-1.0, period, first)
    for span, step in ((0.02, 1e-4), (0.0004, 1e-5)):  # coarse, then fine
        center = best[1]
        for p in np.arange(center * (1 - span), center * (1 + span), step):
            n = int((et[-1] - beats[0]) / p) + 2
            phases = beats[0] - p + np.arange(0, p, 0.003)
            grid = phases[:, None] + p * np.arange(n)[None, :]
            score = np.interp(grid, et, env).sum(axis=1)
            i = int(score.argmax())
            if score[i] > best[0]:
                best = (float(score[i]), p, float(phases[i]))
    _, period, first = best
    first -= period * np.floor((first - beats[0]) / period + 0.5)  # start near the first beat
    first -= period * max(0, np.floor(first / period))
    steady = first + period * np.arange(int((et[-1] - first) / period) + 1)

    hits = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="time")
    hits = hits[np.interp(hits, et, env) > np.percentile(env, 90)]

    def missed(grid: np.ndarray) -> float:
        pos = np.interp(hits, grid, np.arange(len(grid))) * 2
        return float(np.mean(np.abs(pos - np.round(pos)) / 2 * period > tolerance))

    tracked, even = missed(beats), missed(steady)
    print(f"[beats] drum hits off the grid: tracked {tracked:.0%}, steady {60 / period:.2f} BPM "
          f"{even:.0%} -> {'steady' if even <= tracked else 'tracked'}")
    return steady if even <= tracked else None


@lru_cache(maxsize=2)
def _power(mix: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y, sr = librosa.load(str(mix), sr=22050, mono=True)
    power = librosa.feature.rms(y=y, hop_length=512)[0] ** 2
    return librosa.times_like(power, sr=sr, hop_length=512), power


def beat_loudness(mix: Path, beat_times: np.ndarray) -> np.ndarray:
    """Loudness of the original recording (dB) between each tracked beat and the next."""
    times, power = _power(mix)
    out = []
    for a, b in zip(beat_times, beat_times[1:]):
        seg = power[(times >= a) & (times < b)]
        out.append(10 * np.log10(seg.mean() + 1e-10) if len(seg) else -100.0)
    return np.array(out)


@lru_cache(maxsize=2)
def _drum_onsets(drums: Path) -> tuple[np.ndarray, np.ndarray] | None:
    import librosa

    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    if np.sqrt(np.mean(y**2)) < 2e-3:
        return None
    env = librosa.onset.onset_strength(y=y, sr=sr)
    return librosa.times_like(env, sr=sr), env


def drum_pattern(drums: Path, beat_times: np.ndarray, offset: float) -> list[float] | None:
    """The drums' accented eighth-note positions in a bar (in beats): those hit harder than
    average, plus beat 1. `offset` shifts tracked beats onto bar-aligned beats. None if no drums."""
    onsets = _drum_onsets(drums)
    if onsets is None:
        return None
    times, env = onsets
    pos = 2 * (np.interp(times, beat_times, np.arange(len(beat_times))) + offset)  # in eighths
    near = np.abs(pos - np.round(pos)) < 0.15  # onsets close to an eighth-note position
    weight = np.bincount(np.round(pos[near]).astype(int) % 8, weights=env[near], minlength=8)
    return [i / 2 for i in range(8) if i == 0 or weight[i] > weight.mean()]


def to_beats(notes: list[Note], beat_times: np.ndarray) -> list[Note]:
    """Map seconds onto the beat grid, following tempo drift; extrapolate past the ends."""
    idx = np.arange(len(beat_times), dtype=float)
    first_ibi = beat_times[1] - beat_times[0]
    last_ibi = beat_times[-1] - beat_times[-2]

    def conv(t: float) -> float:
        if t < beat_times[0]:
            return (t - beat_times[0]) / first_ibi
        if t > beat_times[-1]:
            return idx[-1] + (t - beat_times[-1]) / last_ibi
        return float(np.interp(t, beat_times, idx))

    return [Note(conv(n.start), conv(n.end), n.pitch, n.velocity) for n in notes]
