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


# CREPE's confidence that a frame is pitched; breathy singers (Clairo) sit around 0.3, under
# the usual 0.5 cut
CREPE_VOICED = 0.5


def _crepe(y: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """(f0, confidence) per frame from CREPE (a neural pitch tracker for one voice, far steadier
    on singing than pYIN)."""
    import torch
    import torchcrepe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f0, periodicity = torchcrepe.predict(torch.tensor(y[None], dtype=torch.float32), sr, hop, 80,
                                         1200, "full", decoder=torchcrepe.decode.viterbi,
                                         return_periodicity=True, batch_size=1024, device=device)
    periodicity = torchcrepe.filter.median(periodicity, 3)
    return f0[0].cpu().numpy(), periodicity[0].cpu().numpy()


def _pitch_track(path: Path, tracker: str = "pyin") -> tuple[np.ndarray, np.ndarray, float]:
    """(frame times, MIDI pitch per frame with the recording's tuning offset removed, offset).
    pYIN or CREPE f0 (NaN = unvoiced), cached next to the stem."""
    cache = path.with_suffix(".f0.npz" if tracker == "pyin" else f".{tracker}2.npz")
    if cache.exists():
        d = np.load(cache)
        times, f0 = d["times"], d["f0"]
        if "periodicity" in d:
            f0 = np.where(d["periodicity"] > CREPE_VOICED, f0, np.nan)
    else:
        import librosa

        print(f"[transcribe] {path.stem}: tracking pitch with {tracker} ...")
        y, sr = librosa.load(str(path), sr=16000, mono=True)
        if tracker == "crepe":
            f0, periodicity = _crepe(y, sr, 320)
            times = librosa.times_like(f0, sr=sr, hop_length=320)
            np.savez(cache, times=times, f0=f0, periodicity=periodicity)
            f0 = np.where(periodicity > CREPE_VOICED, f0, np.nan)
        else:
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


FRAME = 0.02            # s per pitch-track frame (hop 320 at 16 kHz)
FLICK = (0.04, 0.14)    # s: how long a quick turn away from the note lasts
GRACE_WINDOW = 0.15     # s: a held neighbour note this close to the attack is a grace note
ORNAMENT_EVERY = 16.0   # beats: at most one ornament per 4 bars, so they stay slight


def sung_ornaments(vocals: Path, to_sec, tune: list[Note], tracker: str = "pyin",
                   min_len: float = 0.75, every: float = ORNAMENT_EVERY) -> dict[int, tuple[str, int]]:
    """The singer's little turns that the notes smooth away, per tune note index:
    ("grace", semitones) when the voice lands on a neighbour note first and steps onto the note,
    ("upper"/"lower", 0) for a quick flick up or down and back within the note (a mordent),
    ("trill", semitones) when it flicks to the same neighbour and back 3+ times.
    Only notes of `min_len` beats or more, the clearest one per `every` beats."""
    times, midi, _ = _pitch_track(vocals, tracker)
    found: dict[int, tuple[float, int, tuple[str, int]]] = {}
    for i, n in enumerate(tune):
        if n.end - n.start < min_len:
            continue
        s, e = to_sec(n.start), to_sec(n.end)
        a, b = np.searchsorted(times, [s - GRACE_WINDOW / 2, e])
        seg = midi[a:b]
        if len(seg) < 8 or np.mean(np.isnan(seg)) > 0.3:
            continue
        # distance from the written note, folded to the nearest octave (the tune may sit an
        # octave or two from the voice)
        d = (seg - n.pitch + 6) % 12 - 6
        step = np.where(np.isnan(d), 0, np.round(d)).astype(int)
        on = np.abs(d) < 0.5
        runs, k = [], 0
        while k < len(step):  # runs of frames on one neighbour note (1-2 semitones away)
            j = k
            while j < len(step) and step[j] == step[k]:
                j += 1
            if step[k] in (-2, -1, 1, 2):
                runs.append((k, j, int(step[k])))
            k = j
        # back and forth to a neighbour 3+ times in quick flicks: the voice is trilling
        quick = [(k, j, st) for k, j, st in runs if FLICK[0] <= (j - k) * FRAME <= FLICK[1]]
        for first in range(len(quick) - 2):
            group = quick[first:first + 3]
            same = len({st for _, _, st in group}) == 1
            close = all(b[0] - a[1] <= int(FLICK[1] / FRAME) for a, b in zip(group, group[1:]))
            if same and close:
                slot = int(n.start // every)
                found[slot] = (1.0, i, ("trill", group[0][2]))  # outranks a single flick
                break
        else:
            group = None
        if group and found.get(int(n.start // every), (0, -1))[1] == i:
            continue
        for k, j, st in runs:
            dur = (j - k) * FRAME
            steady = np.nanstd(d[k:j]) < 0.35  # a held neighbour, not a glide through it
            settled = on[j:j + 4].sum() >= 3   # and the voice comes back onto the note
            if not (steady and settled and FLICK[0] <= dur):
                continue
            if times[a + k] < s + GRACE_WINDOW / 2 and dur <= GRACE_WINDOW + 0.05:
                orn, strength = ("grace", st), dur
            elif k > 0 and on[max(0, k - 3):k].sum() >= 2 and dur <= FLICK[1]:
                orn, strength = ("upper" if st > 0 else "lower", 0), dur
            else:
                continue
            slot = int(n.start // every)
            if slot not in found or strength > found[slot][0]:
                found[slot] = (strength, i, orn)
            break
    return {i: orn for _, i, orn in found.values()}


@lru_cache(maxsize=2)
def sung_onsets(vocals: Path) -> np.ndarray:
    """Times (s) where the singer starts a note, ignoring quiet bleed in the vocal stem."""
    import librosa

    y, sr = librosa.load(str(vocals), sr=22050, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    return np.array([t for t in onsets if rms[min(len(rms) - 1, np.searchsorted(rms_times, t))] > 0.02])


def sung_line(vocals: Path, to_sec, tracker: str = "pyin"):
    """The singing on the beat grid, for `best_melody`: line(total, step) -> (sung pitch per
    step, NaN where nobody sings; whether the singer starts a syllable in that step)."""
    times, midi, _ = _pitch_track(vocals, tracker)
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


def refine_pitch(path: Path, notes: list[Note], max_move: int = 2, tracker: str = "pyin") -> list[Note]:
    """Re-pitch notes from a pYIN f0 track: the median over each note ignores vibrato and
    slides. Only small corrections are taken, so pYIN's own octave slips can't leak in."""
    times, midi, _ = _pitch_track(path, tracker)
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


def drum_activity(drums: Path, threshold: float = 0.02) -> float:
    """Share of the song (in seconds) where the drum stem is clearly playing."""
    import librosa

    y, sr = librosa.load(str(drums), sr=11025, mono=True)
    return float(np.mean(librosa.feature.rms(y=y, hop_length=sr)[0] > threshold))


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


@lru_cache(maxsize=2)
def _band_chroma(other: Path, bass: Path) -> tuple[np.ndarray, np.ndarray]:
    """(frame times, 12 x frames) pitch-class energy of the band and the bass (at half weight),
    from the audio itself: CQT chroma per frame, weighted by how loud the frame is so silence
    and bleed don't vote."""
    import librosa

    total, times = None, None
    for path, weight in ((other, 1.0), (bass, 0.5)):
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
        rms = librosa.feature.rms(y=y, hop_length=1024)[0][:chroma.shape[1]]
        gate = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0, 1)
        part = weight * chroma[:, :len(gate)] * gate
        if total is None:
            total, times = part, librosa.times_like(part, sr=sr, hop_length=1024)
        else:
            n = min(total.shape[1], part.shape[1])
            total, times = total[:, :n] + part[:, :n], times[:n]
    return times, total


def band_chroma(other: Path, bass: Path, to_sec):
    """The band's harmony per slot, for naming chords from the audio instead of from transcribed
    notes: harmony(total, slot) -> slots x 12, on the same scale as arrange.pc_histogram."""
    times, chroma = _band_chroma(other, bass)

    def harmony(total: float, slot: float) -> np.ndarray:
        n = int(np.ceil(total / slot))
        edges = np.searchsorted(times, [to_sec(k * slot) for k in range(n + 1)])
        w = np.zeros((n, 12))
        for k in range(n):
            if edges[k + 1] > edges[k]:
                w[k] = chroma[:, edges[k]:edges[k + 1]].mean(axis=1) * slot
        return w

    return harmony


@lru_cache(maxsize=2)
def _structure_features(mix: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """(CQT in dB, MFCC, sample rate) of the recording, at a 512-sample hop."""
    import librosa

    y, sr = librosa.load(str(mix), sr=22050, mono=True)
    cqt = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=252)), ref=np.max)
    return cqt, librosa.feature.mfcc(y=y, sr=sr), sr


@lru_cache(maxsize=8)
def song_sections(mix: Path, beats: tuple[float, ...], k: int = 5, min_beats: int = 16) -> np.ndarray:
    """Section label per beat (0..k-1): passages that sound alike (chords, timbre) share a label,
    so a chorus gets the same label every time it comes back. Laplacian structural segmentation
    (McFee & Ellis 2014) on beat-synchronous CQT and MFCC; sections under `min_beats` merge into
    the one before."""
    import librosa
    import scipy.ndimage
    import scipy.sparse.csgraph
    from sklearn.cluster import KMeans

    cqt, mfcc, sr = _structure_features(mix)
    frames = librosa.time_to_frames(np.array(beats), sr=sr, hop_length=512)
    csync = librosa.util.sync(cqt, frames, aggregate=np.median)
    rec = librosa.segment.recurrence_matrix(csync, width=3, mode="affinity", sym=True)
    rec = librosa.segment.timelag_filter(scipy.ndimage.median_filter)(rec, size=(1, 7))
    msync = librosa.util.sync(mfcc, frames)
    dist = np.sum(np.diff(msync, axis=1) ** 2, axis=0)
    sim = np.exp(-dist / np.median(dist))
    path = np.diag(sim, k=1) + np.diag(sim, k=-1)
    d_path, d_rec = path.sum(axis=1), rec.sum(axis=1)
    mu = d_path.dot(d_path + d_rec) / np.sum((d_path + d_rec) ** 2)
    lap = scipy.sparse.csgraph.laplacian(mu * rec + (1 - mu) * path, normed=True)
    _, vecs = np.linalg.eigh(lap)
    vecs = scipy.ndimage.median_filter(vecs, size=(9, 1))
    norm = np.cumsum(vecs ** 2, axis=1) ** 0.5
    x = vecs[:, :k] / (norm[:, k - 1:k] + 1e-9)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)[1:len(beats)]  # col 0: before beat 1
    i = 0
    while i < len(labels):  # short sections are fragments of their neighbour
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i < min_beats and i > 0:
            labels[i:j] = labels[i - 1]
        i = j
    return labels

