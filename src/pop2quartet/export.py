"""Stage 4: write the quartet as MusicXML (sheet music), MIDI, and a rough audio preview."""
import contextlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .transcribe import Note

PARTS = [  # key, display name, GM program, clef
    ("vln1", "Violin I", 40, "treble"),
    ("vln2", "Violin II", 40, "treble"),
    ("vla", "Viola", 41, "alto"),
    ("vc", "Violoncello", 42, "bass"),
]
MUSESCORE = [shutil.which("mscore4"), shutil.which("mscore"), shutil.which("musescore"),
             "/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe"]  # last: Windows, via WSL
SOUNDFONTS = ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2"]


LEVELS = ["pp", "p", "mp", "mf", "f"]
SOFTER = {"vln2": 1, "vla": 1}  # the accompaniment sits a dynamic level under the tune and bass


def level_at(plan, part: str, t: float) -> int:
    """LEVELS index for `part` at beat `t`, from (start beat, tune level) sections."""
    lv = next((lv for start, lv in reversed(plan) if start <= t), plan[0][1])
    return max(0, lv - SOFTER.get(part, 0))


def write_midi(parts: dict[str, list[Note]], bpm: float, path: Path, plan=None) -> None:
    import pretty_midi

    sec = 60.0 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for key, name, program, _ in PARTS:
        inst = pretty_midi.Instrument(program=program, name=name)
        for n in parts[key]:
            vel = 40 + 70 * min(1.0, n.velocity)
            if plan:  # +-8 per dynamic level around mf
                vel += 8 * (level_at(plan, key, n.start) - LEVELS.index("mf"))
            vel = int(min(127, max(1, vel)))
            inst.notes.append(pretty_midi.Note(vel, n.pitch, n.start * sec, n.end * sec))
        pm.instruments.append(inst)
    pm.write(str(path))


# Krumhansl-Kessler key profiles, starting from the tonic.
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NAMES = ["C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]


def detect_key(parts: dict[str, list[Note]], slack: float = 0.02) -> tuple[str, str]:
    """Duration-weighted pitch-class histogram correlated against every major/minor key, among
    the keys whose scale holds (nearly) as much of the music as the best-fitting one. The profile
    alone confuses keys a third apart (C major vs E minor); a scale that lacks the song's notes
    can't be its key."""
    import numpy as np

    hist = np.zeros(12)
    for ns in parts.values():
        for n in ns:
            hist[n.pitch % 12] += n.end - n.start
    steps = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
    fit = {(t, mode): sum(hist[(t + i) % 12] for i in steps[mode]) / (hist.sum() or 1)
           for t in range(12) for mode in steps}
    top = max(fit.values())
    best = max(((np.corrcoef(hist, np.roll(prof, t))[0, 1], NAMES[t], mode)
                for prof, mode in ((MAJOR, "major"), (MINOR, "minor")) for t in range(12)
                if fit[(t, mode)] >= top - slack))
    return best[1], best[2]


DYNAMICS = {"vln1": "mf", "vln2": "p", "vla": "p", "vc": "mf"}  # notation players actually follow
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SUFFIX = {"major": "", "minor": "m", "maj7": "maj7", "7": "7", "m7": "m7"}
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


BOW = 4.0  # longest slur, in beats: what one bow stroke comfortably carries


def slur_groups(notes: list[Note], skip=frozenset()) -> list[list[int]]:
    """Note indices to play under one bow: connected notes (no rest between), at most BOW beats,
    broken before a repeated pitch (a re-attack needs a new bow) and around plucked notes."""
    groups, run = [], []
    for i, n in enumerate(notes):
        prev = notes[run[-1]] if run else None
        if i in skip or (prev and (abs(prev.end - n.start) > 1e-6 or prev.pitch == n.pitch
                                   or n.end - notes[run[0]].start > BOW)):
            groups.append(run)
            run = []
        if i not in skip:
            run.append(i)
    groups.append(run)
    return [g for g in groups if len(g) >= 2]


def pizz_notes(notes: list[Note], plan, short: float = 1.0, share: float = 0.7) -> set[int]:
    """Accompaniment notes to pluck: in the song's quietest dynamic sections, where the part is
    mostly rhythm (short re-strikes), pizzicato sets the verse apart from the bowed chorus."""
    if not plan:
        return set()
    quiet = min(lv for _, lv in plan)
    ends = [s for s, _ in plan[1:]] + [float("inf")]
    out: set[int] = set()
    for (start, lv), end in zip(plan, ends):
        idx = [i for i, n in enumerate(notes) if start <= n.start < end]
        if lv == quiet and idx and sum(notes[i].end - notes[i].start <= short for i in idx) >= share * len(idx):
            out.update(idx)
    return out


def phrase_arcs(notes: list[Note], busy: list[tuple[float, float]], breath: float = 0.5,
                landing: float = 1.5, min_len: float = 3.0) -> list[tuple[int, int, int]]:
    """(first, peak, last) note indices per sung phrase, for a swell up to the phrase's highest
    note and back. A phrase ends at a breath (a rest of `breath` beats) or on a note held for
    `landing` beats. Phrases under `min_len` beats, or overlapping a section hairpin (`busy`
    beat spans), stay flat."""
    arcs, start = [], 0
    for i in range(1, len(notes) + 1):
        if (i < len(notes) and notes[i].start - notes[i - 1].end < breath
                and notes[i - 1].end - notes[i - 1].start < landing):
            continue
        a, b = start, i - 1
        start = i
        if b - a < 2 or notes[b].end - notes[a].start < min_len:
            continue
        if any(notes[a].start < e and notes[b].end > s for s, e in busy):
            continue
        peak = max(range(a, b + 1), key=lambda k: notes[k].pitch)
        if a < peak < b:
            arcs.append((a, peak, b))
    return arcs


def write_musicxml(parts: dict[str, list[Note]], bpm: float, path: Path, title: str,
                   chords=(), plan=None, expression: bool = True, ending: bool = False,
                   accents: bool = False, rubato: bool = False) -> str:
    """`chords` are (beat, (root pitch class, quality)) changes, shown above violin I.
    `plan` is (start beat, LEVELS index) dynamic sections; changes get a 2-bar hairpin into them.
    `expression`: slurs, pizzicato verses for the inner voices, and a swell over each sung phrase.
    `ending`: everyone holds the last chord together, under a fermata, after a 2-bar ritardando.
    `accents`: the inner voices lean on the backbeat (2 and 4) and play quick notes short.
    `rubato` (for the audio preview; it fills the score with tempo marks): the tempo gives a
    little at the end of each sung phrase, as players breathe."""
    from music21 import (articulations, clef, dynamics, expressions, harmony, instrument, metadata,
                         meter, note, spanner, stream, tempo)
    from music21 import key as m21key

    instruments = {"vln1": instrument.Violin, "vln2": instrument.Violin,
                   "vla": instrument.Viola, "vc": instrument.Violoncello}
    clefs = {"treble": clef.TrebleClef, "alto": clef.AltoClef, "bass": clef.BassClef}

    total = max((n.end for ns in parts.values() for n in ns), default=4.0)
    total = 4.0 * -(-total // 4)  # round up to a whole bar
    if ending:  # last notes that end in the final 2 bars ring on to the end, together
        parts = {k: ns[:-1] + [Note(ns[-1].start, total, ns[-1].pitch, ns[-1].velocity)]
                 if ns and ns[-1].end >= total - 8 else ns for k, ns in parts.items()}

    def hidden_tempo(factor: float):
        mark = tempo.MetronomeMark(number=round(bpm * factor))
        mark.style.hideObjectOnPrint = True
        return mark

    detected = m21key.Key(*detect_key(parts))
    use_flats = detected.sharps < 0

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title, composer="arr. pop2quartet")
    for k, name, _, clef_name in PARTS:
        part = stream.Part()
        inst = instruments[k]()
        inst.partName = name
        part.insert(0, inst)
        part.insert(0, clefs[clef_name]())
        part.insert(0, m21key.Key(detected.tonic.name, detected.mode))
        part.insert(0, meter.TimeSignature("4/4"))
        if not plan:
            part.insert(0, dynamics.Dynamic(DYNAMICS[k]))
        else:
            marks = [(start, LEVELS[level_at(plan, k, start)]) for start, _ in plan]
            for i, (start, mark) in enumerate(marks):
                if i == 0 or mark != marks[i - 1][1]:
                    part.insert(start, dynamics.Dynamic(mark))
        if k == "vln1":
            part.insert(0, tempo.MetronomeMark(number=round(bpm)))
            names = FLAT_NAMES if use_flats else SHARP_NAMES
            for beat, (root, quality) in chords:
                part.insert(beat, harmony.ChordSymbol(names[root] + SUFFIX[quality]))
        placed = []
        for n in parts[k]:
            m = note.Note(n.pitch, quarterLength=n.end - n.start)
            if use_flats and m.pitch.accidental is not None and m.pitch.accidental.name == "sharp":
                m.pitch = m.pitch.getEnharmonic()
            part.insert(n.start, m)
            placed.append((n.start, m))
        busy = []  # beat spans of the section hairpins
        if plan:
            for (s0, _), (s1, _) in zip(plan, plan[1:]):
                a, b = level_at(plan, k, s0), level_at(plan, k, s1)
                span = [m for t, m in placed if max(s0, s1 - 8) <= t < s1]
                if a != b and len(span) >= 2:
                    wedge = dynamics.Crescendo if b > a else dynamics.Diminuendo
                    part.insert(0, wedge(span[0], span[-1]))
                    busy.append((max(s0, s1 - 8), s1))
        if expression:
            notes = parts[k]
            plucked = pizz_notes(notes, plan) if k in ("vln2", "vla") else set()
            for i in sorted(plucked):  # mark where plucking starts and where the bow returns
                if i - 1 not in plucked:
                    part.insert(notes[i].start, expressions.TextExpression("pizz."))
                if i + 1 < len(notes) and i + 1 not in plucked:
                    part.insert(notes[i + 1].start, expressions.TextExpression("arco"))
            for g in slur_groups(notes, plucked):
                part.insert(0, spanner.Slur([placed[i][1] for i in g]))
            if k == "vln1":
                for a, peak, b in phrase_arcs(notes, busy):
                    part.insert(0, dynamics.Crescendo(placed[a][1], placed[peak][1]))
                    part.insert(0, dynamics.Diminuendo(placed[peak][1], placed[b][1]))
        if accents and k in ("vln2", "vla"):
            for n, (_, m) in zip(parts[k], placed):
                if n.end - n.start <= 1 and n.start % 2 == 1:  # beats 2 and 4
                    m.articulations.append(articulations.Accent())
                elif n.end - n.start <= 0.5:
                    m.articulations.append(articulations.Staccato())
        if k == "vln1" and rubato:  # a breath at each phrase end, a tempo again on the next note
            notes = parts[k]
            for a, peak, b in phrase_arcs(notes, []):
                if notes[b].end < total - 8:
                    part.insert(notes[b].start, hidden_tempo(0.94))
                    part.insert(notes[b].end, hidden_tempo(1.0))
        if ending:
            if placed:
                placed[-1][1].expressions.append(expressions.Fermata())
            if k == "vln1":
                part.insert(total - 8, expressions.TextExpression("rit."))
                for i in range(1, 8):  # one step per beat, down to ~70% of the tempo
                    part.insert(total - 8 + i, hidden_tempo(1 - 0.043 * i))
        part.makeRests(refStreamOrTimeRange=[0.0, total], fillGaps=True, inPlace=True)
        part = part.makeMeasures()
        part.makeTies(inPlace=True)
        score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return f"{detected.tonic.name} {detected.mode}"


# Muse Sounds solo players by part (Muse Strings ids). MuseScore gives every violin part the same
# player, so the two violins would play identical samples; a project file can pick each one.
# The last value is where each sits, -1 = far left: the usual quartet half circle (the cello
# nearer the middle, as the bass carries the most energy).
MUSE_SOLOISTS = [
    ("violin", "103", "Violin 1 (Solo)", "strings.violin.orchestral:primary", -0.5),
    ("violin", "104", "Violin 2 (Solo)", "strings.violin.orchestral:secondary", -0.2),
    ("viola", "105", "Viola (Solo)", "strings.viola.orchestral", 0.2),
    ("violoncello", "106", "Violoncello (Solo)", "strings.violoncello.orchestral", 0.3),
]


PLAYING = {"pizz.": "pizzicato", "arco": "normal"}  # MuseScore's playing-technique names


def prepare_project(mscz: Path) -> None:
    """Set up a MuseScore project converted from our MusicXML: each part's player and seat
    (MUSE_SOLOISTS, parts 1-4 in score order), and "pizz."/"arco", which MusicXML import leaves
    as plain text, turned into playing techniques so the sound changes."""
    with zipfile.ZipFile(mscz) as z:
        files = {n: z.read(n) for n in z.namelist()}
    for n in files:
        if n.endswith(".mscx"):
            files[n] = re.sub(
                r"<StaffText>(\s*<eid>[^<]*</eid>)?\s*<text>(pizz\.|arco)</text>\s*</StaffText>",
                lambda m: f"<PlayTechAnnotation>{m[1] or ''}<playTechType>{PLAYING[m[2]]}"
                          f"</playTechType><text>{m[2]}</text></PlayTechAnnotation>",
                files[n].decode()).encode()
    settings = json.loads(files.get("audiosettings.json", b"{}") or b"{}")
    settings["activeSoundProfile"] = "MuseSounds"
    settings["tracks"] = [{
        "in": {"resourceMeta": {
            "attributes": {"museCategory": "Muse Strings", "museName": name, "musePack": "Muse Strings",
                           "museUID": uid, "museVendorName": "Muse", "playbackSetupData": setup},
            "hasNativeEditorSupport": False, "id": uid, "type": "muse_sampler_sound_pack",
            "vendor": "MuseSounds"}, "unitConfiguration": {}},
        "instrumentId": instrument, "partId": str(i + 1),
        "out": {"balance": pan, "fxChain": {}, "volumeDb": 0},
        "soloMuteState": {"mute": False, "solo": False},
    } for i, (instrument, uid, name, setup, pan) in enumerate(MUSE_SOLOISTS)]
    files["audiosettings.json"] = json.dumps(settings, indent=4).encode()
    with zipfile.ZipFile(mscz, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)


@contextlib.contextmanager
def one_at_a_time():
    """Several songs may convert at once, but MuseScore (a desktop app) renders one at a time."""
    try:
        import fcntl
    except ImportError:  # Windows: no flock, and nothing to coordinate with
        yield
        return
    with open(Path(tempfile.gettempdir()) / "pop2quartet-musescore.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def render_audio(score: Path, midi: Path, wav: Path) -> str | None:
    """Render the preview: MuseScore plays the score (with Muse Sounds when installed, and it
    follows the dynamics), fluidsynth plays the MIDI as a fallback. Returns which one ran."""
    mscore = next((p for p in MUSESCORE if p and Path(p).exists()), None)
    if mscore:
        def native(p: Path) -> str:  # a Windows MuseScore under WSL needs Windows paths
            if not mscore.endswith(".exe"):
                return str(p)
            return subprocess.run(["wslpath", "-w", str(p.resolve())], check=True,
                                  capture_output=True, text=True).stdout.strip()
        wav.unlink(missing_ok=True)  # so a stale preview can't pass for a fresh one
        project = score.with_suffix(".mscz")  # to choose each part's player (MUSE_SOLOISTS)
        with one_at_a_time():
            made = subprocess.run([mscore, "-o", native(project), native(score)], capture_output=True)
            if made.returncode == 0 and project.exists():
                prepare_project(project)
                score = project
            done = subprocess.run([mscore, "-o", native(wav), native(score)], capture_output=True)
        project.unlink(missing_ok=True)
        if done.returncode == 0 and wav.exists():
            return "MuseScore"
    sf2 = next((p for p in SOUNDFONTS if Path(p).exists()), None)
    if not (shutil.which("fluidsynth") and sf2):
        return None
    subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", "44100", sf2, str(midi)],
                   check=True, stdout=subprocess.DEVNULL)
    return "fluidsynth"


ROOMS = {"studio": None, "hall": (1.8, 0.22), "church": (3.2, 0.32)}  # (seconds to fade 60 dB, wet share)
ROOM_DRUMS = [(0.30, "church"), (0.45, "hall")]  # drums play less of the song -> this room


def room_for(drums_active: float) -> str:
    """Ballads (few or no drums) ring in a church, sparse grooves in a hall, and drum-driven songs
    stay close (studio), as long reverb blurs the groove. Drums are a better guide than the tempo,
    which trackers often read at double (a ballad's hi-hat eighths)."""
    return next((room for limit, room in ROOM_DRUMS if drums_active < limit), "studio")


def _biquad(kind: str, f0: float, gain_db: float, q: float, sr: int):
    """RBJ cookbook EQ: a peaking bell or a high shelf, as (b, a) filter coefficients."""
    A, w = 10 ** (gain_db / 40), 2 * np.pi * f0 / sr
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:  # high shelf
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array(b) / a[0], np.array(a) / a[0]


def ride_level(y: np.ndarray, sr: int, ratio: float = 3.0, window: float = 1.0,
               max_boost: float = 8.0) -> np.ndarray:
    """Even out the loudness like a mixing engineer riding the faders: the gap between the
    smoothed loudness and the song's typical level shrinks by `ratio`. Muse Sounds takes each
    dynamic mark and every entry of a voice at face value, so the quartet swings ~3x as far as
    the recording. Offline, so the gain eases in around a change instead of lagging it, and
    near-silence (breaks, the very start and end) is never pulled up."""
    hop = int(0.1 * sr)
    frames = np.sqrt(np.add.reduceat((y ** 2).mean(axis=1), np.arange(0, len(y), hop))
                     / np.diff(np.append(np.arange(0, len(y), hop), len(y))))
    db = 20 * np.log10(frames + 1e-9)
    k = max(1, int(window / 0.1))
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    typical = np.median(smooth[smooth > smooth.max() - 30])
    gain = (typical - smooth) * (1 - 1 / ratio)
    gain = np.clip(gain, -12, max_boost)
    gain[smooth < typical - 30] = 0.0  # leave real breaks and fades alone
    gain = np.convolve(gain, np.ones(k) / k, mode="same")  # ease every change in
    t = np.arange(len(gain)) * hop + hop / 2
    return y * 10 ** (np.interp(np.arange(len(y)), t, gain) / 20)[:, None]


def add_room(wav: Path, room: str = "studio", level_db: float = -20.0, master: bool = False) -> None:
    """Put the preview in a room (reverb: the recording convolved with a decaying, slightly dark
    noise echo per ear) and bring it to a steady listening level (RMS `level_db`, peaks < 0.95)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, fftconvolve, lfilter

    y, sr = sf.read(str(wav), always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    out = y
    if ROOMS.get(room):
        decay, wet = ROOMS[room]
        t = np.arange(int(decay * sr)) / sr
        tail = np.random.default_rng(0).standard_normal((len(t), 2)) * 10 ** (-3 * t / decay)[:, None]
        tail = lfilter(*butter(2, 6000 / (sr / 2)), tail, axis=0)  # air absorbs the highs
        ir = np.vstack([np.zeros((int(0.02 * sr), 2)), tail])  # 20 ms before the first echo
        ir /= np.sqrt((ir ** 2).sum(axis=0))
        room_y = np.stack([fftconvolve(y[:, c], ir[:, c])[:len(y)] for c in range(2)], axis=1)
        room_y *= np.sqrt((y ** 2).mean() / max((room_y ** 2).mean(), 1e-12))
        out = (1 - wet) * y + wet * room_y
    out = ride_level(out, sr)
    if master:  # less low-mid boxiness, a touch of air, and 2 dB louder with peaks rounded off
        for kind, f0, g, q in (("peak", 300, -2.5, 0.9), ("shelf", 9000, 2.0, 0.7)):
            out = lfilter(*_biquad(kind, f0, g, q, sr), out, axis=0)
        out *= 10 ** ((level_db + 2) / 20) / max(np.sqrt((out ** 2).mean()), 1e-9)
        out = np.where(np.abs(out) > 0.8, np.sign(out) * (0.8 + 0.19 * np.tanh((np.abs(out) - 0.8) / 0.19)), out)
        sf.write(str(wav), out, sr)
        return
    rms = np.sqrt((out ** 2).mean())
    gain = min(10 ** (level_db / 20) / max(rms, 1e-9), 0.95 / max(np.abs(out).max(), 1e-9))
    sf.write(str(wav), out * gain, sr)
