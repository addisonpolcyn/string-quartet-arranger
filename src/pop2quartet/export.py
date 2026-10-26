"""Stage 4: write the quartet as MusicXML (sheet music), MIDI, and a rough audio preview."""
import shutil
import subprocess
from pathlib import Path

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


def write_musicxml(parts: dict[str, list[Note]], bpm: float, path: Path, title: str,
                   chords=(), plan=None) -> str:
    """`chords` are (beat, (root pitch class, quality)) changes, shown above violin I.
    `plan` is (start beat, LEVELS index) dynamic sections; changes get a 2-bar hairpin into them."""
    from music21 import clef, dynamics, harmony, instrument, metadata, meter, note, stream, tempo
    from music21 import key as m21key

    instruments = {"vln1": instrument.Violin, "vln2": instrument.Violin,
                   "vla": instrument.Viola, "vc": instrument.Violoncello}
    clefs = {"treble": clef.TrebleClef, "alto": clef.AltoClef, "bass": clef.BassClef}

    total = max((n.end for ns in parts.values() for n in ns), default=4.0)
    total = 4.0 * -(-total // 4)  # round up to a whole bar

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
        if plan:
            for (s0, _), (s1, _) in zip(plan, plan[1:]):
                a, b = level_at(plan, k, s0), level_at(plan, k, s1)
                span = [m for t, m in placed if max(s0, s1 - 8) <= t < s1]
                if a != b and len(span) >= 2:
                    wedge = dynamics.Crescendo if b > a else dynamics.Diminuendo
                    part.insert(0, wedge(span[0], span[-1]))
        part.makeRests(refStreamOrTimeRange=[0.0, total], fillGaps=True, inPlace=True)
        part = part.makeMeasures()
        part.makeTies(inPlace=True)
        score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return f"{detected.tonic.name} {detected.mode}"


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
        done = subprocess.run([mscore, "-o", native(wav), native(score)], capture_output=True)
        if done.returncode == 0 and wav.exists():
            return "MuseScore"
    sf2 = next((p for p in SOUNDFONTS if Path(p).exists()), None)
    if not (shutil.which("fluidsynth") and sf2):
        return None
    subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", "44100", sf2, str(midi)],
                   check=True, stdout=subprocess.DEVNULL)
    return "fluidsynth"
