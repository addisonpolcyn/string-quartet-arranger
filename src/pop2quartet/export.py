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
COUNTER_VELOCITY = 0.35  # marks violin II's answers to the tune: written a level under its chords
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


SLIDE = 0.25  # beats a slide takes, at the end of the note it leaves
SLIDE_LEAP = (4, 12)  # semitones up: from a major 3rd to an octave
SLIDE_EVERY = 8.0  # beats: at most one slide per two bars, so it stays a gesture


def slide_notes(notes: list[Note]) -> set[int]:
    """Indices of tune notes that slide up into the next one, the way a singer scoops: the next
    note follows without a rest, a leap of SLIDE_LEAP up, both notes held a beat or more."""
    out, last = set(), -SLIDE_EVERY
    for i, (a, b) in enumerate(zip(notes, notes[1:])):
        if (abs(a.end - b.start) < 1e-6 and SLIDE_LEAP[0] <= b.pitch - a.pitch <= SLIDE_LEAP[1]
                and a.end - a.start >= 1 and b.end - b.start >= 1 and a.start - last >= SLIDE_EVERY):
            out.add(i)
            last = a.start
    return out


def preview_seconds(beats, bpm: float, marks: list) -> np.ndarray:
    """Where beats fall in the rendered preview, following its hidden (beat, factor) tempo marks."""
    at_b, at_t, b0, f, t = [0.0], [0.0], 0.0, 1.0, 0.0
    for b, nf in sorted(marks):
        t += (b - b0) * 60 / (round(bpm) * f)
        at_b.append(b)
        at_t.append(t)
        b0, f = b, nf
    at_b.append(b0 + 1e4)
    at_t.append(t + 1e4 * 60 / (round(bpm) * f))
    return np.interp(beats, at_b, at_t)


def write_musicxml(parts: dict[str, list[Note]], bpm: float, path: Path, title: str,
                   chords=(), plan=None, expression: bool = True, ending: bool = False,
                   accents: bool = False, rubato: bool = False, slides: bool = False,
                   tempo_marks: list | None = None, swells: bool = True) -> str:
    """`chords` are (beat, (root pitch class, quality)) changes, shown above violin I.
    `plan` is (start beat, LEVELS index) dynamic sections; changes get a 2-bar hairpin into them.
    `expression`: slurs, pizzicato verses for the inner voices, and (with `swells`) a swell over
    each sung phrase.
    `ending`: everyone holds the last chord together, under a fermata, after a 2-bar ritardando.
    `accents`: the inner voices lean on the backbeat (2 and 4) and play quick notes short.
    `rubato` (for the audio preview; it fills the score with tempo marks): the tempo gives a
    little at the end of each sung phrase, as players breathe.
    `slides` (for the audio preview): violin I slides up into a held note after a leap, as a
    singer would (see slide_notes).
    `tempo_marks`, if given, collects the hidden (beat, tempo factor) changes, to time the audio."""
    from music21 import (articulations, clef, dynamics, expressions, harmony, instrument, metadata,
                         meter, note, spanner, stream, tempo, tie)
    from music21 import key as m21key

    instruments = {"vln1": instrument.Violin, "vln2": instrument.Violin,
                   "vla": instrument.Viola, "vc": instrument.Violoncello}
    clefs = {"treble": clef.TrebleClef, "alto": clef.AltoClef, "bass": clef.BassClef}

    total = max((n.end for ns in parts.values() for n in ns), default=4.0)
    total = 4.0 * -(-total // 4)  # round up to a whole bar
    if ending:  # last notes that end in the final 2 bars ring on to the end, together
        parts = {k: ns[:-1] + [Note(ns[-1].start, total, ns[-1].pitch, ns[-1].velocity)]
                 if ns and ns[-1].end >= total - 8 else ns for k, ns in parts.items()}

    def hidden_tempo(factor: float, beat: float):
        if tempo_marks is not None:
            tempo_marks.append((beat, factor))
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
        placed, tails = [], {}
        slide = slide_notes(parts[k]) if slides and k == "vln1" else set()
        for i, n in enumerate(parts[k]):
            # a slide is played on the note's last moment: Muse Sounds glides through the whole
            # note it starts on, so the slide starts on a short tied tail
            head = n.end - n.start - (SLIDE if i in slide else 0)
            m = note.Note(n.pitch, quarterLength=head)
            if use_flats and m.pitch.accidental is not None and m.pitch.accidental.name == "sharp":
                m.pitch = m.pitch.getEnharmonic()
            part.insert(n.start, m)
            placed.append((n.start, m))
            if i in slide:
                tails[i] = note.Note(m.pitch, quarterLength=SLIDE)
                m.tie, tails[i].tie = tie.Tie("start"), tie.Tie("stop")
                part.insert(n.start + head, tails[i])
        for i, tail in tails.items():
            part.insert(0, spanner.Glissando([tail, placed[i + 1][1]]))
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
            if k == "vln1" and swells:
                for a, peak, b in phrase_arcs(notes, busy):
                    part.insert(0, dynamics.Crescendo(placed[a][1], placed[peak][1]))
                    part.insert(0, dynamics.Diminuendo(placed[peak][1], placed[b][1]))
        if k == "vln2":  # the answers to the tune sit a level under violin II's chords
            soft = [abs(n.velocity - COUNTER_VELOCITY) < 1e-6 for n in parts[k]]
            for i, n in enumerate(parts[k]):
                if not soft[i] or (i and soft[i - 1]):
                    continue
                j = i
                while j + 1 < len(soft) and soft[j + 1]:
                    j += 1
                lv = level_at(plan, k, n.start) if plan else LEVELS.index(DYNAMICS[k])
                part.insert(n.start, dynamics.Dynamic(LEVELS[max(0, lv - 1)]))
                if j + 1 < len(soft):
                    part.insert(parts[k][j + 1].start, dynamics.Dynamic(LEVELS[lv]))
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
                    part.insert(notes[b].start, hidden_tempo(0.94, notes[b].start))
                    part.insert(notes[b].end, hidden_tempo(1.0, notes[b].end))
        if ending:
            if placed:
                placed[-1][1].expressions.append(expressions.Fermata())
            if k == "vln1":
                part.insert(total - 8, expressions.TextExpression("rit."))
                for i in range(1, 8):  # one step per beat, down to ~70% of the tempo
                    part.insert(total - 8 + i, hidden_tempo(1 - 0.043 * i, total - 8 + i))
        part.makeRests(refStreamOrTimeRange=[0.0, total], fillGaps=True, inPlace=True)
        part = part.makeMeasures()
        part.makeTies(inPlace=True)
        score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return f"{detected.tonic.name} {detected.mode}"


# Muse Sounds solo players by part (Muse Strings ids). MuseScore gives every violin part the same
# player, so the two violins would play identical samples; a project file can pick each one.
# Then where each sits (-1 = far left: the usual quartet half circle, the cello nearer the middle
# as the bass carries the most energy) and its fader in dB: Muse Strings' cello sounds ~7 dB
# louder than its violin at the same dynamic, so the cello comes down and the tune up, ~1.5 dB on top.
MUSE_SOLOISTS = [
    ("violin", "103", "Violin 1 (Solo)", "strings.violin.orchestral:primary", -0.5, 6.5),
    ("violin", "104", "Violin 2 (Solo)", "strings.violin.orchestral:secondary", -0.2, 0.0),
    ("viola", "105", "Viola (Solo)", "strings.viola.orchestral", 0.2, 0.0),
    ("violoncello", "106", "Violoncello (Solo)", "strings.violoncello.orchestral", 0.3, -4.0),
]


PLAYING = {"pizz.": "pizzicato", "arco": "normal"}  # MuseScore's playing-technique names


INTONATION = {4: -10.0, 3: 8.0, 7: 2.0, 10: -8.0}  # cents, by interval above the chord's root:
# players lower a major 3rd and raise a minor 3rd toward pure intervals, lean a 7th low
LEADING_TONE = 10.0  # cents sharp: the tune's 7th degree pulls up toward the tonic
INTONATION_SPREAD = 4.0  # cents: nobody plays the same note exactly in tune twice


def intonation(parts: dict[str, list[Note]], chords, seed: int = 0) -> dict[str, list[float]]:
    """Cents off equal temperament for every note, the way a quartet tunes: by its place in the
    chord sounding at the time (INTONATION), the tune's leading tone sharp, and a small random
    spread per note. Muse Sounds otherwise plays every note dead in tune, a giveaway."""
    from bisect import bisect_right

    tonic = NAMES.index(detect_key(parts)[0])
    beats = [b for b, _ in chords]
    rng = np.random.default_rng(seed)
    out = {}
    for k, ns in parts.items():
        cents = []
        for n in ns:
            i = bisect_right(beats, n.start) - 1
            c = INTONATION.get((n.pitch - chords[i][1][0]) % 12, 0.0) if i >= 0 else 0.0
            if k == "vln1" and (n.pitch - tonic) % 12 == 11:
                c = LEADING_TONE
            cents.append(float(np.clip(c + rng.normal(0, INTONATION_SPREAD), -18, 18)))
        out[k] = cents
    return out


def tune_notes(mscx: str, tunings: list[list[float]]) -> str:
    """Write each staff's cents onto its notes, in order; a tied continuation keeps its note's
    tuning. A staff whose note count doesn't match is left in equal temperament."""
    starts = [m.start() for m in re.finditer(r'\n    <Staff id="\d+">', mscx)] + [len(mscx)]
    out = [mscx[:starts[0]]]
    for k, (a, b) in enumerate(zip(starts, starts[1:])):
        staff, cents = mscx[a:b], tunings[k] if k < len(tunings) else []
        notes = list(re.finditer(r"<Note>.*?</Note>", staff, re.S))
        tied = [bool(re.search(r'<Spanner type="Tie">\s*<prev>', m[0])) for m in notes]
        if sum(not t for t in tied) != len(cents):
            print(f"[export] intonation: staff {k + 1} has {sum(not t for t in tied)} notes, "
                  f"expected {len(cents)}; left in equal temperament")
            out.append(staff)
            continue
        pieces, at, i, c = [], 0, -1, 0.0
        for m, t in zip(notes, tied):
            if not t:
                i += 1
                c = cents[i]
            tuned = re.sub(r"(<pitch>\d+</pitch>)", rf"\1<tuning>{c:.1f}</tuning>", m[0], count=1)
            pieces += [staff[at:m.start()], tuned]
            at = m.end()
        out.append("".join(pieces) + staff[at:])
    return "".join(out)


def prepare_project(mscz: Path, tunings: list[list[float]] | None = None) -> None:
    """Set up a MuseScore project converted from our MusicXML: each part's player and seat
    (MUSE_SOLOISTS, parts 1-4 in score order), and "pizz."/"arco", which MusicXML import leaves
    as plain text, turned into playing techniques so the sound changes. `tunings`: cents per
    note for each part, in score order (see intonation); MusicXML can't carry them."""
    with zipfile.ZipFile(mscz) as z:
        files = {n: z.read(n) for n in z.namelist()}
    for n in files:
        if n.endswith(".mscx"):
            files[n] = re.sub(
                r"<StaffText>(\s*<eid>[^<]*</eid>)?\s*<text>(pizz\.|arco)</text>\s*</StaffText>",
                lambda m: f"<PlayTechAnnotation>{m[1] or ''}<playTechType>{PLAYING[m[2]]}"
                          f"</playTechType><text>{m[2]}</text></PlayTechAnnotation>",
                files[n].decode()).encode()
            if tunings:
                files[n] = tune_notes(files[n].decode(), tunings).encode()
    settings = json.loads(files.get("audiosettings.json", b"{}") or b"{}")
    settings["activeSoundProfile"] = "MuseSounds"
    settings["tracks"] = [{
        "in": {"resourceMeta": {
            "attributes": {"museCategory": "Muse Strings", "museName": name, "musePack": "Muse Strings",
                           "museUID": uid, "museVendorName": "Muse", "playbackSetupData": setup},
            "hasNativeEditorSupport": False, "id": uid, "type": "muse_sampler_sound_pack",
            "vendor": "MuseSounds"}, "unitConfiguration": {}},
        "instrumentId": instrument, "partId": str(i + 1),
        "out": {"balance": pan, "fxChain": {}, "volumeDb": fader},
        "soloMuteState": {"mute": False, "solo": False},
    } for i, (instrument, uid, name, setup, pan, fader) in enumerate(MUSE_SOLOISTS)]
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


def solo_project(mscz: Path, dst: Path, part: int) -> None:
    """A copy of a prepared project with every part but `part` (0-3) muted."""
    with zipfile.ZipFile(mscz) as z:
        files = {n: z.read(n) for n in z.namelist()}
    settings = json.loads(files["audiosettings.json"])
    for i, track in enumerate(settings["tracks"]):
        track["soloMuteState"]["mute"] = i != part
    files["audiosettings.json"] = json.dumps(settings, indent=4).encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)


def fractional_read(x: np.ndarray, at: np.ndarray, taps: int = 32) -> np.ndarray:
    """x read at fractional sample positions `at` through a Hann-windowed sinc, which keeps the
    highs (straight-line interpolation between samples dulls everything above ~8 kHz)."""
    n = np.floor(at).astype(np.int64)
    frac = (at - n).astype(np.float32)
    out = np.zeros(len(at), dtype=np.float32)
    half = taps // 2
    for k in range(1 - half, half + 1):
        d = k - frac
        w = np.sinc(d) * 0.5 * (1 + np.cos(np.pi * d / half))
        out += x[np.clip(n + k, 0, len(x) - 1)] * w
    return out


BALANCE_CUT = 8.0  # dB: the most --lead-ahead turns the accompaniment down


def balance(parts: list[np.ndarray], sr: int, target: float) -> tuple[float, float]:
    """How far violin I (parts[0]) sits above the other three together, median over the seconds it
    plays, and the fixed gain (dB, <= 0) for the other three that brings it to `target`."""
    sec = len(parts[0]) // sr
    lv = np.array([[10 * np.log10(np.mean(p[t * sr:(t + 1) * sr] ** 2) + 1e-12) for t in range(sec)]
                   for p in parts])
    rest = 10 * np.log10(sum(10 ** (lv[i] / 10) for i in range(1, len(parts))))
    ahead = float(np.median((lv[0] - rest)[lv[0] > lv[0].max() - 35]))
    return ahead, max(-BALANCE_CUT, min(0.0, ahead - target))


SLIP_AFTER_REST, SLIP_IN_LINE = 0.35, 0.12  # chance a note's attack slips: entries are loosest
SLIP_SETTLE = 0.5  # share of a slip still left on the next note: players find each other again
SLIP_XFADE = 0.01  # s: the crossfade between offsets, ending just before the attack


def note_slips(onsets: list[tuple[float, bool]], ms: float, rng) -> np.ndarray:
    """Timing offset (s) for each note of one player: mostly together (within ~1 ms); now and
    then an attack slips by up to `ms`, more often entering after a rest, and the slip settles
    over the next few notes."""
    out, prev = [], 0.0
    for _, after_rest in onsets:
        slip = rng.random() < (SLIP_AFTER_REST if after_rest else SLIP_IN_LINE)
        d = SLIP_SETTLE * prev + (rng.normal(0, ms / 2) if slip else rng.normal(0, 0.5))
        prev = float(np.clip(d, -ms, ms))
        out.append(prev / 1000)
    return np.array(out)


def slipped(x: np.ndarray, sr: int, onsets: list[tuple[float, bool]], ms: float, rng) -> np.ndarray:
    """One player's render with each attack moved by its slip. The offset changes in a short
    crossfade just before each attack, while both readings still hold the previous note, so
    there's no click and no pitch bend."""
    times = np.array([t for t, _ in onsets])
    slips = note_slips(onsets, ms, rng)
    n, idx = len(x), np.arange(len(x), dtype=float)
    cur, prev, w = np.zeros(n), np.zeros(n), np.ones(n)
    last = 0.0
    for t, d in zip(times, slips):
        a = int(max(0.0, t + min(last, d) - SLIP_XFADE - 0.002) * sr)
        b = min(n, a + int(SLIP_XFADE * sr))
        cur[a:], prev[a:] = d, last
        w[a:b] = np.linspace(0, 1, b - a)
        last = d
    read = lambda delay: np.stack([fractional_read(x[:, c], idx - delay * sr) for c in range(x.shape[1])], axis=1)
    return read(cur) * w[:, None] + read(prev) * (1 - w[:, None])


def loosen(wav: Path, stems: list[Path], ms: float, lead_ahead: float = 0.0, onsets=None) -> None:
    """Four players never land exactly together: each part drifts slowly against the others, by
    up to `ms`. MuseScore ignores per-note timing written into a project, so the parts are
    rendered one at a time, each shifted along its own slow random curve (20-40 s swings, so the
    pitch moves < 3 cents), and mixed. MuseScore's engine compresses the full mix, which no part
    gets alone, so the mix takes on the full render's moment-to-moment gain.
    `lead_ahead` (dB, 0 = off): where violin I sits less than this far above the other three
    (a busy accompaniment, like fingerpicking copied into the inner voices), they come down by
    one fixed amount for the whole song, as a mixing engineer would set the faders.
    `onsets` (per part, (seconds, after a rest) per note): instead of drifting, the players are
    together and slip on single notes, then find each other again (see note_slips)."""
    import soundfile as sf
    from scipy.ndimage import uniform_filter1d

    y, sr = sf.read(str(wav), always_2d=True, dtype="float32")
    parts = [sf.read(str(p), always_2d=True, dtype="float32")[0] for p in stems]
    n = min(len(y), *map(len, parts))
    y, parts = y[:n], [p[:n] for p in parts]
    cut = 0.0
    if lead_ahead:
        ahead, cut = balance([p.mean(axis=1) for p in parts], sr, lead_ahead)
        print(f"[export] violin I sits {ahead:+.1f} dB over the others; "
              + (f"they come down {-cut:.1f} dB" if cut else "balance left alone"))
    rng = np.random.default_rng(0)
    t, idx = np.arange(n) / sr, np.arange(n, dtype=float)
    mix = np.zeros_like(y)
    for i, p in enumerate(parts):
        if i and cut:
            p = p * 10 ** (cut / 20)
        if onsets:
            mix += slipped(p, sr, onsets[i], ms, rng)
            continue
        swings = [np.sin(2 * np.pi * t / rng.uniform(20, 40) + rng.uniform(0, 2 * np.pi)) for _ in range(3)]
        drift = 0.5 * rng.uniform(-1, 1) + 0.5 * np.mean(swings, axis=0)  # within -1..1
        at = idx - drift * ms / 1000 * sr
        mix += np.stack([fractional_read(p[:, c], at) for c in range(p.shape[1])], axis=1)

    def env(a):  # over 300 ms: a gain that follows every 10 ms flutters by ~1.3 dB, audible grit
        return np.sqrt(uniform_filter1d((a ** 2).mean(axis=1), int(0.3 * sr)) + 1e-12)

    # float, so a peak over full scale survives to the mastering stage instead of clipping here
    sf.write(str(wav), mix * (env(y) / env(sum(parts)))[:, None], sr, subtype="FLOAT")


def render_audio(score: Path, midi: Path, wav: Path, humanize: float = 0.0,
                 lead_ahead: float = 0.0, tunings: list[list[float]] | None = None,
                 onsets=None) -> str | None:
    """Render the preview: MuseScore plays the score (with Muse Sounds when installed, and it
    follows the dynamics), fluidsynth plays the MIDI as a fallback. Returns which one ran.
    `humanize` (ms, MuseScore only): the players drift apart by up to this much, and with
    `lead_ahead` the accompaniment comes down under the tune (see loosen). `tunings`: cents
    per note per part (see intonation); `onsets`: per-note slips instead of drift (see loosen)."""
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
                prepare_project(project, tunings)
                score = project
            done = subprocess.run([mscore, "-o", native(wav), native(score)], capture_output=True)
            stems = []
            if (humanize or lead_ahead) and score == project and done.returncode == 0:
                for k in range(len(MUSE_SOLOISTS)):
                    solo, stem = project.with_name(f".solo{k}.mscz"), wav.with_name(f".{wav.stem}{k}.wav")
                    solo_project(project, solo, k)
                    subprocess.run([mscore, "-o", native(stem), native(solo)], capture_output=True)
                    solo.unlink()
                    stems.append(stem)
        project.unlink(missing_ok=True)
        if done.returncode == 0 and wav.exists():
            if stems and all(p.exists() for p in stems):
                loosen(wav, stems, humanize, lead_ahead, onsets)
            for p in stems:
                p.unlink(missing_ok=True)
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
               max_boost: float = 8.0, target=None) -> np.ndarray:
    """Even out the loudness like a mixing engineer riding the faders: the gap between the
    smoothed loudness and the song's typical level shrinks by `ratio`. Muse Sounds takes each
    dynamic mark and every entry of a voice at face value, so the quartet swings ~3x as far as
    the recording. Offline, so the gain eases in around a change instead of lagging it, and
    near-silence (breaks, the very start and end) is never pulled up.
    `target` ((seconds, dB) points) steers toward a loudness curve, such as the recording's,
    instead of one steady level, so a chorus keeps its lift; the gain then also moves slowly."""
    hop = int(0.1 * sr)
    frames = np.sqrt(np.add.reduceat((y ** 2).mean(axis=1), np.arange(0, len(y), hop))
                     / np.diff(np.append(np.arange(0, len(y), hop), len(y))))
    db = 20 * np.log10(frames + 1e-9)
    k = max(1, int(window / 0.1))
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    typical = np.median(smooth[smooth > smooth.max() - 30])
    t = np.arange(len(smooth)) * hop + hop / 2
    goal = typical + (np.interp(t / sr, *target) if target is not None else 0.0)
    gain = (goal - smooth) * (1 - 1 / ratio)
    gain = np.clip(gain, -12, max_boost)
    gain[smooth < typical - 30] = 0.0  # leave real breaks and fades alone
    gain = np.convolve(gain, np.ones(k) / k, mode="same")  # ease every change in
    return y * 10 ** (np.interp(np.arange(len(y)), t, gain) / 20)[:, None]


FADE = 2.5  # seconds the final chord dies away over, as players let it go
FADE_TO = -24.0  # dB it has faded to when the bows stop; the room rings on from there


def music_end(y: np.ndarray, sr: int) -> int:
    """Sample where the music stops: the end of the last 0.1 s within 20 dB of its typical level."""
    hop = int(0.1 * sr)
    frames = np.sqrt(np.add.reduceat((y ** 2).mean(axis=1), np.arange(0, len(y), hop)) / hop)
    db = 20 * np.log10(frames + 1e-9)
    typical = np.median(db[db > db.max() - 30])
    return min(len(y), (int(np.nonzero(db > typical - 20)[0][-1]) + 1) * hop)


def phrase_swell(n: int, sr: int, phrases, depth_db: float) -> np.ndarray:
    """Gain (dB) per sample: up `depth_db` from each (start, peak, end) phrase's start to its peak
    and back down by its end, on cosine curves."""
    g = np.zeros(n)
    for s, p, e in phrases:
        s, p, e = (min(n, int(x * sr)) for x in (s, p, e))
        if s < p:
            g[s:p] = np.maximum(g[s:p], depth_db * (1 - np.cos(np.linspace(0, np.pi, p - s))) / 2)
        if p < e:
            g[p:e] = np.maximum(g[p:e], depth_db * (1 + np.cos(np.linspace(0, np.pi, e - p))) / 2)
    return g


ROOM_IRS = {  # room -> (impulse response recorded there, wet share): Voxengo's free IR pack
    "studio": ("French 18th Century Salon.wav", 0.15),  # a real chamber-music salon, just a touch
    "hall": ("Musikvereinsaal.wav", 0.22),  # Vienna's hall, famous for strings
    "church": ("St Nicolaes Church.wav", 0.32),
}


def room_response(room: str, sr: int, irs: Path | None) -> tuple[np.ndarray, float] | None:
    """(stereo impulse response, wet share) for `room`: recorded in a real space when `irs` holds
    it (ROOM_IRS), else ROOMS' decaying, slightly dark noise; None for no room."""
    import soundfile as sf
    from scipy.signal import butter, lfilter, resample_poly

    if irs and room in ROOM_IRS and (irs / ROOM_IRS[room][0]).exists():
        ir, ir_sr = sf.read(str(irs / ROOM_IRS[room][0]), always_2d=True)
        if ir_sr != sr:
            ir = resample_poly(ir, sr, ir_sr, axis=0)
        ir = np.repeat(ir, 2, axis=1) if ir.shape[1] == 1 else ir[:, :2]
        wet = ROOM_IRS[room][1]
    elif ROOMS.get(room):
        decay, wet = ROOMS[room]
        t = np.arange(int(decay * sr)) / sr
        tail = np.random.default_rng(0).standard_normal((len(t), 2)) * 10 ** (-3 * t / decay)[:, None]
        tail = lfilter(*butter(2, 6000 / (sr / 2)), tail, axis=0)  # air absorbs the highs
        ir = np.vstack([np.zeros((int(0.02 * sr), 2)), tail])  # 20 ms before the first echo
    else:
        return None
    return ir / np.sqrt((ir ** 2).sum(axis=0)), wet


def add_room(wav: Path, room: str = "studio", level_db: float = -20.0, master: bool = False,
             contour=None, swells=(), swell_db: float = 0.0, irs: Path | None = None) -> None:
    """Put the preview in a room (reverb: the recording convolved with a decaying, slightly dark
    noise echo per ear) and bring it to a steady listening level (RMS `level_db`, peaks < 0.95).
    `contour` ((seconds, dB) points): the loudness to follow instead of one level (ride_level).
    The last chord fades over FADE seconds and the room rings out after it, instead of the
    music stopping dead where the render ends. `swells` ((start, peak, end) seconds per sung
    phrase): the whole quartet swells `swell_db` up to each phrase's peak and back, together, so
    the balance never changes (Muse Sounds plays a written hairpin a full dynamic level, ~6 dB,
    which buried the tune at every phrase end). `irs`: a folder of recorded impulse responses
    (ROOM_IRS), for a real room instead of the noise one."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve, lfilter

    y, sr = sf.read(str(wav), always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    end = music_end(y, sr)
    response = room_response(room, sr, irs)
    ring = (len(response[0]) / sr if response else 0.0) + 0.5
    y = np.vstack([y, np.zeros((int(ring * sr), 2))])  # room for the reverb's tail
    out = y
    if response:
        ir, wet = response
        room_y = np.stack([fftconvolve(y[:, c], ir[:, c])[:len(y)] for c in range(2)], axis=1)
        room_y *= np.sqrt((y ** 2).mean() / max((room_y ** 2).mean(), 1e-12))
        out = (1 - wet) * y + wet * room_y
    out = ride_level(out, sr) if contour is None else ride_level(out, sr, target=contour)
    # after the leveler, which would otherwise ride the swells and the fade back out
    if swells and swell_db:
        out = out * 10 ** (phrase_swell(len(out), sr, swells, swell_db) / 20)[:, None]
    start = max(0, end - int(FADE * sr))
    fade = np.ones(len(out))
    fade[start:end] = 10 ** (FADE_TO / 20 * (1 - np.cos(np.linspace(0, np.pi, end - start))) / 2)
    fade[end:] = 10 ** (FADE_TO / 20)
    out = out * fade[:, None]
    if master:  # less low-mid boxiness, less edge, a touch of air, and 2 dB louder with peaks rounded off
        # (the quartet has ~3x a pop mix's share of 2-6 kHz, where the ear is most sensitive, so it
        # sounded louder and harsher than its level; -3 dB at 3.5 kHz was preferred in an A/B)
        for kind, f0, g, q in (("peak", 300, -2.5, 0.9), ("peak", 3500, -3.0, 0.8), ("shelf", 9000, 2.0, 0.7)):
            out = lfilter(*_biquad(kind, f0, g, q, sr), out, axis=0)
        out *= 10 ** ((level_db + 2) / 20) / max(np.sqrt((out ** 2).mean()), 1e-9)
        out = np.where(np.abs(out) > 0.8, np.sign(out) * (0.8 + 0.19 * np.tanh((np.abs(out) - 0.8) / 0.19)), out)
        sf.write(str(wav), out, sr)
        return
    rms = np.sqrt((out ** 2).mean())
    gain = min(10 ** (level_db / 20) / max(rms, 1e-9), 0.95 / max(np.abs(out).max(), 1e-9))
    sf.write(str(wav), out * gain, sr)


RELEASE_LUFS = -18.0  # Spotify's -14 and Apple's -16 both sounded too loud next to everything else
RELEASE_CEILING_DB = -1.3  # limiter ceiling; margin under Spotify's -1 dBTP after resampling


def loudness(wav: Path) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP), measured by ffmpeg's EBU R128 scanner."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav),
                          "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = log[log.rfind("Summary:"):]
    return (float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary)[1]),
            float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)[1]))


def master_for_release(src: Path, dst: Path, lufs: float = RELEASE_LUFS) -> tuple[float, float]:
    """Ready a finished preview for streaming: bring it to `lufs` (integrated loudness), round off the peaks with a
    gentle limiter run at 4x the sample rate (so it catches the peaks between samples, which a
    lossy encode would clip), trim dead air at both ends, and write 24-bit / 44.1 kHz WAV, which
    every distributor accepts. Returns the result's (LUFS, dBTP)."""
    gain = lufs - loudness(src)[0]
    chain = ",".join([
        "silenceremove=start_periods=1:start_threshold=-70dB",  # dead air before the first note
        "areverse,silenceremove=start_periods=1:start_threshold=-70dB,areverse",  # ...and after the tail
        "adelay=150|150", "afade=t=in:d=0.01",  # a short breath before the downbeat, no click
        f"volume={gain:.2f}dB",
        "aresample=176400:resampler=soxr",
        f"alimiter=limit={10 ** (RELEASE_CEILING_DB / 20):.4f}:attack=5:release=80:level=disabled",
        "aresample=44100:resampler=soxr:precision=28",
    ])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}")  # ffmpeg can't write over its own input
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-af", chain, "-c:a", "pcm_s24le", "-ar", "44100", "-f", "wav", str(tmp)],
                   check=True)
    tmp.replace(dst)
    return loudness(dst)

