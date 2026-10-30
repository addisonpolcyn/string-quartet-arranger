"""Cuban dance bands. The rhythm section is written from the song's chords: a tumbao in the
bass (it anticipates each bar's chord on beat 4) and a 2-bar piano montuno in two lines, under
the tune. Charanga: flute on the tune, piano, plucked cello. Buena Vista Social Club: trumpet,
guitar on the montuno figure (the tres's guajeo), piano, plucked upright bass."""
from ..arrange import CHORDS
from ..transcribe import Note
from . import Ensemble, Voice

TUMBAO = ((1.5, 3.0, "fifth"), (3.0, 5.5, "next root"))  # the and of 2, then 4 tied over the bar
MONTUNO = (0.5, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.5, 7.5)  # 2 bars, son feel (beats)
PIANO = "167"  # Muse Keys grand piano


def chord_at(changes, t: float):
    """(root pitch class, quality) sounding at beat `t`, from (beat, chord) changes."""
    now = None
    for beat, c in changes:
        if beat > t:
            break
        now = c
    return now or (changes[0][1] if changes else (0, "major"))


def near(pc: int, centre: int) -> int:
    """The pitch of pitch class `pc` closest to `centre`."""
    return min(range(pc, 128, 12), key=lambda p: abs(p - centre))


def tumbao(changes, total: float, centre: int = 43) -> list[Note]:
    out = []
    for bar in range(int(total // 4)):
        b = 4.0 * bar
        root, q = chord_at(changes, b)
        nxt, _ = chord_at(changes, b + 4)
        for a, e, what in TUMBAO:
            pc = (root + CHORDS[q][2]) % 12 if what == "fifth" else nxt
            out.append(Note(b + a, min(b + e, total), near(pc, centre), 0.8))
    return out


def montuno(changes, total: float, top: int = 67) -> tuple[list[Note], list[Note]]:
    """Two lines in the same syncopated rhythm (whole 8ths: MuseScore rejects odd lengths): chord
    tones on top, rising and falling through the chord, and the chord tone below each."""
    hi, lo = [], []
    for two in range(0, int(total // 8) + 1):
        for k, off in enumerate(MONTUNO):
            t = 8.0 * two + off
            if t >= total:
                break
            root, q = chord_at(changes, t)
            tones = [(root + i) % 12 for i in CHORDS[q][:3]]
            p = near(tones[(k if (k // 3) % 2 == 0 else -k) % 3], top)
            below = max((near(x, p - 4) for x in tones if near(x, p - 4) < p), default=p - 12)
            hi.append(Note(t, t + 0.5, p, 0.6))
            lo.append(Note(t, t + 0.5, below, 0.6))
    return hi, lo


def rhythm_section(parts, chords, total):
    out = dict(parts)
    out["vln2"], out["vla"] = montuno(chords, total)
    out["vc"] = tumbao(chords, total)
    return out


# a dance band's touches, not a string quartet's: no octave drop (the tune rides on top), no
# tremolo/harmonics/double bass, and the piano can't bend, so no intonation
DANCE_BAND = {"lower_high": False, "intonation": False, "slides": False, "lead_boost": 0.0,
              "cello_boost": 0.0, "tremolo_builds": False, "harmonic_ending": False,
              "double_bass": False, "mutes": False}

CHARANGA = Ensemble(
    "cuban_charanga",
    {"lead": Voice("Flute", "Flute", "118", 4.0, (60, 96)),
     "harmony1": Voice("Piano", "Piano", PIANO, 2.0, (21, 108), full=True),
     "harmony2": Voice("Piano", "Piano", PIANO, 2.0, (21, 108), full=True),
     "bass": Voice("Violoncello", "Violoncello", "106", -4.0, (36, 67), pizz=True)},
    rhythm_section, DANCE_BAND)

BUENA_VISTA = Ensemble(
    "bvsc",
    {"lead": Voice("Trumpet", "Trumpet", None, 2.0, (54, 82)),
     "harmony1": Voice("AcousticGuitar", "Guitar", None, 0.0, (52, 76)),
     "harmony2": Voice("Piano", "Piano", PIANO, 0.0, (21, 108), full=True),
     "bass": Voice("Contrabass", "Contrabass", "95", 0.0, (28, 60), pizz=True)},
    rhythm_section, DANCE_BAND)
