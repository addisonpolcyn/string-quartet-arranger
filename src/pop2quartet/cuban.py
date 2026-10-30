"""Experiment: Cuban rhythm sections built from the song's chords. A tumbao for the bass (it
anticipates each bar's chord on beat 4), a 2-bar piano montuno in two lines, and cha-cha-cha
string stabs, all in 4/4 on beats."""
from .arrange import CHORDS
from .transcribe import Note

TUMBAO = ((1.5, 3.0, "fifth"), (3.0, 5.5, "next root"))  # on the and of 2, then 4 tied over
MONTUNO = (0.5, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.5, 7.5)  # 2 bars, son feel (beats)
CHACHA = (0.0, 1.0, 2.0, 3.0, 3.5)  # strings: on the beats, then "cha-cha" on 4 and 4-and


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


def tumbao(changes, total: float, centre: int = 43, velocity: float = 0.8) -> list[Note]:
    out = []
    for bar in range(int(total // 4)):
        b = 4.0 * bar
        root, q = chord_at(changes, b)
        nxt, _ = chord_at(changes, b + 4)
        for a, e, what in TUMBAO:
            pc = (root + CHORDS[q][2]) % 12 if what == "fifth" else nxt
            out.append(Note(b + a, min(b + e, total), near(pc, centre), velocity))
    return out


def montuno(changes, total: float, top: int = 76, velocity: float = 0.6) -> tuple[list[Note], list[Note]]:
    """Two lines in the same syncopated rhythm: chord tones on top (rising and falling through the
    chord) and the chord tone below each, so together they sound as the montuno's thirds/sixths."""
    hi, lo = [], []
    for two in range(0, int(total // 8) + 1):
        for k, off in enumerate(MONTUNO):
            t = 8.0 * two + off
            if t >= total:
                break
            root, q = chord_at(changes, t)
            tones = [(root + i) % 12 for i in CHORDS[q][:3]]
            pc = tones[(k if (k // 3) % 2 == 0 else -k) % 3]
            p = near(pc, top)
            below = max((near(x, p - 4) for x in tones if near(x, p - 4) < p), default=p - 12)
            hi.append(Note(t, t + 0.5, p, velocity))
            lo.append(Note(t, t + 0.5, below, velocity))
    return hi, lo


def chacha(changes, total: float, centre: int, which: int, velocity: float = 0.6) -> list[Note]:
    """Short string stabs on the cha-cha-cha rhythm: chord tone `which` (1 = 3rd, 2 = 5th)."""
    out = []
    for bar in range(int(total // 4)):
        for off in CHACHA:
            t = 4.0 * bar + off
            root, q = chord_at(changes, t)
            out.append(Note(t, t + 0.5, near((root + CHORDS[q][which]) % 12, centre), velocity))
    return out
