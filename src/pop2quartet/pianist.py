"""Experiment: the parts a pianist would play from the song's chords, for --ensemble solo_piano.
Left hand: the bass on 1 and 3 (root, then fifth) and a broken chord rolling in 8ths. Right hand:
the tune, with a chord tone a 3rd-6th under each held note."""
from .arrange import CHORDS
from .cuban import chord_at, near
from .transcribe import Note


def bass(changes, total: float, centre: int = 40) -> list[Note]:
    out = []
    for bar in range(int(total // 4)):
        for off, which in ((0.0, 0), (2.0, 2)):
            t = 4.0 * bar + off
            root, q = chord_at(changes, t)
            out.append(Note(t, t + 2.0, near((root + CHORDS[q][which]) % 12, centre), 0.7))
    return out


def broken_chord(changes, total: float, centre: int = 55) -> list[Note]:
    """Root, 5th, 10th, 5th... in 8ths, around `centre`, re-voiced on every chord."""
    out = []
    for k in range(int(total * 2)):
        t = 0.5 * k
        root, q = chord_at(changes, t)
        r = near(root, centre - 5)
        steps = (r, r + CHORDS[q][2], r + 12 + CHORDS[q][1], r + CHORDS[q][2])
        out.append(Note(t, t + 0.5, steps[k % 4], 0.5))
    return out


def under_tune(tune: list[Note], changes, min_len: float = 1.0) -> list[Note]:
    """A chord tone 3-9 semitones under each tune note held `min_len` beats or more."""
    out = []
    for n in tune:
        if n.end - n.start < min_len:
            continue
        root, q = chord_at(changes, n.start)
        tones = {(root + i) % 12 for i in CHORDS[q]}
        cands = [n.pitch - d for d in range(3, 10) if (n.pitch - d) % 12 in tones]
        if cands:
            out.append(Note(n.start, n.end, cands[0], n.velocity * 0.8))
    return out
