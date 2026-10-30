"""Experiment: Motown and jazz-combo accompaniment from the song's chords. An organ pad for Motown
(two sustained chord tones per chord), and for jazz a walking bass and Charleston-rhythm piano
comping on shell voicings (3rd and 7th; a 6th on plain triads)."""
from .arrange import CHORDS
from .cuban import chord_at, near
from .transcribe import Note


def _spans(changes, total: float):
    """(start, end, chord) for each chord, cut at least every bar."""
    out, bars = [], [4.0 * b for b in range(int(total // 4) + 1)]
    cuts = sorted({b for b in bars if b < total} | {b for b, _ in changes if 0 <= b < total})
    for a, b in zip(cuts, cuts[1:] + [total]):
        out.append((a, b, chord_at(changes, a)))
    return out


def organ_pad(changes, total: float, centre: int, which: int) -> list[Note]:
    """One chord tone (`which`: 0 root, 1 third, 2 fifth) held through each chord."""
    out = []
    for a, b, (root, q) in _spans(changes, total):
        p = near((root + CHORDS[q][which]) % 12, centre)
        if out and out[-1].pitch == p and abs(out[-1].end - a) < 1e-6:
            out[-1] = Note(out[-1].start, b, p, 0.55)
        else:
            out.append(Note(a, b, p, 0.55))
    return out


def walking_bass(changes, total: float, centre: int = 40) -> list[Note]:
    """Quarter notes: root, a chord tone, the fifth, then a half-step into the next bar's root."""
    out = []
    for bar in range(int(total // 4)):
        b = 4.0 * bar
        root, q = chord_at(changes, b)
        nxt, _ = chord_at(changes, b + 4)
        r = near(root, centre)
        line = [r, r + CHORDS[q][1], r + CHORDS[q][2]]
        target = near(nxt, r)
        line.append(target - 1 if target - 1 != line[-1] else target + 1)
        out += [Note(b + k, b + k + 1, p, 0.75) for k, p in enumerate(line)]
    return out


def comping(changes, total: float, centre: int, upper: bool) -> list[Note]:
    """Charleston rhythm (on 1, then the and of 2) on a shell voicing: the 3rd, and above it the
    7th (or the 6th on a plain triad)."""
    out = []
    for bar in range(int(total // 4)):
        for off, dur in ((0.0, 1.0), (1.5, 0.5)):
            t = 4.0 * bar + off
            root, q = chord_at(changes, t)
            iv = CHORDS[q]
            third = near((root + iv[1]) % 12, centre)
            seventh = (root + (iv[3] if len(iv) > 3 else 9)) % 12
            top = min((p for p in range(third + 1, third + 12) if p % 12 == seventh), default=third + 7)
            out.append(Note(t, t + dur, top if upper else third, 0.55))
    return out
