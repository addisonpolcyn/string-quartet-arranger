"""Other ensembles than the string quartet. An Ensemble says, per role, which instrument plays it
(its Muse Sounds uid, fader and range) and how the accompaniment is played: `accompany` rewrites
the arranged parts from the song's tune and chords. The arranger still works in quartet roles
(ROLES), so a flute on the tune is the "lead" role, the part the quartet gives violin I."""
from dataclasses import dataclass, field
from typing import Callable

from ..transcribe import Note

ROLES = {"lead": "vln1", "harmony1": "vln2", "harmony2": "vla", "bass": "vc"}


@dataclass(frozen=True)
class Voice:
    instrument: str          # music21 instrument class
    name: str                # name on the score
    uid: str | None          # Muse Sounds uid; None = MuseScore's own default sound for it
    fader: float             # dB
    range: tuple[int, int]   # sounding MIDI pitches the part is folded into
    pizz: bool = False       # plucked throughout
    full: bool = False       # play at the tune's dynamic level, not a level under (a piano bed)


@dataclass(frozen=True)
class Ensemble:
    name: str
    voices: dict[str, Voice]                      # role -> Voice; roles not named keep strings
    accompany: Callable | None = None             # (parts, chords, total) -> parts
    overrides: dict = field(default_factory=dict) # settings that don't apply (e.g. no string bass)

    def part(self, key: str) -> Voice | None:
        """The Voice playing quartet part `key` ("vln1"...), if this ensemble replaces it."""
        return next((v for r, v in self.voices.items() if ROLES[r] == key), None)

    def fold(self, parts: dict[str, list[Note]]) -> dict[str, list[Note]]:
        out = dict(parts)
        for role, v in self.voices.items():
            lo, hi = v.range
            out[ROLES[role]] = [Note(n.start, n.end, _into(n.pitch, lo, hi), n.velocity)
                                for n in parts[ROLES[role]]]
        return out


def _into(p: int, lo: int, hi: int) -> int:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def registry() -> dict[str, Ensemble]:
    from .cuban import BUENA_VISTA, CHARANGA
    return {e.name: e for e in (CHARANGA, BUENA_VISTA)}
