"""Stage 3: turn transcribed stems (in beats) into four playable string parts.

  Violin I  <- vocal melody (instrumental hooks fill the gaps when nobody sings)
  Violin II <- upper chord tones from the "other" stem
  Viola     <- lower chord tones from the "other" stem
  Cello     <- bass line
"""
import math
from collections import defaultdict

import numpy as np

from .export import COUNTER_VELOCITY, LEVELS, NAMES, detect_key
from .transcribe import Note

# Comfortable MIDI ranges (lo, hi) and a "sweet spot" centre for octave fitting.
RANGES = {
    "vln1": (55, 93, 72),  # G3-A6
    "vln2": (55, 84, 67),  # G3-C6
    "vla": (48, 76, 60),   # C3-E5
    "vc": (36, 67, 48),    # C2-G4
}


def quantize(notes: list[Note], grid: float) -> list[Note]:
    out = []
    for n in notes:
        s = round(n.start / grid) * grid
        e = max(round(n.end / grid) * grid, s + grid)
        out.append(Note(s, e, n.pitch, n.velocity))
    return out


def drop_harmonics(notes: list[Note], max_len: float = 1.0) -> list[Note]:
    """Remove short ghost notes an octave/twelfth/two octaves above a stronger sounding note."""
    out = []
    for n in notes:
        ghost = n.end - n.start <= max_len and any(
            n.pitch - m.pitch in (12, 19, 24)
            and m.start <= n.start + 0.25 and m.end >= n.start
            and (m.end - m.start >= n.end - n.start or m.velocity > n.velocity)
            for m in notes if m is not n)
        if not ghost:
            out.append(n)
    return out


def bar_phase(bass: list[Note], harmony: list[Note]) -> int:
    """Guess which beat (0-3) is the downbeat: bass and chord changes tend to land on beat 1."""
    votes = np.zeros(4)
    line = monophonic(quantize(bass, 1.0), pick=lambda g: min(g, key=lambda n: n.pitch))
    for prev, n in zip(line, line[1:]):
        if n.pitch % 12 != prev.pitch % 12:
            votes[int(n.start) % 4] += 1
    total = max((n.end for n in harmony), default=0.0)
    chords = chord_per_slot(harmony, [], total, 1.0)
    for k in range(1, len(chords)):
        if chords[k] and chords[k - 1] and chords[k] != chords[k - 1]:
            votes[k % 4] += 0.5
    return int(np.argmax(votes))


def monophonic(notes: list[Note], pick, legato_gap: float = 0.0) -> list[Note]:
    """Reduce to one note at a time. `pick` chooses among notes with the same onset."""
    by_start = defaultdict(list)
    for n in notes:
        by_start[n.start].append(n)
    line = [pick(group) for _, group in sorted(by_start.items())]
    out: list[Note] = []
    for i, n in enumerate(line):
        end = n.end
        if i + 1 < len(line):
            nxt = line[i + 1].start
            end = min(end, nxt)
            if 0 < nxt - end <= legato_gap:  # close tiny gaps: strings like to slur
                end = nxt
        if end <= n.start:
            continue
        if (out and out[-1].pitch == n.pitch and out[-1].end == n.start
                and end - n.start <= legato_gap):  # re-triggered fragment of the same note
            out[-1] = Note(out[-1].start, end, n.pitch, out[-1].velocity)
        else:
            out.append(Note(n.start, end, n.pitch, n.velocity))
    return out


SHRILL = {"vln1": 86}  # D6: sustained above this a violin turns thin and piercing


def fit_range(notes: list[Note], part: str, shift: int | None = None) -> list[Note]:
    """Transpose by whole octaves to best fit the instrument, then fold stragglers in. For the tune
    (parts in SHRILL) the octave weighs notes out of range (cheap: they fold in), time in the
    shrill top register and distance from the sweet spot, so a few low notes can't push the whole
    tune up an octave; other parts take the octave with the most notes in range."""
    lo, hi, centre = RANGES[part]
    if not notes:
        return []
    pitches = np.array([n.pitch for n in notes])
    if shift is None and part in SHRILL:
        def cost(k):
            p = pitches + 12 * k
            return (3 * np.mean((p < lo) | (p > hi)) + 3 * np.mean(p >= SHRILL[part])
                    + abs(np.median(p) - centre) / 12)
        shift = min(range(-3, 4), key=cost)
    elif shift is None:
        def score(k):
            p = pitches + 12 * k
            return (np.mean((p >= lo) & (p <= hi)), -abs(np.median(p) - centre))
        shift = max(range(-3, 4), key=score)
    out = []
    for n in notes:
        p = n.pitch + 12 * shift
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        out.append(Note(n.start, n.end, p, n.velocity))
    return out


def fill_gaps(melody: list[Note], filler: list[Note], min_gap: float) -> list[Note]:
    """Insert filler notes wherever the melody rests for at least `min_gap` beats."""
    if not filler:
        return melody
    bounds = [(-math.inf, -math.inf)] + [(n.start, n.end) for n in melody] + [(math.inf, math.inf)]
    added = []
    for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
        if next_start - prev_end < min_gap:
            continue
        for f in filler:
            if f.start >= prev_end and f.start < next_start:
                added.append(Note(f.start, min(f.end, next_start), f.pitch, f.velocity))
    return sorted(melody + added, key=lambda n: n.start)


def sounding(notes: list[Note], s: float, e: float) -> list[int]:
    return [n.pitch for n in notes if n.start < e and n.end > s]


def pc_histogram(harmony: list[Note], bass: list[Note], total: float, slot: float) -> np.ndarray:
    """Duration- and velocity-weighted pitch-class histogram per slot (bass at half weight)."""
    n_slots = math.ceil(total / slot)
    w = np.zeros((n_slots, 12))
    for src, weight in ((harmony, 1.0), (bass, 0.5)):
        for n in src:
            for k in range(max(0, int(n.start // slot)), min(n_slots, math.ceil(n.end / slot))):
                ov = min(n.end, (k + 1) * slot) - max(n.start, k * slot)
                if ov > 0:
                    w[k, n.pitch % 12] += ov * n.velocity * weight
    return w


def chord_per_slot(harmony: list[Note], bass: list[Note], total: float, slot: float,
                   threshold: float = 0.35, max_pcs: int = 4) -> list[set[int] | None]:
    """Pitch-class histogram per slot -> its strongest pitch classes."""
    w = pc_histogram(harmony, bass, total, slot)
    chords = []
    for row in w:
        if row.max() < 0.05 * slot:
            chords.append(None)
            continue
        top = np.argsort(row)[::-1][:max_pcs]
        chords.append({int(pc) for pc in top if row[pc] >= threshold * row.max()})
    return chords


MELODY_MIN = 0.5  # shortest melody note, in beats: 16th-note flickers sound nervous on a violin
INNER_VELOCITY = 0.6  # the dynamics plan puts the accompaniment a level under (export.SOFTER)
CHORDS = {  # quality -> intervals from the root; the 3rd is always second
    "major": (0, 4, 7), "minor": (0, 3, 7),
    "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10),
}
SEVENTH = 0.97  # a 7th chord must beat its triad clearly, so plain songs stay plain
Chord = tuple[int, str]  # (root pitch class, quality)


def label_chords(w: np.ndarray, slot: float, scale: set[int] | None = None,
                 stay: float = 0.9, borrowed: float = 0.9) -> list[Chord | None]:
    """Name each slot's chord: cosine similarity between its histogram and each chord shape
    (triads and 7ths). The previous chord is kept while it fits nearly as well as the best, so
    labels don't flicker, and chords from outside the key must clearly win (`borrowed`)."""
    labels: list[Chord | None] = []
    prev = None
    for row in w:
        if row.max() < 0.05 * slot:
            labels.append(None)
            prev = None
            continue
        norm = np.linalg.norm(row)
        scores = {(r, q): sum(row[(r + i) % 12] for i in iv) / (norm * math.sqrt(len(iv)))
                      * (1 if not scale or {(r + i) % 12 for i in iv} <= scale else borrowed)
                      * (SEVENTH if len(iv) == 4 else 1)
                  for r in range(12) for q, iv in CHORDS.items()}
        best = max(scores, key=scores.get)
        if prev and scores[prev] >= stay * scores[best]:
            best = prev
        labels.append(best)
        prev = best
    return labels


def chord_tones(chord: Chord | None) -> set[int] | None:
    return {(chord[0] + i) % 12 for i in CHORDS[chord[1]]} if chord else None


def validate(notes: list[Note], labels: list[Chord | None], slot: float, part: str,
             scale: set[int] | None = None, min_len: float = 1.0) -> tuple[list[Note], int]:
    """Fix notes that don't belong to the chord under them. With `scale` (the melody), every
    out-of-key note moves into the key (unless the chord itself has it, e.g. a borrowed chord),
    and long in-key notes rubbing a half step against a chord tone settle on it; sus4s, 9ths and
    the like a whole step away survive. Otherwise (the bass) long notes become the chord's root."""
    lo, hi, _ = RANGES[part]
    out, fixed = [], 0
    for n in notes:
        ks = range(int(n.start // slot), min(len(labels), math.ceil(n.end / slot)))
        spans = [(labels[k], min(n.end, (k + 1) * slot) - max(n.start, k * slot)) for k in ks]
        spans = [(c, ov) for c, ov in spans if c and ov > 0]
        off = sum(ov for c, ov in spans if n.pitch % 12 not in chord_tones(c))
        clash = spans and off > 0.5 * sum(ov for _, ov in spans)
        chord = max(spans, key=lambda s: s[1])[0] if spans else None
        tones = chord_tones(chord) or set()
        near = [pc for pc in tones if (pc - n.pitch) % 12 in (1, 11)]
        targets = []
        if scale is None:
            if clash and n.end - n.start >= min_len:
                targets = [chord[0]]
        elif n.pitch % 12 not in scale and (clash or not spans):
            targets = near or [pc for pc in scale if (pc - n.pitch) % 12 in (1, 11)]
        elif clash and n.end - n.start >= 2 * min_len:
            targets = near
        cands = [q for q in range(lo, hi + 1) if q % 12 in targets]
        p = min(cands, key=lambda q: abs(q - n.pitch)) if cands else n.pitch
        fixed += p != n.pitch
        out.append(Note(n.start, n.end, p, n.velocity))
    return out, fixed


DYNAMIC_RANGES = {  # name -> (quietest level, quantile cuts between levels)
    "gentle": (2, (0.15, 0.8)),  # mp, mf, f
    "full": (1, (0.15, 0.45, 0.8)),  # p, mp, mf, f
}


def dynamics_plan(loudness: np.ndarray, min_bars: int = 4,
                  dynamic_range: str = "gentle") -> list[tuple[float, int]]:
    """Per-beat loudness of the song -> (start beat, LEVELS index) sections for the tune.
    Bars are ranked against the rest of the song (gentle: quietest 15% mp, loudest 20% f, mf in
    between), and sections shorter than `min_bars` merge into the neighbour closest in level."""
    lowest, cuts = DYNAMIC_RANGES[dynamic_range]
    n_bars = max(1, math.ceil(len(loudness) / 4))
    bar_db = np.array([np.mean(loudness[4 * b:4 * b + 4]) for b in range(n_bars)])
    bar_db = np.convolve(np.pad(bar_db, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    levels = lowest + np.searchsorted(np.quantile(bar_db, cuts), bar_db)
    runs = []  # [level, length in bars]
    for lv in levels:
        if runs and runs[-1][0] == lv:
            runs[-1][1] += 1
        else:
            runs.append([int(lv), 1])
    while len(runs) > 1 and min(r[1] for r in runs) < min_bars:
        i = min(range(len(runs)), key=lambda j: runs[j][1])
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = min(nbrs, key=lambda j: abs(runs[j][0] - runs[i][0]))
        runs[j][1] += runs[i][1]
        del runs[i]
        merged = []
        for r in runs:  # neighbours may now share a level
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] += r[1]
            else:
                merged.append(r)
        runs = merged
    plan, bar = [], 0
    for lv, length in runs:
        plan.append((4.0 * bar, lv))
        bar += length
    return plan


def groove_pattern(band: list[Note], grid: float = 0.5) -> list[float]:
    """Fallback when there are no drums: the band's above-average attack positions in a bar
    (in beats), plus beat 1."""
    counts = np.zeros(round(4 / grid))
    for n in band:
        counts[round(n.start / grid) % len(counts)] += n.velocity
    return [i * grid for i in range(len(counts)) if i == 0 or counts[i] > counts.mean()]


def groove(notes: list[Note], pattern: list[float], skip=None, slot: float = 1.0) -> list[Note]:
    """Re-strike held notes on the groove's attack positions (not in `skip` slots)."""
    out = []
    for n in notes:
        hits = {4 * bar + p for bar in range(int(n.start // 4), int(n.end // 4) + 1) for p in pattern}
        cuts = sorted({n.start, n.end} | {t for t in hits if n.start < t < n.end
                                           and not (skip and skip[min(len(skip) - 1, int(t // slot))])})
        out += [Note(a, b, n.pitch, n.velocity) for a, b in zip(cuts, cuts[1:])]
    return out


def quiet_bars(band: list[Note], total: float, slot: float, fraction: float,
               min_bars: int = 2) -> list[bool]:
    """Per slot: is this among the quietest `fraction` of bars (by how much the band plays),
    in a run of at least `min_bars`? Those passages thin out to three voices."""
    n_bars = max(1, math.ceil(total / 4))
    energy = np.zeros(n_bars)
    for n in band:
        energy[min(n_bars - 1, max(0, int(n.start // 4)))] += (n.end - n.start) * n.velocity
    played = energy[energy > 0]
    quiet = energy <= (np.quantile(played, fraction) if len(played) else 0)
    i = 0
    while i < n_bars:  # a single quiet bar would just flicker the viola in and out
        j = i
        while j < n_bars and quiet[j] == quiet[i]:
            j += 1
        if quiet[i] and j - i < min_bars:
            quiet[i:j] = False
        i = j
    return [bool(quiet[min(n_bars - 1, int(k * slot // 4))]) for k in range(math.ceil(total / slot))]


def voice_inner(chords, melody, bass, slot, essentials=None, solo=None):
    """Pick (violin II, viola) pitches per slot: chord tones, smooth motion, no crossing,
    and no half-step clashes with the melody or bass. Notes are held while they still fit.
    In `solo` slots the viola rests and violin II alone fills in the chord (three voices)."""
    lo2, hi2, c2 = RANGES["vln2"]
    loa, hia, ca = RANGES["vla"]
    prev2, preva, prev_k = c2, ca, -2
    vln2, vla = [], []
    for k, pcs in enumerate(chords):
        s, e = k * slot, (k + 1) * slot
        if not pcs:
            continue
        others = [(n.pitch, min(n.end, e) - max(n.start, s))
                  for n in melody + bass if n.start < e and n.end > s]
        mel = sounding(melody, s, e)
        bas = sounding(bass, s, e)
        ceiling = min(mel) if mel else 128
        floor = max(bas) if bas else 0
        held = {p % 12 for p in mel + bas}

        tune = [(n.pitch, min(n.end, e) - max(n.start, s))
                for n in melody if n.start < e and n.end > s]

        def clash(p):  # fraction of the slot this pitch rubs against the tune or bass: a half
            # step against either, or a tritone against the tune (it exaggerates the song's colour)
            return (sum(ov for q, ov in others if abs(p - q) % 12 in (1, 11))
                    + sum(ov for q, ov in tune if abs(p - q) % 12 == 6)) / slot

        # root and 3rd make the chord, unless the tune sits a half step or tritone off them
        need = {pc for pc in (essentials[k] if essentials else ())
                if all((pc - q) % 12 not in (1, 6, 11) for q in mel)}

        three = bool(solo and solo[k])
        low = (lambda a, b: a) if three else (lambda a, b: b)  # lowest inner voice

        def missing(a, b):
            return len(need - held - {a % 12} - ({b % 12} if b is not None else set()))

        def fits(a, b):
            return (all(p % 12 in pcs and clash(p) == 0 for p in (a, b) if p is not None)
                    and floor < low(a, b) and (b is None or b < a) and a < ceiling
                    and missing(a, b) == 0)

        if prev_k == k - 1 and fits(prev2, None if three else preva):  # keep holding
            a, b = prev2, None if three else preva
        else:
            cand2 = [p for p in range(lo2, hi2 + 1) if p % 12 in pcs]
            canda = [None] if three else [p for p in range(loa, hia + 1) if p % 12 in pcs]

            def cost(a, b):
                c = 1.5 * abs(a - prev2) - 1.5 * (a == prev2) + 0.15 * abs(a - c2)
                if b is not None:
                    c += 1.5 * abs(b - preva)  # smooth voice leading
                    c -= 1.5 * (b == preva)  # hold common tones
                    c += 0.15 * abs(b - ca)  # stay near the middle of the range
                    c += 4 * (a % 12 == b % 12)  # don't double each other
                    c += 2 * (a - b < 3) + 0.5 * max(0, a - b - 12)  # keep spacing sensible
                    c += 2 * (b < 64 and a - b < 5)  # inner pair a 4th+ apart low down
                    c += 12 * clash(b)
                tones = {a % 12} | ({b % 12} if b is not None else set())
                c -= 3 * len((tones - held) & pcs)  # reward filling in the chord
                c += 8 * missing(a, b)  # the root and 3rd must be heard somewhere
                # close intervals blur low down, so keep the bottom open above the cello
                c += 4 * (low(a, b) < 60 and low(a, b) - floor < 7)
                c += 12 * clash(a)  # avoid rubbing against the tune or bass
                c += 100 * (a >= ceiling) + 100 * (low(a, b) <= floor)
                return c

            pairs = [(a, b) for a in cand2 for b in canda if b is None or b < a]
            if not pairs:
                continue
            a, b = min(pairs, key=lambda ab: cost(*ab))
        vln2.append(Note(s, e, a, INNER_VELOCITY))
        if b is not None:
            vla.append(Note(s, e, b, INNER_VELOCITY))
            preva = b
        prev2, prev_k = a, k
    return vln2, vla


MELODY_COST = {  # what the best melody trades off; all per grid step unless noted
    "off_pitch": 1.0,   # x (semitones away from the sung pitch)^2, capped at 3 semitones
    "out_of_key": 1.5,  # a pitch outside the key, unless the chord has it
    "non_chord": 0.15,  # an in-key pitch that isn't in the chord
    "rest_sung": 3.0,   # resting while the singer sings
    "hold_unsung": 0.4, # holding a note through a breath
    "change": 1.0,      # per new note: keeps notes held instead of flickering
    "change_onset": 0.1,  # ... when the singer starts a syllable right there
    "leap": 0.0,        # per semitone jumped between notes (tunes mostly move by step)
    "past_octave": 1.0, # extra for a jump wider than an octave, when "leap" is on
}


def best_melody(f0: np.ndarray, starts: np.ndarray, labels: list[Chord | None], slot: float,
                scale: set[int], step: float, cost=MELODY_COST) -> list[Note]:
    """The whole tune at once: the note sequence (a pitch or a rest per grid step) with the lowest
    total cost (MELODY_COST) over the song, found with the Viterbi algorithm. Notes split where
    the singer re-attacks the same pitch (a repeated syllable)."""
    sung = ~np.isnan(f0)
    if not sung.any():
        return []
    pitches = np.arange(int(np.nanmin(f0)) - 2, int(np.nanmax(f0)) + 3)
    n, rest = len(f0), len(pitches)  # state index `rest` = no note
    pc = pitches % 12
    in_key = np.isin(pc, list(scale))
    local = np.zeros((n, rest + 1))
    for t in range(n):
        k = int(t * step // slot)
        tones = chord_tones(labels[k]) if k < len(labels) else None
        in_chord = np.isin(pc, list(tones)) if tones else np.ones(rest, bool)
        local[t, :rest] = (cost["out_of_key"] * (~in_key & ~in_chord)
                           + cost["non_chord"] * (in_key & ~in_chord))
        if sung[t]:
            local[t, :rest] += cost["off_pitch"] * np.minimum(np.abs(pitches - f0[t]), 3) ** 2
            local[t, rest] = cost["rest_sung"]
        else:
            local[t, :rest] += cost["hold_unsung"]
    total = local[0].copy()
    back = np.zeros((n, rest + 1), int)
    keep = np.arange(rest + 1)
    if cost["leap"]:  # from-to cost of every jump between pitches (rests are free to enter/leave)
        moved = np.zeros((rest + 1, rest + 1))
        d = np.abs(pitches[:, None] - pitches[None, :])
        moved[:rest, :rest] = cost["leap"] * d + cost["past_octave"] * (d > 12)
        new = 1.0 - np.eye(rest + 1)
    for t in range(1, n):
        change = cost["change_onset"] if starts[t] else cost["change"]
        if cost["leap"]:
            via = total[:, None] + change * new + moved
            back[t] = via.argmin(axis=0)
            total = via[back[t], keep] + local[t]
            continue
        j = int(total.argmin())
        jump = total[j] + change
        back[t] = np.where(jump < total, j, keep)
        total = np.minimum(total, jump) + local[t]
    path = np.empty(n, int)
    path[-1] = int(total.argmin())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    notes: list[Note] = []
    t = 0
    while t < n:
        u = t + 1
        while u < n and path[u] == path[t]:
            u += 1
        if path[t] != rest:
            cuts = [t] + [v for v in range(t + 1, u) if starts[v]
                          and (v - t) * step >= MELODY_MIN and (u - v) * step >= MELODY_MIN] + [u]
            for a, b in zip(cuts, cuts[1:]):
                notes.append(Note(a * step, b * step, int(pitches[path[t]]), 0.8))
        t = u
    return notes


def fill_roots(cello: list[Note], labels: list[Chord | None], band: list[Note],
               slot: float, low: int = 40) -> tuple[list[Note], int]:
    """Where the bass is silent but the band plays chords, the cello plays the chord's root
    (E2-Eb3), one note per chord, so songs without a bass player (guitar and voice) or with
    bass-less breakdowns still have a floor. Where the whole band rests, so does the cello."""
    added: list[Note] = []
    for k, chord in enumerate(labels):
        s, e = k * slot, (k + 1) * slot
        if not chord or sounding(cello, s, e) or not sounding(band, s, e):
            continue
        pitch = low + (chord[0] - low) % 12
        if added and added[-1].pitch == pitch and added[-1].end == s:
            added[-1] = Note(added[-1].start, e, pitch, added[-1].velocity)
        else:
            added.append(Note(s, e, pitch, 0.7))
    return sorted(cello + added, key=lambda n: n.start), len(added)


def lift_last_chorus(parts: dict[str, list[Note]], plan) -> tuple[dict[str, list[Note]], float]:
    """The climax: through the song's last loudest section, violin II leaves its chords to play the
    tune an octave below violin I (where that fits its range; it keeps its chords in the tune's
    rests). Returns the new parts and the beat the lift starts (or -1 if there's no plan)."""
    if not plan:
        return parts, -1.0
    top = max(lv for _, lv in plan)
    k = max(i for i, (_, lv) in enumerate(plan) if lv == top)
    start = plan[k][0]
    end = plan[k + 1][0] if k + 1 < len(plan) else float("inf")
    lo = RANGES["vln2"][0]
    tune = [n for n in parts["vln1"] if start <= n.start < end]
    double = [Note(n.start, n.end, n.pitch - 12, n.velocity) for n in tune if n.pitch - 12 >= lo]
    keep = [n for n in parts["vln2"] if not (start <= n.start < end)
            or not any(t.start < n.end and t.end > n.start for t in double)]
    return {**parts, "vln2": sorted(keep + double, key=lambda n: n.start)}, start


def first_chorus(plan) -> float:
    """Beat where the song first reaches its loudest dynamic level (0 if it starts there or there
    is no plan): the end of the first verse."""
    if not plan:
        return 0.0
    top = max(lv for _, lv in plan)
    return next(start for start, lv in plan if lv == top)


def countermelody(melody: list[Note], hooks: list[Note], inner: list[Note], scale: set[int],
                  min_gap: float = 2.0, min_notes: int = 3, after: float = 0.0) -> tuple[list[Note], int]:
    """Where the singer rests at least `min_gap` beats, violin II answers with the band's own hook
    from that moment (in key, notes of half a beat or more, moved into violin II's upper range)
    instead of holding chords; only from beat `after` on. Returns violin II's new line and how
    many answers it plays."""
    lo, hi, _ = RANGES["vln2"]
    out, answers = list(inner), 0
    for a, b in zip(melody, melody[1:]):
        s, e = a.end, b.start
        if e - s < min_gap or s < after:
            continue
        phrase = [n for n in hooks if s <= n.start and n.end <= e
                  and n.pitch % 12 in scale and n.end - n.start >= 0.5]
        if len(phrase) < min_notes:
            continue
        mid = float(np.median([n.pitch for n in phrase]))
        shift = 12 * round((73 - mid) / 12)  # centre the answer around C#5, near the tune
        phrase = [Note(n.start, n.end, n.pitch + shift, COUNTER_VELOCITY) for n in phrase]
        if min(n.pitch for n in phrase) < lo or max(n.pitch for n in phrase) > hi:
            continue
        out = [n if n.end <= s or n.start >= e else Note(n.start, s, n.pitch, n.velocity)
               for n in out if not s <= n.start < e]
        out = [n for n in out if n.end > n.start] + phrase
        answers += 1
    return sorted(out, key=lambda n: n.start), answers


def smooth_octaves(notes: list[Note], part: str) -> list[Note]:
    """Undo octave errors: a leap of more than an octave goes to the nearest octave instead."""
    lo, hi, _ = RANGES[part]
    out: list[Note] = []
    for n in notes:
        p = n.pitch
        if out and abs(p - out[-1].pitch) > 12:
            p = min((q for q in range(p % 12, 128, 12) if lo <= q <= hi),
                    key=lambda q: abs(q - out[-1].pitch))
        out.append(Note(n.start, n.end, p, n.velocity))
    return out


def simplify(notes: list[Note], min_len: float) -> list[Note]:
    """Close rests shorter than `min_len` and fold notes shorter than it into the previous one."""
    out: list[Note] = []
    for n in notes:
        if out and 0 < n.start - out[-1].end < min_len:
            out[-1] = Note(out[-1].start, n.start, out[-1].pitch, out[-1].velocity)
        if out and out[-1].end == n.start and (n.end - n.start < min_len or n.pitch == out[-1].pitch
                                                and out[-1].end - out[-1].start < min_len):
            out[-1] = Note(out[-1].start, n.end, out[-1].pitch, out[-1].velocity)
        else:
            out.append(n)
    return out


def hold(notes: list[Note], max_gap: float, chords=None, slot: float = 1.0) -> list[Note]:
    """Sustain each note into rests up to `max_gap` beats, so only real phrase breaks stay.
    With `chords`, let go as soon as the harmony no longer contains the held pitch."""
    out = list(notes)
    for i in range(len(out) - 1):
        n, nxt = out[i], out[i + 1]
        if not 0 < nxt.start - n.end <= max_gap:
            continue
        end = nxt.start
        if chords is not None:
            k = int(n.end // slot)
            while k * slot < end and k < len(chords) and chords[k] and n.pitch % 12 in chords[k]:
                k += 1
            end = min(end, max(n.end, k * slot))
        out[i] = Note(n.start, end, n.pitch, n.velocity)
    return out


def cap_repeats(notes: list[Note], min_gap: float) -> tuple[list[Note], int]:
    """Merge a same-pitch re-attack that comes less than `min_gap` beats after the previous one
    into it. Samples repeat note-for-note identically (a "machine gun"), and no player re-bows
    that fast; this matters most when the tempo was read at double speed."""
    out: list[Note] = []
    merged = 0
    for n in notes:
        p = out[-1] if out else None
        if p and p.pitch == n.pitch and abs(p.end - n.start) < 1e-6 and n.start - p.start < min_gap:
            out[-1] = Note(p.start, n.end, p.pitch, p.velocity)
            merged += 1
        else:
            out.append(n)
    return out, merged


def merge_repeats(notes: list[Note]) -> list[Note]:
    """Tie consecutive identical pitches into one sustained note."""
    out = []
    for n in notes:
        if out and out[-1].pitch == n.pitch and abs(out[-1].end - n.start) < 1e-6:
            out[-1] = Note(out[-1].start, n.end, n.pitch, out[-1].velocity)
        else:
            out.append(n)
    return out


def pulse(notes: list[Note], step: float) -> list[Note]:
    """Re-strike each held note every `step` beats (driving eighth-note accompaniment)."""
    out = []
    for n in notes:
        t = n.start
        while t < n.end - 1e-6:
            out.append(Note(t, min(t + step, n.end), n.pitch, n.velocity))
            t += step
    return out


def arrange(vocals, bass, other, *, grid=0.25, harmony_slot=1.0, inner_style="auto",
            fill=False, melody_shift=None, bass_shift=None, min_note=0.25, hold_gap=1.5,
            tie_repeats=False, bass_hold=4.0, three_voice=0.3, pattern=None, repair=None, sung=None,
            leap=0.0, root_fill=False, counter=False, repeat_gap=0.0, sparse_until=0.0):
    """Returns (parts, chord changes as (beat, (root pitch class, quality))). With `sung` (from
    transcribe.sung_line), the tune comes from best_melody instead of the transcribed notes.
    Before beat `sparse_until` (the first verse) the inner voices hold their chords instead of
    playing the groove and don't answer the tune, so the song has somewhere to build to."""
    vocals, bass, other = (quantize(x, grid) for x in (vocals, bass, other))
    vocals, bass = drop_harmonics(vocals), drop_harmonics(bass)

    total = max((n.end for n in vocals + bass + other), default=0.0)
    tonic, mode = detect_key({"vocals": vocals, "bass": bass, "other": other})
    scale = {(NAMES.index(tonic) + i) % 12
             for i in ((0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10))}
    labels = label_chords(pc_histogram(other, bass, total, harmony_slot), harmony_slot, scale)
    chords = [chord_tones(c) for c in labels]

    hooks = monophonic(drop_harmonics(other), pick=lambda g: max(g, key=lambda n: n.pitch))
    if vocals and sung:
        melody = best_melody(*sung(total, grid), labels, harmony_slot, scale, grid,
                             {**MELODY_COST, "leap": leap})
        print(f"[melody] best path through the sung pitch line: {len(melody)} notes")
    elif vocals:
        melody = monophonic(vocals, pick=lambda g: max(g, key=lambda n: n.velocity), legato_gap=grid)
    else:  # instrumental track: the top line of the band is the best guess at the tune
        melody, fill = hooks, False
    melody = fit_range(melody, "vln1", melody_shift)
    if fill:
        melody = fill_gaps(melody, fit_range(hooks, "vln1"), min_gap=4.0)
        melody = monophonic(melody, pick=lambda g: g[0], legato_gap=grid)

    cello = monophonic(bass, pick=lambda g: min(g, key=lambda n: n.pitch), legato_gap=grid)
    cello = fit_range(cello, "vc", bass_shift)
    if min_note:
        melody = simplify(smooth_octaves(melody, "vln1"), max(min_note, MELODY_MIN))
        cello = simplify(smooth_octaves(cello, "vc"), min_note)
    if hold_gap:
        melody = hold(melody, hold_gap)
    if tie_repeats:
        melody = merge_repeats(melody)

    melody, fixed_mel = validate(melody, labels, harmony_slot, "vln1", scale)
    cello, fixed_vc = validate(cello, labels, harmony_slot, "vc")
    if root_fill:
        cello, filled = fill_roots(cello, labels, other, harmony_slot)
        print(f"[arrange] cello plays the chord root in {filled} bass-less spots")
    print(f"[chords] fixed {fixed_mel} melody and {fixed_vc} cello notes that clashed with the chord")
    if repair:  # last word on the tune goes to the recording itself
        melody, matched = repair(melody, scale)
        print(f"[melody] moved {matched} notes onto the pitch the voice actually sings")
    if bass_hold:  # let the bass ring like a sustain pedal, released when the chord changes
        cello = merge_repeats(hold(cello, bass_hold, chords, harmony_slot))
    band = chord_per_slot(other, bass, total, harmony_slot)  # the band's 7ths, 9ths, ... too
    voicing = [c | (b or set()) if c else None for c, b in zip(chords, band)]
    solo = quiet_bars(other, total, harmony_slot, three_voice) if three_voice else None
    if solo:
        print(f"[arrange] three voices (viola rests) in {sum(solo) * harmony_slot / 4:.0f} "
              f"of {len(solo) * harmony_slot / 4:.0f} bars")
    vln2, vla = voice_inner(voicing, melody, cello, harmony_slot,
                            [{c[0], (c[0] + CHORDS[c[1]][1]) % 12} if c else None for c in labels],
                            solo)
    vln2, vla = merge_repeats(vln2), merge_repeats(vla)
    if inner_style == "pulse":
        vln2, vla = pulse(vln2, 0.5), pulse(vla, 0.5)
    elif inner_style in ("groove", "auto"):  # auto: sustained chords in the quiet sections
        pattern = pattern or groove_pattern(other + bass)
        print("[arrange] groove: " + "".join("x" if i / 2 in pattern else "." for i in range(8))
              + " (eighths in a bar)")
        skip = solo if inner_style == "auto" else None
        if sparse_until:
            n_slots = math.ceil(total / harmony_slot)
            skip = [bool(skip and skip[k]) or k * harmony_slot < sparse_until for k in range(n_slots)]
        vln2, vla = groove(vln2, pattern, skip, harmony_slot), groove(vla, pattern, skip, harmony_slot)

    if counter:
        vln2, answers = countermelody(melody, fit_range(hooks, "vln2"), vln2, scale, after=sparse_until)
        print(f"[arrange] violin II answers the tune with the band's hook {answers} times")
    if repeat_gap:
        melody, a = cap_repeats(melody, repeat_gap)
        vln2, b = cap_repeats(vln2, repeat_gap)
        vla, c = cap_repeats(vla, repeat_gap)
        print(f"[arrange] merged {a + b + c} same-note repeats faster than {repeat_gap:.2f} beats")
    parts = {"vln1": melody, "vln2": vln2, "vla": vla, "vc": cello}
    chord_report(parts, labels, harmony_slot)
    changes = [(k * harmony_slot, c) for k, c in enumerate(labels)
               if c and (k == 0 or labels[k - 1] != c)]
    return parts, changes


def chord_report(parts: dict[str, list[Note]], labels: list[Chord | None], slot: float) -> None:
    """Check the quartet against the named chords, sampling the middle of every slot."""
    named = complete = clean = 0
    for k, chord in enumerate(labels):
        if not chord:
            continue
        t = (k + 0.5) * slot
        sounding = {p: [n.pitch % 12 for n in ns if n.start <= t < n.end] for p, ns in parts.items()}
        heard = {pc for pcs in sounding.values() for pc in pcs}
        named += 1
        complete += chord[0] in heard and (chord[0] + CHORDS[chord[1]][1]) % 12 in heard
        clean += all(pc in chord_tones(chord) for p in ("vln2", "vla", "vc") for pc in sounding[p])
    if named:
        print(f"[chords] {named * slot:.0f} beats with a chord: root+3rd sounding in "
              f"{complete / named:.0%}, accompaniment all chord tones in {clean / named:.0%}")
