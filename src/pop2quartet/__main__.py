"""python -m pop2quartet song.mp3 [options]"""
import argparse
import contextlib
import io
import itertools
import math
from pathlib import Path

import numpy as np

from .arrange import LEVELS, arrange, cello_sings, loudness_tiers, second_verse, suspensions, bar_phase, dynamics_plan, first_chorus, lift_last_chorus
from .export import (PARTS, RELEASE_LUFS, ROOMS, add_room, intonation, master_for_release,
                     phrase_arcs, preview_seconds, render_audio, room_for, write_midi, write_musicxml)
from .quality import WEIGHTS, evaluate
from .separate import separate
from .transcribe import (Note, band_chroma, beat_loudness, chroma_repair, drum_activity, drum_pattern,
                         refine_pitch, song_sections, sung_line, sung_ornaments, to_beats, track_beats,
                         transcribe)


def build(args, raw, stems, beat_times, grid, slot, verbose=False) -> dict:
    """One arrangement of the transcribed stems on a given beat grid and settings."""
    notes = {k: to_beats(ns, beat_times) for k, ns in raw.items()}
    # Shift so the music starts at bar 1 (notes before the first tracked beat land in a pickup)
    # and the detected downbeat falls on beat 1 of each bar.
    first = min((n.start for ns in notes.values() for n in ns), default=0.0)
    offset = -4 * math.floor(min(first, 0) / 4)
    phase = bar_phase(notes["bass"], notes["other"])
    offset += (-phase) % 4 + args.pickup
    notes = {k: [Note(n.start + offset, n.end + offset, n.pitch, n.velocity) for n in ns]
             for k, ns in notes.items()}

    # dynamics follow the recording's loudness, bar by bar (shifted like the notes)
    shift = round(offset)
    loud = beat_loudness(args.audio, beat_times)
    loud = np.concatenate([np.full(max(0, shift), loud.min()), loud[max(0, -shift):]])
    sections = None
    if args.sections == "repeats":
        labels = song_sections(args.audio, tuple(float(b) for b in beat_times))
        sections = np.concatenate([np.full(max(0, shift), labels[0]), labels[max(0, -shift):]])
    plan = (dynamics_plan(loud, dynamic_range=args.dynamics, sections=sections)
            if args.dynamics != "off" else None)
    tempo = 60.0 / float(np.median(np.diff(beat_times)))
    if verbose:
        print(f"[beats] tempo ~{tempo:.1f} BPM, downbeat on tracked beat {phase} (mod 4)")
        print("[dynamics] " + (" ".join(f"bar {int(b // 4) + 1}: {LEVELS[lv]}" for b, lv in plan)
                               if plan else "off (mf tune and cello, p inner voices)"))

    idx = np.arange(len(beat_times), dtype=float)

    def to_sec(b: float) -> float:
        return float(np.interp(b - offset, idx, beat_times))

    parts, chords = arrange(notes["vocals"], notes["bass"], notes["other"], grid=grid,
                            harmony_slot=slot, inner_style=args.inner,
                            fill=args.fill, melody_shift=args.melody_octave,
                            bass_shift=args.bass_octave, min_note=args.min_note, hold_gap=args.hold,
                            tie_repeats=args.tie_repeats, bass_hold=args.bass_hold,
                            three_voice=args.three_voice,
                            pattern=drum_pattern(stems["drums"], beat_times, offset),
                            repair=chroma_repair(stems["vocals"], to_sec) if raw["vocals"] else None,
                            sung=sung_line(stems["vocals"], to_sec, args.pitch) if args.melody == "best" else None,
                            leap=args.leap_cost, change=args.change_cost, melody_min=args.melody_min, bass_min=args.bass_min,
                            breathe=args.breathe, peak=args.peak_min,
                            tiers=loudness_tiers(loud) if args.breathe or args.peak_min else None, root_fill=args.root_fill, counter=args.counter,
                            repeat_gap=args.max_repeat * float(np.median(np.diff(beat_times))) ** -1
                            if args.max_repeat else 0.0,
                            sparse_until=first_chorus(plan) if args.vary else 0.0,
                            lower_high=args.lower_high, high_share=args.high_share,
                            harmony=band_chroma(stems["other"], stems["bass"], to_sec)
                            if args.chords == "audio" else None)
    if args.cello_verse:
        span = second_verse(plan, loudness_tiers(loud))
        if span:
            parts = cello_sings(parts, span)
            if verbose:
                print(f"[arrange] the cello sings the second verse: bars {int(span[0] // 4) + 1}-{int(span[1] // 4)}")
    if args.suspensions:
        made = 0
        for k in ("vln2", "vla"):
            parts[k], n = suspensions(parts[k], parts["vln1"])
            made += n
        if verbose:
            print(f"[arrange] {made} suspensions: inner voices hold into the next chord, then resolve")
    if args.lift:
        parts, start = lift_last_chorus(parts, plan)
        if verbose and start >= 0:
            print(f"[arrange] last chorus lift from bar {int(start // 4) + 1}: violin II doubles the tune")
    return {"parts": parts, "chords": chords, "plan": plan, "tempo": tempo, "to_sec": to_sec,
            "loud": loud}


MAX_BPM = 160  # a pop song's beat is rarely faster; a grid above this is double time, and its
# 16ths come out as frantic flurries of tiny notes and machine-gun repeats


def bpm_of(beat_times: np.ndarray) -> float:
    return 60.0 / float(np.median(np.diff(beat_times)))


SWELL_DB = 1.5  # --swells mix: gentle; the score's hairpins moved the tune ~6 dB

SONG_RIDE = 0.2  # --ride song: the quartet's loudness moves this share as far as the recording's
# (0.6, 0.35 and 0.23 faded too much while hairpins buried the tune; with gentle mix swells, 0.2)


def song_contour(loud: np.ndarray, tempo: float, marks: list,
                 amount: float = SONG_RIDE) -> tuple[np.ndarray, np.ndarray]:
    """(preview seconds, dB) the leveler steers toward: the recording's loudness per beat, relative
    to its typical level, smoothed over four bars, `amount` of its range."""
    rel = loud - np.median(loud[loud > loud.max() - 30])
    rel = np.convolve(np.pad(np.clip(rel, -10, 8), 8, mode="edge"), np.ones(17) / 17, mode="valid")
    return preview_seconds(np.arange(len(loud)) + 0.5, tempo, marks), amount * rel


def main() -> None:
    ap = argparse.ArgumentParser(prog="pop2quartet", description=__doc__)
    ap.add_argument("audio", type=Path, help="input song (mp3/wav/flac/...)")
    ap.add_argument("-o", "--out", type=Path, help="output dir (default: out/<song>)")
    ap.add_argument("--cache", type=Path, default=Path(".cache"), help="stems/transcription cache")
    ap.add_argument("--melody", choices=["notes", "best"], default="best",
                    help="how the tune is found: notes = transcribed notes, cleaned up; best = the "
                         "lowest-cost path through the sung pitch line, weighing pitch, key, chord "
                         "and held notes all at once (Viterbi)")
    ap.add_argument("--leap-cost", type=float, default=0.15,
                    help="best melody: extra cost per semitone jumped between notes, so slips of a "
                         "4th, 5th or octave lose to stepwise motion (0 = off)")
    ap.add_argument("--change-cost", type=float, default=0.5,
                    help="best melody: cost of a new note in the middle of a sung syllable; lower "
                         "follows more of the singer's turns, higher holds and repeats notes more")
    ap.add_argument("--melody-min", type=float, default=0.5,
                    help="shortest tune note in beats; shorter turns fold into the note before "
                         "(0.25 keeps the singer's 16th-note turns)")
    ap.add_argument("--breathe", type=float, default=1.0,
                    help="shortest tune note in the song's quietest third (by section), e.g. 1 = quarter "
                         "notes there and --melody-min in the middle third (0 = off)")
    ap.add_argument("--peak-min", type=float, default=0.0,
                    help="shortest tune note in the song's loudest third (by section), e.g. 0.25 = 16ths "
                         "there; needs a 16th grid (--grid 0.25) (0 = off)")
    ap.add_argument("--bass-min", type=float, default=0.5,
                    help="shortest cello note in beats; quicker bass notes fold into the one before "
                         "(0.25 keeps the bassist's 16th-note pickups)")
    ap.add_argument("--separator", default="htdemucs_ft",
                    help="Demucs model for the stems (htdemucs_ft: cleaner vocals; htdemucs: ~4x faster)")
    ap.add_argument("--bpm", type=float, help="override detected tempo (fixes half/double-time)")
    ap.add_argument("--pickup", type=float, default=0.0,
                    help="shift the barlines by N more beats if auto-detected bar 1 is still wrong")
    ap.add_argument("--grid", type=float,
                    help="rhythmic quantization in beats (0.25 = 16ths, 0.5 = 8ths; default: "
                         "chosen by --auto, else 0.25)")
    ap.add_argument("--harmony-slot", type=float,
                    help="how often violin II / viola may change chord, in beats (they hold notes "
                         "while the harmony allows; default: chosen by --auto, else 1)")
    ap.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True,
                    help="try half/double tempo, grid 0.25/0.5 and harmony slot 1/2 (whichever "
                         "you didn't set), score each arrangement, and keep the best")
    ap.add_argument("--min-note", type=float, default=0.25,
                    help="clean-up: fold notes and rests shorter than this many beats into "
                         "their neighbours, fix octave jumps (0 = off, keep every detail)")
    ap.add_argument("--hold", type=float, default=1.5,
                    help="sustain melody notes through rests up to this many beats, so only "
                         "real phrase breaks stay (0 = off)")
    ap.add_argument("--tie-repeats", action=argparse.BooleanOptionalAction, default=False,
                    help="tie repeated melody notes (one per sung syllable) into one long note")
    ap.add_argument("--dynamics", choices=["gentle", "full", "off"], default="gentle",
                    help="dynamics follow the recording's loudness: gentle = mp to f, full = p to f, "
                         "off = one fixed level")
    ap.add_argument("--three-voice", type=float, default=0.3,
                    help="fraction of bars, the quietest ones, where the viola rests and the "
                         "texture thins to three voices (0 = always four)")
    ap.add_argument("--bass-hold", type=float, default=4.0,
                    help="sustain cello notes through rests up to this many beats, like a "
                         "sustain pedal (e.g. 4 = ring through a bar; 0 = off)")
    ap.add_argument("--inner", choices=["auto", "groove", "pad", "pulse"], default="auto",
                    help="inner voices: play the band's rhythm (groove), sustained chords (pad), "
                         "repeated eighths (pulse), or auto = groove, with pads in quiet sections")
    ap.add_argument("--root-fill", action=argparse.BooleanOptionalAction, default=True,
                    help="where the bass is silent but the band plays chords, the cello plays "
                         "the chord's root (for songs with no bass line, and bass-less breakdowns)")
    ap.add_argument("--melody-octave", type=int, help="force melody octave shift (e.g. 1)")
    ap.add_argument("--bass-octave", type=int, help="force cello octave shift (e.g. 1)")
    ap.add_argument("--fill", action=argparse.BooleanOptionalAction, default=False,
                    help="fill melody rests with instrumental hooks (off: violin I sticks to the "
                         "vocal tune)")
    ap.add_argument("--expression", action=argparse.BooleanOptionalAction, default=True,
                    help="slurs, pizzicato inner voices in the quietest verses, and a swell over "
                         "each sung phrase")
    ap.add_argument("--room", choices=["auto", *ROOMS], default="auto",
                    help="reverb for the preview: studio (only the samples' own room), hall (~1.8 s), "
                         "church (~3.2 s), or auto = by how much the drums play (ballads: church)")
    ap.add_argument("--max-repeat", type=float, default=0.3,
                    help="fastest same-note re-attack, in seconds: quicker repeats merge into one "
                         "note, as samples repeat identically (0 = off)")
    ap.add_argument("--lift", action=argparse.BooleanOptionalAction, default=False,
                    help="last loudest section: violin II doubles the tune an octave below")
    ap.add_argument("--counter", action=argparse.BooleanOptionalAction, default=True,
                    help="violin II answers the tune with the band's hook where the singer rests")
    ap.add_argument("--ending", action=argparse.BooleanOptionalAction, default=True,
                    help="a 2-bar ritardando into a held final chord under a fermata")
    ap.add_argument("--accents", action=argparse.BooleanOptionalAction, default=True,
                    help="inner voices accent the backbeat (2 and 4) and play quick notes short")
    ap.add_argument("--rubato", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: the tempo gives a little at the end of each sung phrase")
    ap.add_argument("--master", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: EQ (less low-mid mud, more air) and a louder, limited level")
    ap.add_argument("--release", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: ready for streaming at --lufs, peaks under -1 dBTP, 24-bit")
    ap.add_argument("--lufs", type=float, default=RELEASE_LUFS,
                    help="release loudness (Spotify plays at -14, Apple Music at -16; quieter is "
                         "played quieter, louder is turned down)")
    ap.add_argument("--vary", action=argparse.BooleanOptionalAction, default=True,
                    help="first verse sparse (held chords, no answers), so later verses and choruses build")
    ap.add_argument("--slides", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: violin I slides up into a held note after a leap, like a singer")
    ap.add_argument("--ornaments", action=argparse.BooleanOptionalAction, default=True,
                    help="violin I copies the singer's little turns: a grace note where the voice "
                         "flips in from a neighbour note, a mordent for a quick flick, a trill")
    ap.add_argument("--grace", action=argparse.BooleanOptionalAction, default=True,
                    help="--ornaments: include grace notes (--no-grace keeps only mordents and trills)")
    ap.add_argument("--ornament-bars", type=float, default=4.0,
                    help="--ornaments: at most one ornament per this many bars (2 = twice as many)")
    ap.add_argument("--trills", action=argparse.BooleanOptionalAction, default=False,
                    help="violin I trills a long note ending a sung phrase, at most once per 16 bars")
    ap.add_argument("--slide-max", type=int, default=7,
                    help="--slides: widest leap (semitones) violin I slides across (7 = a 5th; wider "
                         "leaps are played clean)")
    ap.add_argument("--humanize", type=float, default=0.0,
                    help="preview only: the four players drift apart by up to this many ms (MuseScore; "
                         "renders each part separately, so ~4x slower; 0 = off)")
    ap.add_argument("--ride", choices=["flat", "song"], default="song",
                    help="preview only: the leveler evens the quartet out to one level (flat) or steers "
                         "it along the recording's loudness, so choruses keep their lift (song)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="play the quartet at this share of the recording's tempo (0.85 = 15%% slower)")
    ap.add_argument("--ride-amount", type=float, default=SONG_RIDE,
                    help="--ride song: how far the quartet follows the recording's loudness changes "
                         "(0.2 = 20%% as far; 0 = as flat as possible)")
    ap.add_argument("--swells", choices=["mix", "score", "off"], default="mix",
                    help="a swell up to the top of each sung phrase and back: mix = the whole quartet "
                         "together by --swell-db in the preview's mix; score = violin I hairpins, which "
                         "Muse Sounds plays a full dynamic level (~6 dB) and buried the tune; off")
    ap.add_argument("--swell-db", type=float, default=SWELL_DB,
                    help="--swells mix: how far the quartet swells at a phrase's peak, in dB")
    ap.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: violin I EQ'd warmer and less nasal (renders violin I once more on "
                         "its own, so ~2x slower)")
    ap.add_argument("--warm-master", action=argparse.BooleanOptionalAction, default=False,
                    help="preview only, with --master: keep more body (300 Hz), no added air, light tape saturation")
    ap.add_argument("--cello-verse", action=argparse.BooleanOptionalAction, default=False,
                    help="the cello sings the tune in the second verse; violin I rests and returns for the chorus")
    ap.add_argument("--suspensions", action=argparse.BooleanOptionalAction, default=False,
                    help="violin II / viola hold a note into the next chord and resolve down (once per 2 bars each)")
    ap.add_argument("--lead-boost", type=float, default=3.0,
                    help="preview only: dB added to violin I's fader (on top of its +6.5)")
    ap.add_argument("--cello-boost", type=float, default=5.0,
                    help="preview only: dB added to the cello's fader (on top of its -4)")
    ap.add_argument("--lead-ahead", type=float, default=0.0,
                    help="preview only: if violin I sits less than this many dB over the other three "
                         "(a busy accompaniment), turn them down for the song to get it there (0 = off)")
    ap.add_argument("--lower-high", action=argparse.BooleanOptionalAction, default=True,
                    help="a tune spending 8%%+ of its time at A5 or above plays the whole song an "
                         "octave lower (violin II and its answers follow)")
    ap.add_argument("--high-share", type=float, default=0.0,
                    help="--lower-high: share of the tune's time at A5+ that takes it down an octave "
                         "(0 = lower any tune that still fits on the violin)")
    ap.add_argument("--intonation", action=argparse.BooleanOptionalAction, default=True,
                    help="preview only: tune notes the way a quartet does (thirds toward pure, the "
                         "leading tone sharp, a few cents of spread) instead of dead in tune")
    ap.add_argument("--pitch", choices=["pyin", "crepe"], default="crepe",
                    help="pitch tracker for the sung line: pyin, or crepe (a neural tracker, steadier on voices)")
    ap.add_argument("--chords", choices=["notes", "audio"], default="audio",
                    help="name chords from the band's transcribed notes, or from its audio (chroma)")
    ap.add_argument("--sections", choices=["loudness", "repeats"], default="loudness",
                    help="dynamics change by loudness alone, or at detected sections (passages that "
                         "repeat), with the loudest returning one as the chorus")
    ap.add_argument("--reverb", choices=["noise", "real"], default="real",
                    help="preview only: the room's echo synthesized from noise, or recorded in real "
                         "spaces (impulse responses in <cache>/ir)")
    ap.add_argument("--timing", choices=["drift", "notes"], default="notes",
                    help="preview only, with --humanize: the players drift slowly apart, or stay together "
                         "and slip on single notes (mostly entries), then find each other again")
    ap.add_argument("--no-render", action="store_true", help="skip the WAV preview")
    args = ap.parse_args()

    out = args.out or Path("out") / args.audio.stem
    out.mkdir(parents=True, exist_ok=True)

    stems = separate(args.audio, args.cache, args.separator)
    _, beat_times = track_beats(stems["drums"], args.audio, args.bpm)

    raw = {k: transcribe(stems[k], k) for k in ("vocals", "bass", "other")}
    if raw["vocals"]:
        raw["vocals"] = refine_pitch(stems["vocals"], raw["vocals"], tracker=args.pitch)
    for k, ns in raw.items():
        print(f"[transcribe] {k}: {len(ns)} notes")

    vocals = stems["vocals"] if raw["vocals"] else None
    grids = {"detected": beat_times}
    if not args.bpm and (args.auto or bpm_of(beat_times) > MAX_BPM):
        # beat trackers often lock onto half or double the tempo
        grids["half"] = beat_times[::2]
        grids["half, offbeat"] = beat_times[1::2]
        if args.auto:
            grids["double"] = np.sort(np.concatenate([beat_times, (beat_times[1:] + beat_times[:-1]) / 2]))
        playable = {k: v for k, v in grids.items() if bpm_of(v) <= MAX_BPM}
        if len(playable) < len(grids):
            print(f"[auto] skipping tempos over {MAX_BPM} BPM: "
                  + ", ".join(f"{k} ({bpm_of(v):.0f})" for k, v in grids.items() if k not in playable))
        grids = playable or {k: v for k, v in grids.items() if k.startswith("half")}
    grid_opts = [args.grid] if args.grid else [0.25, 0.5] if args.auto else [0.25]
    slot_opts = [args.harmony_slot] if args.harmony_slot else [1.0, 2.0] if args.auto else [1.0]
    options = list(itertools.product(grids, grid_opts, slot_opts))
    if len(options) > 1:
        print(f"[auto] scoring {len(options)} arrangements (tempo x grid x harmony slot) ...")
        cols = ["fidelity", *WEIGHTS, "total"]
        print("[auto] " + f"{'tempo':>14} {'bpm':>4} grid slot " + " ".join(f"{c[:8]:>8}" for c in cols))
        scored = []
        for name, grid, slot in options:
            with contextlib.redirect_stdout(io.StringIO()):  # keep the per-run chatter quiet
                cand = build(args, raw, stems, grids[name], grid, slot)
                q = evaluate(cand["parts"], cand["chords"], cand["to_sec"], args.audio,
                             cand["tempo"], vocals)
            scored.append((q["total"], name, grid, slot))
            print("[auto] " + f"{name:>14} {cand['tempo']:4.0f} {grid:4} {slot:4} "
                  + " ".join(f"{q[c]:8.3f}" for c in cols))
        _, name, grid, slot = max(scored)
        print(f"[auto] best: tempo {name}, grid {grid}, harmony slot {slot}")
    else:
        name, grid, slot = options[0]
    cand = build(args, raw, stems, grids[name], grid, slot, verbose=True)
    parts, chords, plan, tempo = cand["parts"], cand["chords"], cand["plan"], cand["tempo"]
    loud = cand["loud"]
    for k, ns in parts.items():
        print(f"[arrange] {k}: {len(ns)} notes")
    q = evaluate(parts, chords, cand["to_sec"], args.audio, tempo, vocals)
    print("[quality] " + ", ".join(f"{k} {v:.2f}" for k, v in q.items())
          + "  (fidelity: 1 = matches the recording; the rest: 0 = clean)")
    if args.speed != 1.0:
        tempo *= args.speed
        print(f"[export] played at {args.speed:.0%} of the recording's tempo: {tempo:.0f} BPM")

    ornaments = None
    if args.ornaments and parts["vln1"]:
        ornaments = sung_ornaments(stems["vocals"], cand["to_sec"], parts["vln1"], args.pitch,
                                   every=4 * args.ornament_bars)
        if not args.grace:
            ornaments = {i: o for i, o in ornaments.items() if o[0] != "grace"}
        kinds = [k for k, _ in ornaments.values()]
        print(f"[arrange] the singer's turns on violin I: {kinds.count('grace')} grace notes, "
              f"{kinds.count('upper') + kinds.count('lower')} mordents, {kinds.count('trill')} trills")

    midi = out / "quartet.mid"
    write_midi(parts, tempo, midi, plan)
    key = write_musicxml(parts, tempo, out / "quartet.musicxml", args.audio.stem, chords, plan,
                         args.expression, args.ending, args.accents, swells=args.swells != "off",
                         trills=args.trills, ornaments=ornaments)
    print(f"[export] key: {key}")
    print(f"[export] {out / 'quartet.musicxml'}\n[export] {midi}")
    if not args.no_render:
        # MuseScore plays chord symbols on a piano, so the preview gets a copy without them
        bare = out / ".preview.musicxml"
        marks = []
        write_musicxml(parts, tempo, bare, args.audio.stem, (), plan, args.expression, args.ending,
                       args.accents, args.rubato, args.slides, marks, args.swells == "score",
                       trills=args.trills, ornaments=ornaments, slide_max=args.slide_max)
        tunings = intonation(parts, chords) if args.intonation else None
        onsets = None
        if args.timing == "notes":
            onsets = [[(float(preview_seconds(n.start, tempo, marks)), i == 0 or n.start - ns[i - 1].end >= 0.5)
                       for i, n in enumerate(ns)] for ns in (parts[k] for k, *_ in PARTS)]
        how = render_audio(bare, midi, out / "preview.wav", args.humanize, args.lead_ahead,
                           [tunings[k] for k, *_ in PARTS] if tunings else None, onsets,
                           (args.lead_boost, 0.0, 0.0, args.cello_boost), args.warm)
        bare.unlink()
        if how:
            room = room_for(drum_activity(stems["drums"])) if args.room == "auto" else args.room
            contour = song_contour(loud, tempo, marks, args.ride_amount) if args.ride == "song" else None
            tune = parts["vln1"]
            phrases = [tuple(preview_seconds(np.array([tune[a].start, tune[p].start, tune[b].end]),
                                             tempo, marks))
                       for a, p, b in phrase_arcs(tune, [])] if args.swells == "mix" else []
            add_room(out / "preview.wav", room, master=args.master, contour=contour, swells=phrases,
                     swell_db=args.swell_db, irs=args.cache / "ir" if args.reverb == "real" else None,
                     warm_master=args.warm_master)
            level = ""
            if args.release:
                lufs, peak = master_for_release(out / "preview.wav", out / "preview.wav", args.lufs)
                level = f", {lufs:.1f} LUFS, {peak:.1f} dBTP"
            print(f"[export] {out / 'preview.wav'} (played by {how}, {room} room{level})")


if __name__ == "__main__":
    main()
