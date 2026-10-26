"""python -m pop2quartet song.mp3 [options]"""
import argparse
import contextlib
import io
import itertools
import math
from pathlib import Path

import numpy as np

from .arrange import LEVELS, arrange, bar_phase, dynamics_plan
from .export import render_audio, write_midi, write_musicxml
from .quality import WEIGHTS, evaluate
from .separate import separate
from .transcribe import (Note, beat_loudness, chroma_repair, drum_pattern, refine_pitch, to_beats,
                         track_beats, transcribe)


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
    plan = dynamics_plan(loud, dynamic_range=args.dynamics) if args.dynamics != "off" else None
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
                            repair=chroma_repair(stems["vocals"], to_sec) if raw["vocals"] else None)
    return {"parts": parts, "chords": chords, "plan": plan, "tempo": tempo, "to_sec": to_sec}


def main() -> None:
    ap = argparse.ArgumentParser(prog="pop2quartet", description=__doc__)
    ap.add_argument("audio", type=Path, help="input song (mp3/wav/flac/...)")
    ap.add_argument("-o", "--out", type=Path, help="output dir (default: out/<song>)")
    ap.add_argument("--cache", type=Path, default=Path(".cache"), help="stems/transcription cache")
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
    ap.add_argument("--melody-octave", type=int, help="force melody octave shift (e.g. 1)")
    ap.add_argument("--bass-octave", type=int, help="force cello octave shift (e.g. 1)")
    ap.add_argument("--fill", action=argparse.BooleanOptionalAction, default=False,
                    help="fill melody rests with instrumental hooks (off: violin I sticks to the "
                         "vocal tune)")
    ap.add_argument("--no-render", action="store_true", help="skip the WAV preview")
    args = ap.parse_args()

    out = args.out or Path("out") / args.audio.stem
    out.mkdir(parents=True, exist_ok=True)

    stems = separate(args.audio, args.cache)
    _, beat_times = track_beats(stems["drums"], args.audio, args.bpm)

    raw = {k: transcribe(stems[k], k) for k in ("vocals", "bass", "other")}
    if raw["vocals"]:
        raw["vocals"] = refine_pitch(stems["vocals"], raw["vocals"])
    for k, ns in raw.items():
        print(f"[transcribe] {k}: {len(ns)} notes")

    vocals = stems["vocals"] if raw["vocals"] else None
    grids = {"detected": beat_times}
    if args.auto and not args.bpm:  # beat trackers often lock onto half or double the tempo
        grids["half"] = beat_times[::2]
        grids["half, offbeat"] = beat_times[1::2]
        grids["double"] = np.sort(np.concatenate([beat_times, (beat_times[1:] + beat_times[:-1]) / 2]))
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
    for k, ns in parts.items():
        print(f"[arrange] {k}: {len(ns)} notes")
    q = evaluate(parts, chords, cand["to_sec"], args.audio, tempo, vocals)
    print("[quality] " + ", ".join(f"{k} {v:.2f}" for k, v in q.items())
          + "  (fidelity: 1 = matches the recording; the rest: 0 = clean)")

    midi = out / "quartet.mid"
    write_midi(parts, tempo, midi, plan)
    key = write_musicxml(parts, tempo, out / "quartet.musicxml", args.audio.stem, chords, plan)
    print(f"[export] key: {key}")
    print(f"[export] {out / 'quartet.musicxml'}\n[export] {midi}")
    if not args.no_render:
        # MuseScore plays chord symbols on a piano, so the preview gets a copy without them
        bare = out / ".preview.musicxml"
        write_musicxml(parts, tempo, bare, args.audio.stem, (), plan)
        how = render_audio(bare, midi, out / "preview.wav")
        bare.unlink()
        if how:
            print(f"[export] {out / 'preview.wav'} (played by {how})")


if __name__ == "__main__":
    main()
