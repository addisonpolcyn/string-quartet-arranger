"""Synthesize a short pop-ish test song (C major, I-V-vi-IV, 100 BPM) with known content."""
import subprocess
import sys

import pretty_midi

out = sys.argv[1]
bpm = 100
b = 60 / bpm
pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
lead = pretty_midi.Instrument(program=65, name="voice")  # alto sax: fast attack like a voice
bass = pretty_midi.Instrument(program=33, name="bass")
keys = pretty_midi.Instrument(program=0, name="piano")
drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

prog = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])]
melody = [  # (beat offset in bar, length, pitch) for each chord, male-vocal register
    [(0, 1, 52), (1, 1, 55), (2, 2, 60)],
    [(0, 1.5, 59), (1.5, 0.5, 57), (2, 2, 55)],
    [(0, 1, 57), (1, 1, 60), (2, 1, 64), (3, 1, 62)],
    [(0, 2, 60), (2, 2, 57)],
]
bars = 16
for bar in range(bars):
    root, chord = prog[bar % 4]
    t0 = (bar + 1) * 4 * b  # one bar count-in of drums only
    for beat in range(4):
        bass.notes.append(pretty_midi.Note(95, root, t0 + beat * b, t0 + (beat + 0.9) * b))
    for p in chord:
        for half in (0, 2):
            keys.notes.append(pretty_midi.Note(70, p, t0 + half * b, t0 + (half + 1.9) * b))
    if bar % 8 < 6:  # melody rests for 2 bars out of every 8
        for off, ln, p in melody[bar % 4]:
            lead.notes.append(pretty_midi.Note(105, p, t0 + off * b, t0 + (off + ln) * b - 0.03))
for beat in range((bars + 1) * 4):
    t = beat * b
    drums.notes.append(pretty_midi.Note(100, 36 if beat % 2 == 0 else 38, t, t + 0.1))
    for e in (0, 0.5):
        drums.notes.append(pretty_midi.Note(70, 42, t + e * b, t + e * b + 0.05))
pm.instruments += [lead, bass, keys, drums]
pm.write(out + ".mid")
subprocess.run(["fluidsynth", "-ni", "-q", "-F", out + ".wav", "-r", "44100",
                "/usr/share/sounds/sf2/FluidR3_GM.sf2", out + ".mid"], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".wav", out + ".mp3"], check=True)
