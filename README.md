# pop2quartet

Turn a pop song recording into a string quartet arrangement (sheet music + MIDI + audio preview).

```
song.mp3 ─► Demucs ─► vocals / bass / other / drums stems
                         │       │      │       └─► beat tracking (librosa)
                         ▼       ▼      ▼
                      basic-pitch note transcription (per stem)
                         │       │      │
                         ▼       ▼      ▼
                   Violin I   Cello   Violin II + Viola
                   (melody)   (bass)  (chords, voice-led)
                         └───────┴──────┴─► quartet.musicxml / quartet.mid / preview.wav
```

## How it works

1. **Split the song** – [Demucs](https://github.com/facebookresearch/demucs) (the fine-tuned `htdemucs_ft` model) separates the mix
   into vocals, bass, drums and "other" (the rest of the band). Runs on the GPU if available.
2. **Find the beat** – librosa tracks beats on the drum stem (or the full mix if there are no
   drums). Note times are mapped onto the actual beats, so the score follows tempo drift.
3. **Turn audio into notes** – Spotify's [basic-pitch](https://github.com/spotify/basic-pitch)
   transcribes each stem, with per-stem frequency limits to cut down on octave errors. It runs on
   the CPU on purpose: TensorFlow's GPU mode crashes when PyTorch's CUDA libraries are loaded.
   Near-silent stems are skipped instead of transcribing noise. Vocal notes are then re-pitched
   from a pYIN pitch track (the median over each note ignores vibrato and slides, and the
   recording's tuning offset from A440 is measured and removed); where the two
   disagree by more than a step, the pitch that fits the surrounding tune wins.
4. **Arrange** (`arrange.py`):
   - **Violin I** plays the vocal melody, held through short breaths. `--tie-repeats` also ties
     repeated syllables into long notes (off by default: it dropped ~10% of the sung note starts). With `--fill` it also takes instrumental riffs (the top line of the "other"
     stem) during rests of a bar or more. Songs with no vocals use the top line throughout.
   - **Cello** plays the bass line, ringing through rests like a sustain pedal until the chord
     changes.
   - **Chords** are named per beat by fitting the "other" and bass stems to major/minor triads
     and maj7/7/m7 chords (cosine similarity), preferring chords in the key. A validation pass moves long cello notes that aren't in the
     chord to its root, and long melody notes that are out of key and a half step off a chord tone
     onto it. The chord names are written above violin I in the score.
   - **Melody repair:** a last check compares each tune note with the vocal stem's chroma and
     moves it a step or two when the voice is clearly elsewhere (never out of the key).
   - **Violin II and Viola** play those chords with smooth voice leading: small steps, held
     common tones, no crossing the melody or bass, root and 3rd always heard unless the tune
     sits right against them. The run prints how often that holds. They play in the rhythm of
     the drums' accents (e.g. `x.x.xxx.` in eighths), with sustained chords in the quietest
     sections, where the viola also rests so the texture thins to three voices.
   - **Dynamics** follow the loudness of the original, bar by bar, in sections of 4+ bars.
   - The **key** is the best Krumhansl match among keys whose scale holds the song's notes.
   - Every part is transposed by whole octaves into its instrument's comfortable range.
   - Short "ghost" notes (false overtones an octave or a twelfth above a real note) are removed,
     and bar 1 is placed where the bass and chords tend to change.
5. **Export** – the key is detected with a duration-weighted Krumhansl key-finder, and the
   quartet is written as MusicXML (music21), MIDI, and an audio preview. The preview is played by MuseScore
   when it is installed (with Muse Sounds if you have them, following the dynamics), otherwise by
   fluidsynth from the MIDI.

## How long it takes

Measured on an RTX 4070 Ti:

| run | time |
|---|---|
| 3½-minute song, first run | ~15–20 s (Demucs on the GPU takes only ~3 s; the rest is model loading and transcription) |
| re-run with different arrangement options | a few seconds (stems and notes come from `.cache/`) |

Without a GPU, Demucs is much slower, so expect a few minutes per song.

## Setup

```bash
sudo apt install ffmpeg fluidsynth fluid-soundfont-gm   # fluidsynth: fallback preview renderer
uv sync
```

For a realistic `preview.wav`, install [MuseScore 4](https://musescore.org) and the Muse Strings
library (through MuseHub). Under WSL, the Windows install is found and used automatically.

## Usage

```bash
uv run pop2quartet song.mp3                 # -> out/song/
uv run pop2quartet song.mp3 --inner pad     # sustained chords instead of the song's groove
uv run pop2quartet song.mp3 --bpm 120       # fix half/double-time tempo detection
uv run pop2quartet song.mp3 --pickup 1      # nudge barlines if bar 1 is on the wrong beat
```

Open `quartet.musicxml` in [MuseScore](https://musescore.org) (free) to view, edit, print and
play back the score. Stems and transcriptions are cached in `.cache/`, so re-running with
different arrangement options takes seconds.

| option | default | effect |
|---|---|---|
| `--melody` | best | how the tune is found: `best` = the lowest-cost note path through the sung pitch line, weighing pitch, key, chord and held notes at once (Viterbi); `notes` = the transcribed notes, cleaned up |
| `--separator` | htdemucs_ft | Demucs model for the stems; the fine-tuned one separates vocals better (quality +0.03–0.04 on the songs tried), `htdemucs` is ~4x faster on a first run |
| `--leap-cost` | 0.15 | best melody: extra cost per semitone jumped between notes, so pitch slips of a 4th, 5th or octave lose to stepwise motion (0 = off) |
| `--root-fill` / `--no-root-fill` | on | where the bass is silent but the band plays chords, the cello plays the chord root (songs with no bass line, bass-less breakdowns) |
| `--expression` / `--no-expression` | on | slurs over connected notes (one bow), pizzicato inner voices in the quietest verses, a swell to the top of each sung phrase |
| `--room` | auto | reverb for the preview: `studio` (only the samples' own room), `hall` (~1.8 s), `church` (~3.2 s); `auto` picks by how much of the song has drums: church under 30% (ballads), hall under 45%, studio above |
| `--max-repeat` | 0.3 | fastest same-note re-attack in seconds; quicker repeats merge into one note (samples repeat identically, a "machine gun"; 0 = off) |
| `--lift` / `--no-lift` | on | the last loudest section: violin II doubles the tune an octave below |
| `--counter` / `--no-counter` | on | where the singer rests, violin II answers with the band's own hook |
| `--ending` / `--no-ending` | on | a 2-bar ritardando into a held final chord under a fermata |
| `--accents` / `--no-accents` | on | inner voices accent the backbeat (2 and 4) and play quick notes short |
| `--rubato` / `--no-rubato` | on | preview only: the tempo gives a little at the end of each sung phrase |
| `--master` / `--no-master` | on | preview only: EQ (less low-mid mud, 3 dB less edge at 3.5 kHz, a touch of air) and a louder, limited level |
| `--release` / `--no-release` | on | preview only: ready for Spotify and other streaming services at `--lufs`, true peaks under -1 dBTP, dead air trimmed, 24-bit / 44.1 kHz |
| `--lufs` | -18 | release loudness: Spotify plays at -14, Apple Music at -16 (both sounded too loud next to everything else) |
| `--ride` | song | preview only: the leveler steers the quartet along the recording's own loudness (13% of its range), so choruses keep their lift; `flat` evens everything to one level |
| `--ride-amount` | 0.13 | `--ride song`: how far the quartet follows the recording's loudness changes (0 = as flat as possible) |
| `--humanize` | 10 | preview only: the four players drift apart by up to this many ms, so attacks don't land like one machine (MuseScore: renders each part separately, ~4x slower; 0 = off) |
| `--speed` | 1.0 | play the quartet at this share of the recording's tempo (0.85 = 15% slower) |
| `--vary` / `--no-vary` | on | first verse sparse (held chords, no answers), so later verses and choruses build |
| `--slides` / `--no-slides` | on | preview only: violin I slides up into a held note after a leap, like a singer |
| `--auto` / `--no-auto` | on | try the tempo as detected, half and double (never above 160 BPM: faster is double time, frantic on strings), grid 0.25/0.5 and harmony slot 1/2 (any you set yourself stay fixed), score each arrangement and keep the best |
| `--grid` | auto (0.25) | rhythmic quantization in beats (0.5 = simpler, eighth-note rhythms) |
| `--harmony-slot` | auto (1) | how often the inner voices may change chord; they hold notes while the harmony allows |
| `--min-note` | 0.25 | clean-up: fold notes/rests shorter than this (in beats) into neighbours and fix octave jumps; `0` keeps every detail |
| `--hold` | 1.5 | sustain melody notes through rests up to this many beats; longer rests stay as phrase breaks (`0` = off) |
| `--tie-repeats` / `--no-tie-repeats` | off | tie repeated melody notes (one per sung syllable) into one long note |
| `--bass-hold` | 4 | sustain cello notes through rests up to this many beats, like a sustain pedal (`4` = ring through a bar, `0` = off) |
| `--dynamics` | gentle | dynamics follow the recording's loudness: `gentle` = *mp* to *f*, `full` = *p* to *f*, `off` = one fixed level |
| `--three-voice` | 0.3 | fraction of bars (the quietest) where the viola rests and violin II carries the chord alone (`0` = always four voices) |
| `--inner` | auto | `groove` = inner voices play the drums' accent pattern, `pad` = sustained chords, `pulse` = repeated eighths, `auto` = groove with pads in the three-voice (quiet) sections |
| `--melody-octave` / `--bass-octave` | auto | force octave transposition of violin I / cello |
| `--fill` / `--no-fill` | off | fill vocal rests with instrumental hooks in violin I |

## Quality checks

Every run scores the arrangement before writing it (`quality.py`), and `--auto` uses the score to
pick settings:

| check | meaning |
|---|---|
| `fidelity` | chroma similarity between the arrangement and the original recording, in half-second windows (1 = same pitch content) |
| `rubs` | share of time two parts are a minor 2nd/9th apart (major 7ths are allowed: tunes sit on them over the bass) |
| `out_of_key` | share of melody time on notes outside the key, usually transcription errors |
| `incomplete` | share of beats where the chord's root and 3rd aren't both heard |
| `flicker` | chord changes beyond 2 per bar, a sign of a double-time beat grid |
| `leaps` | share of melody steps bigger than an octave |
| `tempo` | octaves outside 70–160 BPM, where half or double time is more likely |
| `rhythm` | share of the singer's note starts (onsets in the vocal stem) that violin I doesn't play within 0.1 s |

`total` = fidelity minus the weighted penalties. Treat it as a comparison between settings of the
same song, and roughly between songs: with the rhythm check, *Save Your Tears* (sounds good)
scores about 0.26 and *A Couple Minutes* (sounds off) about 0.05.

## What to expect

The output is a solid **first draft**, not a finished arrangement. Automatic transcription
gets the pitches and harmony mostly right, but expect to clean up in MuseScore:
stray/ghost notes, rhythms in melismatic singing, and the occasional wrong chord. Everything
is assumed to be in 4/4. Dynamics follow the recording's loudness in sections of at least 4 bars
(*mp* to *f* by default, with hairpins into each change; inner voices a level softer); bowings and
articulations aren't generated.

Tips for real songs:

- Tempo detected at half or double speed → `--bpm`
- Bar 1 starts on the wrong beat → `--pickup 1` (or 2, 3)
- Rhythms too fussy → `--grid 0.5`
- Accompaniment too busy → `--inner pad`; too static → `--inner pulse`; chords change too often → `--harmony-slot 2` or `4`
- Melody in an awkward octave → `--melody-octave`

## Testing

`tools/make_test_song.py` synthesizes a song with known content (C major, I–V–vi–IV, 100 BPM,
melody, bass, piano, drums) for checking changes to the pipeline:

```bash
uv run python tools/make_test_song.py /tmp/testsong   # writes testsong.mid/.wav/.mp3
uv run pop2quartet /tmp/testsong.mp3
```

On this song the pipeline recovers the bass line, chords, bar lines and key. It has also been run
on short clips of real songs with vocals from the free
[MUSDB18](https://sigsep.github.io/datasets/musdb.html) sample set.

## Ideas for later

- Automatic time-signature detection (currently assumes 4/4)
- More inner-voice textures (arpeggios, counter-melodies) and articulations (slurs, pizzicato)

## Code map

- `separate.py` – Demucs stem separation
- `transcribe.py` – basic-pitch transcription, beat tracking, seconds → beats
- `arrange.py` – quantization, ghost-note removal, downbeat guess, octave fitting, chord
  detection, inner-voice voice leading
- `quality.py` – rule-based checks and the score `--auto` uses to pick settings
- `export.py` – key detection, MusicXML (music21), MIDI, preview (MuseScore or fluidsynth)
