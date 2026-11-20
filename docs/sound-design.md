# Sound design

How the voices are made, where the samples come from and how loud the pieces are. For what the styles do, see [how-it-works.md](how-it-works.md).

## The voice library

All six styles draw on one library of 26 voices, each with a character, a gain and a brightness of
its own (no blanket filter, no per-note normalising): a warm round sub, a Moog-style ladder bass, a
growling Reese (two or three detuned band-limited saws, waveshaped before a moving resonant low-pass,
with a clean sub under it), an FM growl (the index moves, never the pitch), an acid bass; a string
machine, a Juno-style pad, a supersaw, tape strings, a tape choir, a glassy wavetable pad, a bowed
pad, a drawbar organ; a singing analog lead and a flute-like lead (vibrato at most 15 cents), an FM
electric piano, a felt piano, a soft grand piano, a sampled FM piano, a Juno pluck, a nylon
Karplus-Strong pluck and a strumstick, a cello section for a slow line; a Moog-style sequencer voice, a blip and a chord stab. Every
oscillator is band-limited (a wavetable that holds only the partials that fit), a note never glides
(a vibrato is at most 15 cents and the kick's thump falls below 150 Hz), and a long pad is made
once and looped where the tone is in phase, which is what lets a 20 minute piece render on a
Raspberry Pi. `scripts/voice_report.py` measures every voice: centroid, attack, harmonic profile,
modulation and the energy that is not on a harmonic (under 2 percent for the synthesised ones).

## Samples

Only CC0 and public domain samples are bundled (about 6.4 MB of OGG), and every license was read at
the original repository, not at a mirror: the upright piano and the string sections of VSCO 2
Community Edition, the FM electric piano, strumstick, drum one-shots, slit drum and woodblock of the VCSL (all by
Versilian Studios, CC0 1.0) and the Splendid Grand Piano (public domain, AKAI). The list with the
source, author and license of every folder, and of what was looked at and refused, is in
`entropy_fm/samples/LICENSES.md`; `scripts/fetch_samples.py` rebuilds all of it. The Mellotron archive
could not be verified and is not used: the Mellotron-like strings and choir are tape strings (a
sample with the wow and flutter of a tape) and a synthesised choir.

## Loudness

Every soundscape is gain-staged before the master, so that the limiter has little to do. Each layer
of each style is trimmed to a design level (`entropy_fm/levels.py`, measured by
`scripts/gain_staging.py`), a gain estimated from the notes of the score holds the loudness of the
piece the same way for every seek (a sparse night is lifted, a dense afternoon is held down), a
slow trim of at most 2 dB follows, and a look-ahead peak limiter with a ceiling of -2.4 dBFS
finishes. The styles with drums have a gentle compressor in front of the limiter. The aims:
integrated loudness -18 to -20 LUFS for Ambient, -16 to -18 for Berlin School and Lo-fi, -15 to -17
for Minimal, -18 to -20 for Ritual, -14.5 to -15.5 for Cinematic Glitch; the loudest three seconds at most one LU above the integrated loudness; true peak under
-1 dBTP; the limiter taking at most 2 dB. `scripts/loudness_report.py` measures them, and tests check
them. The numbers of the last run are in [the notes of the demo files](development.md#run-the-demo).

| style | piece | integrated LUFS | target | max short-term | true peak dBTP | LRA | mean GR dB | max GR dB | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ambient | demo excerpts | -19.7 | -20 to -18 | -18.0 (+1.7) | -9.6 | 3.3 | 0.00 | 0.0 | short-term -18.0 more than 1 LU over |
| ambient | kitchen excerpts | -20.5 | -20 to -18 | -18.3 (+2.2) | -8.0 | 3.5 | 0.00 | 0.7 | short-term -18.3 more than 1 LU over |
| ambient | full demo piece | -19.9 | -20 to -18 | -17.7 (+2.2) | -8.3 | 3.6 | 0.00 | 0.0 | short-term -17.7 more than 1 LU over |
| berlin_school | demo excerpts | -17.8 | -18 to -16 | -16.7 (+1.1) | -5.1 | 1.4 | 0.00 | 0.0 | short-term -16.7 more than 1 LU over |
| berlin_school | kitchen excerpts | -17.6 | -18 to -16 | -16.2 (+1.4) | -4.6 | 2.2 | 0.00 | 0.0 | short-term -16.2 more than 1 LU over |
| berlin_school | full demo piece | -17.6 | -18 to -16 | -15.9 (+1.7) | -3.9 | 1.5 | 0.00 | 0.0 | short-term -15.9 more than 1 LU over |
| lofi | demo excerpts | -16.2 | -18 to -16 | -15.1 (+1.1) | -5.7 | 0.7 | 0.00 | 0.0 | short-term -15.1 more than 1 LU over |
| lofi | kitchen excerpts | -17.8 | -18 to -16 | -15.8 (+2.0) | -4.8 | 5.8 | 0.00 | 0.0 | short-term -15.8 more than 1 LU over |
| minimal | demo excerpts | -16.3 | -17 to -15 | -15.6 (+0.7) | -1.2 | 2.0 | 0.00 | 0.2 | ok |
| minimal | kitchen excerpts | -16.8 | -17 to -15 | -15.5 (+1.3) | -4.2 | 2.0 | 0.00 | 0.0 | short-term -15.5 more than 1 LU over |
| ritual | demo excerpts | -18.7 | -20 to -18 | -17.1 (+1.6) | -8.5 | 3.1 | 0.00 | 0.0 | short-term -17.1 more than 1 LU over |
| ritual | kitchen excerpts | -18.9 | -20 to -18 | -16.9 (+2.0) | -8.3 | 2.9 | 0.00 | 0.0 | short-term -16.9 more than 1 LU over |
| glitch | demo excerpts | -14.7 | -15.5 to -14.5 | -11.3 (+3.4) | -2.4 | 5.3 | 0.00 | 1.1 | ok (its own caps: short-term at most 4 LU over, limiter at most 3 dB) |
| glitch | kitchen excerpts | -15.3 | -15.5 to -14.5 | -11.4 (+3.9) | -2.3 | 5.2 | 0.00 | 2.1 | ok (its own caps) |

On the short-term column: the brief asked for the loudest short-term value to stay at least 3 LU below integrated plus 4, which this table reads as at most +1 LU. Most pieces are 0.7 to 2.2 LU over, so that cap is not met everywhere; Ambient is the furthest.
