# Design: generative soundscapes

Status: implemented. This replaces the song form, the eight styles and their kits.

The decision behind it: after a day of chasing radio-quality tracks in eight genres (and getting a mishmash of bells,
disjointed melodies and styles that all sounded alike) the project went back to its roots, which is a soundscape.
One long, slowly evolving piece per room and day, in five styles that can be told apart from the first bar.

## Rules that every part follows

1. **The house never writes a note.** Data shapes texture, density, brightness, space and movement. The notes come
   from systems that run over time, over material that was composed. There is a test that fifty times as many events
   give under two and a half times the notes.
2. **Composed vocabulary, generative grammar.** Per style a library of cells, voicings, loops, progressions and
   timbres, chosen and transposed per day, developed slowly: layering, filter, register, rhythmic displacement,
   mutation by rule. Repetition with slow variation is the goal.
3. **Seeded, so that seeking and caching work.** Everything is seeded from the style, the room and the day. A score is
   made at once from the data (a few hundred milliseconds), and audio is made from absolute positions in time, so any
   second can be rendered without the ones before it (a seek renders eight seconds first to warm up the filters and the
   room). Two days are never alike because the days are dealt out in shuffled rounds of each family (`variation.py`).
4. **Band-limited, no glides, no blanket filters.** Oscillators read wavetables that hold only the partials that fit.
   Pitch is constant for the length of a note; a vibrato is at most 15 cents; the kick falls below 150 Hz. Every voice
   has a gain and a brightness of its own, and no note is scaled to a peak.
5. **Samples only if CC0 or public domain, read at the source.** See `entropy_fm/samples/LICENSES.md`.

## From data to a piece

```
history -> Series (timeline) -> DayAnalysis (day_analysis: activity curve, 5 to 7 stretches, moments)
        -> Plan (plan.py: length, arc, curves of the house, deals)
        -> Score (scape_<style>.py: notes, layers, curves)       <- the generative system
        -> Renderer (render.py: buses, filters, room, leveler, limiter) -> PCM, one second at a time
```

**Length.** 720 s plus up to 480 s for the busyness of the day plus 30 s for each of up to two anomalies, clamped to 600 to
1320 s.

**Arc.** The day is spread over the piece linearly: night (0 to 6 h), morning (6 to 11), daytime (11 to 17), evening
(17 to 22), late night (22 to 24). A partial day (today) spreads what has happened.

**Curves**, a value every tenth of a second of the piece: activity (the activity curve of the day stretched by its rank, so
that a day with one spike still has a quiet half and a lively half), power (appliances above a fridge's duty cycle),
brightness (temperature, sun and time), humidity, lights, the evening, runs of a dishwasher, a washer and the hob, and
the anomalies.

**Layers.** A layer is a bus: a gain and a presence curve, an optional low-pass whose cutoff the house moves, and sends to
the reverb and the echo. A note names its layer.

## The five systems

See the module docstrings of `scape_ambient.py`, `scape_berlin.py`, `scape_lofi.py`, `scape_ritual.py` and `scape_glitch.py`; each says
what the system is and how the house steers it. In short: tape loops that phase (Ambient), a step sequencer with
mutation rules and transposition (Berlin School), a band that varies a composed loop (Lo-fi), and a plucked call-and-response line answered by a slit drum and a woodblock of other loop lengths over a
breathing choir drone with stretched washes, a chant leader, a heartbeat and taiko (Ritual), and a film score of swells, brass-like stabs and a growling bass over half-time drums that the house cuts up, bar after bar, with stutters, rolls, reversals, pitch repeats, bit drops, tape stops, brakes and freezes, on the drums and on the instruments (Cinematic Glitch).

Ritual has three rules of its own. A voice that sustains has a hint of vibrato (4 to 6 cents, 4.5 to 5.5 Hz) that
fades in after a second or two, with a rate and a phase of each note's own (`dsp.pitch_vibrato`, the `vibrato` argument of
`voices.render_note`); it is far below the glide test (a few cents is not a glide). The stretched textures are the one
expensive thing, so they are made once from a few recipes per day and played again from the cache of notes
(`stretch.py`). And Ritual has no noise: the washes are rolled off in the spectrum of the stretch and the drums
are sines and damped modes, so the only broadband sound is the knock of a skin.

Cinematic Glitch has rules of its own. The data steers parameters and never writes a note: the heat of the house (activity and power, in the hours of the day) picks the row of the growl, the pattern of the kit and its level, while the notes come from the chord loop, the bass phrases and the patterns that the day deals. Its renderer (`glitch_render.py`) runs three seconds behind the clock so that an edit (a stutter, a tape stop, a brake, a freeze, a gate) is computed once from the unedited audio of a bus and overlaid, the same for every seek. The master gain is a constant (the arrangement is the dynamics: the night is quiet because it has no drums, not because a leveler pumps), the drum bus is glued and saturated, and the other layers duck from every kick. The kit is cut up often: where it plays, a bar in every one to two gets an edit on a busy day and one in every two to four on a quiet one (activity and the energy of the day move the chance), at a position on the bar that suits its kind (the ramps end on the bar line), and no kind plays more than twice in a row. Besides the older kinds there are a reversed beat or half beat, a pitch-shifted repeat (an octave up, down, or slowed: a jump, never a sweep), a harsh bit drop and an accelerating roll, and most edits cut the whole mix and about a fifth are on the strings, the growl or the stabs alone; the hottest one or two peaks of a busy stretch go all the way (never where the house is calm), with an edit in nearly every bar and gates that cut to silence. The edits that the events of the house bring are still rate-limited (at least two and a half bars apart with the kit, four to seven without), and none overlaps another on a bus. A brake is a fast exponential fall of the read speed over 60 to 150 ms, a cut of 1.5 ms, silence up to the next grid point and a snap back; it and the tape stop are the only glides the style has (the glide test allows them).

## Loudness and gain staging

The voices are brought to the same loudness by their own gains (`voices.gain_of`, RMS 0.09 at a reference note). A layer
is trimmed to its design level (`levels.DESIGN_DB`, measured by `scripts/gain_staging.py` into `levels.TRIMS`). The score
carries a gain curve estimated from its notes (`levels.feed_forward`, from the measured energy of every voice and hit), which
is what holds a sparse night and a dense afternoon at one loudness without pumping and is the same for every seek.
A trim of at most 2 dB follows (fast down, slow up), a gentle compressor sits before the limiter in the styles with drums,
and the look-ahead limiter has a ceiling of -2.4 dBFS, which leaves room for the inter-sample peaks under -1 dBTP.

## Cost

A pad is made once as a *core* (4.8 seconds with its release) and looped: the end of the sustain fades into the part that
precedes the loop point, chosen by correlation so that the tone is in phase. A note of 20 seconds then costs a few
milliseconds. Cores are cached (as float16), short notes and hits are cached, and a stream holds the notes that sound. The
tiers drop layers, saws and filters; Light runs two ladder filters at most, a Freeverb and no chorus.

## What was not done, and why

- The Mellotron archive (a conversion of a commercial CD-ROM set on archive.org): no license could be verified, so the
  Mellotron-like strings are a tape treatment of the VSCO strings and the choir is synthesised.
- Rhodes samples: jRhodes3d has contradictory licenses (CC0 and CC-BY-NC), and the Greg Sullivan e-pianos that the Discord GM
  bank labels CC0 are CC-BY at their source. The electric piano is the FM piano of the VCSL (CC0) or an FM voice.
- Nobody has listened to the result in this sandbox: every claim about sound is a measurement, not a judgment.
