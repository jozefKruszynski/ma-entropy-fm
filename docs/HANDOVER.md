# Handover: make days sound different

Written 2026-11-20 at the move from the cloud session to a local Claude Code session on the Mac.
This is a plan, not a record of finished work. Nothing below the "Plan" heading is built yet.

## Why

Jozef's verdict after testing: everything sounds samey and empty. Days and rooms are hard to tell
apart, even for the person who knows what happened in the house.

The cause is the design, not one bug. Since the start, house data only steers *parameters* inside a
style: density, filter brightness, reverb size, a layer more or less. The identity of a piece
(tempo, key and mode, voices, groove, form) is chosen by `plan.deal(...)`, which rotates material in
shuffled rounds per room and day. That gives variety, but it is random variety: it has nothing to do
with the day, and inside one style the result sounds like the same track with slightly different
weather.

The goal now: **the data makes big, audible choices**. A quiet Sunday and a busy Monday should
differ like two tracks on the same album, not like one track at two volumes. Jozef asked for this
explicitly and asked that every structural change follow music theory, so the result is still
something a person would want to hear.

## Plan

### 1. A day profile that drives identity

Build one compact `DayCharacter` per room and day from what the provider already reads (the
`energy.py` curves, `references.py` per-room scales, `day_analysis.py` facts):

- energy: overall level against the room's own week (low, medium, high), and its shape (morning,
  evening, all day, a single peak);
- warmth: temperature, sun and weather against the season;
- rhythm of life: how many distinct active stretches, how regular they are;
- events: showers, anomalies, appliance runs, a late night;
- weekday or day off (workday sensor or calendar when there is one, else the weekday).

Each feature must move clearly between Jozef's real days (see "Test data" below). If a feature
barely varies across a week, it is not allowed to drive a big choice.

### 2. Data picks the identity, the dealer only breaks ties

Replace random dealing of identity material with choices mapped from `DayCharacter`. Keep the
dealer only for choosing among options that the data rates as equally fitting, so two similar days
still differ in detail.

Per style, the choices that must follow the data (each with music-theory constraints):

| Choice | Driven by | Music-theory rules |
|---|---|---|
| Tempo | energy level and shape | Stay inside the style's idiomatic range (Lo-fi 70 to 90, Berlin 100 to 140 sequencer rates, Glitch half-time 68 to 74). Change tempo only between pieces, never mid-piece. |
| Key and mode | warmth, day off, season | Bright modes (lydian, ionian, mixolydian) for warm, busy days; dorian, aeolian, phrygian for cold, quiet or late days. Keep the circle-of-fifths walk between consecutive days so adjacent days stay related. Ritual stays in its pentatonics (hirajoshi, in, kumoi) and picks among them. |
| Harmonic rhythm and progression family | energy shape | Slow harmonic rhythm (4 to 8 bars a chord) on quiet days, faster (1 to 2 bars) on busy ones. Use functional progressions where the style is tonal (Lo-fi ii-V-I and backdoor moves, Ambient modal plateaus, Berlin pedal-point with chord-root moves by fourth or fifth). Voice-lead between chords: common tones held, other voices by step. |
| Lead and bass voices | warmth and energy | Warm days: rounder voices (felt piano, Juno, strings). Cold or late: glassier, darker voices. Busy days may bring the growl bass in Berlin and Glitch. Keep register separation: bass below C3, pads in the middle, lead above. |
| Groove and kit | energy and rhythm of life | Quiet days: half-time or no kit, more swing in Lo-fi. Busy days: full kit, straighter, more syncopation. Patterns from real genre vocabularies (boom-bap, 140 half-time), never random hits. |
| Form | energy shape | A morning-heavy day builds early and fades; an evening day builds late; a single-peak day has one clear climax at the peak's position; a flat day is a plateau with slow evolution. Sections in multiples of 4 or 8 bars, transitions on phrase boundaries, a build always resolves. |
| Room signature | the room (fixed) | A short motif (3 to 5 notes) or a signature timbre per room, transposed into the day's key and mode, appearing at least once per piece. It makes "this is the kitchen" recognisable across days and styles. |
| Contrast range | the room's own week | Quiet days are really sparse and slow, busy days really full. Use the full range of the parameters, not 10 % either way. |

Events (showers, anomalies, runs) keep their current jobs, but should also be allowed to cause a
section change (a breakdown, a modulation to the relative key, a new layer entering on a phrase
boundary).

### 3. Proof: the contrast reel

Before any tuning by ear, build `scripts/contrast_reel.py`. It renders 20 seconds from the busiest
stretch of six room and day pairs, back to back, about two minutes in all, with a one-line label of
the day's character in the log. Pairs: the same room on its quietest and busiest day, two rooms on
the same day, and Whole House on a weekday and a day off.

**The work counts as successful only if the reel sounds clearly different within two minutes.**
Also report a measured difference (tempo, key, voices, notes per minute, spectral centroid) per pair.

### 4. Test data: Jozef's real house

Render from his real Home Assistant history, not only the synthetic sparse house. His house has no
motion sensors. What moves daily: CO2 in the living room, bedroom and office, bathroom humidity
(showers), TV and desk plugs, total power, kitchen, office and living-room lights, the dishwasher
door. Contrasting days from the week of 11 to 19 Nov 2026:

| Room | Quiet day | Busy day | Why |
|---|---|---|---|
| Living Room | Sat 14 Nov | Mon 16 Nov | CO2 max 760 ppm against 2,600 ppm |
| Bedroom | Sun 15 Nov | Tue 17 Nov | CO2 mean 520 ppm against a 3,700 ppm overnight peak |
| Bathroom | Sun 15 Nov | Thu 19 Nov | no shower against an 84 % spike |
| Office | Sun 15 Nov | Thu 19 Nov | day off against a workday |

Home Assistant keeps about 10 days of history, so pull these days into a local fixture early (numeric
series only, generic entity ids, kept out of the shipped package).

## Constraints that still hold

- Keep the voices Jozef likes; change which voices a day uses, not how they are designed. No bells,
  no chimes, no broadband noise beds, no continuous pitch glides above 150 Hz.
- Loudness bands per style and the look-ahead limiter stay. CPU budgets per tier stay
  (`scripts/bench.py`; Light at most 20 ms per audio second).
- Bump `ENGINE_VERSION` in `const.py` with every change to the sound, or cached days keep the old
  audio. (It was forgotten from Wednesday to Friday, which hid all sound changes in Jozef's tests.)
- Additive data rules: motion and occupancy stay the main presence signal where a house has them;
  CO2, plugs, lights and showers add to it. Dead or constant sensors are ignored.
- Never write into Music Assistant's library or database. Asking MA to sync this provider's own
  items is the one allowed exception (rooms or length changed).
- The rules in `AGENTS.md` and the community catalog rules.

## Decided so far (don't redo)

- Library: the sync shows the shown rooms, the recent days and all five tracks of each day; hearts
  keep days (pins). The library button does nothing special.
- Settings: "Rooms to show" (multi-select), "Keep every day forever", "Piece length" (Condensed,
  5 to 7 minutes, or Soundscape), "Pre-render past days", "Render quality" (Auto benchmarks the
  heaviest offered style).
- Track titles are generated per room, day and style; descriptions carry the style's history.
  Soundscape tracks (the long piece) have their own id (a last part `soundscape`) and the version "Soundscape".
- Covers are drawn only; AI covers and "Redraw covers" are hidden behind `AI_COVERS_ENABLED`.
  Room images are 24-hour clock figures from the room's week.
- Ritual: one pluck line, slit drum and woodblock (VCSL, CC0), shishi-odoshi clack, drones, washes,
  chant leader, heartbeat, taiko.

## Known limitations that need Music Assistant core changes

- MA never refreshes a stored item from a streaming provider during a sync (names, durations,
  descriptions, images); `_library_item_needs_update` only reacts to mapping or date_added changes.
  This breaks AI covers and makes old library tracks keep stale lengths and titles.
- The frontend shows no track descriptions and labels album descriptions "Review".

## How Jozef works

- Direct and concise; no em dashes in anything written for him.
- Delegate coding to smaller models (subagents) where the tool allows it.
- Treat plans as plans until he confirms they are done.
- He reviews all code himself before any catalog submission (Open Home Foundation AI policy).
- Durable decisions also go to his Memory MCP server (`memory_store`).
