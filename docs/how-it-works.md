# How Entropy FM works

The long version of what the [README](../README.md) says in short: how the house steers the music, what each style does, how an entity gets its sound, and how the library, the pins and the cache work. How the voices are made and how loud the pieces are is in [sound-design.md](sound-design.md).

## How the house drives the music

The house never plays a note. Each style has a vocabulary that was composed (motif cells,
sequences, chord voicings, loops, euclidean rhythms, timbres) and a generative system that plays it
over time: random processes with rules, seeded from the room, the day and the style, so a piece is
the same every time it is played and no two minutes are alike. What the house gives are *curves*, a
value every tenth of a second of the piece, and they set the parameters of the system:

| The house | Steers |
|---|---|
| Activity of the house (people about, appliances working, things happening, lights on; see *Rooms with and without motion sensors*) | How dense the layers are, how loud they play, how fast the sequences mutate, how many elements the groove has, and which of the extra layers come in on top of the base |
| Power of appliances | The presence of the bass and the drone, the cutoff and the accent of the sequencer; when it is high, the **growling bass** enters (often in Minimal and Berlin School, rarely in Ambient) |
| Temperature, the sun, the time | Brightness: the cutoff of the filters and the warmth of the sound |
| Humidity | The size of the reverb and the echo: the space. A spike far above what the room usually reads (a shower) is also a texture moment of its own |
| Lights and the evening | The harmonic colour (more extended chords, a string machine or a Juno pad that swells in) |
| An appliance running | A layer that fades in and out: the dishwasher brings in the second Berlin line, the drum loop of Lo-fi, the clicks and percussion of Minimal (Ritual does not use it); the washer the third line or the acid line |
| An unusual moment (a door at 03:00, a wash on a strange day) or a shower | A distinct texture event: a dip of the filters like a tape slowing down, a swell, a step of the harmony. Never a pitch glide. At most four a piece, of which at most two are showers. |

The material is dealt per room and day from a library: the days are dealt out in shuffled rounds of
the whole family, so a progression, a sequence cell, a voice or a patch never plays on two days
in a row and comes back as late as the size of the family allows.

### The six styles

| Style | The generative system | Voices (the core, and what a day may swap in) |
|---|---|---|
| **Ambient** (Eno, Stars of the Lid, Budd, Yoshimura) | **Tape loops** of 17 to 40 seconds that share no common measure and phase against each other (Music for Airports): each loop plays one long note and sometimes records a new one from the pitch set of the day; a Markov chain moves the pitch set between harmonic centres, so the harmony changes pass by pass; a felt piano is a Poisson process on the same set. No beat. | Tape strings, tape choir, bowed pad, glass pad, Juno pad, string machine; a felt piano (CC0 sample); sub, bowed or organ drone; the growl rarely |
| **Berlin School** (Tangerine Dream, Schulze) | A **step sequencer with rules**: three lines of 16, 12 and 8 or fewer steps on the same grid interlock and phase; every four bars each may drop a step, add one from the chord, move one, rotate or be pulled back; the whole sequence is transposed when the chord root moves (a progression of the day). Strings and choir hold the chord, a drawbar organ is the bed, a slow lead plays composed phrases of long notes into a tape echo. No drum kit, a soft pulse when busy. | Moog-style sequencer voice, Juno pluck or blip, tape strings or string machine, tape choir, drawbar organ, analog or flute lead, Reese or FM growl |
| **Lo-fi** (lo-fi hip hop, chillhop) | A **band that never plays the loop the same way twice**: a 70 to 88 BPM swung boom-bap bar, electric piano on jazzy 7th, 9th, 11th and 13th progressions, a comping rhythm, a round bass. The band swaps a chord for a related one, drifts the comping, adds ghost notes, fills, a chord chop or a plucked line, by chance that follows the house. Vinyl crackle, tape hiss and wobble at a whisper. At night the drums rest and it becomes an ambient piece of piano. | Sampled or FM electric piano, sub or Moog bass, strumstick or nylon pluck, a string machine or Juno pad in the evening, soft kick and cracked snare of the CC0 drum one-shots |
| **Minimal** (Hood, Hawtin, Villalobos) | **Euclidean percussion** with a shuffle and a probability on every step, a short **figure that mutates** (a step moves, an accent moves, a note drops out or returns) while its filter, its echo and its level walk at random, and a **state machine** that adds or removes one element every 8 or 16 bars (kick, hats, clicks, percussion, shaker, figure, bass, acid line, pad) towards the number of elements the activity asks for. 118 to 124 BPM, a soft deep kick, a deep rolling bass. | Blip or chord stab, sub or Moog bass, acid bass, Reese or FM growl, string machine, Juno or glass pad |
| **Ritual** (Hiroshi Yoshimura, Stars of the Lid, taiko at rest) | **One plucked line**, never two plucked voices at once: a short call-and-response cell (6 to 7.2 seconds, eight pulses) in a Japanese pentatonic (hirajoshi, in or kumoi on D, E, G or A, dealt per day) on a strumstick, a nylon string or a felt piano (dealt per day), round and low-passed; about one repeat in three moves one note by a step of the scale and never further than that from the cell. **Two woods answer it off the beat**: a slit drum (the log drum of the VCSL, read onto the scale) in a slow, sparse pattern of 9, 11 or 13 pulses and a woodblock with a few accents in 15 to 19 pulses, every hit on an off-beat of the pluck's pulse, so that they rest where the pluck plays, phase against it because their loops are of other lengths, and are dropped when they would fall within 120 ms of a plucked note; the activity of the house calls in the extra hits of each pattern. A few times in a piece, at a calm moment or in the quiet after an anomaly or a shower, a **shishi-odoshi clack** falls (a soft tick of the woodblock, then a hollow, lowered knock of the slit drum). Under them a **breathing drone** (cello section and Juno pad with a hint of vibrato, 4 to 6 cents at 4.5 to 5.5 Hz, that begins after a second or two and is different for every note) whose low-pass opens and closes over a dozen seconds and whose level swells over eleven, moving root, fifth, root, fourth over minutes; **stretched washes** of cello and violin sections and felt piano (Paulstretch, made once and cached); a **chant leader**, a cello section that answers the end of the pluck cell with two or three long notes of the scale and rests; and, in the busy stretches only, a soft **heartbeat** drum with a few **taiko** strikes on its pulse and a roll that accelerates into a strike at an unusual moment. The night is drone and washes alone; the pluck enters as the day wakes, then the slit drum and the woodblock, and they leave in the evening. No bells, no chimes, no noise bed. | Strumstick, nylon or felt piano pluck; slit drum and woodblock (VCSL samples); tape strings and Juno pad drone with a sub; stretched cello, violin and felt piano; chant cello section; odaiko, shime-daiko, wooden rim and a heartbeat tom, all synthesised |
| **Cinematic Glitch** (film trailer scores, half-time electronic, glitch edits) | **A film score cut up by the house.** A chord loop in a dark minor (aeolian or phrygian, on a root the day deals; two bars a chord, four while the house sleeps) carries saw swells that open as they rise, cello and violin sections, stretched washes and brass-like stabs. A **growling bass** (an oversampled saw stack through a vowel band-pass that wobbles in time with the tempo, with a sine sub; six rows from a held note to the full drop) enters first, then **hard half-time drums** (68 to 72 BPM, counted at 140: a punchy kick, a layered snare with a dark plate, hats, rim, ticks, ghost snares, five patterns of two variants each, a glued and saturated drum bus; the bass, pads, strings and stabs duck from every kick). The doors, switches and appliances of the house bring **glitch edits** on the grid, at most one in a couple of bars: stutters, tape stops, hard brakes of 60 to 150 ms (a fast fall to nothing, a cut, a gap, a snap back on the grid, sometimes doubled), grain and spectral freezes, a crush, gate chops, beat repeats. The peaks of the day and the anomalies are **drop sequences** (reversed swell, snare roll, reversed snare, a tape stop on the whole mix, silence, the drop). The night is swells, washes and distant booms alone; the kit leaves in the evening. | Its own voices (`glitch:*`: swells, stabs, booms, growl, kick, snare, hats), tape strings, cello and violin washes, the odaiko |

## Surprise and the character of the day

A baseline per entity is built from the previous 7 days
(the hours in which it is usually active, how often, and the scale of its values, see *Rooms with and without motion sensors*); routine fades into the texture and anomalies
come forward: a door at 03:00, a washing machine on a day or hour it does not usually run, a day far
warmer or colder than usual. They become texture events (at most three a piece). The day picks its
palette: weekday or weekend sets how dense it is, the mean temperature (or the weather) the mode,
how much the house was doing the density, and the hours with motion how much it moves. The key walks the circle of
fifths one step per week, up to three fifths either way and back over twelve weeks, and the season
brightens or darkens the sound: two Tuesdays a month apart sound related, not identical.

## Rooms with and without motion sensors

Motion and occupancy sensors are the primary sign that someone is in a room, and a house that has them is
read as it always was. The other sources are added alongside, and each is only there when the room has its
sensors: the activity of a room is the weighted mean of the terms that exist, so a room is blended by what it has.

| Source | What it says | Weight |
|---|---|---|
| Motion, occupancy, doors, windows, media players (binary) | How much happens | 0.4 |
| **CO2** (`sensor` with device class `carbon_dioxide`, or ppm with `co2` in its id) | The **level** against the room's usual range is how many people are there; a **rise** within twenty minutes is someone coming in and counts a little more. A level that builds in the night is somebody asleep: present, and counted at 45 percent. Never a note | 0.25 next to motion, 0.5 when there is none |
| Power: plugs, meters, a TV, a desk feed, the total | What is switched on. A load that cycles all day by itself (a fridge, by its role) is the floor and never counts. It also still feeds the power curve (the bass and the drone) | 0.4 (the loudest of them) |
| Lights | How long they are on, **how often they are switched**, and against the hours the light usually burns: a lamp that stays on says less than a kitchen light that comes on three times, and a day with more light than usual says more | 0.2 |
| Humidity over its usual level | Steam: someone is there (a shower, a pot) | 0.3 |
| Appliance phases, the hob | Something is being cooked or washed | 0.3 |

**A room is its own measure.** Numeric sensors are not scaled with the extremes of one day any more but with the
percentiles (5th, 50th, 95th, weighted by how long each value lasted) of the same entity over the previous 7 days,
which the baseline keeps (the same days as for surprise). A CO2 monitor in a bedroom that reaches 3700 ppm and one
in an office that reaches 1200 are both at the top of their scale on their busiest day, a quiet day is a low curve
and not stretched to the full scale, and a house with many sensors is not pinned at the maximum. A power sensor is
measured against the highest 98th percentile of a day in the week (the level of its loudest run), a temperature against
the usual level of its sensor. The curve of the day is stretched to at least 40 percent of its range only, and the rank
of the day (which is what gives a quiet day its highs and lows) is a smaller part of the activity. With fewer than
three days of history, or when the setting *Character of the day* is off, a sensor keeps the scale of its own day.

**Sensors that say nothing are left out.** A numeric sensor that reads the same all day and has not moved on the days
before (a temperature sensor that is stuck, a plant sensor that is unavailable), and a light or door that has not
changed in any of the days and today (a lamp that is always on), are dropped before the plan is made. A TV that was
off all day but was on the days before is only quiet, not dead. When every sensor of a track would be dropped, none is.

**A shower is an event.** A stretch of at most three hours in which a humidity sensor is at least 12 points (and
2.5 times its usual spread) over its usual level is a shower: a texture moment at the moment it climbs, in
every style (the same texture machinery as an anomaly: a dip of the filters, a swell, a step of the harmony), at most two a piece
and four texture moments together with the anomalies. The humidity still sets the reverb as before, and
humidity that stays up all day (the weather) is not a shower.

**A calm day is still a full piece.** Every style has a base: layers that always play, at a floor of their
strength, and the house moves how loud, dense, bright and moving they are, and brings in the layers that are special.
The appliance runs, the vacuum and the blinds are such special moments, never needed for a full sound.

| Style | The base that plays on any day | What the house brings in on top |
|---|---|---|
| Ambient | Three tape loops (a fourth and fifth on a busy day), the drone, the felt piano at its own rate, a quiet string machine triad all day | More loops, a denser piano, the evening strings swelling in, the growl rarely |
| Berlin School | Two sequencer lines (the second at 70 percent at least, and only on every second step while it is calm), strings, the organ bed, the choir at half strength, the slow lead | The third line, loudness and mutation of the lines, the pulse, toms and shaker, the growl |
| Lo-fi | Keys, bass, the kit at 60 percent by day (it rests at night), comping, quiet strings, the plucked line | The kit at full strength, ghost notes, chops, the evening strings swelling in |
| Minimal | Kick, hats, figure and bass from the first bar and never leaving; the state machine counts four elements on a calm day, up to eight | Clicks, percussion, shaker, acid line, pad |
| Ritual | Drone, sub, washes, the plucked line by day and in the evening (the night is drone and washes alone), the chant, the core hits of the slit drum | The extra hits of the slit drum and the woodblock, the heartbeat, the taiko, how often the cell moves |
| Cinematic Glitch | Swells, strings, washes, the growling bass and the first drums by day (the night is swells, washes and distant booms) | The harder patterns, the stabs, the glitch edits, the drop sequences |

## Browsing, the library, pins and the cache

- **Browsing and search** show every day that Home Assistant still has (the setting *Days to
  browse*, up to 10, the default of the recorder) for every room and the whole house, in every
  style, and every day that you have pinned, however old. Browsing reads no history of Home
  Assistant; the tracks of a day (and so the library, which lists one track per day of its
  window) read that day once, for its title and description, and the server keeps what it read.
- **Which rooms are offered.** The setting *Rooms to show* (nothing ticked is every room, also rooms added
  later) and *Hide quiet rooms* (on by default) decide which rooms are artists in browsing, search
  and the library, and which are pre-rendered. A room that is not offered is hidden, never deleted:
  its pins, the days you added and its cached audio stay where they are, and come back when the
  room is offered again. It can still be looked up by id, so a queue that holds it plays on. The
  settings win over labels (a label can only mute entities, not bring a room back). *Whole House*
  is made of the ticked rooms when *Whole House uses only these rooms* is on (the default), and of
  every room when it is off; the rooms it is made of are part of the key of its cached audio.
  The server only notices a change of the offered rooms at its next library sync, so when the
  provider loads (the server reloads it after a change of settings) and the offered rooms differ
  from the ones stored last time in the storage of the provider, it asks the music controller of
  the server, a few seconds later and in the background, for a sync of this source only. Nothing
  is asked when Home Assistant cannot tell the rooms or no room is offered. The library itself is
  never written to.
- **Quiet rooms.** A room is quiet when none of its entities that make a note or an event (after the
  role filtering: no muted, diagnostic or excluded entity) changed state in the window that can be
  browsed. Temperature, humidity, the sun and the like only colour a day, so a room that has only
  those is quiet. It is read from `last_changed` of the current states, which are read for the
  catalog anyway, so nothing is read from the history; an entity that Home Assistant does not say
  anything about counts as having changed, so a room is only hidden when it is known to be quiet.
  Only a state that last changed before the first day of the window did not change in it, so a
  restart of Home Assistant inside the window (which moves `last_changed` of many entities) keeps
  rooms visible until it is out of the window: the check errs towards showing a room. A room that is
  merely quiet on one day still offers that day.
- **The library is rooms, days and soundscapes.** Every room (a Home Assistant area that has
  something that can play) is an **artist**, plus *Whole House*, so the Artists overview is filled
  without doing anything. An **album** is a day of a room. The **tracks** of an album are that
  day's *Soundscape*, once in each style (six tracks, the style is the version). Single entities
  are never tracks: they only shape the sound. The source is a streaming provider in the sense of
  Music Assistant: what it offers is larger than the library.
- **The library holds a window of days.** For every room and for the whole house, the albums of the
  last *Days in library* days (default 7, never more than *Days to browse* and so than the
  recorder keeps) are in the library, counted back from today, so the window rolls on at
  midnight. Every track of those albums is in the library too: the soundscape in each style that
  is offered. Music Assistant syncs and shows all of it; what you keep is what you heart.
- **There is no library edit.** The source declares the heart features for artists, albums and tracks
  (`FAVORITE_*_EDIT`) and not the `LIBRARY_*_EDIT` ones, so the library button of Music Assistant only
  acts on its own database and the next sync lists the days again. An earlier version let you remove an
  album or track and left it out for good; that list is gone, and the file of the provider is cleaned
  of it at the first start (the albums, tracks and followed rooms that were in it are hearts now).
- **A change of *Piece length* or of the rooms asks for a sync** of this source, once, at the next
  load. Music Assistant adds new items from a sync, but it does not rewrite the name, the length or
  the description of a library track it already has (see the README).
- **A heart pins the day.** Music Assistant calls `set_favorite` when you heart an album or a track,
  and the event timeline of that room on that day, compressed (a few hundred bytes to a few
  kilobytes), is written to the folder `entropy_fm` of the storage path of the server, never into
  the folder of the provider. Hearting the whole house pins every room. A pinned day stays in the
  library (with all its tracks) after it leaves the window and after Home Assistant has purged it.
  The provider reports the heart back (`favorite` is true for what you hearted, and never false, which
  Music Assistant reads as a dislike). Taking the heart off deletes the pins that nothing else needs,
  and the day ages out of the library with the window; a dislike counts as no heart.
- **A heart on a room keeps every day of it,** like *Keep every day forever*, and the same check pins
  each new day when it is over.
- **Keep every day forever** (a setting with a list of your rooms) pins every day of the
  chosen rooms, the days that can be browsed at once and each new day when it is over (a check
  every six hours, which also pins again a day that was pinned while it was still going on).
  A room that is not offered (see above) is not pinned while it is hidden. Taking a room out of the list (or the heart off a room) lets its pins go, unless an album, a track or the other of the two needs them.
- **The sync of Music Assistant keeps pinned days.** A library sync lists the window and every
  pinned day. For a streaming provider the server never deletes an item that is no longer
  listed: it marks it as no longer in the library and keeps the record, so nothing a pin keeps is
  lost (a test runs the sync).
- **Pinned days play after Home Assistant has purged them,** from the pin. They are rendered with
  the engine of the day you play them, so a better engine makes old days sound better. The pin
  stores the version of the engine of the day it was made, and no audio.
- **The audio cache** keeps the rendered track of a *past* day as Opus audio (96 kbit/s) in an Ogg
  file in the cache folder of the server (`entropy_fm`); Music Assistant plays it as an Ogg file.
  A track that plays from the start to its very end is recorded while it plays, through ffmpeg;
  the next time it is played from the file, which costs next to no CPU and can be seeked. Today is
  never cached, a track that is stopped halfway leaves nothing (the file is written under a
  temporary name and moved into place at the end), a file that looks damaged is deleted and
  rendered again, and when the folder is over the limit the files that were used least recently
  go first. The size is a setting; 0 turns the cache off. The key of a file holds the engine
  version, the tier, the track (room, day, style), the piece length, for the whole house the rooms
  it is made of, the key and scale, the activity offset and the other settings that change the sound.
- **The pre-render** fills that cache before anyone asks (setting *Pre-render past days*, pinned
  rooms by default). Shortly after midnight of the house (ten minutes plus up to half an hour of
  jitter) and once three minutes after the start, it takes the tracks of yesterday that the cache
  does not have: with *Pinned rooms*, every offered style of the rooms in *Keep every day forever*
  and of the albums hearted for that day, and the tracks that were hearted; with
  *All rooms*, also every room and the whole house in the default style. It renders one track at
  a time through the same recorder and the same renderer as a live stream (so the file is the one
  a stream would have made), in a worker thread of low priority, with a pause after every second
  of audio. It waits while any Entropy FM stream is rendering, stops at once when the provider
  unloads, does nothing when the cache is off, and never fills the cache past 80 percent, so that
  it does not push out what you played. It logs one line per run.

- **Earlier versions.** Tracks that were added as songs or as full-day timelines are moved to the soundscape of the
  same day and style, with the pins of their day. The timeline track and the *Time compression* setting
  are gone.

## Piece length

*Piece length* is *Full* (the default: 10 to 22 minutes, as long as the day was busy) or
*Condensed*. A condensed piece is made from the same house data, the same plan (the same
curves, deals, key and mode) and the same composer; only the clock is faster: the day is spread over
about 7 minutes in the slow styles (Ambient, Berlin School, Ritual) and about 5 in the fast ones
(Lo-fi, Minimal, Cinematic Glitch). The length is fixed for a style, so the track shows the
length it plays. What keeps it from getting crowded:

- Unusual moments (anomalies) are kept, the strongest first, at most three and at least 25 seconds
  apart.
- The little things (a switch, a door, an appliance starting) are thinned to one every 6 seconds,
  appliances, hobs, phases and doors before lights, motion and blips. Only Cinematic Glitch turns
  them into edits, and it keeps its own rate limits (a gap of two and a half to seven bars between edits) and, when
  condensed, at most three drops.
- The curves of activity, brightness, lights and power are smoothed over at least 12 seconds, so that
  a build or a fade of a layer is never faster than a few seconds; the fades at the start and the
  end are the same as in a full piece.
- The density of the notes is not driven by the number of events but by the activity curve, so it
  stays about what it is in a full piece (a test checks that notes per minute stay within 30 percent).

The soundscape of today still needs three hours of the day to have happened, whatever the length.
Full and Condensed are cached and pre-rendered apart (the length is in the audio cache key).

## How entities get their sound

The role of an entity does not depend on the language of its name. Layers are tried in order and
the first that has an answer wins:

1. **You.** A Home Assistant **label** named `entropy_fm_<role>` on the entity or on its device, or a
   line `entity_id=role` in the *Role overrides* setting. The settings win over a label.
2. **The integration.** The translation key of the entity, its platform and the model of its device,
   for power and energy sensors. A small table of appliance words in English, which is how
   integrations such as Home Connect, Miele, SmartThings or LG ThinQ describe their devices. Best
   effort: it only helps when the integration exposes the appliance type.
3. **Domain and device class.** Power, energy, temperature, humidity, carbon dioxide, motion, occupancy, door,
   window, opening, light, switch, `sun.sun`. Language independent.
4. **Name keywords, a last resort**, only to tell a dishwasher, washer, dryer, fridge or oven from a
   plain power meter. English, German, Dutch, French, Spanish, Italian and Polish.

To tell Entropy FM what a power sensor is, create a label in Home Assistant (Settings > Areas,
labels & zones > Labels) called `entropy_fm_dishwasher`, `entropy_fm_washer`, `entropy_fm_dryer`,
`entropy_fm_fridge`, `entropy_fm_oven`, `entropy_fm_drone`, `entropy_fm_pluck`, `entropy_fm_chord`,
`entropy_fm_blip`, `entropy_fm_pad`, `entropy_fm_temperature`, `entropy_fm_humidity`, `entropy_fm_co2` or
`entropy_fm_mute`, and put it on the entity or its device. `mute` keeps an entity out of the music.

Every track says why an entity plays the part that it does, in its description, for example
`sensor.kitchen_machine_power: role: dishwasher (label)`. The diagnostics of the provider list the
role of every entity.

## What each kind of entity does

An entity never plays a note. Its role (above) decides which curve it feeds, and the curve steers the
system of the style (see *How the house drives the music*):

| Entity | Feeds |
|---|---|
| `sensor` power or energy (W, kW; kWh as a rate): oven, TV, a desk feed, the total, a plain meter | **Power**: the bass and the drone, the cutoff and the accent of the sequencer; and **activity**, as what is switched on (against the highest run of the week). A fridge that cycles all day is the floor: it counts for nothing, and only what rises above it is load. |
| `sensor` power of a **dishwasher** | **A run**: while it works, a layer of its own fades in (the second Berlin line, the Lo-fi drum loop, the Minimal clicks and percussion) |
| `sensor` power of a **washer or dryer** | **A run**: the third Berlin line, the acid line of Minimal |
| `sensor` enum (operation state, program phase), `select`, the power level of a **hob** zone | **A run** (cooking) and activity |
| `sensor` **carbon dioxide** (ppm) | **Presence**, where there are no motion sensors and alongside them where there are: the level is how many people, a rise is someone entering, a build in the night is a sleeper. Never a note |
| `sensor` temperature | **Brightness**, with the sun and the time of day (against the usual level of the sensor) |
| `sensor` humidity | **Space**: the size of the room and the echo. A spike far over the usual level is a **shower**, a texture moment of its own, and steam is a little activity |
| `binary_sensor` motion, occupancy, door, window, opening; `media_player`; other on/off entities | **Activity** (how many things happen). Never a note: fifty times as many events give less than two and a half times the notes. |
| `light`, `switch` | **Lights**: how extended the chords are and when strings and choir swell in; and **activity**: how long they are on, how often they are switched, against the hours of the week. A light that never changes is left out |
| `sun.sun`, or `sensor.sun_solar_elevation` when you have it | **Brightness** and the arc of the day |
| `cover`, `sensor` progress and the like | Activity, or nothing |

Events are rate limited (one per entity every ten seconds), the changes of one device that come together
count once, and sensors that say the same thing count once. The provider logs a hint when it drops one,
with the label `entropy_fm_mute` to leave it out for good. Diagnostics (signal, connectivity, firmware, update,
versions, wifi, voltage, current, entities that Home Assistant files as diagnostic or config) and the
energy totals of devices that have a power sensor are ignored. Home Assistant keeps no attributes in
its compact history, so the elevation of the sun is only known with `sensor.sun_solar_elevation`
(disabled by default in Home Assistant); without it the sun is a smoothed day and night.
