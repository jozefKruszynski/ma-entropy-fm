# Entropy FM

Entropy FM is a community provider for [Music Assistant](https://music-assistant.io) that turns the
history of your [Home Assistant](https://www.home-assistant.io) into generative soundscapes: one long,
slowly evolving piece for each room and each day, in five styles. Your house does not write the
notes. It steers the texture, the density, the brightness, the space and the movement of music that
a generative system plays. The audio is made on your server while it plays, from the history it
reads from your own Home Assistant and from CC0 samples that ship with the provider. Nothing leaves
your server; what it keeps is the event timeline of the days you heart and, if you
want it, a cache of the days that were played, and the drawn covers.

Entropy FM is made and maintained by [@jozefKruszynski](https://github.com/jozefKruszynski), not by
the Music Assistant team. Questions and problems go to the
[issues of this repository](https://github.com/jozefKruszynski/ma-entropy-fm/issues).

## What it plays

| In Music Assistant | In Entropy FM |
|---|---|
| Artist | A Home Assistant area, such as "Kitchen", and one more, "Whole House" |
| Album | One day of one area: "Kitchen, Tue 17 Nov" |
| Track | The soundscape of that day, 10 to 22 minutes (5 to 7 when *Piece length* is Condensed), with a title of its own: "Dishwasher at Dusk (Ambient)". Each style you offer is its own version, so an album has up to five tracks. |

**Titles and descriptions.** Every track is named from what the day was like: the busiest hour, the
kind of thing that ran (a dishwasher, a hob, a door at night), the temperature, the weekday and how busy it
was, in the words of its style: "Dishwasher at Dusk (Ambient)", "Sequencer for a Busy Kitchen (Berlin School)",
"Door Fault 14:12 (Cinematic Glitch)", ceremonial for Ritual, warm for Lo-fi.
The same room, day and style always give the same title; the title of today can change as the day goes on, but
only once an hour. Titles hold kinds of things and times, never an entity id or a name, and a day without
history, or Home Assistant being away, gives a plain title of the style. The description of a track is a
short history of the style (Eno's tape loops, the Berlin School, lo-fi hip hop, Reich's phasing, the mood of
the Akira soundtrack, cinematic neurofunk), how the house drives that style, and one line about the day ("A busy
afternoon; a door at 03:12."). The description does not list which entity plays which part: the role of every
entity, and why it has it, is in the diagnostics of the provider. An album has a short review of its own (the room,
the weekday, what kind of day it was). Titles and descriptions follow the language of the server when it is Dutch,
and are English otherwise (see *Known limitations* for when a changed title shows).

A soundscape is as long as its day was busy, between 10 and 22 minutes (or, as a condensed piece,
about 7 minutes in Ambient, Berlin School and Ritual and about 5 in Lo-fi and Cinematic
Glitch, whatever the day), and the day is spread over it in order: the night is sparse and dark, the morning brightens, the busy part of the day is
denser and warmer, the evening is fuller, the late night thins out again. A soundscape of today
needs three hours of the day to have happened. Tracks can be seeked, and the same room, day and
style always sound the same.

| Style | What you hear |
|---|---|
| Ambient | Tape loops of different lengths that phase against each other, a felt piano, no beat |
| Berlin School | Interlocking sequencer lines that mutate by rule, strings, choir and a slow lead, and in the busy day a soft kick, warm toms and a light shaker under the grid |
| Lo-fi | A boom-bap band that never plays the loop the same way twice, electric piano, a little vinyl crackle |
| Ritual | One plucked line in a Japanese scale, answered off the beat by a slit drum and a woodblock, over a breathing drone, stretched strings, a chant, a soft heartbeat and taiko |
| Cinematic Glitch | Film-score swells and brass stabs, a growling bass and hard half-time drums, cut up by the doors, switches and appliances of the house |

**No motion sensors needed.** A motion or occupancy sensor is still the first sign that someone is in a room, but a
house without them is read from what it has: the CO2 of its rooms (the level is how many people, a rise is someone
coming in), plugs and meters, how long and how often the lights are on, and the humidity (a shower is a texture moment
of its own). Every sensor is measured against its own previous 7 days, so a quiet room still has highs and lows,
sensors that are dead or stuck are left out, and every style keeps a full base arrangement on a calm day: the house
shapes the density, the colour and the peaks.

How the house steers each style (a table of what every driver does in every style), and how an entity gets its sound, is in
[docs/how-it-works.md](docs/how-it-works.md); the voices, the samples and the loudness are in
[docs/sound-design.md](docs/sound-design.md).

The pieces are levelled to about -14.5 to -20 LUFS depending on the style, with a limiter at
-2.4 dBFS and a true peak under -1 dBTP, so they play at the level of ordinary music. Cinematic
Glitch has real low bass and drums: turn a speaker up slowly the first time.

## Add it

Entropy FM comes from the community catalog of Music Assistant. Everything in the community catalog
is used at your own risk.

1. Open the Marketplace in Music Assistant (Settings > Marketplace). An administrator enables the
   community catalog there once.
2. Find Entropy FM and install it. Music Assistant installs its two requirements, `numpy` and
   `pedalboard`, as wheels (Linux x86-64 and aarch64, Python 3.14).
3. Set up the **Home Assistant** plugin of Music Assistant first (Settings > Providers) if you have
   not: Entropy FM uses its connection. Without the plugin, turn off *Use the Home Assistant plugin*
   in the settings of Entropy FM and enter a URL and a long-lived access token.
4. Add Entropy FM under Settings > Music sources. The rooms are in the Artists overview at once;
   play a day, or heart an album to keep that day for good.

### The library and hearts

The overviews of Music Assistant (artists, albums and tracks) show the recent days of the rooms you show: every room and
the whole house is an artist, each of the last days an album, and each album has the day's soundscape as a track in
every style that is offered. Heart an album or a track to keep its day for good (it stays after it leaves the window and
after Home Assistant purges it); heart a room to keep every day of it; take the heart off and the day ages out as every
other day does. There is nothing to add to or remove from the library by hand.

## Settings

Every setting, with the label that Music Assistant shows, its default and what it does. Settings
marked *advanced* only show when Music Assistant shows advanced settings.

### Connection

| Setting | Default | Values | What it does |
|---|---|---|---|
| Use the Home Assistant plugin | on | on, off | Read Home Assistant through the plugin of Music Assistant. Off: connect with the URL and the token below. Changing it reloads the provider. |
| Home Assistant URL | empty | text | The address of Home Assistant, such as `http://homeassistant.local:8123`. Only shown and used without the plugin. |
| Long-lived access token | empty | secret text | A token from your Home Assistant profile. Only shown and used without the plugin. |
| Verify the certificate | on | on, off | Check the certificate of Home Assistant when the URL uses https. Advanced, and only shown without the plugin. |

### Library

| Setting | Default | Values | What it does |
|---|---|---|---|
| Days to browse | 10 | 1 to 10 | How many days back you can browse and search. The recorder of Home Assistant keeps 10 days by default. Days you hearted stay, however old. |
| Days in library | 7 | 1 to 10 | How many of the last days of every room are in the library, counted back from today (never more than *Days to browse*). The window rolls on at midnight. Every style of every one of these days is a track in the library. Days you heart stay for good, however old; the window only decides which recent days the library shows. To hide a room, untick it in *Rooms to show*. |
| Rooms to show | all (nothing ticked) | any of your rooms | The rooms that appear in Music Assistant: in browsing, search, the library and the pre-render. Nothing ticked shows every room, including rooms that Home Assistant gets later. A room you leave out disappears from those lists; what you pinned or cached for it is kept (hidden, not deleted) and is back when you tick the room again. A room is still found by its id, so a queue that holds it keeps playing. When this changes, the provider asks Music Assistant to re-sync its own items, so that the library views follow. |
| Whole House uses only these rooms | on | on, off | On: the *Whole House* soundscape is made of the ticked rooms only (every room when none is ticked). Off: it is made of every room, whatever is ticked. |
| Hide quiet rooms | on | on, off | Do not offer a room in which nothing that would make a note or an event (a door, a motion sensor, a light, an appliance and the like) has changed over the days you can browse. A room that has only temperature, humidity, sun or other static sensors after the role filtering counts as quiet too. A room that is merely quiet on one day still offers that day. |
| Keep every day forever | none | the rooms that are shown | Days of these rooms are saved permanently (pinned) as soon as they are over, even after Home Assistant purges its history. Hearting a room does the same. It does not show or hide rooms; the list only offers the rooms that are shown when some are ticked in *Rooms to show*. |
| Styles offered | all five | any of the five styles | Which styles each day comes in, as versions of the same soundscape. |
| Default style | Ambient | one of the five styles | The style that an id without a version plays, and that the *All rooms* pre-render uses for every room. It is always offered. The library lists every style that is offered, not only this one. |

### Sound

| Setting | Default | Values | What it does |
|---|---|---|---|
| Piece length | Full | Full (10 to 22 minutes), Condensed (5 to 7 minutes) | Full: as long as the day was busy, 10 to 22 minutes. Condensed: the same house data and plan in about 7 minutes (Ambient, Berlin School, Ritual) or about 5 minutes (Lo-fi, Cinematic Glitch). The hours of the day pass faster, the little things of the house are thinned out and unusual moments are kept (see [docs/how-it-works.md](docs/how-it-works.md#piece-length)). Tracks show the length that they play (a full piece is read from its day when it is listed). |
| Key | D | C, C#, D, D#, E, F, F#, G, G#, A, A#, B | The root note that the pitches are tuned to; the key of a day walks away from it and back. |
| Scale | Dorian | Dorian, Natural minor (Aeolian), Mixolydian, Lydian, Major (Ionian), Minor pentatonic, Major pentatonic | The scale of the pieces. A style that does not have the scale uses its own: Berlin School is in a minor mode, Ritual keeps a Japanese pentatonic, Cinematic Glitch is in aeolian or phrygian. |
| Activity offset | 0 | -50 to 50 | Points added to how lively the house seems. The activity brings layers in and out and sets how dense and how fast they are; it never writes notes. |
| Character of the day | on | on, off | Baseline, surprise, the palette of the day and the drift of the key. Off: the plain sound, and no extra history is read. |
| Include presence | off | on, off | Also play persons and device trackers, which tells who is home (advanced). |
| Role overrides | empty | `entity_id=role` lines | Choose the role of an entity yourself, one line each (advanced). Roles: dishwasher, washer, washing_machine, dryer, drum, drone, fridge, oven, temperature, humidity, co2, air, pluck, chord, blip, pad, event, hob, media, cover, progress, sun, mute. A label in Home Assistant (`entropy_fm_<role>`) is the other way and wins over this list. |
| Extra exclusions | empty | comma separated patterns | Entities to keep out, such as `sensor.bedroom_*, light.hall_*` (advanced). |

### Covers

| Setting | Default | Values | What it does |
|---|---|---|---|
| Covers | Drawn from the day | Off, Drawn from the day | Off: no covers and no room images. Drawn: a picture made on your server from the events of the day, in the look of the covers of Music Assistant; it is the same every time. |

AI covers (an image engine of Music Assistant drawing the covers, with a style, a prompt, a title overlay, an hourly limit
and a *Redraw covers* button) are built but hidden in this release, see *Known limitations*. A *Covers* value of "AI image
engine" that an earlier install stored is read as drawn covers.

### Performance

| Setting | Default | Values | What it does |
|---|---|---|---|
| Render quality | Auto | Auto, Light, Standard, Rich | What a second of music may cost in processor time. Auto benchmarks at start and picks the richest tier that fits (see below). |
| Audio cache size (MB) | 500 | 0 to 20000 | Rendered tracks of past days are kept as Opus audio in Ogg files in the cache folder of the server, so playing them again costs no processor and can be seeked. The least recently used go first above this size. 0 turns the cache off. |
| Pre-render past days | Pinned rooms | Off, Pinned rooms, All rooms | Render the tracks of yesterday into the audio cache shortly after midnight and once after the start (see below). |

The settings that change the sound (piece length, key, scale, activity offset, character of the day,
presence, roles, exclusions, quality) are part of the key of an audio cache file, and so are the
rooms that the whole house is made of, so a changed setting plays a fresh render and never an old
file; Full and Condensed never share a file.

## Render quality, and small machines

Music Assistant often runs on a Raspberry Pi, so the engine has three tiers with hard caps on
layers, voices and effects. A second of audio costs, on one core of an x86-64 server (CPU
milliseconds per second of audio, the kitchen of an anomaly day of the demo house, the whole piece
from the first second to the last, with the cache of notes empty so that making every note once is
in it; measured with `scripts/bench.py`):

CPU milliseconds per second of audio, kitchen, cold, one core:

| style | Light | Standard | Rich |
|---|---|---|---|
| ambient | 13.8 | 22.9 | 24.2 |
| berlin_school | 18.0 | 43.2 | 46.0 |
| lofi | 17.5 | 30.3 | 30.5 |
| ritual | 14.9 | 34.6 | 32.6 |
| glitch (Cinematic Glitch) | 15.8 | 46.6 | 45.5 |
| budget | 20 | 60 | 120 |

The budget is what a tier aims at. Light meets its 20 ms in every style, and every
style is inside the budget of Standard and Rich. Berlin School is the dearest style in Light and, by a hair, in Rich;
Cinematic Glitch is in Standard. Glitch's swells and its growling bass are its dearest notes: each is made once and kept (the cache
of notes keeps them in a pool of their own, so that nothing pushes them out and none is made twice), and the swells of
one chord share the saws that they are made from. A few numbers move by a few percent from one run to the next (this
table is from one run of `scripts/bench.py` on a two-core x86-64 server).

| | Light | Standard | Rich |
|---|---|---|---|
| Layers that play at the same time | 4 | 6 | 9 |
| Saws in the supersaw, the Reese and the choir | 3, 2, 2 | 5, 3, 3 | 7, 3, 3 |
| Room | one Freeverb | convolution with a 2.6 second room | convolution with a 5 second hall |
| Filters (ladder low-pass) on buses | the two busiest | all | all |
| Chorus, tape wobble | no | yes | yes |
| Granular shimmer (an octave up, in the reverb) | no | no | yes |
| Notes kept in memory | 64 MB | 128 MB | 256 MB |

Cinematic Glitch has its own room (a 4.5 second hall) from Standard up. Its swells, stabs and
growl notes live in a pool of the cache of notes of their own (up to 85 percent on top of the size
in the table), and the saw stacks of the swells use up to 48 MB more; it needs about 150 MB at
Standard for a whole day.

**Auto** (the default) renders a few seconds of a small invented house in each tier when the
provider starts and measures the CPU time one second of audio takes on this machine. It does that
with the dearest style that is offered (Cinematic Glitch when it is; otherwise Berlin School, then
Ritual, Lo-fi, Ambient), and picks the richest tier that would use less than 30 percent
of one core in real time. A tier that Auto picks therefore fits every style that is offered, not
only the cheap ones. The benchmark takes about 20 seconds once, on a background thread (it renders a short stretch of
the invented house twice per tier and times the second pass), and its
result is kept in the cache of the server for a month (per engine version and heaviest style).

**For a small machine:** choose **Light**, or leave Auto. The notes are the same in every tier (the
composer decides what plays; a tier only drops the second line of a sequence, the choir and the
pulse of Berlin School and the like where it has fewer layers). Leave the **audio cache** on and
play a day to the end once: after that it comes from a file and costs almost nothing. A soundscape
of 20 minutes at 96 kbit/s is 15 MB, so 500 MB is about 30 of them.

**Pre-render.** With *Pre-render past days* on (the default is *Pinned rooms*), Entropy FM renders
yesterday's tracks into the audio cache so that they are ready when you want them: shortly after
midnight of the house (10 to 40 minutes after it, at random, so that servers do not start
together) and once about three minutes after the start. Only offered rooms are rendered (see
*Rooms to show* and *Hide quiet rooms*). *Pinned rooms* renders the rooms in *Keep
every day forever*, in the default style first and then every style that is offered, and
the albums and tracks you added for that day; *All rooms* also renders every room and the whole
house in the default style. It works on one track at a time, in a worker thread of low priority,
with a pause after every second of audio, and waits while any Entropy FM stream is playing. It
stops when the provider unloads, does nothing when the cache is off, never evicts anything to make
room (it stops at 80 percent of the cache size) and logs one line per run. A track takes a few
minutes in the background, so five styles of five rooms take hours on a Raspberry Pi: on a small
machine keep *Keep every day forever* short, or choose *Off*.

**Many streams at once.** Each stream renders its own pieces, a block of one second at a time, so
two rooms can play at the same time; they take turns on the processor, and each costs what the
table says.

## Covers and room images

Albums and tracks have a cover in the look of the AI covers of Music Assistant's playlists: a
gradient of two or three colours chosen for the mood, a soft bloom and fine grain, a large white
line-art glyph cropped off the top and right edges, and the title in the lower-left corner.

- **Drawn from the day** (the default, and for now the only kind): a 1024 pixel JPEG made on your server with
  Pillow, with no network at all. The gradient follows the style and the key and mode of the day, and the
  glyph is that of the style, shaped by what happened that day. The same area, day and style always give the
  same cover.
- **Off**: no covers and no room images.
- **AI covers come later.** The code is in the repository behind `AI_COVERS_ENABLED` in
  `entropy_fm/const.py` and is off. Music Assistant keeps the first cover it is given for an item and does
  not ask again, so an AI cover (shown only after the drawn one) would never appear, and a redraw would change
  nothing that can be seen. They come back when core Music Assistant can take a replacement image.

Every room, and the whole house, is an artist with a drawn image of its own: a square one with the name of the room,
and a wide one for the backdrop of the artist page. Its figure is a clock of the typical day of the room, from the
last 7 days that Entropy FM already keeps: 24 spokes, one per hour, as long as the presence (motion, doors, CO2) in
that hour, a wave round it for the lights and small rings for the appliances and the media. A room without kept
history gets a figure of the same kind from its id. The images follow the *Covers* setting, and nothing is read from
Home Assistant for them.

## Known limitations

- **Music Assistant keeps stored items as they were.** Once a room, a day or a track is in the library, Music
  Assistant does not rewrite its name, its length, its description or its images when the provider's data changes
  (a new title, a changed *Piece length*, a redrawn cover). Use *Refresh metadata* on the item, or remove it and add it
  again, to see the new data. New items are added from a library sync, which Entropy FM asks for when the rooms that
  are shown or the piece length change.
- **Tracks stored before the length fix keep their old length until they are synced again.** A track used to be listed
  with a nominal 15 minutes whatever its real length, and Music Assistant never rewrites a stored length. A track is now
  listed with the length it plays (a full piece with the real length of its day, a condensed piece with its fixed one),
  and the length is part of its id and its version ("Ambient, Condensed"), so a changed *Piece length* lists new items
  and the next sync replaces the old ones; hearts follow, because they are kept for the day and the style. Items whose
  ids do not change (full pieces stored before the upgrade) keep the old length until you use *Refresh metadata* on them.
  The soundscape of today is listed with the length of the day so far and can play longer later.
- **The frontend shows no track descriptions**, and labels the description of an album "Review".
- **AI covers are hidden** for the reason above: Music Assistant caches the first cover it receives, which is the
  drawn one, so an AI cover would never show. A change in core Music Assistant is needed first.

## Privacy

- Persons, device trackers, locks, alarm panels and cameras are never used, and neither is any
  entity whose id contains `password` or `code`. The first two can be switched on with *Include
  presence* (which tells who is home, so it is off); the others cannot, and neither can labels or
  role overrides lift them.
- Only entities that are in an area play. Entities without an area do not.
- *Hide quiet rooms* only looks at when the current state of each entity last changed, which
  Home Assistant gives with the states the provider reads anyway; it reads no history for it.
- You can keep more out with *Extra exclusions* or the `entropy_fm_mute` label in Home Assistant.
- The provider only reads from Home Assistant: the registries, the labels, the current states of
  the entities that can play and the history of a day when it plays it, names it or draws its cover. It
  sends nothing anywhere else, and does not write into its own folder.
- What it keeps is only what you ask for: a pin (the event timeline of one area on one day, a few
  kilobytes) for what you heart, in the storage folder of the server; rendered
  audio of past days in the cache folder, if the cache is on; and the drawn covers and room images,
  small JPEGs in the cache folder. Taking the heart off an item deletes its pins; the cache is a
  normal cache folder.

## For developers

This repository started from the
[Music Assistant community provider template](https://github.com/music-assistant/community-provider-template),
whose README explains the development container, the rules of the community catalog and how a
release reaches users. The provider is in `entropy_fm/`; a map of its files, the demos and the
release steps are in [docs/development.md](docs/development.md).

With [uv](https://docs.astral.sh/uv/), run the tests, the lint, the type check and the spelling
check:

    uv sync
    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run codespell

Most tests render audio and read history from a fake Home Assistant. The tests that need a Music
Assistant server come with the fixture that starts it: `pytest_plugins =
["music_assistant.testing"]` in `tests/conftest.py` gives a fixture `mass`, a started server in dev
mode with this repository as its local provider folder.

`scripts/run.sh` starts a server in dev mode with the provider, to try it in the app at
<http://localhost:8095>. The demos need no server: `uv run python scripts/soundscapes.py out/`
renders an invented house to MP3 files, and `uv run python scripts/bench.py` measures the CPU
table above (see [docs/development.md](docs/development.md#run-the-demo)).

## Licence

Apache-2.0, see [LICENSE](LICENSE).
