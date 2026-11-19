# Entropy FM

Entropy FM is a community provider for [Music Assistant](https://music-assistant.io) that turns the
history of your [Home Assistant](https://www.home-assistant.io) into generative soundscapes: one long,
slowly evolving piece for each room and each day, in six styles. Your house does not write the
notes. It steers the texture, the density, the brightness, the space and the movement of music that
a generative system plays. The audio is made on your server while it plays, from the history it
reads from your own Home Assistant and from CC0 samples that ship with the provider. Nothing leaves
your server; what it keeps is the event timeline of the days you add to your library and, if you
want it, a cache of the days that were played.

Entropy FM is made and maintained by [@jozefKruszynski](https://github.com/jozefKruszynski), not by
the Music Assistant team. Questions and problems go to the
[issues of this repository](https://github.com/jozefKruszynski/ma-entropy-fm/issues).

## What it plays

| In Music Assistant | In Entropy FM |
|---|---|
| Artist | A Home Assistant area, such as "Kitchen", and one more, "Whole House" |
| Album | One day of one area: "Kitchen, Tue 17 Nov" |
| Track | The soundscape of that day, 10 to 22 minutes (5 to 7 when *Piece length* is Condensed), with a title of its own: "Dishwasher at Dusk (Ambient)". Each style you offer is its own version, so an album has up to six tracks. |

**Titles and descriptions.** Every track is named from what the day was like: the busiest hour, the
kind of thing that ran (a dishwasher, a hob, a door at night), the temperature, the weekday and how busy it
was, in the words of its style: "Dishwasher at Dusk (Ambient)", "Sequencer for a Busy Kitchen (Berlin School)",
"Door Fault 14:12 (Cinematic Glitch)", ceremonial for Ritual, warm for Lo-fi, sparse and geometric for Minimal.
The same room, day and style always give the same title; the title of today can change as the day goes on, but
only once an hour. Titles hold kinds of things and times, never an entity id or a name, and a day without
history, or Home Assistant being away, gives a plain title of the style. The description of a track is a
short history of the style (Eno's tape loops, the Berlin School, lo-fi hip hop, Reich's phasing, the mood of
the Akira soundtrack, cinematic neurofunk), how the house drives it, and a line about the day ("A busy afternoon;
a door at 03:12."). Titles and descriptions follow the language of the server when it is Dutch, and are
English otherwise. Music Assistant keeps the name, the length and the description of a track from the moment it
was added to the library; to see new titles on a track that is already there, remove it and add it again, or
refresh its metadata.

A soundscape is as long as its day was busy, between 10 and 22 minutes (or, as a condensed piece,
about 7 minutes in Ambient, Berlin School and Ritual and about 5 in Lo-fi, Minimal and Cinematic
Glitch, whatever the day), and the day is spread over it in order: the night is sparse and dark, the morning brightens, the busy part of the day is
denser and warmer, the evening is fuller, the late night thins out again. A soundscape of today
needs three hours of the day to have happened. Tracks can be seeked, and the same room, day and
style always sound the same.

| Style | What you hear |
|---|---|
| Ambient | Tape loops of different lengths that phase against each other, a felt piano, no beat |
| Berlin School | Interlocking sequencer lines that mutate by rule, strings, choir and a slow lead, a soft pulse when busy |
| Lo-fi | A boom-bap band that never plays the loop the same way twice, electric piano, vinyl crackle |
| Minimal | Euclidean percussion with a shuffle, a short figure that mutates, a deep rolling bass, 118 to 124 BPM |
| Ritual | Three plucked loops in a Japanese scale over a breathing drone, stretched strings, a chant, a soft heartbeat and taiko |
| Cinematic Glitch | Film-score swells and brass stabs, a growling bass and hard half-time drums, cut up by the doors, switches and appliances of the house |

How the house steers each style, and how an entity gets its sound, is in
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
   play a day, or add an album to the library to keep that day for good.

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
| Days to browse | 10 | 1 to 10 | How many days back you can browse and search. The recorder of Home Assistant keeps 10 days by default. Days you added to the library stay, however old. |
| Days in library | 7 | 1 to 10 | How many of the last days of every room are in the library, counted back from today (never more than *Days to browse*). The window rolls on at midnight. Days you add stay for good. An album or track you remove stays out of the library, also inside the window, until you add it again. Rooms cannot be removed: untick them in *Rooms to show*. |
| Rooms to show | all (nothing ticked) | any of your rooms | The rooms that appear in Music Assistant: in browsing, search, the library and the pre-render. Nothing ticked shows every room, including rooms that Home Assistant gets later. A room you leave out disappears from those lists; what you pinned or cached for it is kept (hidden, not deleted) and is back when you tick the room again. A room is still found by its id, so a queue that holds it keeps playing. When this changes, the provider asks Music Assistant to re-sync its own items, so that the library views follow. |
| Whole House uses only these rooms | on | on, off | On: the *Whole House* soundscape is made of the ticked rooms only (every room when none is ticked). Off: it is made of every room, whatever is ticked. |
| Hide quiet rooms | on | on, off | Do not offer a room in which nothing that would make a note or an event (a door, a motion sensor, a light, an appliance and the like) has changed over the days you can browse. A room that has only temperature, humidity, sun or other static sensors after the role filtering counts as quiet too. A room that is merely quiet on one day still offers that day. |
| Keep every day forever | none | the rooms that are shown | Days of these rooms are saved permanently (pinned) as soon as they are over, even after Home Assistant purges its history. It does not show or hide rooms; the list only offers the rooms that are shown when some are ticked in *Rooms to show*. |
| Styles offered | all six | any of the six styles | Which styles each day comes in, as versions of the same soundscape. |
| Default style | Ambient | one of the six styles | The style the library lists for every day, and the one an id without a version plays. It is always offered. |

### Sound

| Setting | Default | Values | What it does |
|---|---|---|---|
| Piece length | Full | Full, Condensed | Full: as long as the day was busy, 10 to 22 minutes. Condensed: the same house data and plan in about 7 minutes (Ambient, Berlin School, Ritual) or about 5 minutes (Lo-fi, Minimal, Cinematic Glitch). The hours of the day pass faster, the little things of the house are thinned out and unusual moments are kept (see [docs/how-it-works.md](docs/how-it-works.md#piece-length)). Tracks show the length that they play. |
| Key | D | C, C#, D, D#, E, F, F#, G, G#, A, A#, B | The root note that the pitches are tuned to; the key of a day walks away from it and back. |
| Scale | Dorian | Dorian, Natural minor (Aeolian), Mixolydian, Lydian, Major (Ionian), Minor pentatonic, Major pentatonic | The scale of the pieces. A style that does not have the scale uses its own: Berlin School and Minimal are in a minor mode, Ritual keeps a Japanese pentatonic, Cinematic Glitch is in aeolian or phrygian. |
| Activity offset | 0 | -50 to 50 | Points added to how lively the house seems. The activity brings layers in and out and sets how dense and how fast they are; it never writes notes. |
| Character of the day | on | on, off | Baseline, surprise, the palette of the day and the drift of the key. Off: the plain sound, and no extra history is read. |
| Include presence | off | on, off | Also play persons and device trackers, which tells who is home (advanced). |
| Role overrides | empty | `entity_id=role` lines | Choose the role of an entity yourself, one line each (advanced). A label in Home Assistant is the other way. |
| Extra exclusions | empty | comma separated patterns | Entities to keep out, such as `sensor.bedroom_*, light.hall_*` (advanced). |

### Covers

| Setting | Default | Values | What it does |
|---|---|---|---|
| Covers | Drawn from the day | Off, Drawn from the day, AI image engine | Off: no covers. Drawn: a picture made on your server from the events of the day. AI image engine: only offered when Music Assistant has an image engine. |
| Image engine | default engine | the engines of Music Assistant | Which engine makes the AI covers. Advanced; only with AI covers. |
| Cover style | Music Assistant | Music Assistant, Neon, Bold, Soft Gradient, Risograph, Gouache, Custom | The look of the AI covers. Only with AI covers. |
| Cover prompt | empty | text with `{name}`, `{context}` | The prompt when the style is Custom; empty uses the default of Music Assistant. |
| Title on the cover | on | on, off | On: Entropy FM draws the title over the cover and the engine leaves the picture without text. Off: the engine is asked to draw the title. Only with AI covers. |
| AI covers per hour | 30 | 1 to 120 | The most AI covers asked for in an hour. Only with AI covers. |
| Redraw covers | button | press | Throws away the covers made so far (the cover files of Entropy FM only, never pinned days, audio or other data) and makes new ones as albums are shown; AI covers count against the hourly limit. The cover paths carry a redraw number, so that the image cache of Music Assistant cannot show an old picture, and the provider asks Music Assistant to re-sync its own items so that stored albums and tracks take the new paths. |

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
| ambient | 15.1 | 25.6 | 27.6 |
| berlin_school | 23.7 | 41.1 | 43.1 |
| lofi | 17.9 | 31.9 | 34.8 |
| minimal | 13.9 | 26.8 | 30.5 |
| ritual | 17.4 | 41.0 | 37.8 |
| glitch (Cinematic Glitch) | 18.2 | 50.6 | 50.5 |
| budget | 20 | 60 | 120 |

The budget is what a tier aims at. Light meets its 20 ms except in Berlin School, and every style
is inside the budget of Standard and Rich. Cinematic Glitch is the dearest style. Its swells and
its growling bass are the dearest notes: each is made once and kept (the cache of notes keeps
them in a pool of their own, so that nothing pushes them out and none is made twice), and the
swells of one chord share the saws that they are made from. A few numbers move by a few percent
from one run to the next.


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
Ritual, Lo-fi, Ambient, Minimal), and picks the richest tier that would use less than 30 percent
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
minutes in the background, so six styles of five rooms take hours on a Raspberry Pi: on a small
machine keep *Keep every day forever* short, or choose *Off*.

**Many streams at once.** Each stream renders its own pieces, a block of one second at a time, so
two rooms can play at the same time; they take turns on the processor, and each costs what the
table says.

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
  the entities that can play and the history of a day when you play it. It sends nothing anywhere
  else, with one opt-in exception (AI covers, below), and does not write into its own folder.
- What it keeps is only what you ask for: a pin (the event timeline of one area on one day, a few
  kilobytes) for what you add to the library, in the storage folder of the server; and rendered
  audio of past days in the cache folder, if the cache is on. Removing an item from the library
  deletes its pins; the cache is a normal cache folder.

### Covers

Albums and tracks have a cover in the look of the AI covers of Music Assistant's playlists: a
gradient of two or three colours chosen for the mood, a soft bloom and fine grain, a large white
line-art glyph cropped off the top and right edges, and the title in the lower-left corner.

- **Drawn from the day** (the default): a 1024 pixel JPEG made on your server with Pillow, with no
  network at all. The gradient follows the style and the key and mode of the day, and the glyph is
  that of the style, shaped by what happened that day. The same area, day and style always give
  the same cover.
- **AI image engine** (off by default, and only offered when Music Assistant has an image engine
  from a plugin): the engine you pick makes the cover with the presets of Music Assistant's own
  covers, and the title is drawn over it afterwards. A cover is made in the background the first
  time something asks for it (the drawn cover is shown meanwhile), kept, and made again only when
  its title, style, prompt or engine changes. At most 30 an hour by default; a failure is retried
  after 1 hour, 6 hours and 24 hours, then only when something changes.
- **Off**: no covers.

*Redraw covers* throws away every cover that Entropy FM made, asks the server for a library sync
of this source and counts a redraw, which is part of every cover path (`cover|<room>|<day>|<style>|g<n>`;
paths without it still resolve). New covers are made as albums are shown, within the AI limit.

What the image engine is sent, when you turn AI covers on: the name of the area (the label you gave
it in Home Assistant, never the date, an entity or a person); one short description made only of
mood words from a fixed list (the style, a colour and a mood from the key and mode of the day,
whether the day was quiet or busy, a line about light, and up to three abstract motifs); and the
preset text of the style you picked or your own prompt. It never contains an entity id, a name, a
person, a time or a count, and a test checks that with a house full of personal names. The engine
is a plugin of your own Music Assistant, so what leaves your server is decided by that plugin and by
you.

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
