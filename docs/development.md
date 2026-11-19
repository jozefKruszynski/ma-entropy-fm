# Development notes

The short version is in the [README](../README.md#for-developers). This page has the code map, the demos and the release steps.

## Code map

```sh
uv sync                     # on a network that can reach download.pytorch.org
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run codespell
```

The synthesis (`dsp.py`, `voices.py`, `drums.py`, `sampler.py`), the generative systems (`scape_*.py`,
`gen.py`, `plan.py`, `score.py`, `levels.py`, `render.py`), the library (`library.py`, `pins.py`,
`audiocache.py`) and the benchmark import nothing from Music Assistant, so they are tested and used
from the demo on their own.

| File | What it does |
|---|---|
| `entropy_fm/__init__.py` | The provider: settings, browsing, library, pins, streaming, the cache. |
| `entropy_fm/library.py` | Areas, days and entities to artists, albums and tracks; ids; durations; which rooms are quiet. |
| `entropy_fm/pins.py` | The days that were hearted, kept after Home Assistant forgets them. |
| `entropy_fm/audiocache.py` | Ogg/Opus files of rendered past days, with least-recently-used eviction. |
| `entropy_fm/prerender.py` | Which tracks of yesterday to render ahead into the cache, and when. |
| `entropy_fm/roles.py` | The layered classifier that gives an entity its role. |
| `entropy_fm/hass_access.py`, `history.py` | Registries, states and history, through the core plugin or a client of our own. |
| `entropy_fm/timeline.py` | History to normalised control signals. |
| `entropy_fm/day_analysis.py`, `energy.py` | The day read as a curve of activity, five to seven stretches and salient moments. |
| `entropy_fm/baseline.py`, `surprise.py`, `palette.py`, `variation.py` | What is usual, what is unusual, the character of the day, the dealing of material. |
| `entropy_fm/plan.py` | The length of the piece (full, or condensed per style), the arc of the day over it, the curves the house gives, the thinning of events when condensed, the deals. |
| `entropy_fm/scape_ambient.py`, `scape_berlin.py`, `scape_lofi.py`, `scape_minimal.py`, `scape_ritual.py`, `scape_glitch.py` | The six generative systems and their composed vocabulary. |
| `entropy_fm/glitch.py`, `glitch_drums.py`, `glitch_voices.py`, `glitch_render.py` | The sounds of Cinematic Glitch (swells, stabs, growl, kit), the edits, and its own renderer (a block pipeline with a latency of three seconds, so that an edit can read the material before it). |
| `entropy_fm/gen.py`, `score.py` | Weighted choices, Markov chains, euclid, random walks, event curves; notes, layers and the score. |
| `entropy_fm/voices.py`, `drums.py`, `dsp.py`, `sampler.py`, `samples/` | The voice library, the percussion, the band-limited oscillators and filters, the vibrato, the sampler and its CC0 samples. |
| `entropy_fm/stretch.py`, `taiko.py` | The stretched textures of Ritual (Paulstretch, cached) and its synthesised taiko and heartbeat drum. |
| `entropy_fm/levels.py` | The gain staging: design levels, measured trims, the loudness estimate of a score. |
| `entropy_fm/render.py`, `limiter.py` | The render loop, the buses, the room, the leveler, the limiter. |
| `entropy_fm/quality.py`, `autotune.py` | The three tiers, and the benchmark that Auto uses (it renders the dearest style that is offered). |
| `entropy_fm/covers.py`, `maart.py`, `imageapi.py` | Album covers. |
| `scripts/` | `soundscapes.py`, `bench.py`, `loudness_report.py`, `glide_report.py`, `voice_report.py`, `gain_staging.py`, `fetch_samples.py`, `analyze_audio.py`, the demo house. |

## Run the demo

The demos need no Music Assistant and no Home Assistant. They render an invented fortnight of a
Kitchen and a Living Room (a fridge that cycles every 40 minutes, a kettle, a door, an oven at dinner,
a washing machine, a dishwasher from 20:30 to 22:30, motion at meals, lights in the evening, a
temperature that drifts from 20 to 23 degrees, humidity and the sun). The weekends start later, the
12th is a heatwave, and the last day, a Tuesday, has a door at 03:00 and a wash at 21:30 that the
fortnight has no equivalent of; and a kitchen modelled on a real house (a Home Connect dishwasher with
its phases, an oven, a hob cooked on, a Sonos, blinds, a scene of twenty lights, no motion sensor).

```sh
uv sync
uv run python scripts/soundscapes.py out/        # the excerpts, the parts, the full pieces, the voice library, soundscapes.md
uv run python scripts/analyze_audio.py --table out/*.mp3
uv run python scripts/loudness_report.py         # integrated LUFS, short-term, true peak, limiter
uv run python scripts/glide_report.py out/*.mp3  # the glide tracker of the tests
uv run python scripts/voice_report.py            # the fingerprint of every voice
uv run python scripts/bench.py                   # CPU milliseconds per second of audio, per style and tier
```

`scripts/soundscapes.py` writes, for each style, `entropy_fm_<style>_demo_excerpts.mp3` and
`entropy_fm_<style>_kitchen_excerpts.mp3` (three minutes: one from the night, one from the active
daytime, one from the evening of the piece, with short cross fades, so that the arc of the day can
be heard quickly), `entropy_fm_<style>_parts.mp3` (each layer alone for a few seconds, then all
together), full pieces of Ambient and Berlin School (Ritual and Cinematic Glitch have excerpts and parts), `entropy_fm_voice_library.mp3` (every voice
alone for four seconds on a short phrase) and `soundscapes.md`, which says for each piece how long it
is, what plays when, what the day chose, how the house shaped it and which voices it used.
`analyze_audio.py` prints the levels, the DC offset, the share of energy above 6 kHz, the hiss floor
(the level above 6 kHz in the quietest 20 percent of 50 ms frames, which is where steady hiss shows and
a hat does not), the spectral centroid and the longest quiet stretch, as a stand-in for listening.
Nothing here can say that a piece sounds good: only a person can.

## Release and submit

1. Check that the GitHub handle and the repository address in `entropy_fm/manifest.json` are yours.
2. Make sure CI is green, including the catalog checks.
3. Publish a **GitHub release**, not a pre-release, whose tag is the version, such as `v0.1.0`.
4. Submit the repository with the form
   [Submit a provider](https://github.com/music-assistant/community-catalog/issues/new?template=submit-provider.yml).
   It needs a public repository with an open source licence and a release that passes the checks.
   After admission, publish a release whenever you want.
