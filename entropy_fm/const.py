"""Settings keys and fixed values of Entropy FM; imports nothing from Music Assistant."""

from __future__ import annotations

CONF_USE_PLUGIN = "use_ha_plugin"
CONF_URL = "url"
CONF_TOKEN = "token"  # noqa: S105 - the name of a setting, not a secret
CONF_VERIFY_SSL = "verify_ssl"
CONF_DAYS = "days"
CONF_KEY = "key"
CONF_SCALE = "scale"
CONF_INCLUDE_PRESENCE = "include_presence"
CONF_EXCLUDE = "exclude"
CONF_ROLES = "role_overrides"
CONF_STYLES = "styles"
CONF_DEFAULT_STYLE = "default_style"
CONF_ENERGY_OFFSET = "energy_offset"
CONF_DAY_CHARACTER = "day_character"
CONF_QUALITY = "render_quality"
CONF_AUDIO_CACHE_MB = "audio_cache_mb"
CONF_PRERENDER = "prerender"
CONF_LIBRARY_DAYS = "library_days"
CONF_REDRAW_COVERS = "redraw_covers"
CONF_KEEP_ROOMS = "keep_rooms"
CONF_ROOMS = "rooms"
CONF_HOUSE_SELECTED = "house_selected_rooms"
CONF_HIDE_QUIET = "hide_quiet_rooms"
CONF_LENGTH = "piece_length"
DEFAULT_LIBRARY_DAYS = 7
CONF_COVERS = "covers"
CONF_COVER_ENGINE = "cover_engine"
CONF_COVER_LIMIT = "cover_ai_per_hour"
CONF_COVER_STYLE = "cover_style"
CONF_COVER_PROMPT = "cover_prompt"
CONF_COVER_TITLE = "cover_title"
COVER_PATH_PREFIX = "cover"
# the images of the artists (rooms): `artist|<area id>|thumb or fanart|g<number of the redraw>`
ARTIST_PATH_PREFIX = "artist"

DEFAULT_DAYS = 10
MAX_DAYS = 10
DEFAULT_AUDIO_CACHE_MB = 500
# changes whenever the audible result of the engine changes; pins and cached audio carry it
ENGINE_VERSION = 6
STORAGE_FOLDER = "entropy_fm"
# a day is playable from the moment that this many seconds of it per second of the shortest track have happened
PLAYABLE_COMPRESSION = 60
DEFAULT_KEY = "D"
DEFAULT_SCALE = "dorian"
MAX_ENERGY_OFFSET = 50

HASS_DOMAIN = "hass"
WHOLE_HOUSE_ID = "_house"
WHOLE_HOUSE_NAME = "Whole House"
# the soundscape of a day is the track; the ids of the song and the timeline of earlier versions are moved to it
SCAPE_ID = "_scape"
LEGACY_TARGETS = frozenset({"_song", "_ensemble"})
# a soundscape is listed with this length until its day has been read; the real one (10 to 22 minutes) is told when it plays
NOMINAL_SCAPE_SECONDS = 900
# a soundscape of today needs this much of the day to have happened
MIN_SCAPE_SECONDS_OF_DAY = 3 * 3600

CACHE_CATEGORY_HISTORY = 4
CACHE_CATEGORY_BASELINE = 5
# the recorder of Home Assistant keeps ten days by default
PAST_DAY_EXPIRATION = 7 * 86400
TODAY_EXPIRATION = 300
CACHE_SCHEMA = 1
# the baseline of a day is built from the days before it, which no longer change
BASELINE_DAYS = 7
# how many of the last days the image of a room is drawn from, when they are kept
ARTIST_PROFILE_DAYS = 7
BASELINE_EXPIRATION = 86400
BASELINE_SCHEMA = 2
HISTORY_CHUNK = 25

# after a load the server is given this long to register the provider, before a library sync is asked
# for when the rooms that are offered have changed
LIBRARY_SYNC_DELAY = 10.0
OFFERED_FILE = "offered_rooms.json"
# the number of the last redraw of the covers: it is part of every cover path, so that the image cache
# of the server cannot show a cover that was thrown away
COVER_GENERATION_FILE = "cover_generation.json"
