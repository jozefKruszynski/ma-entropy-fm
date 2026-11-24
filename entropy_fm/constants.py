"""Settings keys and fixed values of Entropy FM; imports nothing from Music Assistant."""

from __future__ import annotations

CONF_USE_PLUGIN = "use_ha_plugin"
CONF_URL = "url"
CONF_TOKEN = "token"  # noqa: S105 - the name of a setting, not a secret
CONF_VERIFY_SSL = "verify_ssl"
CONF_DAYS = "days"
CONF_INCLUDE_PRESENCE = "include_presence"
CONF_EXCLUDE = "exclude"
CONF_ROLES = "role_overrides"
CONF_STYLES = "styles"
CONF_DAY_CHARACTER = "day_character"
CONF_QUALITY = "render_quality"
CONF_AUDIO_CACHE_MB = "audio_cache_mb"
CONF_KEEP_ROOMS = "keep_rooms"
CONF_ROOMS = "rooms"
COVER_PATH_PREFIX = "cover"
# marks the cover path of a day that is not over yet
COVER_LIVE_MARK = "live"
# where the mark sits in a cover path split at the bars, after the style
COVER_LIVE_INDEX = 4
# the parts of a cover path without the mark: prefix, area, day, style, day character, engine and generation
COVER_PATH_PARTS = 7
# the images of the artists (rooms): `artist|<area id>|thumb or fanart|<figure>|g<cover generation>`
ARTIST_PATH_PREFIX = "artist"
# the parts of such a path: prefix, area, thumb or fanart, figure and generation
ARTIST_PATH_PARTS = 5
# where the parts sit in a cover path split at the bars (the part after the prefix is the area)
COVER_AREA_INDEX = 1
COVER_DAY_INDEX = 2
COVER_STYLE_INDEX = 3
COVER_CHARACTER_INDEX = 4
COVER_ENGINE_INDEX = 5
# the day-character part of a cover path: whether the cover was drawn from the character of the day
COVER_CHARACTER_ON = "c1"
COVER_CHARACTER_OFF = "c0"
# the engine part of a cover path: `e` and the engine version, since the character of a day depends on it
ENGINE_PATTERN = r"e\d+"
# and in an artist path: the area, thumb or fanart, and the figure
ARTIST_AREA_INDEX = 1
ARTIST_KIND_INDEX = 2
ARTIST_FIGURE_INDEX = 3
# the last part of both paths: `g` and the cover generation
GENERATION_PATTERN = r"g\d+"
# the folder of a room in `browse`, and the scope of the sun in the history cache and the pins
BROWSE_AREA_FOLDER = "area"
SUN_SCOPE = "_sun"
# the seed of the random generator of a cover or room image is a hash taken modulo this
IMAGE_SEED_MODULUS = 2**32
# how the whole house is called in a title, by language
HOUSE_ROOM_NAMES = {"en": "House", "nl": "Huis"}

DEFAULT_DAYS = 10
MAX_DAYS = 10
DEFAULT_AUDIO_CACHE_MB = 500
MAX_AUDIO_CACHE_MB = 20000
# changes whenever the audible result of the engine changes; pins and cached audio carry it
ENGINE_VERSION = 12
STORAGE_FOLDER = "entropy_fm"
# the audio that is rendered and cached: 16 bit stereo
AUDIO_CHANNELS = 2
AUDIO_BIT_DEPTH = 16
# a moment of a day is unusual enough to become a swell from this surprise score on
ANOMALY_MIN_SCORE = 0.5
# the key and mode the days move away from
HOME_KEY = "D"
HOME_MODE = "dorian"

HASS_DOMAIN = "hass"
WHOLE_HOUSE_ID = "_house"
WHOLE_HOUSE_NAME = "Whole House"
# the soundscape of a day is the track
SCAPE_ID = "_scape"
# a soundscape of today needs this much of the day to have happened
MIN_SCAPE_SECONDS_OF_DAY = 3 * 3600

CACHE_CATEGORY_HISTORY = 4
CACHE_CATEGORY_BASELINE = 5
CACHE_CATEGORY_AUTO = 6
CACHE_CATEGORY_FACTS = 7
CACHE_CATEGORY_COVER = 8
# the recorder of Home Assistant keeps ten days by default
PAST_DAY_EXPIRATION = 7 * 86400
TODAY_EXPIRATION = 300
CACHE_SCHEMA = 1
# how many characters of a hash make the checksum of a cached reading
CHECKSUM_LENGTH = 16
FACTS_SCHEMA = 1
# the schema of the character and traits that a cover is made from (raise it when what they hold changes)
COVER_MOOD_SCHEMA = 1
# today's reading of a day is read again every hour
FACTS_TODAY_EXPIRATION = 3600
# the soundscape of today is read again every this many seconds of the day
SECONDS_BUCKET = 600
# the baseline of a day is built from the days before it, which no longer change
BASELINE_DAYS = 7
# how many of the last days the image of a room is drawn from, when they are kept
ARTIST_PROFILE_DAYS = 7
BASELINE_EXPIRATION = 86400
BASELINE_SCHEMA = 2
WEEK_ENERGY_SCHEMA = 1
HISTORY_CHUNK = 25

# where the rooms of the last library sync request are kept
OFFERED_FILE = "offered_rooms.json"
# soundscapes that may be rendered at once on the CPU of the server
MAX_CONCURRENT_STREAMS = 5
# part of every cover path: raise it to make the image cache of the server fetch every cover again
COVER_GENERATION = 2
ARTIST_KEY = "@artist|"
# the target of a track: the soundscape of a day
TRACK_TARGETS = frozenset({SCAPE_ID})
# the keys of the messages of the setup errors, in the `errors` of strings.json
ERROR_PLUGIN_MISSING = "plugin_missing"
ERROR_CONNECTION_MISSING = "connection_missing"
ERROR_LOGIN_REFUSED = "login_refused"
# how many soundscapes are kept between a request for their length and the stream
MAX_KEPT_SCAPES = 16
# how many covers are drawn at once; a burst of requests for them must not hold every thread
MAX_COVER_DRAWS = 2
# how many covers read their day from Home Assistant at once, so that a grid of covers is not a burst of reads
MAX_COVER_READS = 2
# how long the server's own time zone stands in before Home Assistant is asked again
TIME_ZONE_RETRY_SECONDS = 300.0
# how long the benchmark of the Auto quality stays valid, in seconds
AUTO_EXPIRATION = 30 * 86400
# how often the days of the followed areas are pinned, in seconds, and how long after the start the first time is
MAINTENANCE_SECONDS = 6 * 3600
MAINTENANCE_START_SECONDS = 30
# how long a read of the registries is reused, so that a library sync is one read
CATALOG_TTL = 300
# the core plugin may still be loading when this provider starts: the setup waits this long for it
PLUGIN_WAIT_SECONDS = 10.0
# a lost connection of our own is made again after this long, and after twice that when it fails, up to the cap
RECONNECT_FIRST_SECONDS = 2.0
RECONNECT_MAX_SECONDS = 300.0
RECONNECT_FACTOR = 2.0
