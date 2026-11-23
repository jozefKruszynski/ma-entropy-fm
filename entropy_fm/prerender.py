"""
The pre-render of Entropy FM: what to render of the day that just ended, and when.

A soundscape of a past day is rendered in a few minutes and is the same every time, so shortly after
midnight (and once after the start) the provider renders the tracks of yesterday that the audio cache
does not have yet, one at a time, at low priority, and only while nothing plays. This module decides
which tracks and when; the rendering itself is the provider's, through the same recorder as a live
stream.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from .constants import SCAPE_ID
from .library import TrackRef, parse_album_id, parse_track_id

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

# the first run after the start, so that the benchmark of Auto and the first sync are done
STARTUP_DELAY_SECONDS = 180.0
# the nightly run starts this long after midnight, plus up to the jitter, so that many servers do not start together
MIDNIGHT_DELAY_SECONDS = 600.0
MIDNIGHT_JITTER_SECONDS = 1800.0
# the nice level of the thread that renders, where the system allows it
NICE_LEVEL = 15
# the pause after every second of audio that is rendered, which keeps the worker to a small share of one core
YIELD_SECONDS = 0.15
# how long a pause for a stream that is playing waits at the most
GIVE_UP_SECONDS = 1800.0
# the audio cache holds 96 kbit/s; the pre-render never fills it past this share, so that there is room for what is played
BYTES_PER_SECOND = 12_000
CACHE_SHARE = 0.8


def seconds_until_run(now: datetime, delay: float) -> float:
    """
    Return the seconds from now to `delay` seconds after the next midnight of the clock of `now`.

    :param now: The time, in the time zone of the house.
    :param delay: The seconds after midnight.
    """
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), now.tzinfo)
    # in UTC, so that a night with a clock change is as long as it is
    return max((midnight.astimezone(UTC) - now.astimezone(UTC)).total_seconds() + delay, 1.0)


def fits(cache_bytes: int, limit_bytes: int, seconds: float) -> bool:
    """
    Return whether a track of this length may be added without filling the cache.

    :param cache_bytes: What the cache holds.
    :param limit_bytes: The size of the cache.
    :param seconds: The length of the track.
    """
    return cache_bytes + seconds * BYTES_PER_SECOND <= limit_bytes * CACHE_SHARE


def _added_for(
    day: date, rooms: set[str], albums: Iterable[str], tracks: Iterable[str]
) -> tuple[list[str], list[TrackRef]]:
    """Return the rooms of the albums that were added for a day, and the tracks that were added for it."""
    areas: list[str] = []
    for album in sorted(albums):
        try:
            area, album_day = parse_album_id(album)
        except ValueError:
            continue
        if album_day == day and area in rooms:
            areas.append(area)
    singles: list[TrackRef] = []
    for item in sorted(tracks):
        try:
            ref = parse_track_id(item)
        except ValueError:
            continue
        if ref.day == day and ref.area_id in rooms:
            singles.append(ref)
    return areas, singles


def choose_tracks(
    day: date,
    *,
    rooms: Iterable[str],
    pinned_rooms: Iterable[str],
    offered: Iterable[str],
    added_albums: Iterable[str] = (),
    added_tracks: Iterable[str] = (),
) -> list[TrackRef]:
    """
    Return the tracks of a day to pre-render, the most wanted first.

    Only what is pinned is rendered: the rooms that are kept, and the albums and tracks that were added to the
    library for this day. The first tracks are those in the first style that is offered; then every other style
    that is offered follows.

    :param day: The day to render.
    :param rooms: Every room that can be played, the whole house included.
    :param pinned_rooms: The rooms whose every day is kept.
    :param offered: The keys of the styles the library offers, in their order.
    :param added_albums: The ids of the albums that were hearted.
    :param added_tracks: The ids of the tracks that were hearted.
    """
    styles = list(dict.fromkeys(offered))
    if not styles:
        return []
    first = styles[0]
    room_set = set(rooms)
    album_rooms, singles = _added_for(day, room_set, added_albums, added_tracks)
    pinned = list(dict.fromkeys([*sorted(set(pinned_rooms) & room_set), *album_rooms]))
    wanted: dict[tuple[str, str], None] = {}
    for area in pinned:
        wanted.setdefault((area, first), None)
    for ref in singles:
        wanted.setdefault((ref.area_id, ref.style), None)
    for style in styles:
        for area in pinned:
            wanted.setdefault((area, style), None)
    return [TrackRef(area, day, SCAPE_ID, style) for area, style in wanted]
