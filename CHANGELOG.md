# Changelog

## 1.0.1

- Covers show what kind of mix to expect: the energy of the day sets the wave next to the dots, the
  saturation, brightness and bloom of the colours, and how lit and busy the glyph of each style is.
- A strip above the foot of a cover shows the day round the clock, a bar for every quarter of an hour,
  with the busiest in the accent colour and the sun of the horizon glyph on the same hour.
- The key and mode of a cover are those of its track.
- Days of a daylight saving change fill the strip evenly, and their covers are kept once the day is over
  instead of being read again every hour.
- A cover of a finished day is never kept from a failed read of Home Assistant; it is drawn again later.
  Turning the day-character setting on or off draws the covers anew.
- Covers read Home Assistant at most two at a time, so a full grid of covers does not flood it.
- The README opens with the icon.

## 1.0.0

First release: the history of your Home Assistant as generative soundscapes, one piece per room and
day in five styles (Ambient, Berlin School, Lo-fi, Ritual, Cinematic Glitch).

- Every Home Assistant area, and the whole house, is an artist; each recent day an album; each style
  offered a track of 5 to 7 minutes, made on the server while it plays.
- The same room, day and style always give the same music, title and cover, on every platform.
- Tracks are named and described from what the day was like, never from an entity id or a name.
- The character of the day (how busy it was against the room's own days, its surprises, its palette
  and the drift of its key) shapes the music; it can be turned off.
- Hearts keep a day, or every day of a room, after it leaves the window and after Home Assistant
  purges it.
- Drawn covers for every day and style, and a drawn image and backdrop for every room.
- Render quality tiers with an automatic benchmark, and a size-capped audio cache of past days.
- Role overrides, Home Assistant labels, exclusions and an opt-in for presence decide what plays.
- Nothing leaves the server: the history is read from your own Home Assistant and the samples are CC0
  and ship with the provider.
