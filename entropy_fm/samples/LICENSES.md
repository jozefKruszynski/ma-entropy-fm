# Licenses of the bundled samples

Every sample that Entropy FM bundles is CC0 1.0 or in the public domain, taken from the sources named
here, whose own license files and READMEs were read at the original repository (not at a mirror).
Nothing is under a non-commercial, attribution-only or share-alike license. `scripts/fetch_samples.py`
rebuilds all of it; the files are trimmed, converted to mono OGG/Vorbis and (for strings) given a
seamless loop, which are the only changes. `entropy_fm/samples/<instrument>/instrument.json` maps them.

| folder | what it is | source | author | license |
|---|---|---|---|---|
| felt_piano/ | upright piano ("Upright Nr1"), soft and medium layers, C and G of each octave | VSCO 2 Community Edition, Keys/Upright Nr1, https://github.com/sgossner/VSCO-2-CE | Versilian Studios (Sam Gossner, Simon Dalzell) | CC0 1.0, `LICENSE` of the repository |
| strings/ | violin section and cello section, sustained with vibrato, looped | VSCO 2 Community Edition, Strings, https://github.com/sgossner/VSCO-2-CE | Versilian Studios | CC0 1.0, `LICENSE` of the repository |
| ep/ | FM electric piano sampled from a Yamaha TX81Z, three velocity layers | VCSL, Electrophones/TX81Z/FM Piano, https://github.com/sgossner/VCSL | Versilian Studios (patches made with TX81Z Programmer, sampled from the hardware) | CC0 1.0, `LICENSE` and `README` of the repository |
| pluck/ | strumstick played with the finger | VCSL, Chordophones/Composite Chordophones/Strumstick, https://github.com/sgossner/VCSL | Versilian Studios | CC0 1.0 |
| drums/ | soft bass drum, snare, cross stick, claves (one-shots) | VCSL, Membranophones/Struck Membranophones and Idiophones/Struck Idiophones, https://github.com/sgossner/VCSL | Versilian Studios | CC0 1.0 |
| grand_piano/ | Splendid Grand Piano, two velocity layers, a note every three semitones | https://github.com/sfzinstruments/splendidgrandpiano | AKAI (Steinway samples released in the early 2000s), mapped by kinwie | Public Domain (README of the repository: "Public Domain samples by AKAI") |

Everything else in Entropy FM (oscillators, drums that are not listed, noise, vinyl and tape) is made with
numpy and is original work of this project (Apache-2.0).

## Looked at and not used, and why

- **Mellotron samples** (the `archiveorg-mellotron` set that smpldsnds converted from a Mellotron
  CD-ROM archive on archive.org): the mirror has no license statement and archive.org could not be
  reached from the build sandbox, so no license could be verified. Not bundled. The Mellotron-like
  strings and choir are synthesised instead.
- **jRhodes3d**, and the "Electric Piano 1" of the Discord GM bank that is made from it: the sources
  disagree (CC0 and CC-BY-NC). Not bundled.
- **Greg Sullivan's E-Pianos** (CP80, Pianet, Wurlitzer): the Discord GM bank labels them CC0, but the
  original repository says CC-BY 3.0 (attribution). Not bundled; the GM bank's per-instrument license
  lines are not reliable.
- **Salamander Grand Piano**: CC-BY. Not bundled.
- **Sonic Pi samples** (freesound uploads): the CC0 statement is the Sonic Pi project's, and the
  freesound pages could not be checked. Not bundled; the earlier experiment's drums were removed.
- Hydrogen drum kits (GPL), Drum Abuse packs and Retro Drums (no license): not bundled.
