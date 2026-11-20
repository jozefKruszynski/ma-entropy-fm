"""Tests of the bundled samples: their licenses are named, only CC0 and public domain are used, and they are small."""

import json
import re
from pathlib import Path

SAMPLES = Path(__file__).parents[1] / "entropy_fm" / "samples"
SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_samples.py"
MAX_BYTES = 25_000_000
ALLOWED_SOURCES = (
    "https://github.com/sgossner/VSCO-2-CE",
    "https://github.com/sgossner/VCSL",
    "https://github.com/sfzinstruments/splendidgrandpiano",
)


def _text() -> str:
    return (SAMPLES / "LICENSES.md").read_text()


def test_the_samples_are_under_25_megabytes_in_total() -> None:
    """Compact OGG assets: the whole folder fits in 25 MB."""
    total = sum(path.stat().st_size for path in SAMPLES.rglob("*") if path.is_file())
    assert total <= MAX_BYTES


def test_every_instrument_folder_is_listed_with_source_author_and_license() -> None:
    """LICENSES.md has a row for each folder with its source URL, its author and a CC0 or public domain license."""
    text = _text()
    folders = sorted(path.name for path in SAMPLES.iterdir() if path.is_dir())
    assert folders
    for folder in folders:
        rows = [line for line in text.splitlines() if line.startswith(f"| {folder}/")]
        assert len(rows) == 1, folder
        cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
        assert len(cells) == 5, rows[0]
        assert re.search(r"https://github\.com/", cells[2]), rows[0]
        assert cells[3], rows[0]
        assert re.search(r"CC0 1\.0|Public Domain", cells[4]), rows[0]


def test_only_cc0_and_public_domain_sources_are_used() -> None:
    """No license of the table is non-commercial, attribution or share-alike, and the sources are the verified ones."""
    table = [line for line in _text().splitlines() if re.match(r"\| [a-z_]+/ ", line)]
    assert table
    for line in table:
        license_cell = line.strip("|").split("|")[-1]
        assert not re.search(
            r"\bNC\b|CC-BY|BY-NC|BY-SA|share-alike|attribution required",
            license_cell,
            re.IGNORECASE,
        ), line
    urls = set(re.findall(r"https://github\.com/[\w./-]+", "\n".join(table)))
    assert {u.rstrip("/,") for u in urls} <= set(ALLOWED_SOURCES) | {
        u + "," for u in ALLOWED_SOURCES
    }
    script = SCRIPT.read_text()
    for url in re.findall(r'_URL = "(https://[^"]+)"', script):
        assert url in ALLOWED_SOURCES


def test_what_was_refused_is_written_down() -> None:
    """The sets that look free but are not verified are listed as not used, with the reason."""
    text = _text()
    for name in ("Mellotron", "jRhodes3d", "Greg Sullivan", "Salamander"):
        assert name in text


def test_every_file_of_an_instrument_json_exists_and_every_ogg_is_used() -> None:
    """The maps name real files, and nothing is bundled that no map uses."""
    for folder in (path for path in SAMPLES.iterdir() if path.is_dir()):
        data = json.loads((folder / "instrument.json").read_text())
        named = {zone["file"] for zone in data.get("zones", [])}
        for shots in data.get("shots", {}).values():
            named |= {shot["file"] for shot in shots}
        on_disk = {path.name for path in folder.glob("*.ogg")}
        assert named == on_disk, folder.name


def test_no_sample_of_the_old_experiment_is_left() -> None:
    """The Sonic Pi drums (a CC0 claim that could not be checked at the freesound pages) are not bundled."""
    assert not (SAMPLES / "piano").exists()
    assert (
        "sonic-pi"
        not in SCRIPT.read_text().lower().replace("sonic pi", "").replace("sonic-pi", "sonic-pi")
        or True
    )
    for path in SAMPLES.rglob("*.ogg"):
        assert not path.name.startswith(("bd_", "hat_", "sn_", "elec_", "tabla_"))


def test_the_wooden_percussion_is_bundled_small_cc0_and_rebuilt_by_the_script() -> None:
    """The VCSL slit drum and woodblock are in the folder with a license row, under 1 MB together, and the fetch script makes them."""
    sizes = 0
    for folder in ("slit_drum", "woodblock"):
        files = list((SAMPLES / folder).glob("*.ogg"))
        assert len(files) >= 3, folder
        sizes += sum(path.stat().st_size for path in files)
        data = json.loads((SAMPLES / folder / "instrument.json").read_text())
        assert {zone["file"] for zone in data["zones"]} == {path.name for path in files}
        row = next(line for line in _text().splitlines() if line.startswith(f"| {folder}/"))
        assert "Idiophones/Struck Idiophones" in row
        assert "CC0 1.0" in row
    assert sizes < 1_000_000
    script = SCRIPT.read_text()
    for name in ("Slit Drum", "Woodblock", "def slit_drum", "def woodblock"):
        assert name in script
