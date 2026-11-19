"""
Render a demo of a kitchen modelled on a real house: no motion sensors, events everywhere else.

    uv run python scripts/real_house_demo.py [output folder]

The kitchen has a dishwasher that tells its operation state, program phase and progress (as the
Home Connect integration does) and whose door is the only door sensor, an oven with an operation
state, a hob with four zones whose power levels are text states (`power_level_5`), a Sonos that
plays in the evening, window blinds that open at sunrise and close in the evening, at 19:10, a fridge on a smart
plug, twenty lights that a scene switches together, and the sun. The window is 18:25 to 20:25: a
dinner is cooked on the hob and in the oven, and the dishwasher starts at 19:45.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import demo_render as demo_module

DAY = 86400.0
hours = demo_module.hours


def build_real_kitchen(demo: demo_module.Demo) -> list:
    """Return the series of the kitchen of a real house on one day."""
    timeline = demo.timeline
    kinds = timeline.VoiceKind
    rng = np.random.default_rng(7)
    series = []

    def add(  # noqa: PLR0913, PLR0917
        entity_id: str,
        kind: object,
        times: list[float],
        values: list[float],
        device_class: str = "",
        device: str = "",
        *,
        text: bool = False,
    ) -> None:
        if text:
            values = [timeline.category_value(str(v)) or 0.0 for v in values]  # type: ignore[arg-type]
        info = timeline.EntityInfo(
            entity_id, entity_id, "kitchen", kind, device_class=device_class, device_id=device
        )
        series.append(timeline.build_series(info, times, values))

    times, values = demo_module.fridge(rng)
    add("sensor.fridge_plug_power", kinds.DRONE, times, values, "power", "fridge")
    # the dishwasher, as Home Connect reports it
    phases = [
        (0.0, "inactive", "inactive"),
        (hours("19:45"), "run", "pre_rinse"),
        (hours("19:58"), "run", "main_wash"),
        (hours("20:30"), "run", "intermediate_rinse"),
        (hours("20:45"), "run", "heat_drying"),
        (hours("21:20"), "finished", "inactive"),
        (hours("21:30"), "inactive", "inactive"),
    ]
    add(
        "sensor.siemens_dishwasher_operation_state",
        kinds.PHASE,
        [t for t, _, _ in phases],
        [state for _, state, _ in phases],
        "enum",
        "karl",
        text=True,
    )
    add(
        "sensor.siemens_dishwasher_program_phase",
        kinds.PHASE,
        [t for t, _, _ in phases],
        [phase for _, _, phase in phases],
        "enum",
        "karl",
        text=True,
    )
    progress_t = list(np.arange(hours("19:45"), hours("21:20"), 300.0))
    add(
        "sensor.siemens_dishwasher_program_progress",
        kinds.PROGRESS,
        [0.0, *progress_t, hours("21:21")],
        [
            0.0,
            *[100.0 * (t - hours("19:45")) / (hours("21:20") - hours("19:45")) for t in progress_t],
            0.0,
        ],
        "",
        "karl",
    )
    add(
        "binary_sensor.siemens_dishwasher_door",
        kinds.PLUCK,
        *demo_module.intervals_to_events(
            [(hours("19:40"), hours("19:44")), (hours("21:35"), hours("21:40"))]
        ),
        "door",
        "karl",
    )
    # the oven, Bilbo
    add(
        "sensor.bilbo_operation_state",
        kinds.PHASE,
        [0.0, hours("18:40"), hours("19:20")],
        ["inactive", "run", "finished"],
        "enum",
        "bilbo",
        text=True,
    )
    # the hob: four zones, power levels as text, and its estimated power
    zones = {
        1: [(hours("18:30"), 9), (hours("18:42"), 0)],
        2: [(hours("18:35"), 5), (hours("18:50"), 3), (hours("19:05"), 0)],
        3: [(hours("18:45"), 7), (hours("18:58"), 0)],
        4: [(hours("18:48"), 2), (hours("19:12"), 0)],
    }
    total_t, total_w = [0.0], [0.0]
    for zone, steps in zones.items():
        add(
            f"sensor.kitchen_bosch_hob_zone_{zone}_power_level",
            kinds.HOB,
            [0.0, *[t for t, _ in steps]],
            [0.0, *[float(level) for _, level in steps]],
            "",
            "hob",
        )
    for moment in np.arange(hours("18:30"), hours("19:15"), 60.0):
        level = sum(
            [lv for t, lv in steps if t <= moment][-1] if any(t <= moment for t, _ in steps) else 0
            for steps in zones.values()
        )
        total_t.append(float(moment))
        total_w.append(level * 160.0 + rng.normal(0, 20))
    add("sensor.bosch_hob_estimated_power", kinds.DRONE, total_t, total_w, "power", "hob")
    # a Sonos plays in the evening
    add(
        "media_player.kitchen",
        kinds.MEDIA,
        *demo_module.intervals_to_events(
            [(hours("18:50"), hours("19:35")), (hours("19:50"), hours("22:00"))]
        ),
        "",
        "sonos",
    )
    # blinds open at sunrise and close at sunset
    add(
        "cover.kitchen_window_blinds",
        kinds.COVER,
        *demo_module.intervals_to_events([(hours("07:40"), hours("19:10"))]),
        "",
        "blinds",
    )
    # a scene switches twenty lights on together at 17:00 and off at 23:00
    for number in range(20):
        jitter = float(rng.uniform(0.0, 3.0))
        add(
            f"light.kitchen_{number}",
            kinds.CHORD,
            *demo_module.intervals_to_events([(hours("17:00") + jitter, hours("23:00") + jitter)]),
            "",
            f"light_{number}",
        )
    times, values = demo_module.curve(
        [("00:00", 20.0), ("12:00", 21.0), ("19:00", 23.0)], rng, 0.1, 300.0
    )
    add("sensor.kitchen_temperature", kinds.TEMPERATURE, times, values, "temperature", "thermo")
    sun_t = [0.0, hours("07:45"), hours("16:40")]
    sun = timeline.EntityInfo("sun.sun", "Sun", None, kinds.SUN)
    series.append(timeline.build_series(sun, sun_t, [0.0, 1.0, 0.0]))
    return series


def real_score(demo: demo_module.Demo, style: str, tier: object = None) -> tuple[object, object]:
    """Return the score and the plan of the soundscape of the real-house kitchen in a style."""
    return demo.make(build_real_kitchen(demo), demo_module.FIRST_DAY, style, "kitchen", None, tier)


def main() -> None:
    """Print the soundscape of the kitchen of the real house in each style: its length and what it plays."""
    demo = demo_module.Demo()
    for style in demo.styles.STYLES:
        score, _ = real_score(demo, style)
        print(f"{style}: {score.length} s, {len(score.notes)} notes, voices {score.voices_used()}")


if __name__ == "__main__":
    main()
