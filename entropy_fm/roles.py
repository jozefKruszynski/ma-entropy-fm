"""
Decide which voice an entity gets, in layers that do not depend on the language of its name.

The layers, in the order they are tried:

1. A role the user gave: a Home Assistant label `entropy_fm_<role>` on the entity or its
   device, or a line `entity_id=role` in the settings.
2. Hints from the integration: the platform, the translation key of the entity and the
   manufacturer and model of its device. A small table, best effort.
3. The domain and device class: power, energy, temperature, humidity, motion, door, light, ...
4. Only for power and energy sensors, and only as a last resort, keywords in the name or the
   entity id, in several languages: they tell a dishwasher from a washer from a plain meter.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .timeline import (
    DIAGNOSTIC_WORDS,
    IGNORED_BINARY_CLASSES,
    IGNORED_DEVICE_CLASSES,
    PLUCK_DEVICE_CLASSES,
    PRESENCE_DOMAINS,
    SUN_ELEVATION_ENTITY,
    SUN_ENTITY,
    VoiceKind,
    is_number,
)

LABEL_PREFIX = "entropy_fm_"
# role name to what it plays as; a role with None is muted
ROLES: dict[str, VoiceKind | None] = {
    "dishwasher": VoiceKind.DISHWASHER,
    "washer": VoiceKind.DRUM,
    "washing_machine": VoiceKind.DRUM,
    "dryer": VoiceKind.DRUM,
    "drum": VoiceKind.DRUM,
    "drone": VoiceKind.DRONE,
    "fridge": VoiceKind.DRONE,
    "oven": VoiceKind.DRONE,
    "temperature": VoiceKind.TEMPERATURE,
    "humidity": VoiceKind.HUMIDITY,
    "pluck": VoiceKind.PLUCK,
    "chord": VoiceKind.CHORD,
    "blip": VoiceKind.BLIP,
    "pad": VoiceKind.PAD_PARTIAL,
    "event": VoiceKind.PHASE,
    "hob": VoiceKind.HOB,
    "media": VoiceKind.MEDIA,
    "cover": VoiceKind.COVER,
    "progress": VoiceKind.PROGRESS,
    "sun": VoiceKind.SUN,
    "mute": None,
}
# translation keys, platforms and models that name an appliance; checked as substrings of the
# lower case text, which these integrations write in English
INTEGRATION_HINTS: tuple[tuple[str, str], ...] = (
    ("dishwasher", "dishwasher"),
    ("dish_washer", "dishwasher"),
    ("washing_machine", "washer"),
    ("washer", "washer"),
    ("laundry", "washer"),
    ("tumble_dryer", "dryer"),
    ("dryer", "dryer"),
    ("refrigerator", "fridge"),
    ("freezer", "fridge"),
    ("oven", "oven"),
)
# last resort keywords per role, lower case
NAME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dishwasher",
        (
            "dishwasher",
            "dish washer",
            "dish_washer",
            "geschirrspül",
            "geschirrspul",
            "geschirrspuel",
            "spülmaschine",
            "spulmaschine",
            "spuelmaschine",
            "vaatwasser",
            "lave-vaisselle",
            "lave vaisselle",
            "lavavajillas",
            "lavastoviglie",
            "zmywarka",
        ),
    ),
    (
        "dryer",
        (
            "dryer",
            "tumble",
            "trockner",
            "wäschetrockner",
            "droger",
            "sèche-linge",
            "seche-linge",
            "seche_linge",
            "secadora",
            "asciugatrice",
            "suszarka",
        ),
    ),
    (
        "washer",
        (
            "washing",
            "washer",
            "waschmaschine",
            "wasmachine",
            "lave-linge",
            "lave_linge",
            "machine à laver",
            "lavadora",
            "lavatrice",
            "pralka",
        ),
    ),
    ("fridge", ("fridge", "refrigerator", "kühlschrank", "koelkast", "frigo", "nevera", "lodówka")),
    ("oven", ("oven", "backofen", "herd", "four", "horno", "forno", "piekarnik")),
)
# entity categories that never play
SKIPPED_CATEGORIES = frozenset({"diagnostic", "config"})
STATE_WORDS = ("operation_state", "program_phase", "phase", "_state", "program", "mode")
HOB_WORDS = ("hob", "cooktop", "induction")
PROGRESS_WORDS = ("program_progress", "progress")
POWER_UNITS = frozenset({"w", "kw"})
ENERGY_UNITS = frozenset({"wh", "kwh"})
TEMPERATURE_UNITS = frozenset({"°c", "°f", "c", "f"})
WORD = re.compile(r"[^\W_]+|_", re.UNICODE)


@dataclass(frozen=True)
class EntityMeta:
    """What the registries say about an entity, besides its state."""

    platform: str = ""
    translation_key: str = ""
    manufacturer: str = ""
    model: str = ""
    # the names of the labels on the entity and on its device, lower case
    labels: frozenset[str] = field(default_factory=frozenset)
    # "diagnostic" or "config" for entities that are about the device, not what it does
    entity_category: str = ""
    device_id: str = ""


@dataclass(frozen=True)
class Role:
    """The voice of an entity, and why it has it."""

    kind: VoiceKind
    role: str
    # label, override, integration, device class or name
    source: str
    cumulative: bool = False

    @property
    def description(self) -> str:
        """Return a short text such as "role: dishwasher (label)"."""
        return f"role: {self.role} ({self.source})"


MUTED = Role(VoiceKind.BLIP, "mute", "label")


def parse_overrides(text: str) -> dict[str, str]:
    """
    Read role overrides from lines such as `sensor.geschirrspueler_power=dishwasher`.

    :param text: Lines, or comma separated pairs; lines without `=` and unknown roles are skipped.
    """
    found: dict[str, str] = {}
    for part in re.split(r"[\n,;]+", text or ""):
        entity_id, _, role = part.partition("=")
        entity_id, role = entity_id.strip().lower(), role.strip().lower()
        if entity_id and role in ROLES:
            found[entity_id] = role
    return found


def _role_from_labels(labels: frozenset[str]) -> str | None:
    """Return the role named by an `entropy_fm_<role>` label, if there is a known one."""
    for label in sorted(labels):
        if label.startswith(LABEL_PREFIX) and (role := label.removeprefix(LABEL_PREFIX)) in ROLES:
            return role
    return None


def _role_from_integration(meta: EntityMeta) -> str | None:
    """Return a role the integration data names."""
    text = f"{meta.translation_key} {meta.platform} {meta.model}".lower()
    text = text.replace(" ", "_")
    for hint, role in INTEGRATION_HINTS:
        if hint in text:
            return role
    return None


def _role_from_name(text: str) -> str | None:
    """Return a role named by keywords, the last resort."""
    lowered = text.lower()
    for role, words in NAME_KEYWORDS:
        for word in words:
            # a short word must stand alone ("four", "herd"), a long one may sit inside
            if (len(word) <= 5 and word in WORD.findall(lowered)) or (
                len(word) > 5 and word in lowered
            ):
                return role
    return None


def _is_diagnostic(entity_id: str, dclass: str, meta: EntityMeta) -> bool:
    """Return True for an entity about the device itself: signal, firmware, voltage, versions."""
    text = f"{dclass} {meta.translation_key} {entity_id.split('.', 1)[1]}".lower()
    return any(word in text for word in DIAGNOSTIC_WORDS)


def _is_hob_level(text: str) -> bool:
    """Return True for the power level of one zone of a hob."""
    return any(word in text for word in HOB_WORDS) and "power_level" in text


def _is_state_text(state: str) -> bool:
    """Return True for a state that is a word of a state machine, not 'unknown' or a time."""
    lowered = state.lower()
    return (
        lowered not in {"unknown", "unavailable", "none", ""}
        and len(lowered) < 40
        and ":" not in lowered
    )


def _from_role(role: str, source: str, *, cumulative: bool) -> Role | None:
    """Make a Role from a named role; muted roles give None."""
    kind = ROLES[role]
    if kind is None:
        return None
    return Role(kind, role, source, cumulative)


def classify(  # noqa: PLR0911, PLR0912, PLR0913, C901 - a flat decision table
    entity_id: str,
    name: str,
    device_class: str | None,
    unit: str | None,
    state: str | None,
    *,
    meta: EntityMeta | None = None,
    override: str | None = None,
    include_presence: bool = False,
) -> Role | None:
    """
    Decide which voice an entity gets, or None when it makes no sound.

    :param entity_id: The entity id.
    :param name: The friendly name.
    :param device_class: The device class from the attributes.
    :param unit: The unit of measurement.
    :param state: The current state, which tells a numeric sensor from a text one.
    :param meta: What the registries say about the entity.
    :param override: A role from the settings of the user.
    :param include_presence: Whether persons and device trackers play.
    :return: The role, or None for no voice.
    """
    meta = meta or EntityMeta()
    domain = entity_id.split(".", 1)[0]
    dclass = (device_class or "").lower()
    unit_l = (unit or "").lower()
    is_power = domain == "sensor" and (dclass == "power" or unit_l in POWER_UNITS)
    is_energy = domain == "sensor" and (dclass == "energy" or unit_l in ENERGY_UNITS)
    cumulative = is_energy and not is_power

    # 1. the user
    if override and override in ROLES:
        return _from_role(override, "override", cumulative=cumulative)
    if (label_role := _role_from_labels(meta.labels)) is not None:
        return _from_role(label_role, "label", cumulative=cumulative)

    # 2. the integration, for the sensors that measure an appliance
    if (is_power or is_energy) and (hint := _role_from_integration(meta)) is not None:
        return _from_role(hint, "integration", cumulative=cumulative)

    # 3. domain and device class
    if entity_id in (SUN_ENTITY, SUN_ELEVATION_ENTITY):
        return Role(VoiceKind.SUN, "sun", "device class")
    if meta.entity_category in SKIPPED_CATEGORIES or _is_diagnostic(entity_id, dclass, meta):
        return None
    if domain == "media_player":
        return Role(VoiceKind.MEDIA, "media", "domain")
    if domain == "cover":
        return Role(VoiceKind.COVER, "cover", "domain")
    if domain in {"select", "input_select"}:
        return Role(VoiceKind.PHASE, "event", "domain")
    if domain == "sensor":
        if dclass in IGNORED_DEVICE_CLASSES - {"enum"}:
            return None
        text_key = f"{entity_id} {meta.translation_key}".lower()
        if _is_hob_level(text_key):
            return Role(VoiceKind.HOB, "hob", "integration" if meta.platform else "name")
        if unit_l == "%" and any(word in text_key for word in PROGRESS_WORDS):
            return Role(VoiceKind.PROGRESS, "progress", "integration" if meta.platform else "name")
        if dclass == "enum" or (
            state is not None
            and not is_number(state)
            and _is_state_text(state)
            and any(word in text_key for word in STATE_WORDS)
        ):
            return Role(VoiceKind.PHASE, "event", "device class")
        if is_power or is_energy:
            # 4. the last resort tells which appliance it is
            if (word_role := _role_from_name(f"{entity_id} {name}")) is not None:
                return _from_role(word_role, "name", cumulative=cumulative)
            return Role(VoiceKind.DRONE, "drone", "device class", cumulative)
        if dclass == "temperature" or unit_l in TEMPERATURE_UNITS:
            return Role(VoiceKind.TEMPERATURE, "temperature", "device class")
        if dclass == "humidity":
            return Role(VoiceKind.HUMIDITY, "humidity", "device class")
        if is_number(state):
            return Role(VoiceKind.PAD_PARTIAL, "pad", "domain")
        return None
    if domain == "binary_sensor":
        if dclass in PLUCK_DEVICE_CLASSES:
            return Role(VoiceKind.PLUCK, "pluck", "device class")
        if dclass in IGNORED_BINARY_CLASSES or (dclass == "presence" and not include_presence):
            return None
        return Role(VoiceKind.BLIP, "blip", "domain")
    if domain in {"light", "switch"}:
        return Role(VoiceKind.CHORD, "chord", "domain")
    if domain in {"input_boolean", "fan"}:
        return Role(VoiceKind.BLIP, "blip", "domain")
    if domain in PRESENCE_DOMAINS and include_presence:
        return Role(VoiceKind.BLIP, "blip", "domain")
    return None
