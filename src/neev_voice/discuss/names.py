"""Random session name generator for discuss sessions.

Generates 3-word kebab-case names that are filesystem-safe
and human-readable, e.g. 'bright-coral-summit'.
"""

from __future__ import annotations

import random

__all__ = ["generate_session_name"]

_ADJECTIVES = [
    "bright",
    "calm",
    "clear",
    "cool",
    "crisp",
    "dark",
    "deep",
    "fast",
    "firm",
    "flat",
    "fresh",
    "grand",
    "keen",
    "kind",
    "lean",
    "loud",
    "mild",
    "neat",
    "pale",
    "pure",
    "rare",
    "rich",
    "safe",
    "sharp",
    "slim",
    "soft",
    "swift",
    "tall",
    "warm",
    "wide",
    "wild",
    "bold",
    "fair",
    "glad",
    "prime",
    "quiet",
    "rapid",
    "stern",
    "true",
    "vast",
]

_COLORS = [
    "amber",
    "azure",
    "coral",
    "cyan",
    "ivory",
    "jade",
    "lime",
    "maple",
    "ochre",
    "olive",
    "pearl",
    "plum",
    "ruby",
    "sage",
    "slate",
    "teal",
    "topaz",
    "umber",
    "cedar",
    "flint",
    "frost",
    "hazel",
    "iron",
    "linen",
    "onyx",
    "opal",
    "pewter",
    "quartz",
    "sienna",
    "stone",
]

_NOUNS = [
    "arch",
    "bay",
    "cliff",
    "cove",
    "creek",
    "dale",
    "dune",
    "edge",
    "field",
    "ford",
    "gate",
    "glen",
    "grove",
    "haven",
    "hill",
    "isle",
    "knoll",
    "lake",
    "lane",
    "ledge",
    "marsh",
    "mesa",
    "mist",
    "peak",
    "pond",
    "reef",
    "ridge",
    "shore",
    "slope",
    "summit",
    "trail",
    "vale",
    "brook",
    "crest",
    "drift",
    "forge",
    "glade",
    "heath",
    "inlet",
    "point",
]


def generate_session_name(rng: random.Random | None = None) -> str:
    """Generate a random 3-word kebab-case session name.

    Format: ``<adjective>-<color>-<noun>``, e.g. ``bright-coral-summit``.
    All words are lowercase alphanumeric, producing filesystem-safe names.

    Args:
        rng: Optional Random instance for reproducible names (testing).

    Returns:
        A 3-word kebab-case string suitable for use as a session name.
    """
    r = rng or random.Random()
    adjective = r.choice(_ADJECTIVES)
    color = r.choice(_COLORS)
    noun = r.choice(_NOUNS)
    return f"{adjective}-{color}-{noun}"
