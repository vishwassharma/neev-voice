"""Tests for the session name generator."""

from __future__ import annotations

import random
import re

from neev_voice.discuss.names import (
    _ADJECTIVES,
    _COLORS,
    _NOUNS,
    generate_session_name,
)


class TestGenerateSessionName:
    """Tests for generate_session_name function."""

    def test_returns_string(self) -> None:
        """Returns a string."""
        name = generate_session_name()
        assert isinstance(name, str)

    def test_kebab_case_format(self) -> None:
        """Name is in kebab-case with exactly two hyphens."""
        name = generate_session_name()
        parts = name.split("-")
        assert len(parts) == 3

    def test_all_lowercase(self) -> None:
        """All characters are lowercase."""
        name = generate_session_name()
        assert name == name.lower()

    def test_filesystem_safe(self) -> None:
        """Name contains only alphanumerics and hyphens."""
        for _ in range(50):
            name = generate_session_name()
            assert re.match(r"^[a-z0-9-]+$", name), f"Unsafe name: {name}"

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same name."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert generate_session_name(rng1) == generate_session_name(rng2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different names (with high probability)."""
        names = {generate_session_name(random.Random(i)) for i in range(20)}
        assert len(names) > 10  # at least some variety

    def test_words_from_lists(self) -> None:
        """Each word comes from the expected word list."""
        rng = random.Random(99)
        name = generate_session_name(rng)
        adj, color, noun = name.split("-")
        assert adj in _ADJECTIVES
        assert color in _COLORS
        assert noun in _NOUNS

    def test_word_lists_nonempty(self) -> None:
        """Word lists contain sufficient variety."""
        assert len(_ADJECTIVES) >= 20
        assert len(_COLORS) >= 20
        assert len(_NOUNS) >= 20

    def test_word_lists_lowercase_alpha(self) -> None:
        """All words in lists are lowercase alphabetic."""
        for word in _ADJECTIVES + _COLORS + _NOUNS:
            assert word.isalpha(), f"Non-alpha word: {word}"
            assert word.islower(), f"Non-lowercase word: {word}"

    def test_no_duplicate_words_in_lists(self) -> None:
        """No duplicates within each word list."""
        assert len(_ADJECTIVES) == len(set(_ADJECTIVES))
        assert len(_COLORS) == len(set(_COLORS))
        assert len(_NOUNS) == len(set(_NOUNS))

    def test_default_rng_varies(self) -> None:
        """Without explicit RNG, names vary between calls."""
        names = {generate_session_name() for _ in range(10)}
        assert len(names) > 1
