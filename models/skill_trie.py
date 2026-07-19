"""
Skill Trie: Aho-Corasick multi-string matcher for skill extraction.

Replaces regex/SkillNer soup with a flat O(n) trie.
Loads once, matches 1000 resumes in < 3 seconds.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SkillTrie:
    """Aho-Corasick trie for high-speed skill matching."""

    def __init__(self, skill_db_path: Optional[str] = None):
        self.skill_db_path = skill_db_path
        self._trie = None
        self._canonical: Dict[str, str] = {}       # alias.lower() -> canonical name
        self._categories: Dict[str, str] = {}        # canonical name -> category
        self._all_patterns: List[str] = []            # sorted longest-first

        if skill_db_path and Path(skill_db_path).exists():
            self.load(skill_db_path)
            self.build()

    def load(self, path: str) -> "SkillTrie":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._canonical.clear()
        self._categories.clear()
        self._all_patterns.clear()

        for entry in data.get("skills", []):
            name = entry["name"]
            cat = entry.get("category", "other")
            aliases = [a.lower().strip() for a in entry.get("aliases", [name.lower()])]

            for alias in aliases:
                if alias and alias not in self._canonical:
                    self._canonical[alias] = name
                    self._categories[name] = cat
                    self._all_patterns.append(alias)

        self._all_patterns.sort(key=len, reverse=True)
        logger.info(
            "SkillTrie: %d skills, %d aliases",
            len(set(self._canonical.values())), len(self._canonical),
        )
        return self

    def build(self) -> "SkillTrie":
        import ahocorasick
        self._trie = ahocorasick.Automaton()
        for idx, pattern in enumerate(self._all_patterns):
            self._trie.add_word(pattern, (idx, pattern))
        self._trie.make_automaton()
        logger.info("SkillTrie: automaton built (%d patterns)", len(self._all_patterns))
        return self

    def match(self, text: str) -> List[Dict]:
        """Return deduplicated [{skill, category, match}] found in text."""
        if not self._trie or not text:
            return []

        text_lower = text.lower()
        found: Dict[str, tuple] = {}  # canonical -> (category, matched_alias)

        for end_idx, (_, pattern) in self._trie.iter(text_lower):
            canonical = self._canonical[pattern]
            start = end_idx - len(pattern) + 1
            # For short patterns (<4 chars), require word boundary
            if len(pattern) < 4 and not _is_word_boundary_match(text_lower, start, end_idx):
                continue
            if canonical not in found:
                found[canonical] = (self._categories.get(canonical, "other"), pattern)

        return [
            {"skill": name, "category": cat, "match": alias}
            for name, (cat, alias) in found.items()
        ]

    def match_categorized(self, text: str) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for r in self.match(text):
            grouped.setdefault(r["category"], []).append(r["skill"])
        return grouped

    @property
    def skill_count(self) -> int:
        return len(set(self._canonical.values()))

    @property
    def is_loaded(self) -> bool:
        return self._trie is not None


_WORD_CHARS = re.compile(r"[a-z0-9_]")


def _is_word_boundary_match(text: str, start: int, end: int) -> bool:
    """True if the substring text[start:end+1] is delimited by non-word chars."""
    before = text[start - 1] if start > 0 else " "
    after = text[end + 1] if end < len(text) - 1 else " "
    return (not _WORD_CHARS.match(before)) and (not _WORD_CHARS.match(after))
