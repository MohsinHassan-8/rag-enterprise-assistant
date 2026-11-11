"""Normalise place names extracted by spaCy (Week 4)."""

ALIASES = {
    "U.S.": "United States", "US": "United States", "USA": "United States",
    "U.S.A.": "United States", "UK": "United Kingdom", "U.K.": "United Kingdom",
}
# Entities spaCy tags as places that a user would not filter on
STOPLIST = {"Earth", "Middle Ages", "Sun"}


def normalize_places(raw: str | None, max_places: int = 5) -> list[str]:
    """Split a '|'-joined place string, merge aliases, drop junk, de-duplicate."""
    out = []
    for p in (raw or "").split("|"):
        p = ALIASES.get(p.strip(), p.strip())
        if p and p not in STOPLIST and p not in out:
            out.append(p)
    return out[:max_places]
