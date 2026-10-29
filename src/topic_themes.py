"""Map fine-grained BERTopic topic ids to broad, human-readable themes.

Topic ids refer to the run saved in data/processed/doc_topics.parquet
(see reports/topic_review.csv). If scripts/run_topics.py is re-run, the
mapping must be re-checked by hand.
"""

OUTLIER_TOPIC = -1
DATE_TOPIC = -2
UNCLASSIFIED = "Unclassified"
DATES = "Dates and years"

THEME_TOPICS = {
    "Film, TV and anime": [0, 15, 100],
    "Music": [1, 11, 16],
    "Games": [26, 42],
    "Language": [3],
    "Arts and literature": [4, 50],
    "Computing": [2, 76],
    "Politics and government": [5, 60, 67, 73, 80, 95],
    "North America": [6, 43, 92, 106],
    "Europe": [7, 8, 13, 37, 38, 46, 48, 85, 103, 108],
    "Asia and Middle East": [10, 12, 35, 64],
    "Africa": [27],
    "Oceania and Pacific": [28, 105],
    "Latin America and Caribbean": [45, 56, 57, 83],
    "History": [18, 24, 30, 40, 97],
    "Religion, myth and philosophy": [20, 41, 52, 59, 65, 71, 82, 84],
    "Sports": [36, 44, 69, 78, 91, 99],
    "Animals and nature": [9, 31, 72, 77, 89, 101],
    "Human body and medicine": [21, 33, 58, 70, 81, 86, 90, 94],
    "Food and plants": [23, 32],
    "Physics and chemistry": [14, 54, 61, 66, 68, 79, 96],
    "Technology and transport": [25, 39, 53, 75, 93, 107],
    "Earth, space and weather": [19, 29, 34, 74],
    "Math, units and time": [22, 47, 49, 51, 62],
    "Economics and business": [17],
    "Society and everyday life": [55, 63, 87, 88, 98, 102, 104],
}

TOPIC_TO_THEME = {}
for _theme, _ids in THEME_TOPICS.items():
    for _t in _ids:
        if _t in TOPIC_TO_THEME:
            raise ValueError("topic %d mapped twice" % _t)
        TOPIC_TO_THEME[_t] = _theme


def theme_for(topic: int) -> str:
    """Theme name for a topic id. -1 is outliers, -2 is date/year pages."""
    if topic == OUTLIER_TOPIC:
        return UNCLASSIFIED
    if topic == DATE_TOPIC:
        return DATES
    if topic not in TOPIC_TO_THEME:
        raise ValueError("unknown topic id: %r" % (topic,))
    return TOPIC_TO_THEME[topic]
