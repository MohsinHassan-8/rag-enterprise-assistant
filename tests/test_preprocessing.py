"""Unit tests for src/preprocessing.py."""
from src.preprocessing import (
    clean_document,
    clean_whitespace,
    detect_language,
    is_english,
    normalize_unicode,
    remove_trailing_sections,
    strip_html,
)


class TestStripHtml:
    def test_removes_tags(self):
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result
        assert "Hello world" in result

    def test_removes_script_and_style_content(self):
        result = strip_html("<style>p{color:red}</style><script>alert(1)</script>Text")
        assert result.strip() == "Text"

    def test_decodes_entities(self):
        assert strip_html("Tom &amp; Jerry") == "Tom & Jerry"
        assert strip_html("caf&eacute;") == "caf\u00e9"

    def test_leaves_plain_comparisons_alone(self):
        text = "5 < 7 and 9 > 2"
        assert strip_html(text) == text

    def test_block_tags_become_newlines(self):
        result = strip_html("<p>one</p><p>two</p>")
        assert "onetwo" not in result
        assert "one" in result and "two" in result

    def test_empty_string(self):
        assert strip_html("") == ""


class TestNormalizeUnicode:
    def test_expands_ligatures(self):
        assert normalize_unicode("\ufb01ne") == "fine"

    def test_fullwidth_to_ascii(self):
        assert normalize_unicode("\uff28\uff45\uff4c\uff4c\uff4f") == "Hello"

    def test_removes_zero_width_and_soft_hyphen(self):
        assert normalize_unicode("zero\u200bwidth\ufeff") == "zerowidth"
        assert normalize_unicode("soft\u00adhyphen") == "softhyphen"

    def test_nbsp_becomes_space(self):
        assert normalize_unicode("a\xa0b") == "a b"

    def test_straightens_curly_quotes(self):
        assert normalize_unicode("\u201cquoted\u201d and it\u2019s") == '"quoted" and it\'s'

    def test_removes_control_chars(self):
        assert normalize_unicode("a\x00b\x07c") == "abc"

    def test_preserves_accented_letters(self):
        assert normalize_unicode("caf\u00e9") == "caf\u00e9"

    def test_preserves_newlines_and_tabs(self):
        assert normalize_unicode("a\n\tb") == "a\n\tb"


class TestCleanWhitespace:
    def test_collapses_spaces_and_tabs(self):
        assert clean_whitespace("a   b\t\tc") == "a b c"

    def test_caps_blank_lines_at_one(self):
        assert clean_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_each_line(self):
        assert clean_whitespace("  line1  \n  line2  ") == "line1\nline2"

    def test_whitespace_only_lines_count_as_blank(self):
        assert clean_whitespace("a\n   \n   \nb") == "a\n\nb"

    def test_normalizes_crlf(self):
        assert clean_whitespace("a\r\nb") == "a\nb"

    def test_strips_ends(self):
        assert clean_whitespace("\n\n text \n\n") == "text"

    def test_keeps_single_paragraph_break(self):
        assert clean_whitespace("a\n\nb") == "a\n\nb"


class TestLanguage:
    def test_english_detected(self):
        text = "The moon orbits the Earth and causes tides in the sea."
        assert detect_language(text) == "en"
        assert is_english(text)

    def test_french_rejected(self):
        text = "Le chat est assis sur le tapis et regarde par la fen\u00eatre."
        assert detect_language(text) == "fr"
        assert not is_english(text)

    def test_urdu_rejected(self):
        text = "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0627\u0631\u062f\u0648 \u062c\u0645\u0644\u06c1 \u06c1\u06d2 \u062c\u0648 \u0679\u06cc\u0633\u0679 \u06a9\u06d2 \u0644\u06cc\u06d2 \u0644\u06a9\u06be\u0627 \u06af\u06cc\u0627 \u06c1\u06d2\u06d4"
        assert not is_english(text)

    def test_too_short_is_unknown(self):
        assert detect_language("Hi there") == "unknown"
        assert not is_english("Hi")

    def test_empty_is_unknown(self):
        assert detect_language("") == "unknown"

    def test_digits_only_not_english(self):
        assert not is_english("1234567890 " * 5)


class TestRemoveTrailingSections:
    def test_cuts_at_references(self):
        text = "Body text.\n\nReferences\n\nhttp://example.com"
        assert remove_trailing_sections(text) == "Body text."

    def test_cuts_at_other_websites(self):
        assert remove_trailing_sections("Body\n\nOther websites\n x") == "Body"

    def test_no_heading_unchanged(self):
        text = "Just a body with no trailing sections."
        assert remove_trailing_sections(text) == text

    def test_word_inside_sentence_not_cut(self):
        text = "The References section is long.\nMore text."
        assert remove_trailing_sections(text) == text

    def test_case_insensitive(self):
        assert remove_trailing_sections("Body\n\nreferences\nlink") == "Body"


class TestCleanDocument:
    def test_full_pipeline(self):
        raw = "<p>Hello&nbsp;&nbsp; <b>world</b></p>\n\n\n\nBye"
        assert clean_document(raw) == "Hello world\n\nBye"

    def test_wikipedia_style_tail_removed(self):
        raw = (
            "Camden is a borough.\n\nPlaces \n Bloomsbury\n Camden\n\n"
            "References\n\nOther websites\n\n Camden TV"
        )
        result = clean_document(raw)
        assert "Bloomsbury" in result
        assert "References" not in result
        assert "Camden TV" not in result

    def test_idempotent(self):
        raw = "<p>Hello&nbsp;&nbsp; <b>world</b></p>\n\n\n\nBye\n\nReferences\n x"
        once = clean_document(raw)
        assert clean_document(once) == once

    def test_empty_string(self):
        assert clean_document("") == ""