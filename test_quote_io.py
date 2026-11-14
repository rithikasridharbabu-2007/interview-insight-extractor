"""Parser and writer tests. No model download is needed."""
import csv
import io

import pytest

from insights import Pattern
from quote_io import (detect_format, load_quotes, parse_csv, parse_text, parse_transcript,
                      to_csv, to_markdown, write_output)

CSV = 'who,quote,date\nAna,"Setup was hard, really hard",2026-01-02\nSam,How did setup go?,2026-01-02\nAna,,2026-01-03\n'
TRANSCRIPT = """\
# Interview 3
Interviewer: How did the first week go? Anything stand out?
Ana: Setup was hard. I gave up twice before it worked!
and then I asked a colleague for help.
Interviewer: Got it.
Ana: Yeah. The search is great though.
"""


def test_parse_text_skips_blanks_and_comments():
    assert parse_text("# heading\n\n  first quote  \nsecond\n") == ["first quote", "second"]


def test_parse_csv_reads_the_named_column():
    assert parse_csv(CSV) == ["Setup was hard, really hard", "How did setup go?"]
    assert parse_csv(CSV, column="who") == ["Ana", "Sam", "Ana"]


def test_parse_csv_filters_by_speaker():
    assert parse_csv(CSV, speaker_column="who", speaker="ana") == ["Setup was hard, really hard"]


@pytest.mark.parametrize("kwargs, missing", [({"column": "text"}, "text"), ({"speaker_column": "name"}, "name")])
def test_parse_csv_names_the_missing_column(kwargs, missing):
    with pytest.raises(ValueError, match=f"no column '{missing}'.*who, quote, date"):
        parse_csv(CSV, **kwargs)


def test_parse_csv_speaker_needs_a_speaker_column():
    with pytest.raises(ValueError, match="speaker column"):
        parse_csv(CSV, speaker="Ana")


def test_parse_transcript_splits_turns_into_sentences():
    assert parse_transcript(TRANSCRIPT) == [
        "How did the first week go?",
        "Anything stand out?",
        "Setup was hard.",
        "I gave up twice before it worked!",
        "and then I asked a colleague for help.",
        "The search is great though.",
    ]  # "Got it." and "Yeah." are too short to keep


def test_parse_transcript_keeps_one_speaker_and_drops_short_sentences():
    ana = parse_transcript(TRANSCRIPT, speaker="ana")
    assert ana == [
        "Setup was hard.",
        "I gave up twice before it worked!",
        "and then I asked a colleague for help.",
        "The search is great though.",
    ]
    assert parse_transcript(TRANSCRIPT, speaker="Nobody") == []


def test_parse_transcript_rejects_text_without_speakers():
    with pytest.raises(ValueError, match="Speaker: text"):
        parse_transcript("just a line\nand another\n")


def test_detect_format_by_extension():
    assert [detect_format(name) for name in ("a.csv", "a.CSV", "a.transcript", "a.txt", "notes")] == [
        "csv", "csv", "transcript", "text", "text"]


def test_load_quotes_detects_format_and_allows_override(tmp_path):
    (tmp_path / "q.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "call.txt").write_text(TRANSCRIPT, encoding="utf-8")
    assert load_quotes(tmp_path / "q.csv") == parse_csv(CSV)
    assert len(load_quotes(tmp_path / "call.txt")) == 5  # read as plain lines
    assert load_quotes(tmp_path / "call.txt", format="transcript", speaker="Ana") == parse_transcript(TRANSCRIPT, "Ana")
    with pytest.raises(ValueError, match="unknown format"):
        load_quotes(tmp_path / "q.csv", format="pdf")


PATTERNS = [
    Pattern(quotes=["Setup was hard", 'It said "error", then quit'], representative="Setup was hard",
            cohesion=0.5, label="setup, error"),
    Pattern(quotes=["Too pricey", "Costs a lot"], representative="Costs a lot", cohesion=0.4),
]
UNMATCHED = ["I want dark mode"]


def test_markdown_has_a_heading_per_theme():
    text = to_markdown(PATTERNS, UNMATCHED)
    assert "5 quotes, 2 themes, 1 not grouped." in text
    assert "## 1. setup, error (2 quotes)" in text
    assert "## 2. Unlabelled theme (2 quotes)" in text
    assert '**Most typical:** "Costs a lot"' in text
    assert '- It said "error", then quit' in text
    assert text.index("## Not grouped (1)") < text.index("- I want dark mode")
    assert "Not grouped" not in to_markdown(PATTERNS, []).split("\n", 3)[3]


def test_csv_has_one_row_per_quote():
    rows = list(csv.DictReader(io.StringIO(to_csv(PATTERNS, UNMATCHED))))
    assert list(rows[0]) == ["theme_id", "label", "quote", "is_representative"]
    assert [r["quote"] for r in rows] == ["Setup was hard", 'It said "error", then quit',
                                          "Too pricey", "Costs a lot", "I want dark mode"]
    assert [r["theme_id"] for r in rows] == ["1", "1", "2", "2", ""]
    assert [r["is_representative"] for r in rows] == ["true", "false", "false", "true", "false"]
    assert rows[0]["label"] == "setup, error" and rows[4]["label"] == ""


def test_write_output_picks_the_writer_by_extension(tmp_path):
    write_output(tmp_path / "out.md", PATTERNS, UNMATCHED)
    write_output(tmp_path / "out.csv", PATTERNS, UNMATCHED)
    assert (tmp_path / "out.md").read_text(encoding="utf-8") == to_markdown(PATTERNS, UNMATCHED)
    assert (tmp_path / "out.csv").read_text(encoding="utf-8") == to_csv(PATTERNS, UNMATCHED)
    with pytest.raises(ValueError, match=".md or .csv"):
        write_output(tmp_path / "out.docx", PATTERNS, UNMATCHED)
