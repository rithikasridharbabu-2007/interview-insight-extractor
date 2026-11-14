"""Read quotes from text, CSV or transcript files; write themes to Markdown or CSV."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from insights import Pattern

FORMATS = ("text", "csv", "transcript")
_TURN = re.compile(r"^([A-Za-z][\w .'-]{0,39}):\s+(.+)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def parse_text(text: str) -> list[str]:
    """One quote per line. Blank lines and lines starting with # are ignored."""
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def parse_csv(text: str, column: str = "quote", speaker_column: str | None = None,
              speaker: str | None = None) -> list[str]:
    """Quotes from one CSV column, optionally only the rows of one speaker."""
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    for wanted in filter(None, (column, speaker_column)):
        if wanted not in header:
            raise ValueError(f"CSV has no column {wanted!r}. Columns found: {', '.join(header) or 'none'}")
    if speaker and not speaker_column:
        raise ValueError("filtering a CSV by speaker needs a speaker column")
    return [
        row[column].strip() for row in reader
        if (row[column] or "").strip()
        and (not speaker or (row[speaker_column] or "").strip().lower() == speaker.lower())
    ]


def parse_transcript(text: str, speaker: str | None = None, min_words: int = 3) -> list[str]:
    """Sentences from a transcript whose turns look like `Speaker: text`.

    A line without a speaker prefix continues the previous turn. With
    `speaker`, only that person's turns are kept (case-insensitive), which is
    how interviewer questions are left out. Sentences shorter than `min_words`
    ("Yeah.", "Sure.") are dropped because they carry no theme.
    """
    turns: list[list[str]] = []  # [speaker, text]
    for line in parse_text(text):
        match = _TURN.match(line)
        if match:
            turns.append([match[1].strip(), match[2]])
        elif turns:
            turns[-1][1] += " " + line
    if not turns:
        raise ValueError("no 'Speaker: text' lines found; is this a transcript?")
    return [
        sentence.strip()
        for who, said in turns if not speaker or who.lower() == speaker.lower()
        for sentence in _SENTENCE_END.split(said)
        if len(sentence.split()) >= min_words
    ]


def detect_format(path: Path) -> str:
    """.csv is CSV, .transcript is a transcript, anything else is plain text."""
    return {".csv": "csv", ".transcript": "transcript"}.get(Path(path).suffix.lower(), "text")


def load_quotes(path: Path, format: str | None = None, column: str = "quote",
                speaker_column: str | None = None, speaker: str | None = None) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig")
    format = format or detect_format(path)
    if format == "csv":
        return parse_csv(text, column, speaker_column, speaker)
    if format == "transcript":
        return parse_transcript(text, speaker)
    if format == "text":
        return parse_text(text)
    raise ValueError(f"unknown format {format!r}; choose from {', '.join(FORMATS)}")


def to_markdown(patterns: list[Pattern], unmatched: list[str]) -> str:
    """A heading per theme, ready to paste into a doc."""
    total = sum(len(p.quotes) for p in patterns) + len(unmatched)
    out = ["# Interview themes", "", f"{total} quotes, {len(patterns)} themes, {len(unmatched)} not grouped.", ""]
    for number, pattern in enumerate(patterns, 1):
        out += [
            f"## {number}. {pattern.label or 'Unlabelled theme'} ({len(pattern.quotes)} quotes)",
            "",
            f"**Most typical:** \"{pattern.representative}\"",
            "",
            *(f"- {quote}" for quote in pattern.quotes),
            "",
        ]
    if unmatched:
        out += [f"## Not grouped ({len(unmatched)})", "", *(f"- {quote}" for quote in unmatched), ""]
    return "\n".join(out)


def to_csv(patterns: list[Pattern], unmatched: list[str]) -> str:
    """One row per quote. Ungrouped quotes have an empty theme_id and label."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["theme_id", "label", "quote", "is_representative"])
    for number, pattern in enumerate(patterns, 1):
        for quote in pattern.quotes:
            writer.writerow([number, pattern.label, quote, str(quote == pattern.representative).lower()])
    for quote in unmatched:
        writer.writerow(["", "", quote, "false"])
    return buffer.getvalue()


def write_output(path: Path, patterns: list[Pattern], unmatched: list[str]) -> None:
    """Write Markdown or CSV, chosen by the file extension."""
    writers = {".md": to_markdown, ".csv": to_csv}
    suffix = Path(path).suffix.lower()
    if suffix not in writers:
        raise ValueError(f"cannot write {suffix or 'a file with no extension'}; use .md or .csv")
    Path(path).write_text(writers[suffix](patterns, unmatched), encoding="utf-8")
