"""Command-line tool: python3 demo.py [file] [--threshold 0.35] [--output themes.md]

Run `python3 demo.py --help` for input formats and all options.
"""
import argparse
import json
import sys
from pathlib import Path

from insights import DEFAULT_MODEL, DEFAULT_THRESHOLD, embed, find_patterns
from quote_io import FORMATS, load_quotes, write_output

SAMPLE = Path(__file__).with_name("sample_quotes.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Group interview quotes by meaning.")
    parser.add_argument("file", nargs="?", type=Path, default=SAMPLE,
                        help="quotes as text (one per line), CSV or transcript (default: sample_quotes.txt)")
    parser.add_argument("--format", choices=FORMATS,
                        help="input format (default: csv for .csv, transcript for .transcript, else text)")
    parser.add_argument("--column", default="quote", help="CSV column holding the quotes (default: %(default)s)")
    parser.add_argument("--speaker-column", help="CSV column holding the speaker, for use with --speaker")
    parser.add_argument("--speaker", help="keep only this speaker's turns (transcript, or CSV with --speaker-column)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum average similarity to group quotes, 0 to 1 (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--output", type=Path, help="also write the themes to a .md or .csv file")
    args = parser.parse_args()

    if args.output and args.output.suffix.lower() not in (".md", ".csv"):
        parser.error("--output must end in .md or .csv")
    try:
        quotes = load_quotes(args.file, args.format, args.column, args.speaker_column, args.speaker)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not quotes:
        print(f"No quotes found in {args.file}", file=sys.stderr)
        return 1

    patterns, unmatched = find_patterns(quotes, embed(quotes, args.model), args.threshold)
    if args.output:
        write_output(args.output, patterns, unmatched)

    if args.json:
        print(json.dumps({
            "patterns": [vars(p) for p in patterns],
            "unmatched": unmatched,
        }, indent=2))
        return 0

    print(f"Analyzed {len(quotes)} quotes. Found {len(patterns)} patterns.\n")
    for number, pattern in enumerate(patterns, 1):
        print(f"Pattern {number}: {pattern.label or 'unlabelled'}"
              f" ({len(pattern.quotes)} quotes, similarity {pattern.cohesion:.2f})")
        print(f'  Most typical: "{pattern.representative}"')
        for quote in pattern.quotes:
            print(f"  - {quote}")
        print()
    if unmatched:
        print(f"Not grouped ({len(unmatched)}):")
        for quote in unmatched:
            print(f"  - {quote}")
    if args.output:
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
