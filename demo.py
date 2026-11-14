"""Command-line demo: python3 demo.py [quotes.txt] [--threshold 0.35] [--json]"""
import argparse
import json
import sys
from pathlib import Path

from insights import DEFAULT_MODEL, DEFAULT_THRESHOLD, embed, find_patterns

SAMPLE = Path(__file__).with_name("sample_quotes.txt")


def read_quotes(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Group interview quotes by meaning.")
    parser.add_argument("file", nargs="?", type=Path, default=SAMPLE,
                        help="text file with one quote per line (default: sample_quotes.txt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum average similarity to group quotes, 0 to 1 (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    quotes = read_quotes(args.file)
    if not quotes:
        print(f"No quotes found in {args.file}", file=sys.stderr)
        return 1

    patterns, unmatched = find_patterns(quotes, embed(quotes, args.model), args.threshold)

    if args.json:
        print(json.dumps({
            "patterns": [vars(p) for p in patterns],
            "unmatched": unmatched,
        }, indent=2))
        return 0

    print(f"Analyzed {len(quotes)} quotes. Found {len(patterns)} patterns.\n")
    for number, pattern in enumerate(patterns, 1):
        print(f"Pattern {number}: {len(pattern.quotes)} quotes, similarity {pattern.cohesion:.2f}")
        print(f'  Most typical: "{pattern.representative}"')
        for quote in pattern.quotes:
            print(f"  - {quote}")
        print()
    if unmatched:
        print(f"Not grouped ({len(unmatched)}):")
        for quote in unmatched:
            print(f"  - {quote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
