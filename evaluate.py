"""Score the clustering against hand-labelled quotes at several thresholds.

    python3 evaluate.py [eval/labeled_quotes.csv] [--model all-MiniLM-L6-v2]

The CSV needs `quote` and `theme` columns. Prints a Markdown table.
"""
import argparse
import csv
from pathlib import Path

from sklearn.metrics import (adjusted_rand_score, homogeneity_completeness_v_measure,
                             normalized_mutual_info_score)

from insights import DEFAULT_MODEL, DEFAULT_THRESHOLD, embed, find_patterns

THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def score(quotes, themes, embeddings, threshold) -> dict:
    """Compare predicted groups with the labelled themes.

    Every ungrouped quote counts as its own cluster of one, so leaving a
    quote out is not free: it lowers completeness unless its theme really
    has a single quote.
    """
    patterns, unmatched = find_patterns(quotes, embeddings, threshold)
    predicted = {q: n for n, p in enumerate(patterns) for q in p.quotes}
    predicted.update({q: len(patterns) + n for n, q in enumerate(unmatched)})
    labels = [predicted[q] for q in quotes]
    homogeneity, completeness, _ = homogeneity_completeness_v_measure(themes, labels)
    return {
        "threshold": threshold,
        "clusters": len(patterns),
        "ungrouped": len(unmatched),
        "ari": adjusted_rand_score(themes, labels),
        "nmi": normalized_mutual_info_score(themes, labels),
        "homogeneity": homogeneity,
        "completeness": completeness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", type=Path,
                        default=Path(__file__).parent / "eval" / "labeled_quotes.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    with args.file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    quotes, themes = [r["quote"] for r in rows], [r["theme"] for r in rows]
    if len(set(quotes)) != len(quotes):
        raise SystemExit("duplicate quotes in the evaluation file")
    embeddings = embed(quotes, args.model)

    sizes = [themes.count(t) for t in set(themes)]
    print(f"{len(quotes)} quotes, {len(sizes)} labelled themes "
          f"({sizes.count(1)} with a single quote), model {args.model}\n")
    print("| Threshold | Clusters | Ungrouped | ARI | NMI | Homogeneity | Completeness |")
    print("|---|---|---|---|---|---|---|")
    for threshold in THRESHOLDS:
        r = score(quotes, themes, embeddings, threshold)
        name = f"{threshold:.2f}" + (" (default)" if threshold == DEFAULT_THRESHOLD else "")
        print(f"| {name} | {r['clusters']} | {r['ungrouped']} | {r['ari']:.2f} | {r['nmi']:.2f} "
              f"| {r['homogeneity']:.2f} | {r['completeness']:.2f} |")


if __name__ == "__main__":
    main()
