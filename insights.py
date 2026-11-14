"""Group interview quotes by meaning.

The clustering works on any embedding matrix, so it can be tested without
downloading a model. Only `embed()` needs sentence-transformers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.35


@dataclass
class Pattern:
    """A group of quotes that say roughly the same thing."""

    quotes: list[str]
    representative: str  # the quote closest to all the others in the group
    cohesion: float  # mean pairwise cosine similarity inside the group


def embed(quotes: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode(quotes)


def find_patterns(
    quotes: list[str],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[Pattern], list[str]]:
    """Return (patterns, unmatched quotes).

    Quotes are merged with average-linkage clustering: two groups join only
    while their average cosine similarity stays above `threshold`. Unlike
    comparing everything to the first quote seen, the result does not depend
    on the order of the input.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if len(quotes) != len(embeddings):
        raise ValueError("need exactly one embedding per quote")
    if len(quotes) < 2:
        return [], list(quotes)

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - threshold,
    ).fit_predict(embeddings)
    similarity = cosine_similarity(embeddings)

    patterns: list[Pattern] = []
    unmatched: list[str] = []
    for label in sorted(set(labels), key=lambda l: int(np.flatnonzero(labels == l)[0])):
        members = np.flatnonzero(labels == label)
        if len(members) == 1:
            unmatched.append(quotes[members[0]])
            continue
        block = similarity[np.ix_(members, members)]
        off_diagonal = block[~np.eye(len(members), dtype=bool)]
        central = members[int(np.argmax(block.sum(axis=1)))]
        patterns.append(
            Pattern(
                quotes=[quotes[i] for i in members],
                representative=quotes[central],
                cohesion=float(off_diagonal.mean()),
            )
        )

    patterns.sort(key=lambda p: len(p.quotes), reverse=True)
    return patterns, unmatched
