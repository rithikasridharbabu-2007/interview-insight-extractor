"""Group interview quotes by meaning.

The clustering works on any embedding matrix, so it can be tested without
downloading a model. Only `embed()` needs sentence-transformers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.20

# scikit-learn's English list, plus spoken-interview filler it does not cover.
STOP_WORDS = ENGLISH_STOP_WORDS | frozenset(
    "did didn does doesn don wasn weren isn aren couldn wouldn shouldn won can "
    "ve ll im just really actually basically honestly kind sort like lot bit "
    "thing things stuff got get gets getting know knew think thought feel felt "
    "way use used using need want wanted make makes time times".split()
)


@dataclass
class Pattern:
    """A group of quotes that say roughly the same thing."""

    quotes: list[str]
    representative: str  # the quote closest to all the others in the group
    cohesion: float  # mean pairwise cosine similarity inside the group
    label: str = ""  # a few distinctive words, see label_themes()


@lru_cache(maxsize=2)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(quotes: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    return _model(model_name).encode(quotes)


def _terms(text: str) -> list[str]:
    """Content words, plus pairs of content words that are adjacent in the text."""
    words = re.findall(r"[a-z]+", text.lower().replace("'", " "))
    keep = [len(w) > 1 and w not in STOP_WORDS for w in words]
    terms = [w for w, k in zip(words, keep) if k]
    terms += [f"{a} {b}" for a, b, ka, kb in zip(words, words[1:], keep, keep[1:]) if ka and kb]
    return terms


def label_themes(groups: list[list[str]], others: list[str] = (), max_words: int = 5) -> list[str]:
    """Return one short label per group of quotes.

    Each term (a content word or a pair of adjacent content words) is scored
    by its mean TF-IDF weight in the group's quotes minus its mean weight in
    every other quote, so a label is made of words that set the group apart,
    not words that are merely frequent. Terms are added in score order until
    the label has 3 to `max_words` words; a group with few distinctive terms
    gets a shorter label, or an empty one.
    """
    quotes = [q for group in groups for q in group] + list(others)
    owner = np.array([g for g, group in enumerate(groups) for _ in group] + [-1] * len(others))
    try:
        vectorizer = TfidfVectorizer(analyzer=_terms, binary=True, norm=None)
        weights = vectorizer.fit_transform(quotes).toarray()
    except ValueError:  # no usable words at all
        return ["" for _ in groups]
    vocabulary = vectorizer.get_feature_names_out()

    labels = []
    for g in range(len(groups)):
        inside, outside = weights[owner == g], weights[owner != g]
        score = inside.mean(axis=0) - (outside.mean(axis=0) if len(outside) else 0)
        chosen: list[str] = []
        # Best score first; on a tie a word pair reads better than a single word.
        ranked = sorted(range(len(vocabulary)),
                        key=lambda i: (-round(score[i], 9), -vocabulary[i].count(" "), vocabulary[i]))
        for i in ranked:
            if score[i] <= 0 or sum(len(t.split()) for t in chosen) >= 3:
                break
            parts = vocabulary[i].split()
            overlapping = [t for t in chosen if set(t.split()) & set(parts)]
            if any(len(t.split()) >= len(parts) for t in overlapping):
                continue  # these words are already in the label
            # A pair such as "free plan" replaces "free" or "plan" picked singly.
            candidate = [t for t in chosen if t not in overlapping] + [vocabulary[i]]
            if sum(len(t.split()) for t in candidate) <= max_words:
                chosen = candidate
        labels.append(", ".join(chosen))
    return labels


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
    for pattern, label in zip(patterns, label_themes([p.quotes for p in patterns], unmatched)):
        pattern.label = label
    return patterns, unmatched
