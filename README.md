# Interview Insight Extractor

Groups user-interview quotes by meaning, so recurring themes show up without keywords or manual tagging.

Give it a text file with one quote per line. It turns each quote into a sentence embedding, clusters quotes whose average cosine similarity is above a threshold, and prints each group with its most typical quote.

## Example

Output of `python3 demo.py` on the bundled [sample_quotes.txt](sample_quotes.txt):

```
Analyzed 8 quotes. Found 3 patterns.

Pattern 1: 3 quotes, similarity 0.43
  Most typical: "I love how quickly I can find past research"
  - The search feature is incredibly fast and accurate
  - I love how quickly I can find past research
  - Finding old interviews is so much easier now

Pattern 2: 2 quotes, similarity 0.42
  Most typical: "The onboarding was really confusing, I didn't know where to start"
  - The onboarding was really confusing, I didn't know where to start
  - I gave up during setup because there were too many steps

Pattern 3: 2 quotes, similarity 0.38
  Most typical: "The pricing page wasn't clear, I didn't know what I'd be charged"
  - The pricing page wasn't clear, I didn't know what I'd be charged
  - I had no idea what the free plan included

Not grouped (1):
  - I couldn't figure out how to invite my teammates
```

## Run it

Requires Python 3.10 or newer. The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB).

```bash
pip install -r requirements.txt
python3 demo.py                              # bundled sample
python3 demo.py my_quotes.txt --threshold 0.4
python3 demo.py my_quotes.txt --json         # machine-readable output
```

`--threshold` is the minimum average similarity (0 to 1) for quotes to be grouped. Raise it for fewer, tighter groups.

## How it works

1. **Embed:** each quote becomes a vector with [sentence-transformers](https://www.sbert.net/).
2. **Cluster:** average-linkage agglomerative clustering on cosine distance. Groups merge only while their average similarity stays above the threshold, and the result does not depend on the order of the quotes.
3. **Summarise:** each group reports its size, its mean pairwise similarity, and the quote closest to all the others.

The clustering lives in [insights.py](insights.py) and takes any embedding matrix, so it can be reused with a different model.

## Tests

```bash
pip install pytest
pytest
```

The tests use hand-made vectors and need no model download.

## Limits

- Tested on a handful of short quotes, not on real interview transcripts.
- One threshold for everything; long or multi-topic quotes will group poorly. Split transcripts into single-idea quotes first.
- English only with the default model.

## Background

Started as a small demo of the embeddings-plus-cosine-similarity approach I used for pattern detection in REMIND, a Responsible Computing Club app at SJSU, applied to the interview-analysis problem that research tools like Great Question work on.

Rithika Sridharbabu, Computer Engineering, San José State University
