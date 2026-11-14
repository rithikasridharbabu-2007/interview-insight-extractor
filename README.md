# Interview Insight Extractor

Groups user-interview quotes by meaning and gives each group a short label, so recurring themes show up without keywords or manual tagging.

Give it quotes as a text file, a CSV or a transcript. It turns each quote into a sentence embedding, clusters quotes whose average cosine similarity is above a threshold, labels each cluster with its most distinctive words, and prints or exports the result.

## Features

- **Themes by meaning:** sentence embeddings plus average-linkage clustering; the result does not depend on the order of the quotes.
- **Theme labels:** 3 to 5 words per theme, picked with TF-IDF from the words that set that theme apart from the other quotes. No LLM involved.
- **Three input formats:** plain text (one quote per line), CSV (pick the column), and `Speaker: text` transcripts (optionally one speaker only).
- **Export:** Markdown for pasting into a doc, or CSV with one row per quote.
- **Evaluation script:** scores the clustering against a small hand-labelled set (see [Evaluation](#evaluation) for what that does and does not show).
- **Web demo:** a small Gradio page to paste quotes, move the threshold and download the Markdown.

## Example

Output of `python3 demo.py` on the bundled [sample_quotes.txt](sample_quotes.txt):

```
Analyzed 8 quotes. Found 2 patterns.

Pattern 1: free plan, pricing page (5 quotes, similarity 0.26)
  Most typical: "The onboarding was really confusing, I didn't know where to start"
  - The onboarding was really confusing, I didn't know where to start
  - I gave up during setup because there were too many steps
  - I couldn't figure out how to invite my teammates
  - The pricing page wasn't clear, I didn't know what I'd be charged
  - I had no idea what the free plan included

Pattern 2: finding old, incredibly fast (3 quotes, similarity 0.43)
  Most typical: "I love how quickly I can find past research"
  - The search feature is incredibly fast and accurate
  - I love how quickly I can find past research
  - Finding old interviews is so much easier now
```

This shows both sides of the default threshold of 0.20. The search quotes form a clean group. The other five are one broad "getting started is confusing" group that mixes onboarding and pricing, and its label names only the pricing part. Raising the threshold splits it:

```
$ python3 demo.py --threshold 0.35
Analyzed 8 quotes. Found 3 patterns.

Pattern 1: finding old, incredibly fast (3 quotes, similarity 0.43)
  ...
Pattern 2: confusing, gave, onboarding (2 quotes, similarity 0.42)
  - The onboarding was really confusing, I didn't know where to start
  - I gave up during setup because there were too many steps

Pattern 3: free plan, pricing page (2 quotes, similarity 0.38)
  - The pricing page wasn't clear, I didn't know what I'd be charged
  - I had no idea what the free plan included

Not grouped (1):
  - I couldn't figure out how to invite my teammates
```

The labels are keyword lists, not sentences. In a small group whose quotes share few words, many terms tie and the label is a rough hint.

## Run it

Requires Python 3.10 or newer. The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB).

```bash
pip install -r requirements.txt
python3 demo.py                              # bundled sample
python3 demo.py my_quotes.txt --threshold 0.4
python3 demo.py my_quotes.txt --json         # machine-readable output, includes the labels
```

`--threshold` is the minimum average similarity (0 to 1) for quotes to be grouped. Raise it for fewer, tighter groups; lower it for broader themes and fewer ungrouped quotes.

### Input formats

The format is taken from the file extension: `.csv` is CSV, `.transcript` is a transcript, anything else is plain text. `--format text|csv|transcript` overrides it, which is how you read a transcript saved as `.txt`.

| Format | What it expects | Options |
|---|---|---|
| text | One quote per line. Blank lines and lines starting with `#` are ignored. | |
| csv | A header row and one quote per row. | `--column` (default `quote`), `--speaker-column` with `--speaker` to keep one person's rows |
| transcript | Lines like `Speaker: text`. A line with no speaker continues the previous turn. Each turn is split into sentences; sentences under three words ("Yeah.") are dropped. | `--speaker` to keep one person's turns, which is how interviewer questions are left out |

```bash
python3 demo.py quotes.csv --column quote
python3 demo.py quotes.csv --column answer --speaker-column who --speaker Ana
python3 demo.py sample_transcript.txt --format transcript --speaker Participant
```

A missing column is reported with the columns that do exist:

```
$ python3 demo.py eval/labeled_quotes.csv --column text
Error: CSV has no column 'text'. Columns found: quote, theme, note
```

Output of the transcript command above on the bundled, made-up [sample_transcript.txt](sample_transcript.txt):

```
Analyzed 9 quotes. Found 3 patterns.

Pattern 1: old interview, past research (3 quotes, similarity 0.32)
  Most typical: "I can find an old interview in seconds."
  - Search is great.
  - I can find an old interview in seconds.
  - Looking up past research used to take me all morning.

Pattern 2: step came, confusing (2 quotes, similarity 0.30)
  Most typical: "Honestly the setup was confusing."
  - Honestly the setup was confusing.
  - I did not know which step came first.

Pattern 3: paid plan, pricing page (2 quotes, similarity 0.39)
  Most typical: "I could not work out what the paid plan would cost us."
  - I could not work out what the paid plan would cost us.
  - The pricing page never says what you get charged per seat.

Not grouped (2):
  - There were too many screens before I could do anything useful.
  - I would love a dark theme.
```

The interviewer's four questions are gone. One setup sentence ("too many screens") was still left out of the setup group.

### Export

`--output` writes the themes to a file as well as printing them. The extension picks the format.

```bash
python3 demo.py sample_transcript.txt --format transcript --speaker Participant --output themes.md
python3 demo.py eval/labeled_quotes.csv --threshold 0.2 --output themes.csv
```

`themes.md` from the first command starts like this:

```markdown
# Interview themes

9 quotes, 3 themes, 2 not grouped.

## 1. old interview, past research (3 quotes)

**Most typical:** "I can find an old interview in seconds."

- Search is great.
- I can find an old interview in seconds.
- Looking up past research used to take me all morning.
```

`themes.csv` from the second has one row per quote. Ungrouped quotes have an empty `theme_id` and `label`.

```csv
theme_id,label,quote,is_representative
1,"setup, checklist skipped","The first week was rough, I had no idea where to begin",false
1,"setup, checklist skipped",Setup took me three tries before anything worked,true
1,"setup, checklist skipped","There was no walkthrough, so I just clicked around until something happened",false
```

### Web demo

```bash
pip install -r requirements.txt -r requirements-demo.txt
python3 app.py          # then open http://127.0.0.1:7860
```

Paste quotes (one per line) or upload a `.txt`, `.csv` or `.transcript` file, set the threshold with the slider, and press "Find themes". The page shows each theme with its label and the ungrouped quotes, and offers the same Markdown as a download. It runs on your machine and listens on 127.0.0.1 only. Uploaded CSVs must use a `quote` column, and the page has no speaker filter; use the command line for those.

## How it works

1. **Embed:** each quote becomes a vector with [sentence-transformers](https://www.sbert.net/).
2. **Cluster:** average-linkage agglomerative clustering on cosine distance. Groups merge only while their average similarity stays above the threshold, and the result does not depend on the order of the quotes.
3. **Summarise:** each group reports its size, its mean pairwise similarity, and the quote closest to all the others.
4. **Label:** every content word, and every pair of adjacent content words, is scored by its TF-IDF weight inside the group minus its weight in all other quotes. The top terms are joined until the label has 3 to 5 words. An LLM would write nicer labels; this tool deliberately has none, so it runs offline and the labels are reproducible.

The clustering and labelling live in [insights.py](insights.py) and take any embedding matrix, so they can be reused with a different model. Reading and writing files is in [quote_io.py](quote_io.py).

## Evaluation

[eval/labeled_quotes.csv](eval/labeled_quotes.csv) holds 49 short quotes, each assigned to one of 11 themes: 8 themes of 4 to 7 quotes (onboarding, pricing, search, performance, collaboration, integrations, support, mobile) and 3 themes with a single quote. Three quotes are deliberately ambiguous, for example a slow-search complaint filed under performance; the `note` column marks them.

**Read this before the numbers.** The quotes are synthetic. They were written for this evaluation, they are not real interview data, and the same author who wrote them also chose their themes. They are short, clean and one idea each. So the table shows how the method behaves on tidy quotes whose themes were decided in advance; it is not a claim about real transcripts, which are longer, messier and have no agreed set of themes. Nothing was held out: there is one small set and every row below was computed on it.

Output of `python3 evaluate.py` (model `all-MiniLM-L6-v2`, sentence-transformers 6.3.0). Each ungrouped quote counts as its own cluster of one.

| Threshold | Clusters | Ungrouped | ARI | NMI | Homogeneity | Completeness |
|---|---|---|---|---|---|---|
| 0.15 | 7 | 2 | 0.47 | 0.75 | 0.69 | 0.81 |
| 0.20 (default) | 13 | 3 | 0.39 | 0.74 | 0.80 | 0.68 |
| 0.25 | 15 | 8 | 0.26 | 0.72 | 0.84 | 0.63 |
| 0.30 | 15 | 13 | 0.25 | 0.74 | 0.90 | 0.63 |
| 0.35 | 11 | 26 | 0.11 | 0.73 | 0.94 | 0.59 |
| 0.40 | 8 | 32 | 0.07 | 0.72 | 0.94 | 0.58 |
| 0.45 | 5 | 39 | 0.03 | 0.72 | 0.96 | 0.57 |
| 0.50 | 1 | 47 | -0.00 | 0.72 | 0.99 | 0.57 |
| 0.55 | 1 | 47 | -0.00 | 0.72 | 0.99 | 0.57 |
| 0.60 | 0 | 49 | 0.00 | 0.73 | 1.00 | 0.58 |

ARI (Adjusted Rand Index) is 1 for a perfect match and about 0 for a random one. Homogeneity is high when each cluster holds one theme; completeness is high when each theme lands in one cluster.

What the table says:

- **Agreement with the hand labels is modest.** The best ARI is 0.47, at the lowest threshold tried, and it is 0.39 at the default of 0.20.
- **The threshold trades coverage for purity.** At 0.35 the groups are nearly pure (homogeneity 0.94), but 26 of 49 quotes stay ungrouped and only groups of two or three form. At 0.20 only 3 quotes stay ungrouped, and groups mix more: one cluster of eight holds six onboarding quotes and two support quotes.
- **NMI is not informative on this set.** It is 0.73 when nothing is grouped at all (threshold 0.60), because 49 clusters of one are trivially pure. Use ARI and the ungrouped count to compare rows.
- **The default of 0.20 was chosen on this set.** It was 0.35 before this evaluation and was lowered after seeing the table, so the 0.20 row is not a held-out result and says little about other data. Try a few thresholds on your own quotes: lower for short quotes with broad themes, higher when you want tight groups.

## Tests

```bash
pip install pytest
pytest
```

28 tests cover the clustering, the labels, the three parsers and both export writers. They use hand-made vectors and need no model download. The web demo and `evaluate.py` have no automated tests; they were run by hand.

## Limits

- Evaluated only on the synthetic set above, not on real interview transcripts.
- One threshold for everything. The default of 0.20 groups most short quotes but can merge neighbouring themes (see the example at the top). Long or multi-topic quotes will group poorly.
- Transcript sentences are split on `.`, `!` and `?`, so abbreviations such as "e.g." split early, and a sentence that only makes sense next to its neighbour loses that context. Speaker names must match `Name:` at the start of a line; timestamps before the name are not handled.
- Labels are keyword lists from TF-IDF. They are rough for groups of two or three quotes.
- A quote is assigned to one theme only, even if it touches two.
- English only with the default model and stop-word list.

## Background

Started as a small demo of the embeddings-plus-cosine-similarity approach I used for pattern detection in REMIND, a Responsible Computing Club app at SJSU, applied to the interview-analysis problem that research tools like Great Question work on.

Rithika Sridharbabu, Computer Engineering, San José State University
