# Interview Insight Extractor

A semantic pattern detection tool for user interview transcripts — built as a demo for Great Question.

## What it does

Takes raw quotes from user interviews and automatically groups them by meaning using sentence embeddings and cosine similarity. No keywords, no manual tagging — just semantic understanding.

**Example output:**
## Why this matters for Great Question

Great Question analyzes thousands of interview hours to surface insights. This is the same core problem — finding what users are actually saying across many responses — just at a smaller scale.

I built the same embeddings + cosine similarity pipeline for REMIND, an AI web app deployed to SJSU students, where it detects recurring patterns in user decision logs. This demo applies that same logic to interview data.

## How to run

```bash
pip install sentence-transformers scikit-learn numpy
python3 demo.py
```

## Built by

Rithika Sridharbabu — Freshman CE @ SJSU  
github.com/rithikasridharbabu-2007
