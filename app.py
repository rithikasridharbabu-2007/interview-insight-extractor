"""Web demo: python3 app.py, then open http://127.0.0.1:7860

Needs the extra packages in requirements-demo.txt.
"""
import tempfile
from pathlib import Path

import gradio as gr

from insights import DEFAULT_THRESHOLD, embed, find_patterns
from quote_io import load_quotes, parse_text, to_markdown

SAMPLE = Path(__file__).with_name("sample_quotes.txt").read_text(encoding="utf-8")


def analyze(text: str, file: str | None, threshold: float):
    """Return (Markdown report, path of the report file to download)."""
    try:
        quotes = load_quotes(file) if file else parse_text(text or "")
    except ValueError as error:
        raise gr.Error(str(error))
    if len(quotes) < 2:
        raise gr.Error("Paste at least two quotes, one per line, or upload a file.")
    patterns, unmatched = find_patterns(quotes, embed(quotes), threshold)
    report = to_markdown(patterns, unmatched)
    path = Path(tempfile.mkdtemp()) / "interview_themes.md"
    path.write_text(report, encoding="utf-8")
    return report, str(path)


with gr.Blocks(title="Interview Insight Extractor") as demo:
    gr.Markdown("# Interview Insight Extractor\nGroups interview quotes by meaning and names each theme.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Quotes, one per line", lines=12, value=SAMPLE)
            file = gr.File(label="Or upload a .txt, .csv (column `quote`) or .transcript file; it replaces the box above",
                           file_types=[".txt", ".csv", ".transcript"], type="filepath")
            threshold = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.05,
                                  label="Threshold: higher means fewer, tighter themes")
            button = gr.Button("Find themes", variant="primary")
        with gr.Column():
            report = gr.Markdown(label="Themes")
            download = gr.File(label="Download as Markdown")
    button.click(analyze, [text, file, threshold], [report, download], api_name="analyze")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
