"""Save what a script printed as a numbered, timestamped report.

Terminal output scrolls away, is easy to lose, and is hard to compare between
one run and the next. Every user-facing script in this scenario therefore writes
its console output to disk as well as to the screen, into a `results/` folder:

    results/run_003_20260821_143012_evaluate_model.txt   full console transcript
    results/run_003_20260821_143012_evaluate_model.csv   one row per metric
    results/run_003_20260821_143012_evaluate_model.pdf   the transcript, paginated

The run number counts the reports already in the folder, so runs stay in order
even if two are started in the same minute. Nothing is ever overwritten.

The CSV is the file to open in Excel or pandas when comparing runs, because every
score is one row: run number, timestamp, section, model, metric, value. The PDF
is the file to attach to an email or a report appendix.

Usage:

    from run_report import RunReport

    with RunReport("scenario-1", "evaluate_model") as report:
        print("anything at all")
        report.metric("classification", "RandomForest", "accuracy", 0.996)

Nothing else in the script has to change. Every `print` is captured.

This module has no dependencies beyond the standard library, except that the PDF
is written with matplotlib when it is installed. If matplotlib is missing, the
text and CSV reports are still written and the script carries on.
"""

import csv
import datetime
import os
import re
import sys
import textwrap

RESULTS_DIR = "results"
PDF_LINE_WIDTH = 96      # characters per line before wrapping in the PDF
PDF_LINES_PER_PAGE = 60


class _Tee:
    """Write to the real console and to an in-memory transcript at the same time."""

    def __init__(self, stream, transcript):
        self.stream = stream
        self.transcript = transcript

    def write(self, text):
        self.stream.write(text)
        self.transcript.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


class RunReport:
    """Capture a script's output and write it out as .txt, .csv and .pdf."""

    def __init__(self, scenario, script, results_dir=RESULTS_DIR):
        self.scenario = scenario
        self.script = script
        self.results_dir = results_dir
        self.metrics = []
        self.transcript = []
        self.started = datetime.datetime.now()
        self.run_number = self._next_run_number()
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        self.stem = f"run_{self.run_number:03d}_{stamp}_{script}"
        self._saved_stdout = None

    # ------------------------------------------------------------------ setup
    def _next_run_number(self):
        """One higher than the highest run number already in the folder."""
        if not os.path.isdir(self.results_dir):
            return 1
        highest = 0
        for name in os.listdir(self.results_dir):
            found = re.match(r"run_(\d+)_", name)
            if found:
                highest = max(highest, int(found.group(1)))
        return highest + 1

    def __enter__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        self._saved_stdout = sys.stdout
        sys.stdout = _Tee(self._saved_stdout, self.transcript)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A script that stopped early still gets its report, with the reason in it.
        if exc_type is not None:
            print(f"\n[run ended early: {exc_type.__name__}: {exc_value}]")
        sys.stdout = self._saved_stdout
        self.write()
        return False

    # ----------------------------------------------------------------- metrics
    def metric(self, section, model, name, value):
        """Record one number so it lands in the CSV as its own row."""
        self.metrics.append({
            "run_number": self.run_number,
            "timestamp": self.started.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": self.scenario,
            "script": self.script,
            "section": section,
            "model": model,
            "metric": name,
            "value": value,
        })

    # ------------------------------------------------------------------ output
    def write(self):
        text_path = os.path.join(self.results_dir, self.stem + ".txt")
        body = "".join(self.transcript)
        header = (
            f"{self.scenario} - {self.script}\n"
            f"run {self.run_number}, started {self.started:%Y-%m-%d %H:%M:%S}\n"
            + "=" * 74 + "\n\n"
        )
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(header + body)

        written = [text_path]
        if self.metrics:
            written.append(self._write_csv())
        pdf_path = self._write_pdf(header + body)
        if pdf_path:
            written.append(pdf_path)

        print(f"\nSaved this run to {self.results_dir}/ as run {self.run_number}:")
        for path in written:
            print(f"  {path}")

    def _write_csv(self):
        path = os.path.join(self.results_dir, self.stem + ".csv")
        columns = ["run_number", "timestamp", "scenario", "script",
                   "section", "model", "metric", "value"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.metrics)
        return path

    def _write_pdf(self, text):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            return None

        lines = []
        for line in text.split("\n"):
            lines.extend(textwrap.wrap(line, PDF_LINE_WIDTH) or [""])

        path = os.path.join(self.results_dir, self.stem + ".pdf")
        pages = [lines[i:i + PDF_LINES_PER_PAGE]
                 for i in range(0, len(lines), PDF_LINES_PER_PAGE)] or [[""]]
        with PdfPages(path) as pdf:
            for number, page in enumerate(pages, start=1):
                figure = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
                figure.text(0.06, 0.965, "\n".join(page), family="monospace",
                            fontsize=8, va="top", linespacing=1.45)
                figure.text(0.94, 0.02, f"page {number} of {len(pages)}",
                            fontsize=7, ha="right", color="0.35")
                pdf.savefig(figure)
                plt.close(figure)
        return path
