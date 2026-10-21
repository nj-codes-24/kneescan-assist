"""
views.py: the HTML of the KneeScan Assist page. A landing stage with one job (add a study), then a results page whose first lines
say the answer in plain words.

What it does
    Small pure functions that turn the Predictor's results into HTML strings, styled by app/theme.css:
      * `landing_html()` / `landing_note_html(hosted)`: the promise above the upload card, and the privacy and disclaimer line
        below it (worded for this computer, or for the hosted demo)
      * `brand_html()`: the wordmark on the results page
      * `results_html(results, volumes, seconds, wording)`:
          1. a one-sentence verdict ("Signs of an abnormal knee.") with the other findings in a subline, and for the example
             study a note saying where it comes from and what its real diagnosis is
          2. one row per finding: a plain word (Likely / Possible / Unlikely), the calibrated percentage, a meter with the screening line,
             and one sentence that explains the flag, model agreement and validated accuracy (no separate "how to read" section)
          3. "Where it looked": a segmented control showing one finding's three heatmaps at a time (CSS only)
          4. the limitations, the data credits and the disclaimer as one short paragraph
    Every text that comes from data is passed through `html.escape`, and images are PNG data URIs, so nothing loads from the internet.

Why words *and* numbers?
    "Likely / Possible / Unlikely" can be read at a glance. The percentage and the screening line are there for anyone who wants
    precision. The words come from fixed cut-offs in configs/app.yaml and don't change any model output (D-024).

Alternatives
    A larger product would use a front-end framework (React, Svelte) and a JSON API. For a one-page local demo that's more code than value.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from kneescan.app.predictor import FindingResult, Heatmap

FINDING_NAMES = {"abnormal": "Any abnormality", "acl": "ACL tear", "meniscus": "Meniscal tear"}
FINDING_HINTS = {"abnormal": "Anything outside a normal knee", "acl": "Anterior cruciate ligament",
                 "meniscus": "Medial or lateral meniscus"}
PLANE_NAMES = {"sagittal": "Sagittal · side", "coronal": "Coronal · front", "axial": "Axial · top"}
PLANE_ORDER = ("sagittal", "coronal", "axial")
VERDICT = {"likely": "Signs of an abnormal knee.", "possible": "Possibly an abnormal knee.",
           "unlikely": "No clear signs of an abnormal knee."}


@dataclass(frozen=True)
class Wording:
    """Cut-offs that turn a calibrated probability into a word (configs/app.yaml)."""
    likely_from: float = 0.70
    unlikely_below: float = 0.30


def band(probability: float, wording: Wording) -> str:
    """'likely', 'possible' or 'unlikely'."""
    if probability >= wording.likely_from:
        return "likely"
    return "unlikely" if probability < wording.unlikely_below else "possible"


def png_data_uri(image: np.ndarray) -> str:
    """RGB uint8 array -> 'data:image/png;base64,...' so the page needs no image server."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------- landing page
def landing_html() -> str:
    return """
<div class="ks-landing-top"><div class="ks-brand"><span class="ks-mark" aria-hidden="true"></span>KneeScan</div>
  <span class="ks-tag">Research prototype</span></div>
<div class="ks-landing-hero">
  <h1>Knee MRI,<br>explained in seconds.</h1>
  <p>Checks a knee MRI for an <b>ACL tear</b>, a <b>meniscal tear</b> and <b>any other abnormality</b>,
  and shows exactly where it looked.</p>
</div>"""


def landing_note_html(hosted: bool = False) -> str:
    """Where the analysis runs, what happens to the files, and the disclaimer."""
    where = ("Runs on a demo server. <b>Upload anonymised scans only.</b> Patient names and IDs are never read, and files are "
             "deleted right after reading." if hosted else
             "Runs on this computer. Patient names and IDs are never read, and files are deleted right after reading.")
    return f"""
<p class="ks-landing-note">{where}<br>
<b>Not a medical device:</b> for research and education only, not for diagnosis or treatment.</p>"""


def brand_html() -> str:
    return '<div class="ks-brand"><span class="ks-mark" aria-hidden="true"></span>KneeScan</div>'


# ---------------------------------------------------------------- results page
def verdict_html(results: list[FindingResult], wording: Wording) -> str:
    """Headline from the abnormality finding; the subline names the two specific findings in plain words."""
    by_task = {r.task: r for r in results}
    headline = VERDICT[band(by_task["abnormal"].probability, wording)]
    parts = [f'<span class="ks-dot ks-dot-{band(by_task[t].probability, wording)}"></span>'
             f'{html.escape(FINDING_NAMES[t])} {band(by_task[t].probability, wording)}' for t in ("acl", "meniscus") if t in by_task]
    return f'<section class="ks-verdict"><h1>{headline}</h1><p>{" <span class=ks-sep>·</span> ".join(parts)}</p></section>'


def note_html(note: str) -> str:
    """A short line under the verdict, e.g. where the example study comes from. Empty for the user's own upload."""
    return f'<p class="ks-note">{html.escape(note)}</p>' if note else ""


def series_html(volumes: dict[str, np.ndarray], seconds: float) -> str:
    chips = " · ".join(f"{PLANE_NAMES[p].split(' · ')[0]} {volumes[p].shape[0]} slices" for p in PLANE_ORDER if p in volumes)
    return f'<p class="ks-series">{html.escape(chips)} · analysed in {seconds:.1f} s</p>'


def flagged(result: FindingResult) -> bool:
    """The screening flag: most of the 5 models are above the screening threshold."""
    return result.votes_positive * 2 > len(result.fold_probabilities)


def flag_sentence(result: FindingResult, key: str) -> str:
    """What the screening line means for this row, worded so it never contradicts the band word."""
    line = round(100 * result.threshold)
    if not flagged(result):
        return f"Below the screening line ({line}%)."
    if key == "unlikely":
        return f"Unlikely, but above the cautious screening line ({line}%) that is set to catch 9 in 10 cases."
    return f"Above the screening line ({line}%), so worth a closer look."


def explain_line(result: FindingResult, wording: Wording = Wording()) -> str:
    """The 'how to read' guidance, built into each row as one sentence."""
    n = len(result.fold_probabilities)
    agreeing = max(result.votes_positive, n - result.votes_positive)
    flag = flag_sentence(result, band(result.probability, wording))
    return (f"{flag} {agreeing} of {n} models agree. Validated accuracy (AUC) {html.escape(result.cv_auc.split(' ')[0])}, "
            f"where 1.0 is perfect and 0.5 is chance.")


def finding_row(result: FindingResult, wording: Wording) -> str:
    key = band(result.probability, wording)
    percent, line = round(100 * result.probability), round(100 * result.threshold)
    return f"""
<article class="ks-row ks-band-{key}">
  <div class="ks-row-name"><h3>{html.escape(FINDING_NAMES[result.task])}</h3><p>{html.escape(FINDING_HINTS[result.task])}</p></div>
  <div class="ks-row-main">
    <div class="ks-row-word"><span class="ks-word">{key.capitalize()}</span><span class="ks-pct">{percent}%</span></div>
    <div class="ks-meter" role="img" aria-label="probability {percent} percent, screening line {line} percent">
      <div class="ks-meter-fill" style="width:{percent}%"></div><div class="ks-meter-mark" style="left:{line}%"></div>
    </div>
    <p class="ks-row-explain">{explain_line(result, wording)}</p>
  </div>
</article>"""


def heatmap_tile(heatmap: Heatmap) -> str:
    return f"""
<figure class="ks-tile">
  <img src="{png_data_uri(heatmap.image)}" alt="{html.escape(heatmap.caption)}">
  <figcaption><b>{html.escape(PLANE_NAMES[heatmap.plane])}</b>
    <span>slice {heatmap.slice_index + 1} of {heatmap.n_slices} · {round(100 * heatmap.attention)}% of the weight</span></figcaption>
</figure>"""


def heatmap_section(result: FindingResult) -> str:
    tiles = "".join(heatmap_tile(h) for h in sorted(result.heatmaps, key=lambda h: PLANE_ORDER.index(h.plane)))
    return f'<section class="ks-where ks-p-{result.task}"><div class="ks-tiles">{tiles}</div></section>'


def where_it_looked(results: list[FindingResult]) -> str:
    """A segmented control (one pill per finding) that shows one heatmap strip at a time. CSS only: radio buttons + labels."""
    shown = [r for r in results if r.heatmaps]
    if not shown:
        return ""
    radios = "".join(f'<input type="radio" name="ks-where" id="ks-t-{r.task}" class="ks-radio"{" checked" if i == 0 else ""}>'
                     for i, r in enumerate(shown))
    labels = "".join(f'<label for="ks-t-{r.task}" class="ks-seg-{r.task}">{html.escape(FINDING_NAMES[r.task])}</label>'
                     for r in shown)
    sections = "".join(heatmap_section(r) for r in shown)
    return f"""
<section class="ks-evidence">
  <h2>Where it looked</h2>
  <p class="ks-evidence-text">For each view, the slice the model weighted most. Colour marks where the evidence came from:
  where it looked, not proof of a finding.</p>
  {radios}<div class="ks-seg">{labels}</div>
  <div class="ks-panels">{sections}</div>
</section>"""


def limits_html() -> str:
    return """
<p class="ks-limits"><b>Know the limits.</b> Trained on 1,130 knee MRIs from one US hospital (GE scanners, MRNet). On another hospital's
scanner its ACL accuracy fell to an AUC of about 0.70, and it may fail silently on unusual sequences, scanners or children. Heatmaps are
coarse (8 × 8 blocks). Training data: MRNet, provided by the Stanford Center for Artificial Intelligence in Medicine and Imaging
(AIMI). <b>Research prototype, not a medical device:</b> not for diagnosis or treatment.</p>"""


def results_html(results: list[FindingResult], volumes: dict[str, np.ndarray], seconds: float,
                 wording: Wording = Wording(), note: str = "") -> str:
    rows = "".join(finding_row(r, wording) for r in results)
    return f"""
<div class="ks-results">
  {verdict_html(results, wording)}
  {note_html(note)}
  {series_html(volumes, seconds)}
  <div class="ks-rows">{rows}</div>
  {where_it_looked(results)}
  {limits_html()}
</div>"""
