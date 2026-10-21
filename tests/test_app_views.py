"""
test_app_views.py: the page tells the right story from the results. Plain words from the probability bands, a verdict sentence,
the built-in reading guide, the right flag, ordered heatmaps, no web links, escaped data text, and the disclaimer on both screens.
"""

from __future__ import annotations

import numpy as np

from app.views import (Wording, band, explain_line, finding_row, flagged, landing_html, landing_note_html, limits_html,
                       png_data_uri, results_html, verdict_html)
from kneescan.app.predictor import FindingResult, Heatmap

WORDING = Wording(likely_from=0.70, unlikely_below=0.30)


def fake_result(task: str, probability: float, votes: int) -> FindingResult:
    heatmaps = [Heatmap(p, 4, 30, 0.31, np.zeros((8, 8, 3), np.uint8)) for p in ("axial", "sagittal", "coronal")]
    return FindingResult(task=task, probability=probability, fold_probabilities=[0.5] * 5, threshold=0.25, votes_positive=votes,
                         confidence="High", cv_auc="0.852 (0.823–0.880)", heatmaps=heatmaps)


def test_bands_turn_probability_into_words() -> None:
    assert [band(p, WORDING) for p in (0.05, 0.29, 0.30, 0.69, 0.70, 0.99)] == \
           ["unlikely", "unlikely", "possible", "possible", "likely", "likely"]


def test_verdict_leads_with_abnormality_then_names_the_tears() -> None:
    results = [fake_result("abnormal", 0.99, 5), fake_result("acl", 0.16, 5), fake_result("meniscus", 0.62, 5)]
    verdict = verdict_html(results, WORDING)
    assert "Signs of an abnormal knee." in verdict and "ACL tear unlikely" in verdict and "Meniscal tear possible" in verdict


def test_row_shows_word_number_and_built_in_reading_guide() -> None:
    row = finding_row(fake_result("acl", 0.16, 5), WORDING)
    assert "Unlikely" in row and "16%" in row and "ks-band-unlikely" in row
    line = explain_line(fake_result("acl", 0.16, 5), WORDING)
    assert "Unlikely, but above the cautious screening line (25%)" in line          # never "worth a closer look" next to "Unlikely"
    assert "5 of 5 models agree" in line and "0.852" in line
    assert "worth a closer look" in explain_line(fake_result("acl", 0.62, 5), WORDING)
    assert "Below the screening line" in explain_line(fake_result("acl", 0.10, 1), WORDING)


def test_flag_follows_the_majority_of_models() -> None:
    assert flagged(fake_result("acl", 0.3, votes=3)) and not flagged(fake_result("acl", 0.3, votes=2))


def test_results_page_structure() -> None:
    results = [fake_result(t, 0.5, 4) for t in ("abnormal", "acl", "meniscus")]
    volumes = {p: np.zeros((30, 8, 8), np.uint8) for p in ("sagittal", "coronal", "axial")}
    page = results_html(results, volumes, 9.6, WORDING)
    assert page.count('class="ks-row ') == 3 and page.count("<figure") == 9
    assert page.count('type="radio"') == 3 and page.count(" checked") == 1          # one heatmap strip shown at a time
    strip = page.split('class="ks-tiles"')[1]
    assert strip.index("Sagittal") < strip.index("Coronal") < strip.index("Axial")   # side, front, top
    assert "http" not in page and "data:image/png;base64," in page                  # nothing loads from the web
    assert "<details" not in page and "Know the limits" in page                     # limits are in the page, not a dropdown


def test_text_from_data_is_escaped() -> None:
    result = fake_result("acl", 0.5, 4)
    result.cv_auc = "<script>alert(1)</script> x"
    assert "<script>" not in finding_row(result, WORDING)


def test_disclaimer_on_both_screens() -> None:
    assert "Not a medical device" in landing_note_html() and "not a medical device" in limits_html()
    assert "ACL tear" in landing_html()


def test_hosted_page_asks_for_anonymised_scans_and_credits_the_data() -> None:
    assert "Runs on this computer" in landing_note_html() and "anonymised" not in landing_note_html()
    hosted = landing_note_html(hosted=True)
    assert "anonymised scans only" in hosted and "this computer" not in hosted and "Not a medical device" in hosted
    assert "Stanford Center for Artificial Intelligence in Medicine and Imaging" in limits_html()


def test_example_note_is_shown_escaped_and_only_when_given() -> None:
    results = [fake_result(t, 0.5, 4) for t in ("abnormal", "acl", "meniscus")]
    volumes = {"sagittal": np.zeros((30, 8, 8), np.uint8)}
    assert 'class="ks-note"' not in results_html(results, volumes, 1.0, WORDING)
    page = results_html(results, volumes, 1.0, WORDING, note="KneeMRI <exam> 510057")
    assert 'class="ks-note"' in page and "KneeMRI &lt;exam&gt; 510057" in page


def test_png_data_uri() -> None:
    assert png_data_uri(np.full((4, 4, 3), 200, np.uint8)).startswith("data:image/png;base64,")
