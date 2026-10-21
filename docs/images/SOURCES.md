# README figures: what each one shows

The repository audit (`scripts/audit_repo.py`) only accepts a figure in `docs/images/` if it's listed here. **No figure shows a
scan.** MRNet images may not be published (MRNet RUA clause 4), and KneeMRI images with our heatmaps would be adapted material
(CC BY-NC-ND 4.0). Decision D-025. All three are drawn by `scripts/make_readme_images.py`.

| File | Shows | Made from |
|---|---|---|
| `final_results.svg` | AUC bars with 95% CIs for B0, the Reader and the hybrid on the one-time final test (MRNet `valid`, macro AUC) and on KneeMRI (ACL AUC) | `results/final/final.json` |
| `app_results.png` | The demo's results page for the example study: verdict, source note, three finding rows and the limits paragraph. The "where it looked" section (the scan with its heatmap) is left out | `app/views.py` + `app/theme.css`, the final model's output for KneeMRI exam 510057 |
| `app_landing.png` | The hosted demo's landing page (dark mode): headline, upload card, privacy and disclaimer line | a running copy of the app with `configs/deploy.yaml` |
