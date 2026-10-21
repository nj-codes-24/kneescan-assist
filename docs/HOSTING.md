# HOSTING: running the demo on a server (optional)

> **Status 2026-10-21: not deployed** (decision D-027): the repository is the deliverable, without a hosted demo. The container
> build below is finished and tested, and this guide is kept for anyone who wants to host it later. The weights are kept
> private; anyone else would train their own (README, *How to run*).

The demo is the same app as `app/clinic_app.py`, packed into a container and run on Azure Container Apps under an
**Azure for Students** subscription (free, no card). **Research prototype, not a medical device.**
What may be shown publicly is decision D-025, and the hosting choice is D-026, both in docs/decisions.md.

## What goes where

| Piece | Where | Visibility | Made by |
|---|---|---|---|
| Code, docs, results, figures (no scans) | GitHub `nj-codes-24/kneescan-assist` | public (after 22 Oct) | you, by hand |
| The 30 tested weight files + configs (1.4 GB) | HF model repo `nj-codes-24/kneescan-weights` | **private, always** | `scripts/export_weights.py` |
| The running app + one KneeMRI example | Azure Container Apps, app `kneescan` | public link (share it after 22 Oct) | `scripts/build_deploy.py` |

At start-up the container downloads the weights from the private repo with a read-only token stored as an Azure secret.
Visitors can use the model but can't download it. No MRNet scan is shown anywhere public (MRNet RUA clause 4).

## Tested before deploying (2026-10-21, D-026)
The container was built with Docker on this PC and run with Azure's limits (2 CPUs, 4 GB). The port opened after 20 s; peak
memory was 2.0 GiB; a 3-plane upload took 48.5 s, and the example about 5 s. On Azure, start-up also includes downloading 1.4 GB of
weights.

---

## Part A: the private weights on Hugging Face (free)

This is an institute PC that already has someone else's Hugging Face login (`manishkk1`). Keep yours separate: in **every new
PowerShell window**, run the first line before any `hf` command.

```powershell
$env:HF_HOME = "D:\nj_hf_home"
.venv\Scripts\hf.exe auth login                       # paste your Write token (kneescan-upload); it stays hidden
.venv\Scripts\hf.exe auth whoami                      # must say user=nj-codes-24
```

If you haven't done it yet, make the private repo and upload the weights (1.4 GB; if it stops, run the upload again to resume):
```powershell
.venv\Scripts\python.exe scripts\export_weights.py    # only if export\weights doesn't exist yet
.venv\Scripts\hf.exe repos create nj-codes-24/kneescan-weights --type model --private
.venv\Scripts\hf.exe upload nj-codes-24/kneescan-weights export\weights . --type model
```
Check on the website that `huggingface.co/nj-codes-24/kneescan-weights` shows **Private**.

Then make the **read-only token** for the app (website): Settings → Access Tokens → Create new token → **Fine-grained**, name
`kneescan-read`. Under "Repositories permissions" pick `nj-codes-24/kneescan-weights` and tick **Read access to contents of selected
repos**. Copy it somewhere safe; you'll paste it into Azure in step B6. Never paste a token into a chat or a file.

## Part B: the app on Azure (free with Azure for Students)

**B1. Sign up.** Open https://azure.microsoft.com/free/students and sign up with your **college e-mail**. No card is needed.

**B2. Find your allowed regions.** Student subscriptions may deploy to only a few regions (usually 5, different per person).
In https://portal.azure.com search for **Policy** → **Assignments** → open the one named like *Allowed resource deployment regions*
→ look at its parameter list. Pick one of those regions, ideally `centralindia` or `southindia` if it's listed. You need it in B5.

**B3. Make the upload file on this PC.**
```powershell
.venv\Scripts\python.exe -I scripts\build_deploy.py
```
This writes `export\kneescan-deploy.zip` (about 5 MB: code, Dockerfile, the KneeMRI example; no weights).

**B4. Open Cloud Shell** (a terminal in the browser, so nothing needs installing here). In the portal, click the `>_` icon in the
top bar → **Bash** → **No storage account required** → your *Azure for Students* subscription → Apply.
Then **Manage files → Upload** and choose `D:\knee_mri_tool\export\kneescan-deploy.zip`.

**B5. Build and deploy** (in Cloud Shell; replace `<REGION>` with the region from B2). The last command builds the container in
Azure and takes about 5–10 minutes.
```bash
az extension add --name containerapp --upgrade --allow-preview true
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
unzip -o kneescan-deploy.zip -d kneescan && cd kneescan
az containerapp up --name kneescan --resource-group kneescan-rg --location <REGION> --environment kneescan-env --source . --ingress external --target-port 7860
```
The app starts once without the token and stops at the weights download. That's expected; B6 fixes it.
If you see `RequestDisallowedByAzure`, the region isn't on your allowed list: run the last command again with another one from B2.

**B6. Add the token as a secret** (portal, so the token never sits in a command): open the container app **kneescan** →
Settings → **Secrets** → **Add** → Key `hf-token`, Value = your `kneescan-read` token → Add.

**B7. Set the size, the scaling and the token** (Cloud Shell):
```bash
az containerapp update --name kneescan --resource-group kneescan-rg --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 --set-env-vars HF_TOKEN=secretref:hf-token
```
- `--cpu 2 --memory 4Gi`: the size it was tested at.
- `--min-replicas 0`: it sleeps when nobody uses it, so it uses almost no credit.
- `--max-replicas 1`: Gradio needs each visitor to stay on one server.

**B8. Get the link and watch it start** (the first start downloads the weights; give it 2–5 minutes):
```bash
az containerapp show --name kneescan --resource-group kneescan-rg --query properties.configuration.ingress.fqdn -o tsv
az containerapp logs show --name kneescan --resource-group kneescan-rg --tail 40
```
Open `https://` + the address it prints, then click **Try a sample study**. It should say *ACL tear likely, 80%*, like
docs/images/app_results.png. Share the link from 22 Oct, and put it in README.md ("Live demo") and on your resume.

**B9. Optional: a budget alert.** Cost Management → Budgets → Add, for example $5 a month with an e-mail alert. With Azure for Students
nothing is charged to you anyway: when the $100 runs out, the resources stop unless you choose pay-as-you-go.

## Updating the demo later (for example from your laptop)
Change the code, run `build_deploy.py` again, upload the new zip in Cloud Shell, then:
```bash
rm -rf kneescan && unzip -o kneescan-deploy.zip -d kneescan && cd kneescan
az containerapp up --name kneescan --resource-group kneescan-rg --source .
```
On a machine without the KneeMRI data, `build_deploy.py` can't make the example file. Copy `samples\` from `export\deploy\`
first (keep a copy with your weights backup). The weights never need uploading again.

## What to expect
- **Sleeping:** after a few idle minutes it scales to zero. The next visit wakes it: Azure starts the container and downloads the
  weights, which can take a few minutes. After that the example opens in seconds.
- **Uploads:** a 3-plane study takes about 50 s on 2 vCPUs.
- **Free allowance:** about 25 hours awake per month at this size before the $100 credit is touched (Container Apps' monthly free
  grant of 180,000 vCPU-seconds and 360,000 GiB-seconds).
- **Privacy:** uploads are deleted right after reading, and patient tags are never read (kneescan/app/dicom_io.py). The landing page
  still asks for anonymised scans, because the file does travel to the server.

## Cleaning up this PC afterwards
Delete `D:\nj_hf_home`, which holds your Hugging Face login. The other user's login isn't touched. Then delete the `kneescan-upload`
token on the Hugging Face website. Keep `kneescan-read`: the demo uses it.
