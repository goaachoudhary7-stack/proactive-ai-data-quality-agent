# Two-minute demonstration

Use this script with a recruiter or hiring manager. Stay on the finished pages. The goal is to show how an analyst separates a measured data problem from a hypothesis, then show that an uploaded file is profiled on its own.

## Before you start

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/dashboard/app.py
```

Have `data/clean/campaigns.csv` ready to upload. It is 25 synthetic rows, so it opens quickly. Do not upload `interactions.csv` during this timed pass.

The sidebar shows **Overview** and **Analyze a file**. Findings, Investigation, and the paused SQL studio are not destinations.

No API key is required. Do not click **Explain with AI**.

## 0:00–0:25 — Banking overview

The app opens on **Banking demonstration** and **Overview**.

Say:

> This is a fictional commercial bank, built so I can show a warehouse an analyst would actually have to defend. Seven deterministic checks failed. The page counts checks. It does not roll them into a quality score.

Point at the failed-check count and the synthetic-data line. Glance at **Checks by severity**. Mention that one check is critical and the rest are high, using the labels already on the page.

## 0:25–0:55 — One confirmed measurement

Stay on **Overview**. Point at the critical card, then at **DQ004** in the high-severity table.

Say:

> The confirmed measurement for campaign attribution is 6,736 of 39,401 campaign-driven interactions with no campaign id, about 17 percent, against an 8 percent line. Email opens, email clicks, and event attendance are in scope. Organic and relationship-manager activity is not. I would not use this warehouse to rank campaign response until that gap is understood. The check does not say why the id is missing, and it does not estimate a financial loss.

Point at the critical card and say it is the join break: interaction ids that are filled in and still absent from the customer master.

## 0:55–1:20 — Confirmed measurement versus hypothesis

Stay on **Overview**. The numbers on this page are the check results.

Say:

> These figures are confirmed measurements. Saved investigation narratives exist only for DQ001 and DQ004, and those narratives are hypotheses. I do not treat an impact paragraph in those files as a measured loss. The other five checks have no saved narrative.

Do not open a model call. The narrative files are `outputs/ai_investigations/DQ001_investigation.json` and `outputs/ai_investigations/DQ004_investigation.json`.

## 1:20–2:00 — An uploaded file is a different source

Open **Analyze a file**. The five steps are a horizontal selector on that page. Upload `data/clean/campaigns.csv`.

After it loads, the source selector moves to **My uploaded files**. Open **Overview**.

Say:

> This page is the file I just uploaded, not the seven banking checks. It shows row and column counts, missing values, and duplicate counts from this file. Candidate keys and relationships, if any, are labeled as suggestions. There is no severity and no quality score, because this profiler does not invent them.

Point at **Uploaded data** and name the two versions: **Original uploaded data** and **Current cleaned working copy**. They match until a clean is approved.

Say:

> Cleaning is preview, then approve. Undo and reset put the working copy back. The original file in the session stays as uploaded. I am not cleaning this small campaign list in the live two minutes. On a file that needs it, the Clean step shows before-and-after counts for rows, duplicates, and missing cells, and the audit lists only the actions I approved.

Switch **Active data source** back to **Banking demonstration** for one second so it is obvious the case study is still there.

Stop. Offer to open the Clean step or a downloaded workbook if they want to go past two minutes.

## If they ask to go longer

- **Overview** on **My uploaded files** lists measured issues for the selected version. It still has no severity.
- **3. Insights** shows candidate-key suggestions, relationship suggestions, and offline recommendations. **Explain with AI** stays off unless you approve the summary and a key is configured.
- **4. Clean** on a file with a blank or a duplicate: Preview, then approve, then show the before-and-after table. Undo once.
- **5. Export** shows separate downloads for the original profile, the recommendation workbook, the cleaned file, and the audit.
- Saved banking reports on disk: `outputs/dq_findings.json`, `outputs/ai_investigations/DQ001_investigation.json`, and `outputs/ai_investigations/DQ004_investigation.json`.

## What not to do in this pass

- Do not look for Findings, Investigation, or AI SQL Studio in the sidebar. Measured upload issues are on Overview. Recommendations and the optional explanation control are on **3. Insights**.
- Do not click **Explain with AI** or run `run_investigation` without `--dry-run`.
- Do not regenerate `data/` or `outputs/`. That rewrites the figures in the README.
- Do not describe a root cause or a financial impact that the check did not measure.
- If the file picker looks empty after you leave Upload, use the dataset inventory. The file is still in the session. Choosing it again from disk replaces that session.
