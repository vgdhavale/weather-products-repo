# Automated Weather Products: 3× IMD GFS + Skew‑T

This repository automatically generates:

- Three IMD GFS 850 hPa forecast animations:
  - `imd_gfs_downloads/national/imd_gfs_national_850hPa.gif`
  - `imd_gfs_downloads/maharashtra/imd_gfs_maharashtra_850hPa.gif`
  - `imd_gfs_downloads/regional/imd_gfs_regional_850hPa.gif`
- Skew‑T sounding plots for selected Indian stations:
  - `sounding_plots/SkewT_XXXXX.png`

## How it works

- A GitHub Actions workflow runs daily (00:30 UTC).
- It installs Python dependencies from `requirements.txt`.
- It runs `weather_products.py`, which:
  - Downloads three GFS products (national, Maharashtra, regional).
  - Creates three animated GIFs.
  - Fetches upper‑air soundings from Wyoming.
  - Creates Skew‑T plots with CAPE, CIN, LCL, LFC, EL, CCL, and shear.
- The workflow commits and pushes the generated images back to the repo.

## Sharable links (public repo)

After the workflow runs, you can share:

- National animation (raw):
  - `https://raw.githubusercontent.com/<USER>/<REPO>/main/imd_gfs_downloads/national/imd_gfs_national_850hPa.gif`
- Maharashtra animation (raw):
  - `https://raw.githubusercontent.com/<USER>/<REPO>/main/imd_gfs_downloads/maharashtra/imd_gfs_maharashtra_850hPa.gif`
- Regional animation (raw):
  - `https://raw.githubusercontent.com/<USER>/<REPO>/main/imd_gfs_downloads/regional/imd_gfs_regional_850hPa.gif`
- Skew‑T plots (raw):
  - `https://raw.githubusercontent.com/<USER>/<REPO>/main/sounding_plots/SkewT_43003.png`
  - (replace `43003` with other station numbers)

Replace `<USER>` and `<REPO>` with your GitHub username and repository name.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python weather_products.py
```

Outputs will appear in:
- `imd_gfs_downloads/national/`
- `imd_gfs_downloads/maharashtra/`
- `imd_gfs_downloads/regional/`
- `sounding_plots/`
