# LAMMPS Log Reader Streamlit App

Streamlit interface for inspecting LAMMPS simulation logs, plotting multi-run metrics, and exporting trends to accelerate hydrogen storage studies.

## Prerequisites
Python 3.10+ recommended.

## Install (choose one)
```bash
# Option A: virtualenv (venv)
python -m venv .venv
source .venv/bin/activate  # mac/linux
.venv\\Scripts\\Activate.ps1  # Windows PowerShell
pip install -r streamlit_lmp_log_reader/requirements.txt
```

```bash
# Option B: Conda environment
# No conda yet? Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda create -n lmp-log-reader python=3.11 -y
# or conda env create -f environment.yml
conda activate lmp-log-reader
pip install -r streamlit_lmp_log_reader/requirements.txt
```

## Run
```bash
streamlit run streamlit_lmp_log_reader/app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub with `streamlit_lmp_log_reader/app.py` as the entry point.
2. In Streamlit Community Cloud, select the repo, set the main file to `streamlit_lmp_log_reader/app.py`, and deploy.

## Docs
Official setup references: https://docs.streamlit.io/ · https://docs.conda.io/en/latest/ · https://henriasv.github.io/lammps-logfile/

## How this differs from henriasv/lammps-logfile
- Built as an interactive Streamlit dashboard that accepts multiple uploads at once, overlays runs, and streams live log tails instead of providing only a programmatic parser.
- Adds hydrogen storage-focused tooling such as quick presets, custom formulas, running averages, and downsampling (Every Nth or LTTB) for rapid exploratory plots.
- Surfaces warnings, errors, and timing breakdowns in dedicated tabs with one-click CSV/PNG/HTML exports, complementing the lower-level parsing utilities in `lammps-logfile`.
