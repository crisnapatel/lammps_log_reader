# LAMMPS Log Reader Streamlit App

Streamlit interface for exploring LAMMPS simulation logs with multi-run plotting, running averages, and export helpers to support rapid hydrogen storage analysis workflows.

## Prerequisites
Python 3.10+ recommended.

## Install (choose one)
```bash
# Option A: virtualenv (venv)
python -m venv .venv
source .venv/bin/activate  # mac/linux
.venv\\Scripts\\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

```bash
# Option B: Conda environment
# No conda yet? Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda create -n lmp-log-reader python=3.11 -y
# or conda env create -f environment.yml
conda activate lmp-log-reader
pip install -r requirements.txt
```

## Run
```bash
# From the repository root
streamlit run streamlit_lmp_log_reader/app.py
```

## Docs
Official setup guidance: https://docs.streamlit.io/ · https://docs.conda.io/en/latest/ · https://henriasv.github.io/lammps-logfile/