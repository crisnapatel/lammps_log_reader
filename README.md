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
## Docs
Official setup references: https://docs.streamlit.io/ · https://docs.conda.io/en/latest/ · https://henriasv.github.io/lammps-logfile/
