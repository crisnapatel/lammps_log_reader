
# lmp_loglite v2 (Streamlit) — with lammps-logfile integration

This version integrates the official **lammps-logfile** API to mirror its CLI features: per-run selection (`run_num`), multi-Y plotting, running averages, and column browsing. It also adds downsampling and image/HTML export.

## Install & Run
```bash
conda create -n lmpview2 python=3.10 -y
conda activate lmpview2
pip install -r requirements.txt
streamlit run app.py
```

## Key Features mapped from docs
- `File.get(name, run_num)` — read X/Y arrays for a specific run or `-1` for all runs concatenated.
- `File.get_keywords(run_num)` — show available columns.
- `running_mean(data, N)` — running average like the CLI `-a` flag.
- CLI parity: choose X (`-x`), multiple Y (`-y`), print columns (`-c`), export figure (`-o` equivalent via Download buttons).

Docs: https://henriasv.github.io/lammps-logfile/
