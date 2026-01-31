
## Project Structure

- data/raw: original datasets (do not modify)
- data/interim: intermediate cleaned datasets
- data/processed: final feature tables for modeling
- src: reusable modules (pde, tte, sensitivity, policy)
- scripts: reproducible pipeline entry points
- outputs: figures/tables for the paper

## Quick Start

1. Put raw files into `data/raw/battery_usage/`
2. Run:
   - `python scripts/00_make_processed.py`
   - `python scripts/01_fit_pde_params.py`
   - `python scripts/02_predict_tte.py`
   - `python scripts/03_sensitivity.py`
   - `python scripts/04_recommendations.py`
