# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research pipeline that estimates formal/informal employment for workers in the Guadalajara Metropolitan Area Origin–Destination (OD) survey by training scikit-learn classifiers on ENOE (Mexico's national labor survey), which has an informality label. The README documents the methodology, outputs, and how to apply the trained models downstream — read it for the substantive description.

## Running the pipeline

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`, Python ≥3.11). There is no test suite.

```bash
uv sync                      # create .venv with locked deps
uv run jupyter lab           # from the repo root (see cwd note below)
uv run python -c "import src"
```

The committed `outputs/models/*.joblib` are sklearn pickles and only load reliably with the scikit-learn version in `uv.lock`; if you upgrade scikit-learn, re-run notebooks 04–05 to regenerate them. Notebooks 03–05 set `mpl.rc("text", usetex=True)`, so a LaTeX install is required to render figures (or disable that line locally).

The pipeline is five Jupyter notebooks run in order; each reads the previous stage's parquet from `outputs/` and writes its own:

| Stage | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_generate_enoe_od_base_dataframes` | `data/enoe/*.csv`, `data/od/*.csv` | `enoe_workers`, `od_workers` |
| 2 | `02_harmonize_enoe_od` | stage 1 | `enoe_harmonized`, `od_harmonized` |
| 3 | `03_diagnose_enoe_od_dataframes` | stage 2 | figures only |
| 4 | `04_impute_economic_sector` | `od_harmonized` | `od_sector_imputed`, `models/economic_sector_hybrid_model.joblib` |
| 5 | `05_informality_model` | `enoe_harmonized`, `od_sector_imputed` | `od_informality_imputed`, `models/informality_hybrid_model.joblib` |

Notebooks compute `ROOT` (the repo root) from cwd and anchor every `src` import and `data/`/`outputs/` path to it, so they run correctly from either the repo root or `notebooks/` (nbconvert executes with cwd set to the notebook's directory). Keep new I/O paths anchored to `ROOT` too.

Headless execution of one stage:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/04_impute_economic_sector.ipynb --inplace
```

Raw survey CSVs in `data/` are latin1-encoded and not all are committed (`data/enoe/` is gitignored). Stages 4–5 do grid-searched CV over three model families and are slow; `outputs/` already holds the results of a full run, so you can start from any stage.

## Code structure

`src/` is a flat package; `src/__init__.py` re-exports everything so notebooks just `import src` and call `src.<function>`. **When adding a public function to a module, also add it to the `__init__.py` import list** or the notebooks won't see it. Notebooks use `%autoreload 2`, so edits to `src/` take effect without restarting the kernel.

One module per stage, named to match the notebook. The notebooks hold orchestration, printing, and plotting; `src/` holds the reusable logic. Keep it that way — new computation goes in `src/`, not in notebook cells.

### Key conventions that span files

- **Harmonized column names are Spanish and shared by both surveys**: `genero`, `ocupacion`, `edad_num`, `escolaridad`, `municipio`, `estado_civil`, `parentesco`, `tamano_viv_cat`, `sector`. Missing/unknown categories use the literal string `"no_especificado"` (`NO_ESPECIFICADO` in `harmonize_enoe_od_dataframes.py`); `identify_missing_category` in the model modules keys off it. Survey weights are `survey_weight` (ENOE, from `fac_tri`) and `expansion_factor` (OD, from `Ponderador`).
- **OD retains its raw Spanish question-text column names** (e.g. `"Ocupación:"`, `"Durante la semana pasada trabajó:"`, `"Centralidad"`), and the sector model uses some of them directly as features (`OD_SECTOR_FEATURES`). Don't rename them.
- **Four sector classes** are hard-coded identically as `SECTOR_CLASSES` in both `impute_economic_sector.py` and `informality_model.py`; the informality stage validates that `prob_sector_*` columns sum to 1 per row.
- **Hybrid "with/without education" models.** Both modelling stages train two specifications (`*_FEATURES` vs `*_ROBUST_FEATURES`, the latter dropping `escolaridad`) and dispatch per row on whether education is missing. The `.joblib` files are dicts with keys `model_with_education`, `model_without_education`, `features_with_education`, `features_without_education`, each model being an sklearn `Pipeline` whose final step is named `"classifier"`.
- **Informality is probabilistic and marginalizes over sector.** `predict_od_informality` computes `P(informal | sector=s)` for every `s`, multiplies by `prob_sector_s`, and sums to get `prob_informal`. `informal_predicted` (0.5 threshold) is a convenience; `prob_informal` is the primary output.
- **Cross-validation is grouped by household** (`StratifiedGroupKFold`) with weights normalized via `normalize_sample_weights`; the ENOE household key is `ENOE_HOUSEHOLD_COLUMNS`. Preserve this when changing model tuning — random row-level splits would leak.
- Stage 1 filters ENOE to Jalisco (`ent == 14`) and employed persons; stage 3 further restricts both surveys to `TARGET_MUNICIPALITIES` (the metropolitan area) for comparisons.
