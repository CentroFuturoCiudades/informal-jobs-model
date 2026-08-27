# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research pipeline that estimates formal/informal employment for workers in the Guadalajara Metropolitan Area Origin–Destination (OD) survey by training scikit-learn classifiers on ENOE (Mexico's national labor survey), which has an informality label. The README documents the methodology, outputs, and how to apply the trained models downstream — read it for the substantive description.

## Running the pipeline

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`, Python ≥3.13). Survey data comes from two of the user's packages pinned as git sources: `mxcensus` (ENOE) and `eodgdl` (OD survey). There is no test suite.

```bash
uv sync                      # create .venv with locked deps
uv run jupyter lab           # from the repo root (see cwd note below)
uv run python -c "import src"
```

The committed `outputs/models/*.joblib` are sklearn pickles and only load reliably with the scikit-learn version in `uv.lock`; if you upgrade scikit-learn, re-run notebooks 04–05 to regenerate them. Notebooks 03–05 set `mpl.rc("text", usetex=True)`, so a LaTeX install is required to render figures (or disable that line locally).

The pipeline is five Jupyter notebooks run in order; each reads the previous stage's parquet from `outputs/` and writes its own:

| Stage | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_generate_enoe_od_base_dataframes` | `mxcensus` (ENOE `2023t1`, Jalisco), `eodgdl` (OD `hab` ⋈ `viv`) | `enoe_workers`, `od_workers` |
| 2 | `02_harmonize_enoe_od` | stage 1 | `enoe_harmonized`, `od_harmonized` |
| 3 | `03_diagnose_enoe_od_dataframes` | stage 2 | figures only |
| 4 | `04_impute_economic_sector` | `od_harmonized` | `od_sector_imputed`, `models/economic_sector_hybrid_model.joblib` |
| 5 | `05_informality_model` | `enoe_harmonized`, `od_sector_imputed` | `od_informality_imputed`, `models/informality_hybrid_model.joblib` |

Notebooks compute `ROOT` (the repo root) from cwd and anchor every `src` import and `outputs/` path to it, so they run correctly from either the repo root or `notebooks/` (nbconvert executes with cwd set to the notebook's directory). Keep new I/O paths anchored to `ROOT` too.

Headless execution of one stage:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/04_impute_economic_sector.ipynb --inplace
```

There is no `data/` directory: the packages download and cache the raw tables on first use (`~/Library/Caches/mxcensus`, `~/Library/Caches/eodgdl`; override with `MXCENSUS_CACHE_DIR`, `EODGDL_CACHE_DIR`, or point `EODGDL_DATA_DIR` at local IMEPLAN files). Stages 4–5 do grid-searched CV over three model families and are slow; `outputs/` already holds the results of a full run, so you can start from any stage.

`src/compare_outputs.py` compares a run against a reference copy of `outputs/` (row counts, key overlap, weighted totals, harmonized distributions, informality rates): `uv run python -m src.compare_outputs outputs_baseline outputs [--stage 1|2|3|all]`. `outputs_baseline/` (gitignored) holds the pre-migration run of 2026-08-26 and `outputs_ref_849568a/` the post-migration run used as reference for the review-remediation work (`docs/review_remediation_plan.md`); notebook 01 prints the stage-1 comparison automatically when it exists. `generate_enoe_dataframe(employment_filter="clase2"|"p1")` selects the ENOE employment definition: `clase2` (default) = INEGI employed (`clase2 == 1`) on the analytical universe `r_def == 0`, `c_res in {1,3}`, ages 12–98 (floor lowered from INEGI's 15 to match the OD, which records working 12–14 year olds); `p1` = worked ≥1h last week, the original definition, a strict subset (6,793 vs 6,973 workers) that reproduces `outputs_baseline/` exactly.

## Code structure

`src/` is a flat package; `src/__init__.py` re-exports everything so notebooks just `import src` and call `src.<function>`. **When adding a public function to a module, also add it to the `__init__.py` import list** or the notebooks won't see it. Notebooks use `%autoreload 2`, so edits to `src/` take effect without restarting the kernel.

One module per stage, named to match the notebook. The notebooks hold orchestration, printing, and plotting; `src/` holds the reusable logic. Keep it that way — new computation goes in `src/`, not in notebook cells.

### Key conventions that span files

- **Harmonized column names are Spanish and shared by both surveys**: `genero`, `ocupacion`, `edad_num`, `escolaridad`, `municipio`, `estado_civil`, `parentesco`, `tamano_viv_cat`, `sector`. Missing/unknown categories use the literal string `"no_especificado"` (`NO_ESPECIFICADO` in `harmonize_enoe_od_dataframes.py`); `identify_missing_category` in the model modules keys off it. Survey weights are `survey_weight` (ENOE, from `fac_tri`) and `expansion_factor` (OD, from the person-level `ponderador`).
- **OD uses `eodgdl`'s snake_case column names.** Raw OD columns whose names collide with the harmonized attributes (`ocupacion`, `escolaridad`, `municipio`, `estado_civil`, `parentesco`) get a `_raw` suffix at stage 1 (`OD_RAW_COLUMN_RENAMES` in `generate_enoe_od_dataframes.py`); the unsuffixed name is always the harmonized one. The sector model uses raw `ocupacion_raw`, `trabajo_semana_pasada`, `centralidad` directly as features (`OD_SECTOR_FEATURES`), and `folio_vivienda` is the CV group key. ENOE arrives from `mxcensus` as string codes and is cast to `Int64` in stage 1 because the stage-2 mapping dicts key on integers.
- **Stage 2 fails loudly on unmapped categories** (`assert_mapping_covers`): a new ENOE code or eodgdl label must be added to the mapping dict or to the function's `allowed_unmapped` set; stage 1 likewise asserts the eodgdl columns it renames exist and that the ENOE dwelling key resolved by mxcensus matches `ENOE_DWELLING_KEYS`.
- **Shared constants and helpers live in `src/common.py`** (`SECTOR_CLASSES`, `NO_ESPECIFICADO`, `AGE_LABELS`, `identify_missing_category`, weight/probability normalizers, `prepare_model_features`). `build_category_levels()` declares every level each categorical model feature may take in either survey; both stages build their `OneHotEncoder` with these explicit categories and `handle_unknown="error"`, so a new level must be added there (a level with no training support is *marginalized*: `predict_proba_marginalizing` averages the prediction over the supported levels weighted by their training share, stored on each refitted pipeline as `training_level_shares_`; the `*_marginalized_features` output columns record which rows/features this touched — an all-zero one-hot block would otherwise be routed like the residual training level). The informality stage validates that `prob_sector_*` columns sum to 1 per row.
- **Hybrid "with/without education" models.** Both modelling stages train two specifications (`*_FEATURES` vs `*_ROBUST_FEATURES`, the latter dropping `escolaridad`) and dispatch per row on whether education is missing. The `.joblib` files are dicts with keys `model_with_education`, `model_without_education`, `features_with_education`, `features_without_education`, each model being an sklearn `Pipeline` whose final step is named `"classifier"`.
- **Informality is probabilistic and marginalizes over sector.** `predict_od_informality` computes `P(informal | sector=s)` for every `s`, multiplies by `prob_sector_s`, and sums to get `prob_informal`. `informal_predicted` (0.5 threshold) is a convenience; `prob_informal` is the primary output.
- **Cross-validation is grouped by household** (`StratifiedGroupKFold`) with weights normalized via `normalize_sample_weights`; the ENOE household key is `ENOE_HOUSEHOLD_COLUMNS`. Preserve this when changing model tuning — random row-level splits would leak.
- Stage 1 filters ENOE to Jalisco (`ent == 14`) and employed persons; stages 3 and 5 restrict both surveys to the metro municipalities sampled by both (`filter_common_geography` over `AMG_MUNICIPALITIES`, the 9 OD municipalities); the informality model scores OD workers in municipalities ENOE did not sample (Juanacatlán, Zapotlanejo in 2023t1) with `municipio = "otro"` (`training_municipalities` argument of `predict_od_informality`, recorded in the joblib bundle), keeping their real municipality in the output.
