# Review remediation plan

Status: **Phase 0 done**, Phase 1 in progress (2026-08-27); Phases 2–3 planned. Source: full-pipeline review of 2026-08-27 (after the mxcensus/eodgdl migration, commit `849568a`). Each item records *what is wrong*, *why it matters*, *the fix*, and *how to verify*. Items are grouped into phases that should be executed in order, because early phases change the training data and invalidate any tuning done before them.

Legend: **[D]** = a decision the user must make before implementation; **[R]** = requires re-running notebooks 04–05 (slow).

---

## Phase 0 — Safety net ✅ done

### 0.1 Freeze a second baseline
- **Why:** `outputs_baseline/` holds the pre-migration run; phase 1 deliberately changes the training data, so we need the post-migration run (`849568a`) as the new reference to measure each fix's effect.
- **Fix:** `cp -R outputs outputs_ref_849568a` (gitignored, like `outputs_baseline/`); make `compare_outputs` accept any two directories (it already does).
- **Verify:** `python -m src.compare_outputs outputs_ref_849568a outputs` reports zero differences before phase 1 starts. *(done: stage-1/2 outputs byte-identical after adding the assertions)*

### 0.2 Mapping-coverage assertions (fail loudly on drift)
- **Why:** every stage-2 mapping dict falls back to `no_especificado`/`otro`, so an added or renamed source category (new ENOE quarter, eodgdl revision) silently degrades the data. Today the only legitimately unmapped label is OD `estado_civil_raw == "Otros (especifique)"`.
- **Fix:** in `src/harmonize_enoe_od_dataframes.py`, add `_assert_mapping_covers(series, mapping, allowed_unmapped=frozenset())` and call it in every `harmonize_*` function. ENOE side: compare observed codes to `mxcensus.variables_enoe("sdem", gid)[col]["Categorías"]` where available. In stage 1, assert `set(OD_RENAMES) <= set(od.columns)` before renaming and that the `hab ⋈ viv` merge created no `_x/_y` suffixes; validate `period` against the quarters for which `ENOE_DWELLING_KEYS` is valid (2021t3–2025t2) or resolve the key from the loaded frame like `mxcensus._level_key` does.
- **Verify:** pipeline runs unchanged on 2023t1; a unit-style check with a fake label raises. *(done: `assert_mapping_covers` in every harmonizer with explicit allow-lists — ENOE `e_con == 9`, OD `"Otros (especifique)"`; `assert_enoe_dwelling_key` compares the hard-coded key with mxcensus's resolved key instead of a period range; OD rename/merge asserts in stage 1.)*

---

## Phase 1 — Data-correctness fixes (change training data) **[R]**

### 1.1 SCIAN code 9 → services ✅ done
- **Where:** `src/harmonize_enoe_od_dataframes.py:257`.
- **What/why:** ENOE `scian` is the 2-digit SCIAN sector: 8 = Transportes, **9 = Información en medios masivos**, 10–19 = services. Code 9 is currently sent to `gobierno_otro_agricultura`, where it is 15% of the class weight in the metro area — while the OD version of that class is only "Gobierno/sector público". Cross-check: `rama_est2` puts codes 8 and 9 in the same bucket.
- **Fix:** `9: "servicios_transporte"`. **[D]** also decide 2 (Minería) and 3 (Electricidad/agua/gas), currently `gobierno_otro_agricultura` (24 workers); `manufactura_construccion` is the more natural home. Document the intent next to the dict.
- **Verify:** ENOE `gobierno_otro_agricultura` share in the common geography drops from ~5.6% to ~4.75% (OD: 4.07%).
- **Done (2026-08-27):** 9 → `servicios_transporte`, 2 and 3 → `manufactura_construccion`. The whole sector mapping (ENOE `scian` and OD `giro_empresa`) now lives in `src/mappings/sector.yaml` with INEGI labels, old values as comments and a changelog; loaded via `src.load_mapping("sector")`. Effect on the ENOE training set (Jalisco): `gobierno_otro_agricultura` 14.06% → 13.12%, `servicios_transporte` +0.63 pp, `manufactura_construccion` +0.31 pp.

### 1.2 `eda == 98` is "age unspecified", not 98 years ✅ done
- **Where:** `harmonize_enoe_age` (`harmonize:77`); admitted by `ENOE_MAX_AGE = 98` in stage 1.
- **What/why:** INEGI codes 98 = unspecified (12+), 99 = unspecified (0–11). Two workers currently get `edad_num = 98` (a numeric model feature) and `edad_cat = 65_y_mas`. Under `employment_filter="p1"` code 99 would leak too.
- **Fix:** `enoe["edad_num"] = enoe["eda"].where(enoe["eda"] < 98)`; `pd.cut` on NA yields `no_especificado`. Keep `ENOE_MAX_AGE = 98` in the universe filter (INEGI semantics) but comment why.
- **Verify:** `enoe_harmonized.edad_num.max() <= 97`; two rows in `edad_cat == no_especificado`.
- **Done (2026-08-27):** `edad_num = eda.where(eda < 98)`; 2 workers now have `edad_cat = no_especificado`; `edad_num.max() = 97`.

### 1.3 Who counts as an OD worker — 194 self-reported non-workers ✅ done
- **Where:** `generate_od_dataframe` (`generate:86`) filter on `trabajo_semana_pasada`.
- **What/why:** 194 people report working last week but `ocupacion_raw ∈ {Desempleado 21, Hogar 90, Estudiante 61, Jubilado o pensionado 22}`. All 194 lack `giro_empresa`, so they enter sector imputation with occupation levels the sector model never saw (see 1.5), then receive an informality probability. Two internally inconsistent answers; no way to know which is right.
- **Options:** (a) exclude them from the worker set (recommended: 0.7% of rows; document as an inconsistency exclusion); (b) keep them and map `ocupacion → no_especificado` so they are handled by the missing-category path. **User decides.**
- **Verify:** `od_workers` row count 26,913 → 26,719 under (a); no `ocupacion_raw` level outside the six employed categories.
- **Decision (2026-08-27):** keep all 194. Trip records show they are not typical workers (20% made a work trip on the survey day vs 88% of other workers; 40% show any work-related mobility incl. weekend work destinations) but ENOE analogues (working students 63%, 12–17 y 97%, 65+ 72%, <15 h 94% informal) say those who do work are very likely informal, so excluding them would drop a real informal segment. Implemented as `ocupacion = no_especificado` for Hogar/Estudiante/Jubilado/Desempleado in `harmonize_od_occupation`; the count is printed in notebook 02. A part-time/schedule feature from ENOE `hrsocup` was considered and rejected: OD's self-reported "Medio tiempo" (6%) matches no hours cutoff (ENOE <35 h = 21%, <25 h = 11%, <15 h = 4%), and half of the 194 report full-time anyway. Optional Phase 2 experiment: 3-level `jornada` (absent / <25 h ≈ Medio tiempo / full-time) in the informality model only.

### 1.4 `ocupacion == "otro"` means opposite things in the two surveys ✅ done
- **Where:** `harmonize:46-52` (ENOE) and `:58-70` (OD).
- **What/why:** ENOE `pos_ocu == 4` = *trabajadores sin pago* (unpaid family workers, heavily informal). OD `otro` = home-makers/students/retirees/unemployed. Same level, disjoint concept, in a feature of the informality model: the model learns "otro ⇒ informal" from unpaid workers and applies it to retirees.
- **Fix:** ENOE `4 → "sin_pago"`. OD: after 1.3(a) the OD `otro` group is empty and the four labels map to `no_especificado`; under 1.3(b) map them to `"otra_condicion"` so they stay a distinct, honest level. Update README §4.2 category list and notebook 02 markdown.
- **Verify:** no shared `otro` level; stage-3 `ocupacion` comparison figure re-generated.

### 1.5 Categorical levels unseen at fit time are silently zeroed ✅ done
- **Where:** `OneHotEncoder(handle_unknown="ignore")` at `impute_economic_sector.py:139,142` and `informality_model.py:163,166`.
- **What/why:** `handle_unknown="ignore"` emits an all-zero block — a point corresponding to *no* category — for any level absent from training. Sector model: `ocupacion_raw` unseen levels + `parentesco = no_especificado` → 327 imputed workers with visibly shifted probabilities (mean `prob_comercio` 0.47 vs 0.24). Informality model: `no_especificado` never occurs in ENOE for `ocupacion`, `estado_civil`, `parentesco` (2,023 OD rows), plus the unseen municipalities (1.6).
- **Fix:** build the encoder with explicit `categories=` = union of the levels observed in *both* surveys plus `no_especificado` for every categorical feature (compute the lists once in `src/common.py` from the harmonized frames and store them in the joblib bundle). Add a predict-time check that reports (or raises on) values outside the fitted categories. With explicit categories, `no_especificado` becomes a real, learnable (all-zero in training) column rather than an out-of-support point.
- **Verify:** `model.named_steps["preprocessor"].transformers_[...].categories_` contains `no_especificado` for every feature; zero OD rows trigger the unseen-level check.

### 1.6 One geography for stages 3, 4 and 5 ✅ done
- **Where:** `diagnose:3-23` (`filter_common_geography`, 7 municipalities, applied to both surveys), `informality_model.py:71-91` (`select_common_informality_population`, restricts ENOE only), stage 4 (all 9 OD municipalities), `harmonize:160` (`"Tala"` in the OD map that can never fire; ENOE `mun == 83` → `tala`, listed in `TARGET_MUNICIPALITIES`).
- **What/why:** ENOE 2023t1 Jalisco has no Juanacatlán (51) or Zapotlanejo (124) — 1,860 OD workers (6.9%) are scored with an all-zero municipality block; ENOE-only Tala (27 workers) is a phantom level. The ENOE benchmark (7 municipalities) and the OD estimate (9) are presented side by side as the same population.
- **Fix:** define the AMG once (`src/common.py: AMG_MUNICIPALITIES`, the 9 eodgdl municipalities; drop Tala from both maps and from `TARGET_MUNICIPALITIES`, Tala → `otro` in ENOE). Then **[D]** for OD municipalities absent from ENOE: (a) recode to `otro` at prediction time and report the count (recommended; keeps all OD rows), or (b) drop them from the OD estimate. Make `select_common_informality_population` reuse `filter_common_geography` and return both restricted frames. Report the benchmark and the OD estimate on the same geography, with the out-of-support share printed.
- **Verify:** encoder municipality categories == `AMG_MUNICIPALITIES + ["otro"]`; notebook 05 prints "N OD workers in municipalities absent from ENOE".

### 1.7 Household size: consistent definition and collapsed tail ✅ done
- **Where:** `compute_enoe_dwelling_size` (`generate:39-44`), `harmonize:229-243`.
- **What/why:** ENOE counts every SDEM row, including 202 `c_res == 2` (definitively absent) persons, inflating 3.2% of dwellings; OD is a self-reported category capped at "10 y +". Weighted share of 8+ persons: ENOE 8.4% vs OD 1.6% — a 5× shift in a feature used by both models. ENOE `tamano_viv_num` is uncapped while OD's is capped at 10 (latent trap).
- **Fix:** count only `c_res in {1, 3}` rows (and pass the same filter for both employment modes); collapse `tamano_viv_cat` to `1…6, 7_y_mas` in both surveys; cap `tamano_viv_num` at 10 in ENOE too. Re-check the distribution comparison in notebook 03.
- **Verify:** |ENOE − OD| share for the top category < 2 pp; `tamano_viv_num.max() == 10` in both.
- **Done (2026-08-27):** ENOE dwelling size counts `c_res ∈ {1,3}` only (226 workers' dwellings changed); `HOUSEHOLD_SIZE_LABELS = 1…6, 7_y_mas` and `HOUSEHOLD_SIZE_CAP = 10` in `src/common.py`, applied identically to both surveys. Resulting weighted shares of workers: `7_y_mas` ENOE 12.2% vs OD 5.6% (was 8+: 8.4% vs 1.6%); categories 1–6 within 6 pp.

### 1.8 `survey_stratum` is the socio-economic stratum ✅ done
- **Where:** `ENOE_RENAMES` (`generate:18`).
- **What/why:** `est` has 4 values (INEGI estrato socioeconómico); the design stratum is `est_d_tri` (18 values). Anyone using `survey_stratum` + `survey_psu` for design-based variance gets wrong results.
- **Fix:** `est_d_tri → survey_stratum`, keep `est → estrato_socioeconomico`. Update README §1.
- **Verify:** `enoe_workers.survey_stratum.nunique() == 18`.
- **Done (2026-08-27):** `est_d_tri → survey_stratum`, `est → estrato_socioeconomico` (new column, 23 columns in `enoe_workers`).

### 1.9 Dtype casting: weights as float, codes as Int64, once ✅ done
- **Where:** `generate:75-76` (blanket `Int64` over all 22 columns incl. `survey_weight`), duplicated in `harmonize:11-17`.
- **What/why:** `astype("Int64")` on a fractional float raises `TypeError`; it works only because 2023t1 `fac_tri` is integral.
- **Fix:** `ENOE_CODE_COLUMNS → Int64`, `ENOE_WEIGHT_COLUMNS → float64`; delete `prepare_enoe_data_types`' re-cast (keep a dtype assertion instead). Assert no NA in `expansion_factor`/`survey_weight` inside `normalize_sample_weights`.
- **Done (2026-08-27):** as described; `survey_weight` is float64 in the parquet from now on.

### 1.10 Missing ENOE municipality must be `no_especificado`, not `otro` ✅ done
- **Where:** `harmonize:144` (`.fillna("otro")`).
- **Fix:** map NA → `NO_ESPECIFICADO`; only codes present but outside the AMG → `otro`. INEGI masks municipality as 999 in some quarters.

### 1.11 ENOE workers with unspecified sector ✅ done
- **Where:** `informality_model.py:81` drops `sector == no_especificado` (31 workers, 13,669 weighted, 100% informal).
- **What/why:** non-random exclusion in the direction of the model's known bias; it also moves the ENOE benchmark 39.59% → 39.26%. OD forces every worker into 4 sectors, so the two populations differ.
- **Options:** (a) keep dropping, but report `dropped_weighted_share` in `calculate_enoe_informality_benchmark` and quote the all-sector rate as the benchmark; (b) keep them in training with `sector = no_especificado` as a fifth level (harmless once 1.5 is in). Recommend (b) for training and quoting the all-sector benchmark.

---

## Phase 2 — Modelling / validation **[R]**

### 2.1 Group-safe early stopping
- **Where:** `HistGradientBoostingClassifier(early_stopping=True)` at `impute:163`, `informality:187`.
- **Why:** the internal 10% validation split is row-level and ignores households — the leak CLAUDE.md forbids; both shipped with-education models are HGBs.
- **Fix:** `early_stopping=False`; add `max_iter ∈ {100, 200, 400}` to the grid (outer CV is already grouped). Optionally also pass a `category` dtype frame straight to HGB so `categorical_features="from_dtype"` actually engages (today the ColumnTransformer emits dense floats and `is_categorical_ is None`).

### 2.2 Model selection with a tolerance
- **Where:** `impute:230`, `informality:266` (strict `<` on mean CV log loss).
- **Why:** family gaps (0.0015–0.0035) are 5–10× smaller than fold sd (0.010–0.020); the hybrids bolt an HGB arm to an RF arm by coin flip.
- **Fix:** paired per-fold comparison + 1-SE rule preferring the simpler family; or fix one family for both arms. Report all fold values, not just mean ± sd.

### 2.3 Calibration evidence
- **Where:** stage 4 has none; notebook 05 cell 55 `calibration_r2`.
- **Why:** the whole method rests on probabilities. The R² of 10 unweighted bin means is ~0.97 for any rank-correct model and ranks Model B above A, contradicting the prose. Reviewer-computed reliability for the sector model is actually decent in-distribution — that is the evidence worth showing.
- **Fix:** add `calculate_calibration_table` (weighted, shared in `src/common.py`) and a reliability figure to notebook 04; in stage 5 replace R² with weighted ECE and a calibration slope/intercept; drop the duplicated notebook-cell implementation (2.7). Consider `CalibratedClassifierCV(method="isotonic", cv=StratifiedGroupKFold(...))` given the consistent 1.2–1.6 pp under-prediction.

### 2.4 Validate the without-education model on the population it serves
- **Where:** notebook 05 cell 55 (the "Híbrido" curve is bit-identical to "Con escolaridad" because ENOE has 0 test rows without education).
- **Why:** Model B serves 5,195 OD workers (15.8% weighted), 4,522 of whom also have imputed sector — doubly imputed with no validation path.
- **Fix:** masked evaluation: score Model B on the ENOE test fold with education blanked (all rows, and a subsample reweighted to the OD non-respondent profile); report next to Model A. Remove the degenerate hybrid line.

### 2.5 Covariate shift in sector imputation — state MAR, add sensitivity
- **Where:** notebook 04 narrative (cell 6 notes the issue, cell 57 claims proportions "maintained").
- **Why:** unknown-sector workers are far more educated (35.6% vs 19.4% licenciatura+) and missingness co-occurs with a skipped questionnaire block; imputed `gobierno_otro_agricultura` share moves 4.05% → 5.54%.
- **Fix:** report imputed-vs-observed sector shares as a *result with caveat*; add one sensitivity check (reweight known-sector sample to unknown-sector covariate margins, or a delta adjustment on the rare class); state MAR explicitly in README §2.4.

### 2.6 Honest baselines and uncertainty
- **Fix:** replace `log(4)` with the weighted-marginal log loss on the same fold (1.2145 → improvements 11.4% / 8.7%); report all 5 folds and a household-bootstrap CI for the held-out metrics; add a PSU-grouped CV variant (`survey_psu`) as a robustness line.

### 2.7 Threshold series and headline comparison
- **Fix:** remove "OD duro" (0.5 threshold; 7% hard rate on imputed-sector rows is a marginalization-shrinkage artifact) from the comparison figure or annotate it; add a Bernoulli-sampled series (unbiased for the aggregate). Add a one-paragraph decomposition of the ENOE→OD gap (direct standardization on `ocupacion`, `escolaridad`, `sector`, `genero`) and note ENOE is Jan–Mar 2023.

---

## Phase 3 — Code structure and robustness (no numeric change)

### 3.1 `src/common.py`
- Move the duplicated `SECTOR_CLASSES`, `identify_missing_category`, `normalize_sample_weights`, `normalize_predicted_probabilities` (rename to `validate_…` or make it actually renormalize), `ENOE_HOUSEHOLD_KEYS`, `AMG_MUNICIPALITIES`, harmonized-column list, `NO_ESPECIFICADO`, category lists (1.5). One normalization rule for "missing" (strip + lower) used by `identify_missing_category`, `prepare_*_features`, `calculate_missingness`. Keep "No sabe" as its own answer, distinct from item non-response.
- Make `identify_missing_category` return numpy bool (`.fillna(False).astype(bool)`); explicit `NUMERIC_FEATURES` constant instead of `column == "edad_num"`; `zero_division=0` in tuning F1; build the scenario frame once in `predict_od_informality` instead of 4× `od.copy()`.

### 3.2 Self-contained model bundles
- Fold `prepare_sector_features` / `prepare_informality_features` into the sklearn `Pipeline` (`FunctionTransformer`) so the joblibs work on raw harmonized frames; store `metadata` (best params, CV/test metrics, sklearn version, fitted categories, employment filter, ENOE period). Assert `set(model.classes_) == set(SECTOR_CLASSES)` in `impute_missing_sectors_hybrid`. Re-export everything public from `src/__init__.py` (`SECTOR_CLASSES`, `TARGET_MUNICIPALITIES`/`AMG_MUNICIPALITIES`, `NO_ESPECIFICADO`, `AGE_BINS`, `prepare_*_features`, `identify_missing_category`, …).

### 3.3 `compare_outputs`
- Move the CLI to `src/__main__.py` (removes the `RuntimeWarning`); rename `--stage 3` to `--stage models`; alias the full eodgdl rename map so OD column diffs show real changes only; guard `prob_informal`/`informal_predicted` access; guard zero totals; only run key overlap for stage-1 files.

### 3.4 Notebooks
- Generate every number in the markdown from result frames (f-strings / `IPython.display.Markdown`) — the current prose in 04 and 05 reports the wrong winning model, wrong metrics, and the wrong *sign* of the calibration bias.
- Fix `"\%"` → raw strings (03/04/05); nb03: don't rebind `tick_labels` (cell 26), include `sector` in the informality-profile grid; nb04: remove tautological "consistency check" (cell 47), typo "training:a", weight-normalization footnote; nb02: drop `(Tala) → tala`, document the fallback paths (`eda == 98`, `e_con == 9`, "Otros (especifique)", source NA); nb01: mention `EODGDL_CACHE_DIR`.
- Docs: joblib bundles have 5 keys; README §1 `survey_stratum` source; README §4.2 new category names; note `genero` is sex at birth (2.9% of OD workers report a different `genero_identidad`) or rename to `sexo`.

---

## Execution order and verification

1. Phase 0 → commit. 2. Phase 1 in one branch; run 01→02; `compare_outputs outputs_ref_849568a outputs --stage 2` and inspect each expected shift (1.1, 1.2, 1.7) — differences must be explainable item by item. 3. Phase 2 + re-run 03→05; compare `--stage all`; the calibration and masked-validation tables are the acceptance evidence, not distribution matching alone. 4. Phase 3 can be interleaved but must not change numbers (verify with `compare_outputs` = zero diff). Each phase gets its own commit with the `compare_outputs` report pasted into the message.

## Decisions required before Phase 1 **[D]**
| # | Question | Recommendation |
|---|---|---|
| 1.1 | SCIAN 2 (Minería) and 3 (Electricidad) → which class? | `manufactura_construccion` |
| 1.3 | Exclude the 194 self-reported non-workers, or keep as `no_especificado`? | exclude |
| 1.6 | OD municipalities absent from ENOE: recode to `otro` or drop? | recode to `otro`, report share |
| 1.11 | ENOE unspecified-sector workers: drop (report) or keep as 5th level? | keep as 5th level; benchmark on all sectors |
