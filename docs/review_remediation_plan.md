# Review remediation plan

Status: **Phases 0–4 done** (2026-08-28; 4.5 evaluated and rejected). Headline: OD expected informality **36.42%** (sampled 35.86%; raked to the ENOE metro profile 38.42%; ENOE metro benchmark 39.35%; no-trip sensitivity band 36.4–43.0%). Source: full-pipeline review of 2026-08-27 (after the mxcensus/eodgdl migration, commit `849568a`). Each item records *what is wrong*, *why it matters*, *the fix*, and *how to verify*.

Legend: **[D]** = a decision the user must make before implementation; **[R]** = requires re-running notebooks 04–05 (slow).

---

## Phase 0 — Safety net ✅ done

### 0.1 Freeze a second baseline ✅ done
- **Why:** `outputs_baseline/` holds the pre-migration run; phase 1 deliberately changes the training data, so we need the post-migration run (`849568a`) as the new reference to measure each fix's effect.
- **Fix:** `cp -R outputs outputs_ref_849568a` (gitignored, like `outputs_baseline/`); make `compare_outputs` accept any two directories (it already does).
- **Verify:** `python -m src.compare_outputs outputs_ref_849568a outputs` reports zero differences before phase 1 starts. *(done: stage-1/2 outputs byte-identical after adding the assertions)*

### 0.2 Mapping-coverage assertions (fail loudly on drift) ✅ done
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

### 1.12 A zero-support level is not neutral for tree models ✅ done (found after Phase 1 re-run)
- **Where:** `impute_missing_sectors_hybrid`, `predict_od_informality` (both use one-hot + tree ensembles).
- **What/why:** With explicit categories (1.5) an unsupported level (e.g. `ocupacion = no_especificado`, absent from ENOE) becomes an all-zero one-hot block. A tree needs only K−1 indicators, so the all-zero row follows the branch of whichever training level the trees did not split on — in ENOE that is `sin_pago` (100% informal). Measured on the 194 workers of 1.3: P(informal) = 0.93 with `no_especificado` vs 0.53 as `trabajador` and 0.75 as `independiente`. Same mechanism for `estado_civil`/`parentesco = no_especificado` (informality) and `ocupacion_raw ∈ {Hogar, Estudiante, Jubilado, Desempleado}`, `parentesco = no_especificado` (sector model; the 327 rows of the review).
- **Fix:** marginalize, as already done for sector: for rows whose value of feature *f* has no training support, predict once per supported level and average with the weighted training share of each level (`predict_marginalizing_unsupported` in `src/common.py`, used by both prediction functions; several unsupported features handled sequentially under independence). Record which rows were marginalized (`*_marginalized_features` column).
- **Verify:** the 194 score ≈ the share-weighted mix (~0.55), not 0.93; sector probabilities of the 327 rows no longer differ systematically from other imputed rows; `count_levels_without_training_support` output unchanged (it reports, the marginalization handles).
- **Done (2026-08-27):** `compute_training_level_shares` / `attach_training_level_shares` (called by both `refit_*` functions, stored on the pipeline as `training_level_shares_` so the joblib is self-contained) and `predict_proba_marginalizing` in `src/common.py`; used by `impute_missing_sectors_hybrid` (`sector_marginalized_features` column) and `predict_od_informality` (`informality_marginalized_features`). Rows with several unsupported features are expanded over all combinations. Test: the 194 → 0.58 (was 0.93); flag-free rows identical to `predict_proba`. **Interaction with 1.6:** the ENOE training population excludes `otro` (it is restricted to the common municipalities), so the Juanacatlán/Zapotlanejo rows recoded to `otro` are in fact marginalized over the seven sampled municipalities weighted by their training share — a reasonable 'unknown metro municipality' treatment; `municipio_scored` and `informality_marginalized_features` make it visible.

### 1.13 Train the informality model on the whole state ✅ done (user question after Phase 1)
- **What/why:** the training population was the 7 metro municipalities sampled by both surveys (5,280 workers). The rest of Jalisco (1,693 workers, 59% informal vs 40% in the metro, with different sector interactions — e.g. `gobierno_otro_agricultura` 64% vs 23%) was excluded. Experiment (tuned with-education config, same metro held-out households): whole-state training lowers metro-test log loss 0.4904 → 0.4857, AUC 0.835 → 0.837, aggregate gap −0.18 → −0.03 pp, OD estimate unchanged (32.74% vs 32.75%). But `otro` then means non-metro Jalisco, so scoring Juanacatlán/Zapotlanejo as `otro` would raise them from 0.35 to 0.43 — an overestimate for peri-urban metro municipalities.
- **Done (2026-08-27):** `select_common_informality_population` returns `(training_population = whole state, benchmark_population = common metro geography, training_municipalities, geography_summary)`; unsampled metro municipalities are scored with `municipio = no_especificado` and `predict_proba_marginalizing(level_subsets={"municipio": sampled metro municipalities})`, i.e. averaged over the sampled metro municipalities only; the held-out test fold in notebook 05 keeps metro rows only (non-metro rows stay in the training folds); benchmark and feature-missingness comparison use the benchmark population. Supersedes the `otro` recoding of 1.6.
- **Check after the re-run:** the metro held-out fold now shows a +2.7 pp over-prediction (was −1.6 pp). Controlled comparison on the same metro test households with the tuned config: metro-only training gives +2.6 pp on that fold, so the gap is a fold property, not a statewide-training effect. Over 5 folds (metro test rows only): log loss state 0.477/0.530/0.496/0.487/0.498 vs metro-only 0.482/0.527/0.490/0.505/0.509 (mean 0.497 vs 0.503); aggregate gaps +2.7/+0.7/+0.4/−1.9/−0.5 vs +2.6/+0.7/−0.2/−3.1/−0.6. Slightly larger over-prediction for `gobierno_otro_agricultura` with state training (+7.9 vs +4.6 pp on fold 0) is the one place the non-metro regime shows; worth watching in Phase 2.3 (calibration). Final OD estimate 33.23% (Juanacatlán/Zapotlanejo 0.345/0.371).

---

## Phase 2 — Modelling / validation **[R]**

### 2.1 Group-safe early stopping ✅ done
- **Where:** `HistGradientBoostingClassifier(early_stopping=True)` at `impute:163`, `informality:187`.
- **Why:** the internal 10% validation split is row-level and ignores households — the leak CLAUDE.md forbids; both shipped with-education models are HGBs.
- **Fix:** `early_stopping=False`; add `max_iter ∈ {100, 200, 400}` to the grid (outer CV is already grouped). Optionally also pass a `category` dtype frame straight to HGB so `categorical_features="from_dtype"` actually engages (today the ColumnTransformer emits dense floats and `is_categorical_ is None`).

- **Result (2.1):** all three GradientBoosting winners (sector with-education; informality with/without education) selected `max_iter = 100`, the **lower edge of the grid** (learning rate 0.05). The grid must be extended downward (`{50, 100, 200, 400}`) — folded into the 2.2 re-run to avoid a separate 25-minute pass. Metro held-out: informality Model A log loss 0.4774 → 0.4744, AUC 0.8376 → 0.8386; OD expected informality 33.23% → 33.50%.

### 2.2 Model selection with a tolerance ✅ done
- **Where:** `impute:230`, `informality:266` (strict `<` on mean CV log loss).
- **Why:** family gaps (0.0015–0.0035) are 5–10× smaller than fold sd (0.010–0.020); the hybrids bolt an HGB arm to an RF arm by coin flip.
- **Fix:** paired per-fold comparison + 1-SE rule preferring the simpler family; or fix one family for both arms. Report all fold values, not just mean ± sd.

- **Result (2.2):** no family switch — the paired differences GB−RF are small but consistent across folds (sector A +0.0013 ± 0.0009, informality A +0.0029 ± 0.0011, B +0.0025 ± 0.0011), so RF is *not* within one SE; the rule did simplify within families (sector A: `max_iter` 50, L2 1.0, 31 leaves; informality A and B: `max_iter` 100, L2 1.0, 15 leaves — 50 was available and lost). Sector B stays RandomForest. Metro held-out log loss A 0.4744 → 0.4750 (equal within noise). OD expected informality 33.45%; the `gobierno_otro_agricultura` conditional rate keeps drifting up run to run (13.1 → 18.3% across the Phase 1–2 re-runs) — the smallest class, to be examined under 2.3/2.5.

### 2.3 Calibration evidence ✅ done
- **Where:** stage 4 has none; notebook 05 cell 55 `calibration_r2`.
- **Why:** the whole method rests on probabilities. The R² of 10 unweighted bin means is ~0.97 for any rank-correct model and ranks Model B above A, contradicting the prose. Reviewer-computed reliability for the sector model is actually decent in-distribution — that is the evidence worth showing.
- **Fix:** add `calculate_calibration_table` (weighted, shared in `src/common.py`) and a reliability figure to notebook 04; in stage 5 replace R² with weighted ECE and a calibration slope/intercept; drop the duplicated notebook-cell implementation (2.7). Consider `CalibratedClassifierCV(method="isotonic", cv=StratifiedGroupKFold(...))` given the consistent 1.2–1.6 pp under-prediction.

- **Done (2026-08-27):** shared `calculate_calibration_table` and `calibration_metrics` (weighted ECE, calibration-in-the-large gap, logistic slope/intercept) in `src/common.py`; `calculate_sector_calibration` (one-vs-rest per class) with a reliability figure `outputs/figures/sector_calibration.*` in notebook 04; `evaluate_informality_model` reports ECE/slope/intercept; the notebook-05 R² block and the degenerate hybrid curve are gone (final figure shows ECE and the aggregate gap). Isotonic recalibration implemented as `IsotonicCalibratedPipeline` / `fit_isotonic_calibrator` (out-of-fold, household-grouped) with `calibrate_informality_model`, `compare_informality_models` and a `SHIP_CALIBRATED` flag in notebook 05 (`outputs/figures/informality_calibration.*`).
- **Result — sector model (held-out households):** well calibrated in the large: gaps −1.7 (comercio), −0.3, +0.6, +1.3 pp; ECE 0.006–0.021 per class; slopes 0.9–1.14. Both specifications similar.
- **Result — informality, metro held-out fold 0:** A uncalibrated log loss 0.4732 / ECE 0.048 / slope 1.20 / gap +2.9 pp vs isotonic 0.4747 / 0.037 / 1.10 / +2.7 pp; B 0.5053 / 0.043 / 1.23 / +2.9 vs 0.5306 / 0.046 / 1.08 / +2.4. Isotonic improves shape slightly (ECE for A) but worsens log loss (badly for B) and does not remove the fold-specific +2.8 pp gap (OOF predictions on the training households are already calibrated in the aggregate — the gap is a property of fold 0, cf. 1.13). OD estimate 33.45% uncalibrated vs 32.77% isotonic. **Shipped: uncalibrated** (`SHIP_CALIBRATED = False`); comparison kept in the notebook. If a correction is ever wanted, prefer a two-parameter (Platt) map over isotonic at this sample size.

### 2.4 Validate the without-education model on the population it serves ✅ done
- **Where:** notebook 05 cell 55 (the "Híbrido" curve is bit-identical to "Con escolaridad" because ENOE has 0 test rows without education).
- **Why:** Model B serves 5,195 OD workers (15.8% weighted), 4,522 of whom also have imputed sector — doubly imputed with no validation path.
- **Fix:** masked evaluation: score Model B on the ENOE test fold with education blanked (all rows, and a subsample reweighted to the OD non-respondent profile); report next to Model A. Remove the degenerate hybrid line.

- **Done (2026-08-27):** `reweight_to_od_profile` / `evaluate_on_od_profile` in `src/informality_model.py` (weighted logistic density-ratio classifier on the shared features; ENOE test rows reweighted to the OD missing-education profile; ESS 431 of 1,014 rows, top-decile weight share 27%, profile matched on occupation 97% employees / sex / age / household size). Notebook 05 reports A and B on identical metro test rows under both weightings, and the doubly-imputed group in the OD summary (4,522 workers, 14.0% weighted, mean P(informal) 0.279 vs 0.347 for the rest).
- **Result (fold 0):** cost of losing education on the same rows: log loss 0.473 → 0.505, AUC 0.839 → 0.808. On the OD non-respondent profile: observed informality is lower (28.2% vs 37.7% — non-respondents are mostly employees), log loss A 0.4725 / B 0.5068, AUC A 0.794 / B 0.740, aggregate gap A +3.1 / B +3.0 pp, ECE A 0.045 / B 0.040, slope B 1.37 (B is under-confident on this profile). B degrades on the non-respondent profile as much as A does, i.e. the missing-education population is harder for both models rather than specifically unsuited to B; the hybrid's cost is ≈ 0.03 nats and −0.05 AUC for 15.8% of the OD population.

### 2.5 Covariate shift in sector imputation — state MAR, add sensitivity ✅ done
- **Where:** notebook 04 narrative (cell 6 notes the issue, cell 57 claims proportions "maintained").
- **Why:** unknown-sector workers are far more educated (35.6% vs 19.4% licenciatura+) and missingness co-occurs with a skipped questionnaire block; imputed `gobierno_otro_agricultura` share moves 4.05% → 5.54%.
- **Fix:** report imputed-vs-observed sector shares as a *result with caveat*; add one sensitivity check (reweight known-sector sample to unknown-sector covariate margins, or a delta adjustment on the rare class); state MAR explicitly in README §2.4.

- **Done (2026-08-27):** MAR assumption stated in notebook 04 and README §2.4. `reweight_to_target_profile` (generic density-ratio reweighting, `src/common.py`; 2.4's helper now delegates to it), `impute_sectors_under_covariate_shift` (known-sector rows reweighted to the unknown-sector profile on `SECTOR_SHIFT_PROFILE_FEATURES`, education excluded; ESS 6,870 of 17,429) and `adjust_imputed_sector_share` (delta adjustment) in `impute_economic_sector.py`; scenarios written to `outputs/od_sector_imputed_sensitivity.parquet` and carried into notebook 05, which reports the headline under each.
- **Result:** shift-weighted refit reproduces the current imputation (imputed shares 24.7/8.9/30.5/35.9% vs 24.8/8.7/30.3/36.3%; headline −0.01 pp) — the model's response to the observable shift is stable, i.e. the MAR-given-x fit is not driven by the parts of covariate space where non-respondents concentrate. The `gobierno_otro_agricultura` excess among imputed workers (8.7% vs 4.05% observed) comes from education, the one covariate that differs most and cannot be reweighted on (it is the feature that predicts that class): forcing the class back to its observed share (factor 0.47) moves the headline by +0.05 pp and its within-sector rate by +0.9 pp. Bottom line: the OD informality headline is insensitive to the sector-imputation assumption (±0.05 pp); the sector *composition* of imputed workers is the only quantity that moves (±4.6 pp for the rare class).

### 2.6 Honest baselines and uncertainty ✅ done
- **Fix:** replace `log(4)` with the weighted-marginal log loss on the same fold (1.2145 → improvements 11.4% / 8.7%); report all 5 folds and a household-bootstrap CI for the held-out metrics; add a PSU-grouped CV variant (`survey_psu`) as a robustness line.

- **Done (2026-08-27):** `marginal_log_loss`, `fold_table`, `bootstrap_by_group` (cluster bootstrap over test households, 500 resamples) and `cross_validate_grouped` in `src/common.py`; `sector_test_metrics_with_uncertainty` / `informality_test_metrics_with_uncertainty` wrappers; notebooks 04/05 print per-fold tables, the PSU-grouped CV of the selected informality configurations, and bootstrap intervals for the held-out metrics.
- **Result — sector (test fold):** log loss A 1.076 [1.045, 1.115] vs marginal 1.213 → 11.3% improvement (B: 1.108 → 8.7%); weighted accuracy A 0.483 [0.461, 0.506]; macro-F1 0.386 [0.362, 0.411]. **Informality (metro test fold 0):** log loss A 0.473 [0.440, 0.507] vs marginal 0.663 → 28.6% (B: 0.505 → 23.8%); AUC A 0.839 [0.810, 0.869], B 0.808 [0.775, 0.839]; aggregate gap A +2.9 pp [+0.2, +5.4], B +2.9 [−0.03, +5.7] — the fold-0 gap sits at the edge of its own bootstrap interval, consistent with the cross-fold picture (−1.9 … +2.7). PSU-grouped CV: mean log loss A 0.515 vs 0.514 household-grouped, B 0.560 vs 0.556 — no material optimism from household-only grouping.

### 2.7 Threshold series and headline comparison ✅ done
- **Fix:** remove "OD duro" (0.5 threshold; 7% hard rate on imputed-sector rows is a marginalization-shrinkage artifact) from the comparison figure or annotate it; add a Bernoulli-sampled series (unbiased for the aggregate). Add a one-paragraph decomposition of the ENOE→OD gap (direct standardization on `ocupacion`, `escolaridad`, `sector`, `genero`) and note ENOE is Jan–Mar 2023.

- **Done (2026-08-27):** `sample_informality` (one Bernoulli draw per worker, `informal_sampled` in the output parquet) replaces "OD duro" in the final figure/tables as "OD muestreo"; `informal_predicted` stays in the file but is documented as not an estimate. `decompose_enoe_od_gap` (direct standardization of the ENOE benchmark to the OD covariate profile with the density-ratio helper) added to notebook 05 with a paragraph; README §2.5 and §4.3 updated. Periods coincide (ENOE Jan–Mar 2023; OD fieldwork 23 Jan–29 Apr 2023), so no seasonality caveat.
- **Result:** OD muestreo 33.22% vs OD esperado 33.45% vs ENOE 39.59% (the threshold series was 19.65%). Decomposition: ENOE observed 39.59% → model-predicted on ENOE 40.11% (+0.5 pp bias) → reweighted to the OD profile 34.22% (−5.9 pp composition; observed-rate composition −7.8 pp) → OD 33.45% (−0.8 pp residual). Effective sample size of the reweighting is small (246 of 5,280 rows; top decile 26% of weight) because the OD profile — 16% education non-response, more employees, more educated — is far from ENOE's, so the composition term is indicative rather than precise; but the ordering is clear: about 90% of the ENOE→OD gap is composition, not model bias.

---

## Phase 3 — Code structure and robustness (no numeric change)

### 3.1 `src/common.py` ✅ done
- Move the duplicated `SECTOR_CLASSES`, `identify_missing_category`, `normalize_sample_weights`, `normalize_predicted_probabilities` (rename to `validate_…` or make it actually renormalize), `ENOE_HOUSEHOLD_KEYS`, `AMG_MUNICIPALITIES`, harmonized-column list, `NO_ESPECIFICADO`, category lists (1.5). One normalization rule for "missing" (strip + lower) used by `identify_missing_category`, `prepare_*_features`, `calculate_missingness`. Keep "No sabe" as its own answer, distinct from item non-response.
- Make `identify_missing_category` return numpy bool (`.fillna(False).astype(bool)`); explicit `NUMERIC_FEATURES` constant instead of `column == "edad_num"`; `zero_division=0` in tuning F1; build the scenario frame once in `predict_od_informality` instead of 4× `od.copy()`.
- **Done (2026-08-27):** `src/common.py` holds the shared constants and helpers (`HARMONIZED_FEATURES`, `NUMERIC_FEATURES`, `identify_missing_category` used by the model stages *and* `calculate_missingness`, weight/probability normalizers, feature preparation, category levels, marginalization, calibration, selection, bootstrap); `ENOE_HOUSEHOLD_COLUMNS` imports `ENOE_HOUSEHOLD_KEYS`; `zero_division=0` in every F1 call; the informality scenario frame is built once per sector instead of copying the full OD frame; `impute_missing_sectors_hybrid` asserts the model classes and the known-sector classes against `SECTOR_CLASSES`. "No sabe" stays merged with non-response (changing it would alter outputs; noted in notebook 02).

### 3.2 Self-contained model bundles ✅ done
- Fold `prepare_sector_features` / `prepare_informality_features` into the sklearn `Pipeline` (`FunctionTransformer`) so the joblibs work on raw harmonized frames; store `metadata` (best params, CV/test metrics, sklearn version, fitted categories, employment filter, ENOE period). Assert `set(model.classes_) == set(SECTOR_CLASSES)` in `impute_missing_sectors_hybrid`. Re-export everything public from `src/__init__.py` (`SECTOR_CLASSES`, `TARGET_MUNICIPALITIES`/`AMG_MUNICIPALITIES`, `NO_ESPECIFICADO`, `AGE_BINS`, `prepare_*_features`, `identify_missing_category`, …).
- **Done (2026-08-27):** every pipeline starts with a `prepare` step (`FunctionTransformer(prepare_model_features)`), so the pickled bundles apply to a raw harmonized frame; bundles carry `category_levels`, `sector_classes`, `training_municipalities` (informality) and `metadata` (selected configuration and CV loss per arm, test metrics, ENOE period, employment filter, shipped variant, scikit-learn version, random state). Public names re-exported from `src/__init__.py`.

### 3.3 `compare_outputs` ✅ done
- Move the CLI to `src/__main__.py` (removes the `RuntimeWarning`); rename `--stage 3` to `--stage models`; alias the full eodgdl rename map so OD column diffs show real changes only; guard `prob_informal`/`informal_predicted` access; guard zero totals; only run key overlap for stage-1 files.
- **Done (2026-08-27):** CLI moved to `src/__main__.py` (`uv run python -m src <ref> <new> [--stage 1|2|models|all]`; `3` kept as an alias of `models`); OD columns of a pre-migration reference are aliased through eodgdl's full rename map plus `OD_RENAMES`, so column diffs show real changes only; `prob_informal`/`informal_sampled` guarded; zero totals guarded; key overlap only for stage 1.

### 3.4 Notebooks ✅ done
- Generate every number in the markdown from result frames (f-strings / `IPython.display.Markdown`) — the current prose in 04 and 05 reports the wrong winning model, wrong metrics, and the wrong *sign* of the calibration bias.
- Fix `"\%"` → raw strings (03/04/05); nb03: don't rebind `tick_labels` (cell 26), include `sector` in the informality-profile grid; nb04: remove tautological "consistency check" (cell 47), typo "training:a", weight-normalization footnote; nb02: drop `(Tala) → tala`, document the fallback paths (`eda == 98`, `e_con == 9`, "Otros (especifique)", source NA); nb01: mention `EODGDL_CACHE_DIR`.
- Docs: joblib bundles have 5 keys; README §1 `survey_stratum` source; README §4.2 new category names; note `genero` is sex at birth (2.9% of OD workers report a different `genero_identidad`) or rename to `sexo`.
- **Done (2026-08-27):** the prose cells of notebooks 04 and 05 that quoted numbers (selected model, metrics, confusion, calibration, usage, headline, by-sector) are now code cells rendering Markdown from the result frames (`from IPython.display import Markdown`), and the hyperparameter grids are listed from `build_*_models()`; `\%` strings are raw; notebook 03 draws the informality profile on a 3×3 grid including `sector` and no longer rebinds `tick_labels`; notebook 04's tautological consistency check, typo and weight-normalization footnote fixed; notebook 02 documents the fallback paths; notebook 01 mentions `EODGDL_CACHE_DIR`; CLAUDE.md lists the bundle keys and the `prepare` step; README notes that `genero` is sex at birth. Verified by a full 01→05 re-run compared with the pre-Phase-3 outputs (see status line).

---

## Phase 4 — Model improvements (new features, more data) **[R]**

Each item is run as an experiment first — same metro held-out households, paired folds, 1-SE selection, calibration and bootstrap intervals from Phase 2 — and shipped only if it improves the paired-fold log loss; the OD headline and composition are reported before/after in every case.

### 4.1 Pool several ENOE quarters ✅ done
- **Why:** one quarter gives 6,973 Jalisco workers; `mxcensus` serves 2022t1–2023t4 (and later) with identical schemas, so 8 quarters give ~50k workers with the same definitions. Wider intervals in 2.6 (log loss ±0.03, gap ±2.6 pp) come mostly from sample size; the rare sector class and the municipality effects are barely estimable on one quarter.
- **Fix:** `generate_enoe_dataframe(periods=[...])` concatenates quarters with a `period` column, computes dwelling size per quarter, and divides `survey_weight` by the number of quarters (totals stay at population scale; per-quarter benchmark rates reported). The CV/bootstrap group key becomes the **cross-quarter dwelling/household key** (`cd_a, ent, con, v_sel, n_hog, h_mud` — without `tipo`/`mes_cal`, which identify panel visits), so the five visits of a panel household never straddle folds. Benchmark = pooled metro population (per-quarter rates shown).
- **Verify:** paired-fold log loss of the selected configuration on the 2023t1 metro test households, pooled training vs single quarter; interval widths; OD headline.

- **Done (2026-08-27):** `ENOE_PERIODS` = 2022t1–2023t4 (51,707 workers, 38,933 in the metro benchmark; weights ÷ 8), `period` column, cross-quarter group key `ENOE_GROUP_KEYS` for CV/bootstrap, per-quarter benchmark table. Experiment on identical 2023t1 metro test households: pooled better in 4/5 folds, paired log loss −0.0053 ± 0.0035 SE, AUC 0.829 vs 0.824. Shipped run: benchmark 39.35% pooled (2023t1: 39.59%); held-out (pooled metro fold) log loss A 0.497 [0.481, 0.516], AUC 0.827, gap −0.8 pp [−2.3, +0.7] — bootstrap sd of the log loss halves (0.009 vs 0.017). OD expected informality 33.49%.

### 4.2 Place of work as a shared feature ✅ done
- **Why:** ENOE informality is almost determined by the workplace: `tue1` "no establishment" 99–100% informal vs 11–29% in establishments; `ambito1` 69% vs 14%. The OD has the work trip's destination type (Fábrica/taller, Comercio, Oficina, Otra vivienda, Hospital, Escuela, Restaurante…) for ~88% of workers.
- **Fix:** harmonized `lugar_trabajo` with ~5 levels — establecimiento (factory/workshop, office, hospital, school, restaurant), comercio, otra_vivienda (someone else's home), propia_vivienda_o_sin_local (own home / no fixed place), no_especificado (no work trip recorded) — from ENOE `ambito2`/`tue1` and from the OD work-trip destination (`eodgdl` trips, `motivo_viaje == "Trabajar"`, most frequent destination type per worker). Compare weighted distributions in stage 3 before use.
- **Verify:** paired-fold gain; the `no_especificado` share on the OD side and how those workers are scored (marginalized); calibration by level.

- **Done (2026-08-27):** ENOE `lugar_trabajo` from COE1 `p4`, `p4b` (institutions → establecimiento, agriculture → field), `p4e`, `p4f`, `p4h` (codes coerced numerically — raw strings are zero-padded in some quarters); OD from the most frequent work-trip destination type (`compute_od_work_trip_destination`). Shares (ENOE pooled / OD): establecimiento 47/49%, comercio_o_puesto 20/29%, otra_vivienda 13/6%, otro_o_sin_local 19/3%, no_especificado 0.4/13% — the OD's no-trip group mirrors ENOE's home/mobile workers. ENOE informality by level: 29 / 52 / 92 / 67%. Paired-fold experiment (pooled state training, metro test rows): log loss 0.499 → 0.431 (−0.068 ± 0.003 SE, every fold), AUC 0.820 → 0.865. Shipped in both informality arms. Result: held-out log loss A 0.418 [0.400, 0.441], AUC 0.875, gap −0.02 pp [−1.4, +1.2]; **OD expected informality 33.49% → 36.28%** (sampled 35.78%; superseded by 4.7 — this run scored no-trip workers as ENOE's residual level); within services +8 pp, within gobierno/otro/agricultura −4.5 pp. Sensitivity for the 3,290 workers without a work trip (13.1% weighted; marginalized): headline 36.28% vs 42.96% if all were scored as otro_o_sin_local (reported in notebook 05).

### 4.3 Household-roster features (exactly shared) ✅ done
- **Fix:** from both rosters: number of employed persons in the household, number of children under 12, whether the worker is the only earner, and the household head's education (for non-heads). ENOE: SDEM roster by household key; OD: `hab` by `folio_vivienda`. Added to `INFORMALITY_FEATURES`/`ROBUST` (and to the sector model) after a stage-3 comparison.

- **Done (2026-08-27):** `hogar_trabajadores_cat` (employed members, 1/2/3/4+) and `hogar_ninos_cat` (children aged 6–11, the OD roster's floor) from both rosters; distributions comparable (ENOE 23/37/22/17% vs OD 26/39/24/11%; 72/21/8% vs 77/18/5%). Paired-fold gain +0.0001 ± 0.0005 — none. Kept in the harmonized data, not used as features.

### 4.4 DENUE at the work-trip destination (sector model only) ✅ done
- **Why:** the sector model uses `centralidad` (71 dummies) as its only spatial signal. The OD records the destination AGEB of the work trip; `mxcensus.load_denue` gives every establishment with SCIAN code and location. The sector mix (and size mix) of establishments at the destination AGEB is a direct predictor of the worker's sector and is available for imputed workers too.
- **Fix:** per destination AGEB: shares of establishments (and of employment-size classes) in the four harmonized sectors; join to workers by their most frequent work-trip destination; features `dest_share_<sector>`, `dest_establishments`. OD-only, so no ENOE counterpart needed.
- **Verify:** paired-fold log loss of the sector model; calibration-in-the-large by class; effect on imputed sector composition and the informality headline.

- **Done (2026-08-27):** `src/destination_features.py` — work-trip destination code resolved to INEGI `CVEGEO` through `eodgdl.load_imeplan_agebs().CVEGEO_EOD` (13-character codes are AGEBs; 9-character ones are IMEPLAN codes, resolved where the table has them; `999990001–6` are the six access corridors out of the metro zone and `99999000A` the airport, classified through `zona_destino`); DENUE Jalisco release 2022-11 (`mxcensus.load_denue`) aggregated at AGEB and locality level (sector shares via `denue_scian2` in `sector.yaml`, share of establishments with >10 employees, log count) with locality fallback; `destino_ambito` ∈ {ageb_urbana 21,070; localidad_rural 247; aeropuerto 301; fuera_zm 518; desconocido 4,777}. Prototype paired folds (known-sector rows): log loss 1.088 → 0.900 with the raw destination type, → 1.015 with DENUE only, → **0.873** with both (−0.216 ± 0.006). Shipped: held-out log loss A 1.076 → 0.839 [0.802, 0.872] (31% below marginal), accuracy 48.3% → 64.8%, macro-F1 0.386 → 0.592; calibration-in-the-large gaps ≤ 0.9 pp, ECE ≤ 0.025; imputed `gobierno_otro_agricultura` share 8.7% → 6.3% (delta factor 0.64). OD expected informality 36.28% → **35.26%** via the changed sector composition (superseded by 4.7; final 36.42%) (within services 45.8%, gobierno/otro/agric. 12.1%, manufactura 19.2%, comercio 41.0%).

### 4.5 Household income bracket ✅ evaluated — not shipped
- **Why:** ENOE informality by income level: 75% (≤1 MW) → 43% → 29% → 16% (3–5 MW). OD has `ingreso_mensual_hogar` (10 brackets incl. "No sabe"/refused).
- **Fix/risk:** ENOE labour income summed over the household roster and bracketed to the OD cut points (monthly pesos); ordinal with an explicit missing level. Comparability is imperfect (household total vs labour income; ~20–30% non-response in both). Ship only if the stage-3 distributions are close and the paired-fold gain is clear.

- **Evaluated (2026-08-27), rejected:** OD `ingreso_mensual_hogar` is missing for 48% of workers (41% refused, 7% don't know) and, where answered, differs from ENOE's household labour income (share above $25k: OD 3.8% vs ENOE 15.3%; ENOE labour income is zero/unspecified for 34% of the employed). The informality gradient is real (ENOE 88% at $1.5–3k → 40% above $25k) but the measurement mismatch and the non-response would make it a shift hazard on the strongest-looking feature. Not added.

### 4.6 Raking the OD to ENOE margins (reporting) ✅ done
- **Fix:** rake the OD expansion factors to the ENOE metro margins on the shared covariates and report the informality rate under both weightings, as the mirror image of the 2.7 decomposition. No change to shipped outputs.

- **Done (2026-08-27):** notebook 05 rakes the OD expansion factors to the ENOE metro profile (density-ratio weights on sex, occupation, age, education, municipality, marital status, household size, place of work; ESS 6,297 of 26,913). OD expected informality 35.26% as surveyed → **38.39%** raked, vs the ENOE metro benchmark 39.35%: once composition is equalized the model reproduces ENOE to within 1 pp, consistent with the 2.7 decomposition from the other side.

### 4.7 Conditional marginalization for workers without a work trip ✅ done
- **Why:** 3,290 OD workers (13.1% weighted) have no work trip on the survey day and therefore `lugar_trabajo = no_especificado`; they are marginalized over ENOE's *unconditional* place-of-work shares (mean P(informal) 0.41) whereas scoring them all as home/mobile workers gives 0.59 — a 35.3% vs 43.0% band on the headline, now the largest single uncertainty. ENOE observes the place of work for everyone.
- **Fix:** fit an auxiliary model P(lugar_trabajo | x) on ENOE; marginalize each no-trip worker with its own conditional distribution (row-specific shares in `predict_proba_marginalizing`). Same for `destino_trabajo = no_especificado` in the sector model. Report the headline under unconditional vs conditional marginalization.

- **Done (2026-08-28):** `fit_level_model` / `predict_level_shares` (`src/common.py`); `predict_proba_marginalizing(conditional_shares=...)` accepts row-specific level probabilities; `fit_workplace_models` (ENOE, P(lugar_trabajo | x) per specification; held-out log loss 0.586 vs marginal 1.183, accuracy 78%) used by `predict_od_informality(workplace_models=...)`; `fit_destination_models` (OD workers with a trip, P(destino_trabajo | x)) used by `impute_missing_sectors_hybrid(destination_models=...)`; both stored in the bundles. **Bug found on the way:** ENOE has 0.4% `lugar_trabajo = no_especificado`, so the marginalizer had treated the OD's "no work trip" as a supported level and never marginalized it — the missing label is now excluded from the supported set (shares renormalized), which supersedes the 4.2/4.4 headlines. Result for the 3,290 no-trip workers (13.1% weighted): mean P(informal) 0.518 with unconditional shares, **0.495 conditional (shipped)**, 0.591 if all were home/mobile; headline 36.79% / **36.42%** / 42.97%. Sector imputation composition unchanged (imputed gobierno share 6.3%). OD raked to the ENOE profile 38.42% vs benchmark 39.35%.

## Execution order and verification

1. Phase 0 → commit. 2. Phase 1 in one branch; run 01→02; `compare_outputs outputs_ref_849568a outputs --stage 2` and inspect each expected shift (1.1, 1.2, 1.7) — differences must be explainable item by item. 3. Phase 2 + re-run 03→05; compare `--stage all`; the calibration and masked-validation tables are the acceptance evidence, not distribution matching alone. 4. Phase 3 can be interleaved but must not change numbers (verify with `compare_outputs` = zero diff). Each phase gets its own commit with the `compare_outputs` report pasted into the message.

## Decisions required before Phase 1 **[D]**
| # | Question | Recommendation |
|---|---|---|
| 1.1 | SCIAN 2 (Minería) and 3 (Electricidad) → which class? | `manufactura_construccion` |
| 1.3 | Exclude the 194 self-reported non-workers, or keep as `no_especificado`? | exclude |
| 1.6 | OD municipalities absent from ENOE: recode to `otro` or drop? | recode to `otro`, report share |
| 1.11 | ENOE unspecified-sector workers: drop (report) or keep as 5th level? | keep as 5th level; benchmark on all sectors |
