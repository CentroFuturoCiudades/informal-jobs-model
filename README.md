<div align="center">

# Classification Model for Formal and Informal Jobs

This repository estimates whether workers in the Guadalajara Metropolitan Area Origin-Destination Survey are formally or informally employed using machine learning models trained on data from the ENOE. The process harmonizes variables between the two surveys, reconstructs the missing economic sector in the OD survey, and calculates the probability of informality for each worker.

</div>

## 1. Data

Both surveys are loaded through the project's data packages, which download and cache the raw tables on first use, so no survey files need to be stored in this repository:

- **ENOE** (Encuesta Nacional de Ocupación y Empleo, INEGI) via [`mxcensus`](https://github.com/CentroFuturoCiudades/mxcensus). `mxcensus.load_enoe_persons(period=..., ent=14)` returns the sociodemographic roster (SDEM) joined with the two occupation questionnaires (COE1, COE2) for one quarter, restricted to Jalisco; the pipeline pools the eight quarters 2022-T1 to 2023-T4 (`src.ENOE_PERIODS`, ~50k workers; survey weights divided by the number of quarters so totals stay at the average quarterly population, panel visits grouped by the cross-quarter household key), with 2023-T1 as the reference quarter matching the OD fieldwork. The pipeline keeps (`src.ENOE_OUTPUT_COLUMNS`):
    - `tipo`, `mes_cal`, `cd_a`, `ent`, `con`, `v_sel`, `n_hog`, `h_mud`, `n_ren`: dwelling, household and person identifiers
    - `mun`: municipality
    - `survey_weight` (`fac_tri`), `survey_stratum` (`est_d_tri`, sampling-design stratum), `survey_psu` (`upm`): survey design variables; `estrato_socioeconomico` (`est`) is INEGI's socio-economic stratum
    - `sex`: gender
    - `eda`: age
    - `cs_p13_1`: educational level
    - `e_con`: marital status
    - `par_c`: relationship to the head of household
    - `pos_ocu`: position in the occupation
    - `scian`: economic sector (SCIAN grouping)
    - `emp_ppal`: formal or informal employment status, used as the target variable for training the models
    - `dwelling_size`: number of persons in the dwelling, counted over the full SDEM roster

- **Origin–Destination Survey** (IMEPLAN, Guadalajara Metropolitan Area, 2023) via [`eodgdl`](https://github.com/CentroFuturoCiudades/eodgdl). `eodgdl.load_eod()` returns dwellings (`viv`), persons (`hab`), trips and trip legs with snake_case column names. The pipeline joins persons with their dwelling attributes and keeps every person column; the ones used downstream are:
    - `folio_vivienda`, `folio_habitante`: dwelling and person identifiers
    - `expansion_factor` (`ponderador`): person-level survey expansion factor
    - `sexo_nacimiento`: gender
    - `edad`: age
    - `escolaridad_raw`: educational level
    - `estado_civil_raw`: marital status
    - `parentesco_raw`: relationship to the head of household
    - `ocupacion_raw`: occupation / employment position
    - `trabajo_semana_pasada`: employment status last week (used to select workers)
    - `giro_empresa`: economic sector of the employer (missing for most workers)
    - `municipio_raw`, `ageb`, `centralidad`: dwelling geography (from `viv`)
    - `destino_trabajo`, `destino_cvegeo`, `destino_zona`: type, INEGI code and zone of the most frequent work-trip destination (from the trips table); `destino_ambito` (urban AGEB / rural locality / airport / outside the metro zone / unknown) and the DENUE establishment mix at the destination (`dest_share_*`, `dest_share_grandes`, `dest_establecimientos_log`; DENUE release 2022-11 via `mxcensus`, codes resolved through `eodgdl.load_imeplan_agebs` and the Marco Geoestadístico) — used by the sector model only
    - `dwelling_size` (`personas_en_vivienda`): household size category

    Raw OD columns whose `eodgdl` names coincide with the harmonized attributes created in stage 2 (`ocupacion`, `escolaridad`, `municipio`, `estado_civil`, `parentesco`) carry a `_raw` suffix (`src.OD_RAW_COLUMN_RENAMES`); the unsuffixed name always refers to the harmonized attribute.

This survey does not directly identify informal employment status; therefore, this variable must be estimated or assigned using information from the ENOE and the variables shared between both sources.

The unit of analysis is the employed individuals or workers who appear in both data sources. The linkage is performed using available common variables, such as sex, age, educational level, municipality, employment status, and characteristics of employment or mobility.

**Data access.** Dependencies are pinned to package releases (`mxcensus` v0.2.0, `eodgdl` v0.1.0) in `pyproject.toml`. The packages cache their downloads under `~/Library/Caches/mxcensus` and `~/Library/Caches/eodgdl` (override with `MXCENSUS_CACHE_DIR`, `EODGDL_CACHE_DIR`).

## 2. Methodology

### 2.1 Database structure
In this step, we load the ENOE and OD tables through `mxcensus` and `eodgdl`. We then filter the data to work exclusively with employed individuals and with observations corresponding to the area of interest (Jalisco). For the ENOE, workers are INEGI's employed population (`clase2 == 1`: worked in the reference week or had a job and was temporarily absent) within the survey's analytical universe — definitive interview (`r_def == 0`), habitual or new residents (`c_res in {1, 3}`) — and ages 12 to 98 (INEGI reports employment for ages 15+; the floor is lowered to 12 because the OD survey records working 12–14 year olds). Stage-1 settings (pooled quarters, identifier keys, output columns, renames, OD employment categories, DENUE release) are declared in `src/config/enoe.yaml` and `src/config/od.yaml`. Additionally, for the ENOE, we calculate household size by counting the number of people associated with each dwelling using their identifiers, while for the OD, this information is imported directly from the dwelling table.

To validate changes to the data sources, `src/compare_outputs.py` compares the outputs of a run against a reference copy (`uv run python -m src outputs_baseline outputs`).

**Notebook**: `01_generate_enoe_od_base_dataframes.ipynb`
**Module:** `generate_enoe_od_dataframes.py`

### 2.2 Data Standarization
In this section, we harmonize the gender, occupation, age, educational level, municipality, marital status, relationship to the head of household, household size, and economic sector attributes between the ENOE and OD datasets. Additionally, the informality label available in ENOE is standardized for its subsequent use. This step creates a common set of attributes across both datasets, which is necessary for their comparison and for the subsequent training and application of the Machine Learning models.

**Notebook**: `02_harmonize_enoe_od.ipynb`
**Module:** `harmonize_enoe_od_dataframes.py`

### 2.3 ENOE–OD Diagnostics
The next step is to conduct a general diagnostic analysis of the variables that will be used as predictors in the Machine Learning models. After restricting both surveys to their common geographic coverage, we compare the weighted distributions of the harmonized attributes using their corresponding expansion factors. We also analyze how informality varies across predictor categories in ENOE, quantify missing or unspecified information in both datasets, and examine the economic sector separately due to its high proportion of unknown values in OD.

**Notebook**: `03_diagnose_enoe_od_dataframes.ipynb`
**Module:** `diagnose_enoe_od_dataframes.py`

### 2.4 Economic Sector Assignment
Next, three Machine Learning classifiers (Logistic Regression, Random Forest, and Histogram Gradient Boosting) are evaluated to predict the economic sector of OD workers with missing sector information. Two model specifications are considered, with and without educational level, and the best model from each specification is combined into a hybrid strategy. Workers with available education are classified using the education-based model, while workers without this information use the robust alternative. The final dataset includes both an assigned economic sector and the predicted probabilities for all four sector categories.

The imputation assumes that, conditional on the harmonized attributes, workers who did not report a sector are distributed across sectors like workers who did (missing at random given the covariates). The two populations differ — non-respondents are more educated and their missing sector co-occurs with other item non-response — so notebook 04 reports two sensitivity scenarios (a refit on training rows reweighted to the non-respondent profile, and a delta adjustment of the rare `gobierno_otro_agricultura` class to its observed share), and notebook 05 reports the informality headline under each. The shipped outputs use the unadjusted imputation.

**Notebook**: `04_impute_economic_sector.ipynb`
**Module:** `impute_economic_sector.py`

### 2.5 Informality Classification
Analogously to the economic-sector assignment stage, we evaluate three Machine Learning algorithms to estimate formal and informal employment among workers in the Origin-Destination survey, using ENOE as the training source. A hybrid strategy is adopted: when educational information is available, the model including education is used; otherwise, a robust specification excluding education is applied.

The final output is probabilistic. For each worker, the model estimates the probability of informal employment, while uncertainty in the economic sector is propagated by marginalizing over the sector probabilities obtained in the previous stage. The population-level estimate is the expectation of these probabilities; a discrete status is provided as one Bernoulli draw per worker (`informal_sampled`), which is unbiased for weighted aggregates. A 50% threshold classification (`informal_predicted`) is kept in the output for convenience but is not an estimate of informality: it shrinks toward the majority class and understates the rate by more than ten percentage points. The notebook also decomposes the gap between the ENOE benchmark and the OD estimate into model bias, composition (direct standardization of ENOE to the OD covariate profile) and a residual. ENOE 2023-T1 (January–March) and the OD fieldwork (23 January–29 April 2023) cover the same period.

**Notebook**: `05_informality_model.ipynb`
**Module:** `informality_model.py`

## 3. Results and Outputs

All generated results are stored in the `outputs/` directory. The pipeline produces harmonized datasets, probabilistically imputed OD datasets, serialized Machine Learning models, and diagnostic figures used to evaluate the different stages of the methodology.

### 3.1 Main datasets

| Output | Description |
|---|---|
| `outputs/enoe_workers.parquet` | Initial dataframe of employed ENOE workers obtained after merging and filtering the raw ENOE tables. |
| `outputs/od_workers.parquet` | Initial dataframe of employed workers from the Origin-Destination survey. |
| `outputs/enoe_harmonized.parquet` | ENOE worker dataset after harmonizing the attributes shared with the OD survey. |
| `outputs/od_harmonized.parquet` | OD worker dataset after harmonizing the attributes shared with ENOE. |
| `outputs/od_sector_imputed.parquet` | OD dataset after economic-sector imputation. Workers with missing sector information include both the final sector assignment and the predicted probabilities for the four economic-sector categories. |
| `outputs/od_informality_imputed.parquet` | Final OD dataset containing the estimated probability of informal employment for each worker. Sector uncertainty is propagated into the final informality probability through probabilistic marginalization. |

The two principal outputs of the Machine Learning pipeline are therefore:

- **`od_sector_imputed.parquet`**, which reconstructs the missing economic-sector information in the OD survey.
- **`od_informality_imputed.parquet`**, which provides the final probability of informal employment for every OD worker.

### 3.2 Trained models

The final fitted models are stored in `outputs/models/`:

| Output | Description |
|---|---|
| `economic_sector_hybrid_model.joblib` | Final hybrid economic-sector model, combining the specifications with and without education. |
| `informality_hybrid_model.joblib` | Final hybrid informality model trained on ENOE and used to estimate informality probabilities in OD. |

These files allow the final models to be loaded and applied without repeating the complete tuning and training procedure.

### 3.3 Diagnostic figures

Figures generated throughout the analysis are stored in `outputs/figures/` in both PDF and high-resolution PNG formats.

The main figures include:

- `weighted_distribution_comparison`: comparison of weighted predictor distributions between ENOE and OD.
- `informality_profiles`: weighted ENOE informality profiles across the harmonized predictors.
- `sector_known_unknown_profiles`: comparison between OD workers with known and unknown economic sector.
- `sector_comparison`: comparison of observed and modeled sector distributions.
- `sector_imputation_distributions`: hard and probabilistic sector-imputation diagnostics.
- `sector_initial_final_distribution`: OD economic-sector distribution before and after probabilistic imputation.
- `sector_final_distributions_confidence`: final sector distributions and model-prediction confidence.
- `informality_model_enoe_od_comparison`: final comparison between ENOE and OD informality estimates, including overall formal/informal distributions, probability calibration, and the distribution of informal employment across economic sectors.

## 4. Usage

The fitted models can be used to assign economic-sector and informality information to new worker-level datasets, provided that their attributes follow the same definitions and categories used during model training.

### 4.1 Required files

The main files for downstream applications are:

- `outputs/models/economic_sector_hybrid_model.joblib`: used when the economic sector is unknown.
- `outputs/models/informality_hybrid_model.joblib`: used to estimate the probability of informal employment.
- `outputs/od_informality_imputed.parquet`: final modeled OD dataset and reference output of the complete pipeline.

### 4.2 Required attributes

For informality prediction, the relevant harmonized worker attributes are primarily:

- `genero` (sex at birth in both surveys; the OD also records `genero_identidad`, which is not used so that the attribute matches ENOE)
- `ocupacion`
- `edad_num`
- `escolaridad`
- `municipio`
- `estado_civil`
- `parentesco`
- `tamano_viv_cat`
- `sector`
- `lugar_trabajo` (place of work: `establecimiento`, `comercio_o_puesto`, `otra_vivienda`, `otro_o_sin_local`; from the ENOE workplace questions and the OD work-trip destination)

If the economic sector is unavailable, it must first be estimated using the sector model. Direct application of this model additionally requires the OD-specific predictors defined in `src.OD_SECTOR_FEATURES` and `src.OD_ROBUST_SECTOR_FEATURES` (`ocupacion_raw`, `trabajo_semana_pasada`, `centralidad`, using the `eodgdl` category labels, the work-trip destination features `src.OD_DESTINATION_FEATURES`, built by `src.add_destination_features`, and the mobility features `src.OD_MOBILITY_FEATURES`).

All categorical variables should use the same categories established during the harmonization stage.

### 4.3 Recommended outputs

For downstream applications, the main variables to retain are:

- `prob_sector_comercio`
- `prob_sector_gobierno_otro_agricultura`
- `prob_sector_manufactura_construccion`
- `prob_sector_servicios_transporte`
- `sector_final`
- `prob_informal`
- `informal_sampled`
- `informal_predicted`

The primary informality output is `prob_informal`. If a discrete formal/informal status is required, it can either be obtained using the 50% classification threshold or sampled probabilistically as

$$I_i \sim \text{Bernoulli}(p_i),$$

where $p_i$ is the value of `prob_informal` for worker $i$.

For population-generation or simulation applications, probabilistic sampling is recommended because it preserves the uncertainty estimated by the model and reproduces the expected aggregate informality rate more naturally than a fixed classification threshold.