<div align="center">

# Classification Model for Formal and Informal Jobs

This repository estimates whether workers in the Guadalajara Metropolitan Area Origin-Destination Survey are formally or informally employed using machine learning models trained on data from the ENOE. The process harmonizes variables between the two surveys, reconstructs the missing economic sector in the OD survey, and calculates the probability of informality for each worker.

</div>

## 1. Data

The sources of information used throughout the methodology were as follows:

- **ENOE**: provides sociodemographic and employment information on employed individuals. Among the main columns are:
    - `ent`: state
    - `mun`: municipality
    - `loc`: locality
    - `ageb`: basic geostatistical area.
    - `sex`: gender
    - `eda`: age
    - `niv_ins`: educational level
    - `anios_esc`: years of schooling
    - `e_con`: marital status
    - `ur`: type of locality, urban or rural.
    - `clase1`: economic activity status.
    - `pos_ocu`: position in the occupation.
    - `emp_ppal`: characteristics of the main job.
    - `ingocup`: income from work.
    - `hrsocup`: hours worked.
    - `tue1`: size of the economic unit.
    - `formal`: formal or informal employment status, used as a target variable for training the models.

- **Origin-Destination Survey**: provides sociodemographic, employment, and mobility data on the population of the Guadalajara Metropolitan Area. The main columns include:
    - `id_persona`: identifier of the survey respondent
    - `id_hogar`: household identifier.
    - `factor_expansion`: survey expansion factor
    - `sexo`: gender
    - `edad`: age
    - `escolaridad`: educational level
    - `ocupacion`: employment status or occupation
    - `sector_actividad`: economic sector
    - `ingreso`: individual or household income
    - `municipio_origen`: municipality where the trip begins
    - `municipio_destino`: destination municipality
    - `zona_origen`: area of origin of the trip
    - `zona_destino`: destination area of the trip
    - `motivo_viaje`: main reason for the trip
    - `modo_transporte`: mode of transportation used
    - `tiempo_viaje`: duration of the trip
    - `hora_inicio`: time the trip began
    - `hora_fin`: time the trip ends
    - `frecuencia_viaje`: frequency with which the trip is made

This survey does not directly identify informal employment status; therefore, this variable must be estimated or assigned using information from the ENOE and the variables shared between both sources.

The unit of analysis is the employed individuals or workers who appear in both data sources. The linkage is performed using available common variables, such as sex, age, educational level, municipality, employment status, and characteristics of employment or mobility.

**Note:** The original files from both surveys are not necessarily included in this GitHub repository due to their size.

## 2. Methodology

### 2.1 Database structure
In this step, we load the databases corresponding to the ENOE and the OD. We then filter the data to work exclusively with employed individuals and with observations corresponding to the area of interest (Jalisco). Additionally, for the ENOE, we calculate household size by counting the number of people associated with each household using their identifiers, while for the OD, this information is imported directly from the housing database.

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

**Notebook**: `04_impute_economic_sector.ipynb`
**Module:** `impute_economic_sector.py`

### 2.5 Informality Classification
Analogously to the economic-sector assignment stage, we evaluate three Machine Learning algorithms to estimate formal and informal employment among workers in the Origin-Destination survey, using ENOE as the training source. A hybrid strategy is adopted: when educational information is available, the model including education is used; otherwise, a robust specification excluding education is applied.

The final output is probabilistic. For each worker, the model estimates the probability of informal employment, while uncertainty in the economic sector is propagated by marginalizing over the sector probabilities obtained in the previous stage. For comparison, a hard classification is also generated using a 50% threshold, assigning workers as informal when $P(\text{informal}) \geq 0.5$. However, the expected probabilistic approach is retained as the main population-level estimate because it preserves the uncertainty of the predictions.

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

- `genero`
- `ocupacion`
- `edad_num`
- `escolaridad`
- `municipio`
- `estado_civil`
- `parentesco`
- `tamano_viv_cat`
- `sector`

If the economic sector is unavailable, it must first be estimated using the sector model. Direct application of this model additionally requires the OD-specific predictors defined in `src.OD_SECTOR_FEATURES` and `src.OD_ROBUST_SECTOR_FEATURES`.

All categorical variables should use the same categories established during the harmonization stage.

### 4.3 Recommended outputs

For downstream applications, the main variables to retain are:

- `prob_sector_comercio`
- `prob_sector_gobierno_otro_agricultura`
- `prob_sector_manufactura_construccion`
- `prob_sector_servicios_transporte`
- `sector_final`
- `prob_informal`
- `informal_predicted`

The primary informality output is `prob_informal`. If a discrete formal/informal status is required, it can either be obtained using the 50% classification threshold or sampled probabilistically as

$$I_i \sim \operatorname{Bernoulli}(p_i),$$

where $p_i$ is the value of `prob_informal` for worker $i$.

For population-generation or simulation applications, probabilistic sampling is recommended because it preserves the uncertainty estimated by the model and reproduces the expected aggregate informality rate more naturally than a fixed classification threshold.