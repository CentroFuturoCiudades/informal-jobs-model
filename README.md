# Classification Model for Formal and Informal Jobs

## 1. Overview

This repository estimates whether workers in the Guadalajara Metropolitan Area Origin-Destination Survey are formally or informally employed using machine learning models trained on data from the ENOE. The process harmonizes variables between the two surveys, reconstructs the missing economic sector in the OD survey, and calculates the probability of informality for each worker.

## 2. Data

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

## 3. Methodology

### 3.1 Database structure
In this step, we load the databases corresponding to the ENOE and the OD. We then filter the data to work exclusively with employed individuals and with observations corresponding to the area of interest (Jalisco). Additionally, for the ENOE, we calculate household size by counting the number of people associated with each household using their identifiers, while for the OD, this information is imported directly from the housing database.

**Notebook**: `01_generate_enoe_od_base_dataframes.ipynb`
**Module:** `generate_enoe_od_dataframes.py`

### 3.2 Data Standarization
In this section, we harmonize the gender, occupation, age, educational level, municipality, marital status, relationship to the head of household, household size, and economic sector attributes between the ENOE and OD datasets. Additionally, the informality label available in ENOE is standardized for its subsequent use. This step creates a common set of attributes across both datasets, which is necessary for their comparison and for the subsequent training and application of the Machine Learning models.

**Notebook**: `02_harmonize_enoe_od.ipynb`
**Module:** `harmonize_enoe_od_dataframes.py`

### 3.3 ENOE–OD Diagnostics
The next step is to conduct a general diagnostic analysis of the variables that will be used as predictors in the Machine Learning models. After restricting both surveys to their common geographic coverage, we compare the weighted distributions of the harmonized attributes using their corresponding expansion factors. We also analyze how informality varies across predictor categories in ENOE, quantify missing or unspecified information in both datasets, and examine the economic sector separately due to its high proportion of unknown values in OD.

**Notebook**: `03_diagnose_enoe_od_dataframes.ipynb`
**Module:** `diagnose_enoe_od_dataframes.py`

### 3.4 Economic Sector Assignment
Next, three Machine Learning classifiers (Logistic Regression, Random Forest, and Histogram Gradient Boosting) are evaluated to predict the economic sector of OD workers with missing sector information. Two model specifications are considered, with and without educational level, and the best model from each specification is combined into a hybrid strategy. Workers with available education are classified using the education-based model, while workers without this information use the robust alternative. The final dataset includes both an assigned economic sector and the predicted probabilities for all four sector categories.

**Notebook**: `04_impute_economic_sector.ipynb`
**Module:** `impute_economic_sector.py`

### 3.5 Informality Classification

## 4. Results and Outputs

## 5. Repository Structure

## 6. Usage