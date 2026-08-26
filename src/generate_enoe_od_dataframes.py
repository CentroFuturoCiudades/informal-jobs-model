from pathlib import Path
import pandas as pd

def load_enoe_tables(enoe_directory, period="T123"):
    enoe_directory = Path(enoe_directory)
    housing = pd.read_csv(enoe_directory / f"ENOE_VIV{period}.csv", encoding="latin1", low_memory=False)
    households = pd.read_csv(enoe_directory / f"ENOE_HOG{period}.csv", encoding="latin1", low_memory=False)
    sociodemographic = pd.read_csv(enoe_directory / f"ENOE_SDEM{period}.csv", encoding="latin1", low_memory=False)
    questionnaire_1 = pd.read_csv(enoe_directory / f"ENOE_COE1{period}.csv", encoding="latin1", low_memory=False)
    questionnaire_2 = pd.read_csv(enoe_directory / f"ENOE_COE2{period}.csv", encoding="latin1", low_memory=False)

    return housing, households, sociodemographic, questionnaire_1, questionnaire_2

def load_od_population(od_directory):
    od_directory = Path(od_directory)
    population = pd.read_csv(od_directory / "IMEPLAN_Base_Habitantes_Master.csv", encoding="latin1", low_memory=False)

    return population

def load_od_households(od_directory):
    od_directory = Path(od_directory)
    households = pd.read_csv(od_directory / "IMEPLAN_Base_Viviendas_Master.csv", encoding="latin1", low_memory=False)

    return households

def load_od_trips(od_directory):
    od_directory = Path(od_directory)
    trips = pd.read_csv(od_directory / "IMEPLAN_Base_Viajes_Master.csv", encoding="latin1", low_memory=False)

    return trips

def generate_enoe_dataframe(housing, households, sociodemographic, questionnaire_1, questionnaire_2, state_code=14):
    dwelling_keys = ["tipo", "mes_cal", "cd_a", "ent", "con", "v_sel"]
    household_keys = dwelling_keys + ["n_hog", "h_mud"]
    person_keys = household_keys + ["n_ren"]
    enoe = housing.merge(households, on=dwelling_keys, how="right", suffixes=("_housing", "_household"))
    enoe = enoe.merge(sociodemographic, on=household_keys, how="right", suffixes=("", "_sociodemographic"))
    enoe = enoe.merge(questionnaire_1, on=person_keys, how="left", suffixes=("", "_questionnaire_1"))
    enoe = enoe.merge(questionnaire_2, on=person_keys, how="left", suffixes=("", "_questionnaire_2"))
    enoe["dwelling_size"] = enoe.groupby(dwelling_keys)["n_ren"].transform("size") # Compute dwelling size as the number of individuals per dwelling
    enoe = enoe[(enoe["ent"] == state_code) & (enoe["p1_questionnaire_1"] == 1)].copy()
    columns = person_keys + ["mun", "est", "upm", "fac_tri", "sex", "pos_ocu", "scian", "eda", "cs_p13_1", "emp_ppal", "e_con", "par_c", "dwelling_size"]
    enoe = enoe[columns].copy()
    enoe = enoe.rename(columns={"fac_tri": "survey_weight", "est": "survey_stratum", "upm": "survey_psu"})

    return enoe.reset_index(drop=True)

def generate_od_dataframe(population, households):
    dwelling_size_column = "Incluyéndolo, ¿cuántas personas viven permanentemente en su vivienda contando a los bebés y personas adultas mayores?"
    employment_status_column = "Durante la semana pasada trabajó:"
    employed_categories = ["Tiempo completo", "Medio tiempo", "Tenía trabajo, pero no trabajó"]
    od = population.merge(households[["Folio Vivienda", dwelling_size_column]], on="Folio Vivienda", how="left", validate="many_to_one")
    od = od[od[employment_status_column].isin(employed_categories)].copy()
    od = od.rename(columns={"Ponderador": "expansion_factor", dwelling_size_column: "dwelling_size"})

    return od.reset_index(drop=True)