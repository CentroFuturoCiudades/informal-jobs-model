"""Stage 1: build the worker-level base dataframes from the packaged surveys.

ENOE comes from ``mxcensus`` (national labour survey, one quarter) and the
Guadalajara Origin-Destination survey from ``eodgdl``. Both packages download
and cache the raw tables on first use (see README, "Data access").
"""
import eodgdl
import mxcensus
import pandas as pd


# ENOE
ENOE_PERIOD = "2023t1"
ENOE_STATE_CODE = 14  # Jalisco
ENOE_DWELLING_KEYS = ["tipo", "mes_cal", "cd_a", "ent", "con", "v_sel"]
ENOE_HOUSEHOLD_KEYS = ENOE_DWELLING_KEYS + ["n_hog", "h_mud"]
ENOE_PERSON_KEYS = ENOE_HOUSEHOLD_KEYS + ["n_ren"]
ENOE_RENAMES = {"fac_tri": "survey_weight", "est": "survey_stratum", "upm": "survey_psu"}
ENOE_OUTPUT_COLUMNS = ENOE_PERSON_KEYS + ["mun", "survey_stratum", "survey_psu", "survey_weight", "sex", "pos_ocu", "scian", "eda", "cs_p13_1", "emp_ppal", "e_con", "par_c", "dwelling_size"]
ENOE_EMPLOYMENT_FILTERS = ("p1", "clase2")

# OD
OD_EMPLOYED_CATEGORIES = ["Tiempo completo", "Medio tiempo", "Tenía trabajo, pero no trabajó"]
OD_DWELLING_COLUMNS = ["municipio", "ageb", "centralidad", "personas_en_vivienda"]
# Raw eodgdl columns whose names collide with the harmonized columns created in stage 2.
OD_RAW_COLUMN_RENAMES = {
    "ocupacion": "ocupacion_raw",
    "escolaridad": "escolaridad_raw",
    "municipio": "municipio_raw",
    "estado_civil": "estado_civil_raw",
    "parentesco": "parentesco_raw",
}
OD_RENAMES = {"ponderador": "expansion_factor", "personas_en_vivienda": "dwelling_size", **OD_RAW_COLUMN_RENAMES}


def compute_enoe_dwelling_size(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE):
    """Number of persons per dwelling, counted over the full SDEM roster (all ages, all residents)."""
    sdem = mxcensus.load_enoe(table="sdem", period=period, ent=state_code)
    dwelling_size = sdem.groupby(ENOE_DWELLING_KEYS).size().rename("dwelling_size").reset_index()

    return dwelling_size

def load_enoe_employed_persons(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE, employment_filter="p1"):
    """Employed persons from ``mxcensus.load_enoe_persons`` (SDEM joined with COE1/COE2).

    ``employment_filter``:
      - ``"p1"``: no residency/age filter; keep persons with COE1 ``p1 == 1`` (worked at least one hour last week).
      - ``"clase2"``: mxcensus canonical universe (r_def==0, residents, age 15-98) and ``clase2 == 1`` (INEGI employed).
    """
    if employment_filter == "p1":
        persons = mxcensus.load_enoe_persons(period=period, ent=state_code, canonical_filter=False)
        employed = persons["p1"] == "1"
    elif employment_filter == "clase2":
        persons = mxcensus.load_enoe_persons(period=period, ent=state_code, canonical_filter=True)
        employed = persons["is_ocupado"]
    else:
        raise ValueError(f"employment_filter must be one of {ENOE_EMPLOYMENT_FILTERS}, got {employment_filter!r}")

    return persons[employed.fillna(False).astype(bool)].copy()

def generate_enoe_dataframe(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE, employment_filter="p1"):
    enoe = load_enoe_employed_persons(period=period, state_code=state_code, employment_filter=employment_filter)
    dwelling_size = compute_enoe_dwelling_size(period=period, state_code=state_code)
    enoe = enoe.merge(dwelling_size, on=ENOE_DWELLING_KEYS, how="left", validate="many_to_one")
    assert enoe["dwelling_size"].notna().all(), "Every employed person must belong to a dwelling in SDEM"
    enoe = enoe.rename(columns=ENOE_RENAMES)[ENOE_OUTPUT_COLUMNS].copy()
    # mxcensus returns raw INEGI codes as strings (blanks for missing); stage 2 maps integer codes.
    for column in ENOE_OUTPUT_COLUMNS:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("Int64")

    return enoe.reset_index(drop=True)

# OD
def generate_od_dataframe(eod_path=None):
    tables = eodgdl.load_eod(eod_path)
    population = tables.hab.reset_index()
    households = tables.viv[OD_DWELLING_COLUMNS]
    od = population.merge(households, left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    od = od[od["trabajo_semana_pasada"].isin(OD_EMPLOYED_CATEGORIES)].copy()
    od = od.rename(columns=OD_RENAMES)
    # eodgdl delivers pandas Categoricals; plain strings are simpler for mapping, sklearn and parquet.
    categorical_columns = od.columns[od.dtypes.eq("category")]
    od[categorical_columns] = od[categorical_columns].astype("string")

    return od.reset_index(drop=True)
