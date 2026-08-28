"""Stage 1: build the worker-level base dataframes from the packaged surveys.

ENOE comes from ``mxcensus`` (national labour survey, one quarter) and the
Guadalajara Origin-Destination survey from ``eodgdl``. Both packages download
and cache the raw tables on first use (see README, "Data access").
"""
import eodgdl
import mxcensus
import pandas as pd
from mxcensus.enoe import _DWELLING_KEY_SPEC, _level_key


# ENOE
ENOE_PERIOD = "2023t1"                      # reference quarter (matches the OD fieldwork)
ENOE_PERIODS = ["2022t1", "2022t2", "2022t3", "2022t4", "2023t1", "2023t2", "2023t3", "2023t4"]  # quarters pooled for training (review item 4.1)
ENOE_STATE_CODE = 14  # Jalisco
ENOE_DWELLING_KEYS = ["tipo", "mes_cal", "cd_a", "ent", "con", "v_sel"]
ENOE_HOUSEHOLD_KEYS = ENOE_DWELLING_KEYS + ["n_hog", "h_mud"]
ENOE_PERSON_KEYS = ENOE_HOUSEHOLD_KEYS + ["n_ren"]
# est_d_tri is the sampling-design stratum (18 strata in Jalisco 2023t1); est is INEGI's socio-economic stratum (4 levels).
ENOE_RENAMES = {"fac_tri": "survey_weight", "est_d_tri": "survey_stratum", "upm": "survey_psu", "est": "estrato_socioeconomico"}
ENOE_CODE_COLUMNS = ENOE_PERSON_KEYS + ["mun", "survey_stratum", "survey_psu", "estrato_socioeconomico", "sex", "pos_ocu", "scian", "eda", "cs_p13_1", "emp_ppal", "e_con", "par_c", "dwelling_size"]
ENOE_WEIGHT_COLUMNS = ["survey_weight"]
ENOE_OUTPUT_COLUMNS = ENOE_CODE_COLUMNS[:len(ENOE_PERSON_KEYS) + 3] + ENOE_WEIGHT_COLUMNS + ENOE_CODE_COLUMNS[len(ENOE_PERSON_KEYS) + 3:]
# Cross-quarter identifiers: tipo/mes_cal distinguish the panel visits of the same dwelling, so grouping (CV folds,
# bootstrap) across pooled quarters must use the base keys; person rows are unique with the period added.
ENOE_GROUP_KEYS = ["cd_a", "ent", "con", "v_sel", "n_hog", "h_mud"]
ENOE_ROW_KEYS = ["period"] + ENOE_PERSON_KEYS
ENOE_EMPLOYMENT_FILTERS = ("clase2", "p1")
# Analytical universe for the "clase2" filter: definitive interview, habitual/new residents, age 12+ (INEGI uses 15+;
# 12 keeps the working 12-14 year olds that the OD survey also records). The upper bound 98 follows INEGI's own
# filter EDA in [15, 98]: 98 is the sentinel "age unspecified (12+)", kept in the universe and turned into a missing
# age by harmonize_enoe_age.
ENOE_MIN_AGE, ENOE_MAX_AGE = 12, 98

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


def assert_enoe_dwelling_key(frame, period):
    """The hard-coded keys are only valid for the eras where mxcensus resolves the same dwelling key."""
    resolved = _level_key(_DWELLING_KEY_SPEC, frame)
    if sorted(resolved) != sorted(ENOE_DWELLING_KEYS):
        raise ValueError(f"ENOE period {period!r} resolves the dwelling key to {resolved}, but this pipeline expects {ENOE_DWELLING_KEYS}")

def compute_enoe_dwelling_size(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE):
    """Number of persons per dwelling: every SDEM row of the dwelling (all ages, all households) that is a
    habitual or new resident (``c_res`` 1 or 3); persons who moved out (``c_res == 2``) are not counted."""
    sdem = mxcensus.load_enoe(table="sdem", period=period, ent=state_code)
    assert_enoe_dwelling_key(sdem, period)
    sdem = sdem[sdem["c_res"].isin(["1", "3"])]
    dwelling_size = sdem.groupby(ENOE_DWELLING_KEYS).size().rename("dwelling_size").reset_index()

    return dwelling_size

def load_enoe_employed_persons(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE, employment_filter="clase2"):
    """Employed persons from ``mxcensus.load_enoe_persons`` (SDEM joined with COE1/COE2).

    ``employment_filter``:
      - ``"clase2"`` (default): INEGI's employed population (``clase2 == 1``: working, or with a job but temporarily
        absent) on the analytical universe ``r_def == 0`` (definitive interview), ``c_res in {1, 3}`` (habitual or new
        resident) and ``eda`` in [ENOE_MIN_AGE, ENOE_MAX_AGE].
      - ``"p1"``: no residency/age filter; keep persons with COE1 ``p1 == 1`` (worked at least one hour last week).
        This was the definition of the original pipeline; it is a subset of ``"clase2"`` for 2023t1.
    """
    persons = mxcensus.load_enoe_persons(period=period, ent=state_code, canonical_filter=False)
    assert_enoe_dwelling_key(persons, period)
    if employment_filter == "clase2":
        age = pd.to_numeric(persons["eda"], errors="coerce")
        in_universe = (pd.to_numeric(persons["r_def"], errors="coerce") == 0) & persons["c_res"].isin(["1", "3"]) & age.between(ENOE_MIN_AGE, ENOE_MAX_AGE)
        employed = in_universe & (persons["clase2"] == "1")
    elif employment_filter == "p1":
        employed = persons["p1"] == "1"
    else:
        raise ValueError(f"employment_filter must be one of {ENOE_EMPLOYMENT_FILTERS}, got {employment_filter!r}")

    return persons[employed.fillna(False).astype(bool)].copy()

def generate_enoe_dataframe(period=ENOE_PERIOD, state_code=ENOE_STATE_CODE, employment_filter="clase2", periods=ENOE_PERIODS):
    """Employed ENOE workers for one quarter (``period``) or several pooled quarters (``periods``).

    When several quarters are pooled the frame gets a ``period`` column and ``survey_weight`` is divided by the number
    of quarters, so weighted totals remain at population scale (the average quarterly population) and per-quarter
    rates can still be recovered by multiplying back.
    """
    periods = list(periods) if periods else [period]
    frames = [_generate_enoe_quarter(quarter, state_code, employment_filter).assign(period=quarter) for quarter in periods]
    enoe = pd.concat(frames, ignore_index=True)
    enoe["survey_weight"] = enoe["survey_weight"] / len(periods)
    enoe = enoe[["period"] + ENOE_OUTPUT_COLUMNS]
    assert not enoe.duplicated(ENOE_ROW_KEYS).any(), "ENOE person rows must be unique within a quarter"

    return enoe

def _generate_enoe_quarter(period, state_code, employment_filter):
    enoe = load_enoe_employed_persons(period=period, state_code=state_code, employment_filter=employment_filter)
    dwelling_size = compute_enoe_dwelling_size(period=period, state_code=state_code)
    enoe = enoe.merge(dwelling_size, on=ENOE_DWELLING_KEYS, how="left", validate="many_to_one")
    assert enoe["dwelling_size"].notna().all(), "Every employed person must belong to a dwelling in SDEM"
    enoe = enoe.rename(columns=ENOE_RENAMES)[ENOE_OUTPUT_COLUMNS].copy()
    # mxcensus returns raw INEGI codes as strings (blanks for missing); stage 2 maps integer codes. Weights stay float
    # (fac_tri is integral in 2023t1 but need not be in other quarters).
    for column in ENOE_CODE_COLUMNS:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("Int64")
    for column in ENOE_WEIGHT_COLUMNS:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("float64")
    assert enoe[ENOE_WEIGHT_COLUMNS].notna().all().all(), "ENOE survey weights contain missing values"

    return enoe.reset_index(drop=True)

# OD
def generate_od_dataframe(eod_path=None):
    tables = eodgdl.load_eod(eod_path)
    population = tables.hab.reset_index()
    households = tables.viv[OD_DWELLING_COLUMNS]
    od = population.merge(households, left_on="folio_vivienda", right_index=True, how="left", validate="many_to_one")
    suffixed = [column for column in od.columns if column.endswith(("_x", "_y"))]
    assert not suffixed, f"Person and dwelling tables share columns; eodgdl schema changed: {suffixed}"
    missing = sorted(set(OD_RENAMES) - set(od.columns))
    assert not missing, f"Expected eodgdl columns are missing (schema changed?): {missing}"
    od = od[od["trabajo_semana_pasada"].isin(OD_EMPLOYED_CATEGORIES)].copy()
    od = od.rename(columns=OD_RENAMES)
    # eodgdl delivers pandas Categoricals; plain strings are simpler for mapping, sklearn and parquet.
    categorical_columns = od.columns[od.dtypes.eq("category")]
    od[categorical_columns] = od[categorical_columns].astype("string")

    return od.reset_index(drop=True)
