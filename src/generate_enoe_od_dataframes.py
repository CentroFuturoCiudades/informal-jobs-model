"""Stage 1: build the worker-level base dataframes from the packaged surveys.

ENOE comes from ``mxcensus`` (national labour survey, one quarter per call) and the Guadalajara Origin-Destination
survey from ``eodgdl``. Both packages download and cache the raw tables on first use (see README, "Data access").
All source-specific constants (periods, keys, column lists, renames) live in ``src/config/enoe.yaml`` and
``src/config/od.yaml``; this module derives the composite lists from them.
"""

import eodgdl
import mxcensus
import pandas as pd
from mxcensus.enoe import _DWELLING_KEY_SPEC, _level_key

from .common import load_config
from .destination_features import add_destination_features

# ENOE
_ENOE = load_config("enoe")
ENOE_PERIODS = list(_ENOE["periods"])
ENOE_PERIOD = _ENOE["reference_period"]
ENOE_STATE_CODE = _ENOE["state_code"]
ENOE_MIN_AGE, ENOE_MAX_AGE = _ENOE["min_age"], _ENOE["max_age"]
ENOE_DWELLING_KEYS = list(_ENOE["keys"]["dwelling"])
ENOE_HOUSEHOLD_KEYS = ENOE_DWELLING_KEYS + _ENOE["keys"]["household"]
ENOE_PERSON_KEYS = ENOE_HOUSEHOLD_KEYS + _ENOE["keys"]["person"]
# Cross-quarter identifiers: person rows are unique with the period added.
ENOE_GROUP_KEYS = [
    k for k in ENOE_HOUSEHOLD_KEYS if k not in _ENOE["keys"]["panel_visit"]
]
ENOE_ROW_KEYS = ["period"] + ENOE_PERSON_KEYS
ENOE_RENAMES = dict(_ENOE["renames"])
_COLUMNS = _ENOE["columns"]
ENOE_WORKPLACE_COLUMNS = list(_COLUMNS["workplace"])
ENOE_ROSTER_COLUMNS = list(_COLUMNS["roster"])
ENOE_COMPONENT_COLUMNS = list(_COLUMNS["components"])
ENOE_WEIGHT_COLUMNS = list(_COLUMNS["weights"])
ENOE_OUTPUT_COLUMNS = (
    ENOE_PERSON_KEYS
    + _COLUMNS["design"]
    + ENOE_WEIGHT_COLUMNS
    + _COLUMNS["attributes"]
    + ENOE_WORKPLACE_COLUMNS
    + ENOE_ROSTER_COLUMNS
    + ENOE_COMPONENT_COLUMNS
)
ENOE_CODE_COLUMNS = [c for c in ENOE_OUTPUT_COLUMNS if c not in ENOE_WEIGHT_COLUMNS]
# Habitual (1) or new (3) residents; persons who moved out are not counted.
ENOE_RESIDENT_CODES = ["1", "3"]
# Children counted in the household roster: the OD roster starts at age 6, so younger children are not comparable.
ROSTER_CHILD_AGES = (6, 11)

# OD
_OD = load_config("od")
OD_EMPLOYED_CATEGORIES = list(_OD["employed_categories"])
OD_DWELLING_COLUMNS = list(_OD["dwelling_columns"])
OD_WORK_TRIP_PURPOSE = _OD["work_trip_purpose"]
OD_RAW_COLUMN_RENAMES = dict(_OD["raw_column_renames"])
OD_RENAMES = {**_OD["renames"], **OD_RAW_COLUMN_RENAMES}


def assert_enoe_dwelling_key(frame, period):
    """The configured keys are only valid for the eras where mxcensus resolves the same dwelling key."""
    resolved = _level_key(_DWELLING_KEY_SPEC, frame)
    if sorted(resolved) != sorted(ENOE_DWELLING_KEYS):
        raise ValueError(
            f"ENOE period {period!r} resolves the dwelling key to {resolved}, but this pipeline expects {ENOE_DWELLING_KEYS}"
        )


def compute_enoe_dwelling_size(period, state_code=ENOE_STATE_CODE):
    """Per-dwelling and per-household roster aggregates from the full SDEM roster of habitual/new residents:
    ``dwelling_size`` (all ages, all households of the dwelling), ``hogar_trabajadores`` (employed members of the
    household, ``clase2 == 1``) and ``hogar_ninos_6_11`` (household members aged 6-11)."""
    sdem = mxcensus.load_enoe(table="sdem", period=period, ent=state_code)
    assert_enoe_dwelling_key(sdem, period)
    sdem = sdem[sdem["c_res"].isin(ENOE_RESIDENT_CODES)].copy()
    sdem["_employed"] = (sdem["clase2"] == "1").astype(int)
    sdem["_child"] = (
        pd.to_numeric(sdem["eda"], errors="coerce")
        .between(*ROSTER_CHILD_AGES)
        .astype(int)
    )
    dwelling_size = (
        sdem.groupby(ENOE_DWELLING_KEYS).size().rename("dwelling_size").reset_index()
    )
    household = (
        sdem.groupby(ENOE_HOUSEHOLD_KEYS)
        .agg(
            hogar_trabajadores=("_employed", "sum"), hogar_ninos_6_11=("_child", "sum")
        )
        .reset_index()
    )

    return dwelling_size.merge(
        household, on=ENOE_DWELLING_KEYS, how="left", validate="one_to_many"
    )


def load_enoe_employed_persons(period, state_code=ENOE_STATE_CODE):
    """Employed persons from ``mxcensus.load_enoe_persons`` (SDEM joined with COE1/COE2): INEGI's employed
    population (``clase2 == 1``) on the analytical universe ``r_def == 0``, ``c_res in {1, 3}`` and ``eda`` in
    [ENOE_MIN_AGE, ENOE_MAX_AGE] (see ``src/config/enoe.yaml``)."""
    persons = mxcensus.load_enoe_persons(
        period=period, ent=state_code, canonical_filter=False
    )
    assert_enoe_dwelling_key(persons, period)
    age = pd.to_numeric(persons["eda"], errors="coerce")
    employed = (
        (pd.to_numeric(persons["r_def"], errors="coerce") == 0)
        & persons["c_res"].isin(ENOE_RESIDENT_CODES)
        & age.between(ENOE_MIN_AGE, ENOE_MAX_AGE)
        & (persons["clase2"] == "1")
    )

    return persons[employed.fillna(False).astype(bool)].copy()


def generate_enoe_dataframe(periods=ENOE_PERIODS, state_code=ENOE_STATE_CODE):
    """Employed ENOE workers for the pooled quarters ``periods``.

    The frame gets a ``period`` column and ``survey_weight`` is divided by the number of quarters, so weighted
    totals remain at population scale (the average quarterly population) and per-quarter rates can still be
    recovered by multiplying back.
    """
    periods = list(periods)
    frames = [
        _generate_enoe_quarter(quarter, state_code).assign(period=quarter)
        for quarter in periods
    ]
    enoe = pd.concat(frames, ignore_index=True)
    enoe["survey_weight"] = enoe["survey_weight"] / len(periods)
    enoe = enoe[["period"] + ENOE_OUTPUT_COLUMNS]
    assert not enoe.duplicated(ENOE_ROW_KEYS).any(), (
        "ENOE person rows must be unique within a quarter"
    )

    return enoe


def _generate_enoe_quarter(period, state_code):
    enoe = load_enoe_employed_persons(period=period, state_code=state_code)
    dwelling_size = compute_enoe_dwelling_size(period=period, state_code=state_code)
    enoe = enoe.merge(
        dwelling_size, on=ENOE_HOUSEHOLD_KEYS, how="left", validate="many_to_one"
    )
    assert enoe["dwelling_size"].notna().all(), (
        "Every employed person must belong to a dwelling in SDEM"
    )
    enoe = enoe.rename(columns=ENOE_RENAMES)[ENOE_OUTPUT_COLUMNS].copy()
    # mxcensus returns raw INEGI codes as strings (blanks for missing); stage 2 maps integer codes. Weights stay float
    # (fac_tri is integral in 2023t1 but need not be in other quarters).
    for column in ENOE_CODE_COLUMNS:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("Int64")
    for column in ENOE_WEIGHT_COLUMNS:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("float64")
    assert enoe[ENOE_WEIGHT_COLUMNS].notna().all().all(), (
        "ENOE survey weights contain missing values"
    )

    return enoe.reset_index(drop=True)


# OD
def compute_od_work_trip_destination(trips):
    """Most frequent destination type of each person's work trips (``tipo_lugar_destino`` of trips with purpose
    "Trabajar"); persons without a work trip on the survey day are absent (review item 4.2)."""
    work_trips = trips.reset_index()
    work_trips = work_trips[work_trips["motivo_viaje"] == OD_WORK_TRIP_PURPOSE]
    mode = lambda values: values.astype(str).value_counts().index[0]
    destination = work_trips.groupby(["folio_vivienda", "folio_habitante"]).agg(
        destino_trabajo=("tipo_lugar_destino", mode),
        destino_cvegeo=("destino", mode),
        destino_zona=("zona_destino", mode),
        modo_trabajo=("modo_principal", mode),
    )

    return destination.reset_index()


def compute_od_household_roster(population):
    """Household roster aggregates comparable with ENOE's: employed members and children aged 6-11 (4.3)."""
    roster = population.copy()
    roster["_employed"] = (
        roster["trabajo_semana_pasada"].isin(OD_EMPLOYED_CATEGORIES).astype(int)
    )
    roster["_child"] = (
        pd.to_numeric(roster["edad"], errors="coerce")
        .between(*ROSTER_CHILD_AGES)
        .astype(int)
    )

    return (
        roster.groupby("folio_vivienda")
        .agg(
            hogar_trabajadores=("_employed", "sum"), hogar_ninos_6_11=("_child", "sum")
        )
        .reset_index()
    )


def generate_od_dataframe():
    tables = eodgdl.load_eod()
    population = tables.hab.reset_index()
    households = tables.viv[OD_DWELLING_COLUMNS]
    population = population.merge(
        compute_od_household_roster(population),
        on="folio_vivienda",
        how="left",
        validate="many_to_one",
    )
    population = population.merge(
        compute_od_work_trip_destination(tables.trips),
        on=["folio_vivienda", "folio_habitante"],
        how="left",
        validate="one_to_one",
    )
    od = population.merge(
        households,
        left_on="folio_vivienda",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    suffixed = [column for column in od.columns if column.endswith(("_x", "_y"))]
    assert not suffixed, (
        f"Person and dwelling tables share columns; eodgdl schema changed: {suffixed}"
    )
    missing = sorted(set(OD_RENAMES) - set(od.columns))
    assert not missing, (
        f"Expected eodgdl columns are missing (schema changed?): {missing}"
    )
    od = od[od["trabajo_semana_pasada"].isin(OD_EMPLOYED_CATEGORIES)].copy()
    od = od.rename(columns=OD_RENAMES)
    od = add_destination_features(od)
    # eodgdl delivers pandas Categoricals; plain strings are simpler for mapping, sklearn and parquet.
    categorical_columns = od.columns[od.dtypes.eq("category")]
    od[categorical_columns] = od[categorical_columns].astype("string")

    return od.reset_index(drop=True)
