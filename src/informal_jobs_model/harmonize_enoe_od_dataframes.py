import functools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MAPPINGS_DIR = Path(__file__).parent / "mappings"


@functools.cache
def load_mapping(name):
    """Load a harmonization mapping (``src/mappings/<name>.yaml``) as a dict of dicts."""
    with open(MAPPINGS_DIR / f"{name}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


from .common import (
    AGE_LABELS,
    HOUSEHOLD_SIZE_CAP,
    HOUSEHOLD_SIZE_LABELS,
    NO_ESPECIFICADO,
    load_config,
)

_HARMONIZATION = load_config("harmonization")  # see src/config/harmonization.yaml
AGE_BINS = list(_HARMONIZATION["age"]["bins"]) + [np.inf]  # left-closed; last bin open
ENOE_AGE_UNSPECIFIED = _HARMONIZATION["age"]["enoe_unspecified"]
HOUSEHOLD_SIZE_BINS = list(_HARMONIZATION["household_size"]["bins"]) + [np.inf]
INFORMAL_SECTOR_TUE2 = list(_HARMONIZATION["informal_sector_tue2"])


def assert_mapping_covers(values, mapping, allowed_unmapped=(), name=None):
    """Fail loudly when a source category is neither mapped nor explicitly allowed to fall through.

    Every harmonizer falls back to ``no_especificado``/``otro``, so a new or renamed source category
    would otherwise degrade the data silently (new ENOE quarter, eodgdl revision).
    """
    observed = set(pd.Series(values).dropna().unique())
    unmapped = observed - set(mapping) - set(allowed_unmapped)
    if unmapped:
        raise ValueError(
            f"Unmapped categories in {name or getattr(values, 'name', 'series')}: {sorted(map(str, unmapped))}"
        )


# DATA TYPES
def prepare_enoe_data_types(enoe):
    """Stage 1 already delivers Int64 codes and float weights; only verify, so the cast lives in one place."""
    enoe = enoe.copy()
    columns = load_config("enoe")["columns"]
    code_columns = [
        column
        for group, names in columns.items()
        if group != "weights"
        for column in names
    ]
    not_integer = [
        column
        for column in code_columns
        if not pd.api.types.is_integer_dtype(enoe[column])
    ]
    assert not not_integer, (
        f"ENOE code columns must be integer-typed (stage 1 casts them): {not_integer}"
    )
    assert (
        pd.api.types.is_float_dtype(enoe["survey_weight"])
        and enoe["survey_weight"].notna().all()
    ), "survey_weight must be float without missing values"

    return enoe


def prepare_od_data_types(od):
    od = od.copy()
    numeric_columns = ["edad", "expansion_factor"]
    for column in numeric_columns:
        od[column] = pd.to_numeric(od[column], errors="coerce").astype("Int64")
    text_columns = [
        "sexo_nacimiento",
        "ocupacion_raw",
        "escolaridad_raw",
        "municipio_raw",
        "estado_civil_raw",
        "parentesco_raw",
        "giro_empresa",
        "dwelling_size",
    ]
    for column in text_columns:
        od[column] = od[column].astype("string").str.strip()

    return od


# GENDER
def harmonize_enoe_gender(enoe):
    enoe = enoe.copy()
    assert_mapping_covers(enoe["sex"], {1, 2})
    enoe["genero"] = enoe["sex"].map({1: "H", 2: "F"}).fillna(NO_ESPECIFICADO)

    return enoe


def harmonize_od_gender(od):
    od = od.copy()
    assert_mapping_covers(od["sexo_nacimiento"], {"Hombres", "Mujeres"})
    od["genero"] = (
        od["sexo_nacimiento"]
        .map({"Hombres": "H", "Mujeres": "F"})
        .fillna(NO_ESPECIFICADO)
    )

    return od


# OCCUPATION
def harmonize_enoe_occupation(enoe):
    enoe = enoe.copy()
    occupation_mapping = {
        1: "trabajador",
        2: "trabajador",
        3: "independiente",
        4: "sin_pago",  # trabajadores sin pago (unpaid family workers); ENOE-only level, the OD has no counterpart
    }
    assert_mapping_covers(enoe["pos_ocu"], occupation_mapping)
    enoe["ocupacion"] = enoe["pos_ocu"].map(occupation_mapping).fillna(NO_ESPECIFICADO)

    return enoe


def harmonize_od_occupation(od):
    od = od.copy()
    occupation_mapping = {
        "Empleado": "trabajador",
        "Persona trabajadora del hogar": "trabajador",
        "Trabajador del campo": "trabajador",
        "Profesor": "trabajador",
        "Patrón o empresario": "trabajador",
        "Trabajador independiente": "independiente",
        # Respondents who report working last week but give a non-working status: they do work (see
        # docs/review_remediation_plan.md 1.3) but their position in the occupation is unknown.
        "Hogar": NO_ESPECIFICADO,
        "Estudiante": NO_ESPECIFICADO,
        "Jubilado o pensionado": NO_ESPECIFICADO,
        "Desempleado": NO_ESPECIFICADO,
    }
    assert_mapping_covers(od["ocupacion_raw"], occupation_mapping)
    od["ocupacion"] = (
        od["ocupacion_raw"].map(occupation_mapping).fillna(NO_ESPECIFICADO)
    )

    return od


# AGE
def harmonize_enoe_age(enoe):
    enoe = enoe.copy()
    enoe["edad_num"] = enoe["eda"].where(enoe["eda"] < ENOE_AGE_UNSPECIFIED)
    enoe["edad_cat"] = (
        pd.cut(enoe["edad_num"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        .astype("string")
        .fillna(NO_ESPECIFICADO)
    )

    return enoe


def harmonize_od_age(od):
    od = od.copy()
    od["edad_num"] = od["edad"].copy()
    od["edad_cat"] = (
        pd.cut(od["edad_num"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        .astype("string")
        .fillna(NO_ESPECIFICADO)
    )

    return od


# EDUCATION
def harmonize_enoe_education(enoe):
    enoe = enoe.copy()
    education_mapping = {
        0: "sin_instruccion",
        1: "sin_instruccion",
        2: "primaria_o_secundaria",
        3: "primaria_o_secundaria",
        4: "carrera_tecnica_o_preparatoria",
        5: "carrera_tecnica_o_preparatoria",
        6: "carrera_tecnica_o_preparatoria",
        7: "licenciatura",
        8: "postgrado",
        9: "postgrado",
        99: NO_ESPECIFICADO,
    }
    assert_mapping_covers(enoe["cs_p13_1"], education_mapping)
    enoe["escolaridad"] = (
        enoe["cs_p13_1"].map(education_mapping).fillna(NO_ESPECIFICADO)
    )

    return enoe


def harmonize_od_education(od):
    od = od.copy()
    education_mapping = {
        "Ninguno": "sin_instruccion",
        "Kinder": "sin_instruccion",
        "Preescolar": "sin_instruccion",
        "Primaria": "primaria_o_secundaria",
        "Secundaria": "primaria_o_secundaria",
        "Normal básica": "carrera_tecnica_o_preparatoria",
        "Preparatoria o bachillerato": "carrera_tecnica_o_preparatoria",
        "Carrera técnica con secundaria terminada": "carrera_tecnica_o_preparatoria",
        "Carrera técnica con preparatoria terminada": "carrera_tecnica_o_preparatoria",
        "Licenciatura o profesional": "licenciatura",
        "Maestría o doctorado": "postgrado",
        "No sabe": NO_ESPECIFICADO,
    }
    assert_mapping_covers(od["escolaridad_raw"], education_mapping)
    od["escolaridad"] = (
        od["escolaridad_raw"].map(education_mapping).fillna(NO_ESPECIFICADO)
    )

    return od


# MUNICIPALITY
def harmonize_enoe_municipality(enoe):
    enoe = enoe.copy()
    municipality_mapping = {
        39: "guadalajara",
        120: "zapopan",
        98: "tlaquepaque",
        97: "tlajomulco",
        101: "tonala",
        70: "el_salto",
        51: "juanacatlan",
        44: "ixtlahuacan_membrillos",
        124: "zapotlanejo",
    }
    # Codes outside the metro area are "otro" (Jalisco outside the AMG); a missing/masked code is not.
    enoe["municipio"] = (
        enoe["mun"]
        .map(municipality_mapping)
        .fillna("otro")
        .where(enoe["mun"].notna(), NO_ESPECIFICADO)
    )

    return enoe


def harmonize_od_municipality(od):
    od = od.copy()
    municipality_mapping = {
        "Guadalajara": "guadalajara",
        "Zapopan": "zapopan",
        "Tlaquepaque": "tlaquepaque",
        "Tlajomulco": "tlajomulco",
        "Tonalá": "tonala",
        "El Salto": "el_salto",
        "Juanacatlán": "juanacatlan",
        "Ixtlahuacán de los Membrillos": "ixtlahuacan_membrillos",
        "Zapotlanejo": "zapotlanejo",
    }
    assert_mapping_covers(od["municipio_raw"], municipality_mapping)
    od["municipio"] = (
        od["municipio_raw"]
        .map(municipality_mapping)
        .fillna("otro")
        .where(od["municipio_raw"].notna(), NO_ESPECIFICADO)
    )

    return od


# MARITAL STATUS
def harmonize_enoe_marital_status(enoe):
    enoe = enoe.copy()
    marital_status_mapping = {
        1: "union_libre",
        2: "separado",
        3: "divorciado",
        4: "viudo",
        5: "casado",
        6: "soltero",
    }
    assert_mapping_covers(
        enoe["e_con"], marital_status_mapping, allowed_unmapped={9}
    )  # 9 = no sabe
    enoe["estado_civil"] = (
        enoe["e_con"].map(marital_status_mapping).fillna(NO_ESPECIFICADO)
    )

    return enoe


def harmonize_od_marital_status(od):
    od = od.copy()
    marital_status_mapping = {
        "Soltero": "soltero",
        "Casado": "casado",
        "Unión libre": "union_libre",
        "Viudo": "viudo",
        "Separado": "separado",
        "Divorciado": "divorciado",
    }
    assert_mapping_covers(
        od["estado_civil_raw"],
        marital_status_mapping,
        allowed_unmapped={"Otros (especifique)"},
    )
    od["estado_civil"] = (
        od["estado_civil_raw"].map(marital_status_mapping).fillna(NO_ESPECIFICADO)
    )

    return od


# HOUSEHOLD RELATIONSHIP
def harmonize_enoe_relationship(enoe):
    enoe = enoe.copy()
    relationship_mapping = {
        1: "jefe_del_hogar",
        2: "conyuge",
        3: "hijo",
        4: "otro_parentesco",
        5: "sin_parentesco",
        6: "sin_parentesco",
    }
    relationship_code = enoe["par_c"] // 100
    assert_mapping_covers(relationship_code, relationship_mapping, name="par_c // 100")
    enoe["parentesco"] = relationship_code.map(relationship_mapping).fillna(
        NO_ESPECIFICADO
    )

    return enoe


def harmonize_od_relationship(od):
    od = od.copy()
    relationship_mapping = {
        "Jefe del hogar": "jefe_del_hogar",
        "Cónyuge": "conyuge",
        "Compañero": "conyuge",
        "Hijo": "hijo",
        "Nieto": "otro_parentesco",
        "Otro parentesco": "otro_parentesco",
        "Sin parentesco": "sin_parentesco",
    }

    assert_mapping_covers(od["parentesco_raw"], relationship_mapping)
    od["parentesco"] = (
        od["parentesco_raw"].map(relationship_mapping).fillna(NO_ESPECIFICADO)
    )

    return od


# HOUSEHOLD SIZE
def harmonize_enoe_household_size(enoe):
    enoe = enoe.copy()
    enoe["tamano_viv_num"] = enoe["dwelling_size"].clip(upper=HOUSEHOLD_SIZE_CAP)
    enoe["tamano_viv_cat"] = (
        pd.cut(
            enoe["tamano_viv_num"],
            bins=HOUSEHOLD_SIZE_BINS,
            labels=HOUSEHOLD_SIZE_LABELS,
            right=False,
        )
        .astype("string")
        .fillna(NO_ESPECIFICADO)
    )

    return enoe


def harmonize_od_household_size(od):
    od = od.copy()
    household_size = od["dwelling_size"].replace({"10 y +": "10", "10 y más": "10"})
    assert_mapping_covers(
        household_size, [str(n) for n in range(1, 11)], name="dwelling_size"
    )
    od["tamano_viv_num"] = (
        pd.to_numeric(household_size, errors="coerce")
        .astype("Int64")
        .clip(upper=HOUSEHOLD_SIZE_CAP)
    )
    od["tamano_viv_cat"] = (
        pd.cut(
            od["tamano_viv_num"],
            bins=HOUSEHOLD_SIZE_BINS,
            labels=HOUSEHOLD_SIZE_LABELS,
            right=False,
        )
        .astype("string")
        .fillna(NO_ESPECIFICADO)
    )

    return od


# ECONOMIC SECTOR
def harmonize_enoe_sector(enoe):
    enoe = enoe.copy()
    sector_mapping = load_mapping("sector")["enoe_scian"]
    assert_mapping_covers(enoe["scian"], sector_mapping)
    enoe["sector"] = enoe["scian"].map(sector_mapping).fillna(NO_ESPECIFICADO)
    enoe["sector_desconocido"] = enoe["sector"].eq(NO_ESPECIFICADO)

    return enoe


def harmonize_od_sector(od):
    od = od.copy()
    sector_mapping = load_mapping("sector")["od_giro_empresa"]
    assert_mapping_covers(od["giro_empresa"], sector_mapping)
    od["sector"] = od["giro_empresa"].map(sector_mapping).fillna(NO_ESPECIFICADO)
    od["sector_desconocido"] = od["sector"].eq(NO_ESPECIFICADO)

    return od


def attach_sector_probabilities(od, od_giro):
    """Collapse the OD giro model's output (``od_sector``: ``prob_giro_<slug>`` and the ``giro_*`` bookkeeping
    columns, keyed by ``folio_vivienda``/``folio_habitante``) to the harmonized sector classes.

    The giro -> sector map (``sector.yaml`` ``od_giro``) is many-to-one, so ``prob_sector_<class>`` is the sum of the
    giro probabilities (lossless for the informality model). Adds ``prob_sector_<class>``, ``sector_final`` (the
    observed sector, or the arg-max class for imputed rows), ``sector_fue_imputado``, ``sector_model_used``,
    ``sector_prediction_confidence`` and ``sector_marginalized_features``.
    """
    from .common import SECTOR_CLASSES

    giro_to_sector = load_mapping("sector")["od_giro"]
    keys = ["folio_vivienda", "folio_habitante"]
    giro = od_giro.set_index(keys)
    od = od.copy()
    index = pd.MultiIndex.from_frame(od[keys])
    missing = ~index.isin(giro.index)
    assert not missing.any(), f"{int(missing.sum())} OD workers have no giro-model output"
    giro = giro.reindex(index)
    for sector_class in SECTOR_CLASSES:
        columns = [f"prob_giro_{slug}" for slug, target in giro_to_sector.items() if target == sector_class]
        od[f"prob_sector_{sector_class}"] = giro[columns].sum(axis=1).to_numpy()
    probability_columns = [f"prob_sector_{sector_class}" for sector_class in SECTOR_CLASSES]
    imputed = giro["giro_fue_imputado"].to_numpy(dtype=bool)
    assert (imputed == od["sector_desconocido"].to_numpy(dtype=bool)).all(), "giro-model imputation flags disagree with sector_desconocido"
    argmax = np.array(SECTOR_CLASSES)[od[probability_columns].to_numpy().argmax(axis=1)]
    od["sector_final"] = pd.Series(np.where(imputed, argmax, od["sector"].astype(object)), index=od.index, dtype="string")
    od["sector_fue_imputado"] = imputed
    od["sector_model_used"] = giro["giro_model_used"].astype("string").to_numpy()
    od["sector_prediction_confidence"] = np.where(imputed, od[probability_columns].max(axis=1), np.nan)
    od["sector_marginalized_features"] = giro["giro_marginalized_features"].astype("string").to_numpy()
    maximum_error = float(np.max(np.abs(od[probability_columns].sum(axis=1) - 1.0)))
    assert maximum_error < 1e-8, f"Collapsed sector probabilities do not sum to one (max error {maximum_error:.3e})"

    return od


# PLACE OF WORK (review item 4.2)
def harmonize_enoe_workplace(enoe):
    """``lugar_trabajo`` from COE1 section IV. Levels shared with the OD's work-trip destination type:
    establecimiento (premises of a non-commercial unit), comercio_o_puesto (premises in the commerce sector, or a
    fixed/semi-fixed/improvised stall — the OD answer "Comercio, mercado, tienda o centro comercial" covers both),
    otra_vivienda (employer's or client's home, domestic workers), otro_o_sin_local (field, itinerant, vehicle, own
    home, construction site, visiting clients) and no_especificado."""
    enoe = enoe.copy()
    p4e = pd.to_numeric(enoe["p4e"], errors="coerce")
    p4f = pd.to_numeric(enoe["p4f"], errors="coerce")
    p4h = pd.to_numeric(enoe["p4h"], errors="coerce")
    commerce = enoe["scian"].isin([6, 7])
    premises = p4h.isin([1, 2]) | p4e.isin([1, 2, 3])
    p4b = pd.to_numeric(enoe["p4b"], errors="coerce")
    lugar = pd.Series(NO_ESPECIFICADO, index=enoe.index, dtype=object)
    lugar[p4b.isin([2, 3])] = (
        "establecimiento"  # institutions (schools, hospitals, government, non-profits) skip 4e-4h
    )
    lugar[p4b.eq(1)] = "otro_o_sin_local"  # agricultural activity (field)
    lugar[premises] = np.where(
        commerce[premises], "comercio_o_puesto", "establecimiento"
    )
    lugar[p4f.isin([3, 9, 10])] = "comercio_o_puesto"
    lugar[p4f.eq(8) | enoe["p4"].eq(3)] = "otra_vivienda"
    lugar[p4f.isin([1, 2, 4, 5, 6, 7, 11]) | p4h.isin([3, 4])] = "otro_o_sin_local"
    enoe["lugar_trabajo"] = lugar

    return enoe


OD_WORKPLACE_MAPPING = {
    "Fábrica o taller": "establecimiento",
    "Oficina": "establecimiento",
    "Hospital, clínica, consultorio, laboratorio clínico": "establecimiento",
    "Escuela": "establecimiento",
    "Restaurante, bar, cafetería": "establecimiento",
    "Centro cultural o área recreativa": "establecimiento",
    "Deportivo, gimnasio": "establecimiento",
    "Comercio, mercado, tienda o centro comercial": "comercio_o_puesto",
    "Otra vivienda": "otra_vivienda",
    "Su casa": "otro_o_sin_local",
    "Otros (especifique)": "otro_o_sin_local",
}


def harmonize_od_workplace(od):
    """``lugar_trabajo`` from the destination type of the work trips; workers without a work trip on the survey day
    (home-based, mobile, or simply did not travel that day) are ``no_especificado`` and are marginalized at scoring."""
    od = od.copy()
    assert_mapping_covers(od["destino_trabajo"], OD_WORKPLACE_MAPPING)
    od["lugar_trabajo"] = (
        od["destino_trabajo"].map(OD_WORKPLACE_MAPPING).fillna(NO_ESPECIFICADO)
    )

    return od


def generate_enoe_informal_label(enoe):
    """``informal`` (INEGI `emp_ppal`) and its two components: ``informal_sector`` (informal-sector units, paid domestic
    work, subsistence agriculture — identifiable from the type of unit) and ``informal_unprotected`` (informal employment
    inside other units, essentially lack of social security — not observable in the OD)."""
    enoe = enoe.copy()
    assert_mapping_covers(enoe["emp_ppal"], {1, 2})
    enoe["informal"] = enoe["emp_ppal"].map({1: 1, 2: 0}).astype("Int64")
    in_informal_sector = enoe["tue2"].isin(INFORMAL_SECTOR_TUE2)
    enoe["informal_sector"] = (enoe["informal"].eq(1) & in_informal_sector).astype(
        "Int64"
    )
    enoe["informal_unprotected"] = (
        enoe["informal"].eq(1) & ~in_informal_sector
    ).astype("Int64")

    return enoe


# COMPLETE HARMONIZATION
def harmonize_enoe_dataframe(enoe):
    enoe = prepare_enoe_data_types(enoe)
    enoe = harmonize_enoe_gender(enoe)
    enoe = harmonize_enoe_occupation(enoe)
    enoe = harmonize_enoe_age(enoe)
    enoe = harmonize_enoe_education(enoe)
    enoe = harmonize_enoe_municipality(enoe)
    enoe = harmonize_enoe_marital_status(enoe)
    enoe = harmonize_enoe_relationship(enoe)
    enoe = harmonize_enoe_household_size(enoe)
    enoe = harmonize_enoe_sector(enoe)
    enoe = harmonize_enoe_workplace(enoe)
    enoe = generate_enoe_informal_label(enoe)

    return enoe


def harmonize_od_dataframe(od):
    od = prepare_od_data_types(od)
    od = harmonize_od_gender(od)
    od = harmonize_od_occupation(od)
    od = harmonize_od_age(od)
    od = harmonize_od_education(od)
    od = harmonize_od_municipality(od)
    od = harmonize_od_marital_status(od)
    od = harmonize_od_relationship(od)
    od = harmonize_od_household_size(od)
    od = harmonize_od_sector(od)
    od = harmonize_od_workplace(od)

    return od
