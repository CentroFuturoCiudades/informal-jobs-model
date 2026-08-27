"""Shared constants and helpers used by more than one pipeline stage."""
import numpy as np
import pandas as pd

NO_ESPECIFICADO = "no_especificado"
SECTOR_CLASSES = ["comercio", "gobierno_otro_agricultura", "manufactura_construccion", "servicios_transporte"]
AGE_LABELS = ["0_2", "3_4", "5", "6_7", "8_11", "12_14", "15_17", "18_24", "25_49", "50_59", "60_64", "65_y_mas"]
# The nine municipalities of the Guadalajara Metropolitan Area as covered by the OD survey (eodgdl schema).
AMG_MUNICIPALITIES = ["guadalajara", "zapopan", "tlaquepaque", "tlajomulco", "tonala", "el_salto", "juanacatlan", "ixtlahuacan_membrillos", "zapotlanejo"]
NUMERIC_FEATURES = ["edad_num"]


def _eodgdl_levels(column):
    """Category labels eodgdl guarantees for a raw OD column (its pandera schema)."""
    from eodgdl import schemas

    schema = schemas.hab_schema if column in schemas.hab_schema.columns else schemas.viv_schema

    return list(schema.columns[column].dtype.type.categories)

def build_category_levels():
    """Every level a categorical model feature can take in *either* survey.

    This is the contract the one-hot encoders are built with (``OneHotEncoder(categories=...)``): a level
    absent from the training data still gets its own (all-zero, uninformative) column instead of the silent
    all-zero block that ``handle_unknown="ignore"`` would produce, and a value outside the list raises.
    """
    levels = {
        "genero": ["H", "F"],
        "ocupacion": ["trabajador", "independiente", "sin_pago"],
        "escolaridad": ["sin_instruccion", "primaria_o_secundaria", "carrera_tecnica_o_preparatoria", "licenciatura", "postgrado"],
        "municipio": AMG_MUNICIPALITIES + ["otro"],
        "estado_civil": ["union_libre", "separado", "divorciado", "viudo", "casado", "soltero"],
        "parentesco": ["jefe_del_hogar", "conyuge", "hijo", "otro_parentesco", "sin_parentesco"],
        "tamano_viv_cat": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10_y_mas"],
        "edad_cat": AGE_LABELS,
        "sector": SECTOR_CLASSES,
        # raw OD columns used directly by the sector model
        "ocupacion_raw": _eodgdl_levels("ocupacion"),
        "trabajo_semana_pasada": _eodgdl_levels("trabajo_semana_pasada"),
        "centralidad": _eodgdl_levels("centralidad"),
    }

    return {feature: values + [NO_ESPECIFICADO] for feature, values in levels.items()}

def split_feature_types(features):
    numerical = [column for column in features if column in NUMERIC_FEATURES]
    categorical = [column for column in features if column not in numerical]

    return numerical, categorical

def identify_missing_category(series):
    text = series.astype("string").str.strip().str.lower()
    missing = series.isna() | text.eq("").fillna(False) | text.eq(NO_ESPECIFICADO).fillna(False)

    return missing.fillna(False).astype(bool)

def prepare_model_features(dataframe, features):
    """Select the model features: numeric columns coerced, categorical columns as clean strings with a missing label."""
    X = dataframe[features].copy()
    numerical_features, categorical_features = split_feature_types(features)

    for column in numerical_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_features:
        X[column] = X[column].astype("string").str.strip().replace("", NO_ESPECIFICADO).fillna(NO_ESPECIFICADO).astype(object)

    return X

def assert_known_levels(X, category_levels=None):
    """Raise if a categorical feature holds a value outside the declared level list."""
    category_levels = category_levels or build_category_levels()
    unknown = {}
    for column in X.columns:
        if column in category_levels:
            values = set(X[column].dropna().unique()) - set(category_levels[column])
            if values:
                unknown[column] = sorted(map(str, values))
    if unknown:
        raise ValueError(f"Values outside the declared category levels: {unknown}")

def count_levels_without_training_support(X_train, X):
    """Rows of ``X`` whose categorical value never appears in ``X_train`` (no training support), per feature."""
    _, categorical_features = split_feature_types(list(X.columns))
    rows = []
    for column in categorical_features:
        seen = set(X_train[column].dropna().unique())
        unsupported = ~X[column].isin(seen)
        if unsupported.any():
            rows.append({"feature": column, "rows": int(unsupported.sum()), "levels": sorted(map(str, X.loc[unsupported, column].unique()))})

    return pd.DataFrame(rows, columns=["feature", "rows", "levels"])

def normalize_sample_weights(sample_weights):
    if sample_weights.isna().any():
        raise ValueError("Sample weights contain missing values.")
    sample_weights = sample_weights.astype(float)

    return sample_weights / sample_weights.mean()

def normalize_predicted_probabilities(probabilities, tolerance=1e-8):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values.")

    probability_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(probability_sums <= 0):
        raise ValueError("At least one predicted probability row has a non-positive sum.")

    maximum_error = np.max(np.abs(probability_sums.ravel() - 1.0))
    if maximum_error > tolerance:
        raise ValueError(f"Predicted probabilities do not sum to one. Maximum error: {maximum_error:.3e}")

    return probabilities / probability_sums
