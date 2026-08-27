"""Shared constants and helpers used by more than one pipeline stage."""
import numpy as np
import pandas as pd

NO_ESPECIFICADO = "no_especificado"
SECTOR_CLASSES = ["comercio", "gobierno_otro_agricultura", "manufactura_construccion", "servicios_transporte"]
AGE_LABELS = ["0_2", "3_4", "5", "6_7", "8_11", "12_14", "15_17", "18_24", "25_49", "50_59", "60_64", "65_y_mas"]
# The nine municipalities of the Guadalajara Metropolitan Area as covered by the OD survey (eodgdl schema).
AMG_MUNICIPALITIES = ["guadalajara", "zapopan", "tlaquepaque", "tlajomulco", "tonala", "el_salto", "juanacatlan", "ixtlahuacan_membrillos", "zapotlanejo"]
NUMERIC_FEATURES = ["edad_num"]
# Household size is collapsed at 7+: the ENOE roster count and the OD self-report diverge 5x above 8 persons.
HOUSEHOLD_SIZE_LABELS = ["1", "2", "3", "4", "5", "6", "7_y_mas"]
HOUSEHOLD_SIZE_CAP = 10  # tamano_viv_num is capped here in both surveys (the OD answer stops at "10 y +")


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
        "tamano_viv_cat": HOUSEHOLD_SIZE_LABELS,
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


# Marginalization over categorical levels without training support (review item 1.12)
def compute_training_level_shares(X, sample_weights):
    """Weighted share of each level of every categorical feature in the training data."""
    _, categorical_features = split_feature_types(list(X.columns))
    weights = pd.Series(np.asarray(sample_weights, dtype=float), index=X.index)
    shares = {}
    for column in categorical_features:
        totals = weights.groupby(X[column].astype(str)).sum()
        shares[column] = totals / totals.sum()

    return shares

def attach_training_level_shares(model, X, sample_weights):
    """Store the training level shares on the fitted pipeline so the pickled bundle is self-contained."""
    model.training_level_shares_ = compute_training_level_shares(X, sample_weights)

    return model

def predict_proba_marginalizing(model, X, level_subsets=None):
    """``model.predict_proba`` where a categorical value with no training support is marginalized out.

    With one-hot encoding a tree ensemble routes an all-zero block (a declared level that never occurred in
    training) along the branch of whichever training level it did not split on, so the prediction silently
    becomes that level's. Instead, for every row whose value of feature *f* is unsupported, predict once per
    supported level of *f* and average with the training share of each level — the same treatment the pipeline
    gives to an unknown sector. Rows with several unsupported features are expanded over all combinations.

    ``level_subsets`` (feature -> list of levels) restricts the levels a feature is averaged over (shares are
    renormalized within the subset), e.g. an unsampled metro municipality averaged over the sampled metro ones only.

    Returns ``(probabilities, marginalized_features)`` where the second element is a per-row string listing the
    features that were marginalized ("" if none).
    """
    shares = getattr(model, "training_level_shares_", None)
    if not shares:
        raise ValueError("The model has no training_level_shares_; refit it with attach_training_level_shares.")
    if level_subsets:
        shares = dict(shares)
        for feature, levels in level_subsets.items():
            subset = shares[feature].reindex(levels).dropna()
            if subset.sum() <= 0:
                raise ValueError(f"No training support for the requested {feature} levels {list(levels)}")
            shares[feature] = pd.Series(0.0, index=shares[feature].index).add(subset / subset.sum(), fill_value=0.0)
    X = X.reset_index(drop=True)
    features = [column for column in X.columns if column in shares]
    n_classes = len(model.named_steps["classifier"].classes_)
    result = np.zeros((len(X), n_classes))
    marginalized_features = [[] for _ in range(len(X))]

    def fill(rows, weights, remaining):
        if len(rows) == 0:
            return
        if not remaining:
            result[rows.index] += weights[:, None] * model.predict_proba(rows)
            return
        feature, rest = remaining[0], remaining[1:]
        supported = shares[feature][shares[feature] > 0]
        unsupported = ~rows[feature].astype(str).isin(supported.index)
        fill(rows[~unsupported], weights[~unsupported.to_numpy()], rest)
        if unsupported.any():
            for row in rows.index[unsupported]:
                if feature not in marginalized_features[row]:
                    marginalized_features[row].append(feature)
            for level, share in supported.items():
                fill(rows[unsupported].assign(**{feature: level}), weights[unsupported.to_numpy()] * share, rest)

    fill(X, np.ones(len(X)), features)
    marginalized = pd.Series(["+".join(names) for names in marginalized_features], index=X.index, dtype=object)

    return normalize_predicted_probabilities(result), marginalized


# Model selection with a one-standard-error rule on paired folds (review item 2.2)
FAMILY_COMPLEXITY = {"LogisticRegression": 0, "RandomForest": 1, "GradientBoosting": 2}
# +1: larger value = more complex; -1: larger value = simpler (more regularization)
PARAMETER_COMPLEXITY_DIRECTION = {
    "classifier__max_iter": 1, "classifier__max_leaf_nodes": 1, "classifier__learning_rate": 1, "classifier__max_features": 1,
    "classifier__C": 1, "classifier__l2_regularization": -1, "classifier__min_samples_leaf": -1,
}

def complexity_key(model_name, params):
    """Sort key: simpler families first, then simpler hyperparameters (lexicographic over sorted parameter names)."""
    values = []
    for name in sorted(params):
        value = params[name]
        if value == "sqrt":
            value = 0.3
        values.append(PARAMETER_COMPLEXITY_DIRECTION.get(name, 1) * float(value))

    return (FAMILY_COMPLEXITY.get(model_name, 99), tuple(values))

def select_one_se(results):
    """Mark the configuration to keep under a one-standard-error rule.

    ``results`` has one row per candidate with ``model``, ``best_params`` and ``fold_log_losses`` (the same CV folds
    for every row). The best mean log loss is the reference; every candidate whose paired fold-difference to the
    reference is within one standard error of zero is eligible, and the simplest eligible candidate
    (:func:`complexity_key`) is selected. Family gaps of a few thousandths against fold spreads of ~0.015 are noise,
    so strict argmin would pick a family by coin flip.
    """
    table = results.reset_index(drop=True).copy()
    losses = np.array([np.asarray(row, dtype=float) for row in table["fold_log_losses"]])
    table["weighted_log_loss"] = losses.mean(axis=1)
    best = int(table["weighted_log_loss"].idxmin())
    differences = losses - losses[best]
    table["mean_diff_vs_best"] = differences.mean(axis=1)
    table["se_diff_vs_best"] = differences.std(axis=1, ddof=1) / np.sqrt(losses.shape[1])
    table["within_one_se"] = table["mean_diff_vs_best"] <= table["se_diff_vs_best"]
    keys = [complexity_key(model, params) for model, params in zip(table["model"], table["best_params"])]
    table["complexity_rank"] = pd.Series(keys).rank(method="first").astype(int)
    eligible = table.index[table["within_one_se"]]
    selected = min(eligible, key=lambda index: keys[index])
    table["selected"] = False
    table.loc[selected, "selected"] = True

    return table
