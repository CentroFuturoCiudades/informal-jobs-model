import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import NO_ESPECIFICADO, SECTOR_CLASSES, assert_known_levels, bootstrap_by_group, marginal_log_loss, reweight_to_target_profile, calculate_calibration_table, calibration_metrics, select_one_se, attach_training_level_shares, predict_proba_marginalizing, build_category_levels, count_levels_without_training_support, identify_missing_category, normalize_predicted_probabilities, normalize_sample_weights, prepare_model_features, split_feature_types


# Work-trip destination features (review item 4.4): the destination type, its ámbito, and the DENUE establishment mix
# at the destination AGEB/locality. They are OD-only, which is fine here (the sector model trains on OD).
OD_DESTINATION_FEATURES = ["destino_trabajo", "destino_ambito", "dest_establecimientos_log", "dest_share_grandes", "dest_share_comercio", "dest_share_gobierno_otro_agricultura", "dest_share_manufactura_construccion", "dest_share_servicios_transporte"]
OD_SECTOR_FEATURES = ["genero", "edad_num", "escolaridad", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "ocupacion_raw", "trabajo_semana_pasada", "centralidad"] + OD_DESTINATION_FEATURES
OD_ROBUST_SECTOR_FEATURES = ["genero", "edad_num", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "ocupacion_raw", "trabajo_semana_pasada", "centralidad"] + OD_DESTINATION_FEATURES

# Diagnostics
def compare_sector_known_unknown_profiles(od, columns):
    known_sector = od[~od["sector_desconocido"]].copy()
    unknown_sector = od[od["sector_desconocido"]].copy()

    comparisons = []
    for column in columns:
        known_distribution = known_sector.groupby(column, dropna=False)["expansion_factor"].sum().reset_index(name="known_population")
        unknown_distribution = unknown_sector.groupby(column, dropna=False)["expansion_factor"].sum().reset_index(name="unknown_population")

        known_distribution["known_share"] = known_distribution["known_population"] / known_distribution["known_population"].sum()
        unknown_distribution["unknown_share"] = unknown_distribution["unknown_population"] / unknown_distribution["unknown_population"].sum()

        comparison = known_distribution.merge(unknown_distribution, on=column, how="outer")
        comparison["variable"] = column
        comparison["category"] = comparison[column].astype("string")
        comparison["known_share"] = comparison["known_share"].fillna(0)
        comparison["unknown_share"] = comparison["unknown_share"].fillna(0)
        comparison["difference_pp"] = (comparison["unknown_share"] - comparison["known_share"]) * 100

        comparisons.append(comparison[["variable", "category", "known_population", "unknown_population", "known_share", "unknown_share", "difference_pp"]])

    return pd.concat(comparisons, ignore_index=True)

def calculate_od_sector_feature_missingness(od, sector_features=OD_SECTOR_FEATURES):
    known_sector = od[~od["sector_desconocido"]].copy()
    unknown_sector = od[od["sector_desconocido"]].copy()

    results = []
    for column in sector_features:
        known_missing = identify_missing_category(known_sector[column])
        unknown_missing = identify_missing_category(unknown_sector[column])
        known_missing_population = known_sector.loc[known_missing, "expansion_factor"].sum()
        unknown_missing_population = unknown_sector.loc[unknown_missing, "expansion_factor"].sum()
        known_total_population = known_sector["expansion_factor"].sum()
        unknown_total_population = unknown_sector["expansion_factor"].sum()

        results.append({
            "variable": column,
            "known_missing_share": known_missing_population / known_total_population,
            "unknown_missing_share": unknown_missing_population / unknown_total_population
        })

    return pd.DataFrame(results)

# Train-test split
def split_od_sector_known_data(od, n_splits=5, test_fold=0, random_state=42):
    known_sector = od[~od["sector_desconocido"]].copy()
    known_sector = known_sector[known_sector["sector"].isin(SECTOR_CLASSES)].copy()
    known_sector = known_sector.reset_index(drop=True)

    y = known_sector["sector"].copy()
    groups = known_sector["folio_vivienda"].astype("string")
    cross_validation = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cross_validation.split(known_sector, y, groups=groups))
    train_index, test_index = splits[test_fold]

    training_data = known_sector.iloc[train_index].copy()
    test_data = known_sector.iloc[test_index].copy()

    return training_data, test_data

# Feature preparation



def prepare_sector_features(dataframe, sector_features):
    return prepare_model_features(dataframe, sector_features)

def prepare_od_probabilistic_sector_training_data(od, sector_features=OD_SECTOR_FEATURES):
    training_data = od[~od["sector_desconocido"]].copy()
    training_data = training_data[training_data["sector"].isin(SECTOR_CLASSES)].copy()
    training_data = training_data.reset_index(drop=True)

    X = prepare_sector_features(training_data, sector_features)
    y = training_data["sector"].reset_index(drop=True)
    sample_weights = normalize_sample_weights(training_data["expansion_factor"]).reset_index(drop=True)
    groups = training_data["folio_vivienda"].astype("string").reset_index(drop=True)

    return X, y, sample_weights, groups, training_data

# Models
def build_probabilistic_sector_models(sector_features=OD_SECTOR_FEATURES, random_state=42):
    numerical_features, categorical_features = split_feature_types(sector_features)
    category_levels = build_category_levels()
    categories = [category_levels[column] for column in categorical_features]

    linear_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    linear_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error"))])

    tree_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    tree_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False))])

    # The pipelines are self-contained: the first step selects the features and cleans them (numeric coercion,
    # categorical strings with the missing label), so a pickled bundle applies to a raw harmonized frame.
    prepare = FunctionTransformer(prepare_model_features, kw_args={"features": list(sector_features)})
    linear_preprocessor = ColumnTransformer([("numerical", linear_numerical_preprocessor, numerical_features), ("categorical", linear_categorical_preprocessor, categorical_features)])
    tree_preprocessor = ColumnTransformer([("numerical", tree_numerical_preprocessor, numerical_features), ("categorical", tree_categorical_preprocessor, categorical_features)])

    models = {
        "LogisticRegression": {
            "model": Pipeline([("prepare", prepare), ("preprocessor", linear_preprocessor), ("classifier", LogisticRegression(max_iter=2000, random_state=random_state))]),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "model": Pipeline([("prepare", prepare), ("preprocessor", tree_preprocessor), ("classifier", RandomForestClassifier(n_estimators=500, class_weight=None, random_state=random_state, n_jobs=-1))]),
            "params": {
                "classifier__max_leaf_nodes": [25, 50, 100],
                "classifier__max_features": ["sqrt", 0.7],
                "classifier__min_samples_leaf": [1, 5, 10]
            }
        },
        "GradientBoosting": {
            "model": Pipeline([("prepare", prepare), ("preprocessor", tree_preprocessor), ("classifier", HistGradientBoostingClassifier(early_stopping=False, class_weight=None, random_state=random_state))]),  # early stopping would use a row-level split that ignores households; max_iter is tuned in the grouped CV instead
            "params": {
                "classifier__max_iter": [50, 100, 200, 400],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_leaf_nodes": [15, 31],
                "classifier__l2_regularization": [0.0, 1.0]
            }
        }
    }

    return models

# Weighted cross-validation and tuning
def tune_probabilistic_sector_models(X, y, sample_weights, groups, sector_features=OD_SECTOR_FEATURES, cv_splits=5, random_state=42):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sample_weights = sample_weights.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    models = build_probabilistic_sector_models(sector_features=sector_features, random_state=random_state)
    cross_validation = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    splits = list(cross_validation.split(X, y, groups=groups))

    model_results = []
    grid_results = []
    best_models = {}

    for model_name, model_config in models.items():
        family_results = []

        for params in ParameterGrid(model_config["params"]):
            fold_log_loss = []
            fold_balanced_accuracy = []
            fold_accuracy = []
            fold_f1_macro = []

            for train_index, validation_index in splits:
                model = clone(model_config["model"])
                model.set_params(**params)

                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                weights_train = sample_weights.iloc[train_index]

                X_validation = X.iloc[validation_index]
                y_validation = y.iloc[validation_index]
                weights_validation = sample_weights.iloc[validation_index]

                model.fit(X_train, y_train, classifier__sample_weight=weights_train)

                predictions = model.predict(X_validation)
                probabilities = normalize_predicted_probabilities(model.predict_proba(X_validation))
                classes = model.named_steps["classifier"].classes_

                fold_log_loss.append(log_loss(y_validation, probabilities, labels=classes, sample_weight=weights_validation))
                fold_balanced_accuracy.append(balanced_accuracy_score(y_validation, predictions, sample_weight=weights_validation))
                fold_accuracy.append(accuracy_score(y_validation, predictions, sample_weight=weights_validation))
                fold_f1_macro.append(f1_score(y_validation, predictions, average="macro", sample_weight=weights_validation, zero_division=0))

            result = {
                "model": model_name,
                "weighted_log_loss": np.mean(fold_log_loss),
                "weighted_log_loss_std": np.std(fold_log_loss),
                "weighted_balanced_accuracy": np.mean(fold_balanced_accuracy),
                "weighted_accuracy": np.mean(fold_accuracy),
                "weighted_f1_macro": np.mean(fold_f1_macro),
                "best_params": params,
                "fold_log_losses": [float(value) for value in fold_log_loss]
            }

            family_results.append(result)

        # Within the family: simplest configuration within one standard error of the best (paired folds).
        family_table = select_one_se(pd.DataFrame(family_results))
        best_result = family_results[int(family_table.index[family_table["selected"]][0])]

        best_model = clone(model_config["model"])
        best_model.set_params(**best_result["best_params"])
        best_model.fit(X, y, classifier__sample_weight=sample_weights)

        grid_results.extend(family_results)
        model_results.append(best_result)
        best_models[model_name] = best_model

    # Across families: same rule, simplest family within one standard error of the best is selected.
    model_summary = select_one_se(pd.DataFrame(model_results)).sort_values("weighted_log_loss").reset_index(drop=True)
    model_summary.attrs["grid_results"] = pd.DataFrame(grid_results)

    return model_summary, best_models

def get_best_probabilistic_sector_model(model_summary, best_models):
    best_model_name = model_summary.loc[model_summary["selected"], "model"].iloc[0]
    best_model = best_models[best_model_name]

    return best_model_name, best_model

# Test evaluation
def evaluate_probabilistic_sector_model(model, validation_data, sector_features=OD_SECTOR_FEATURES):
    validation_data = validation_data.copy()

    X = prepare_sector_features(validation_data, sector_features)
    y_true = validation_data["sector"].copy()
    sample_weights = validation_data["expansion_factor"].astype(float).copy()

    predictions = model.predict(X)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = model.named_steps["classifier"].classes_

    metrics = pd.DataFrame({
        "accuracy": [accuracy_score(y_true, predictions)],
        "balanced_accuracy": [balanced_accuracy_score(y_true, predictions)],
        "f1_macro": [f1_score(y_true, predictions, average="macro", zero_division=0)],
        "weighted_accuracy": [accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_balanced_accuracy": [balanced_accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_f1_macro": [f1_score(y_true, predictions, average="macro", sample_weight=sample_weights, zero_division=0)],
        "weighted_log_loss": [log_loss(y_true, probabilities, labels=classes, sample_weight=sample_weights)]
    })

    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=SECTOR_CLASSES, sample_weight=sample_weights, zero_division=0)

    class_metrics = pd.DataFrame({
        "sector": SECTOR_CLASSES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_support": support
    })

    weighted_confusion = confusion_matrix(y_true, predictions, labels=SECTOR_CLASSES, sample_weight=sample_weights)
    confusion = pd.DataFrame(weighted_confusion, index=SECTOR_CLASSES, columns=SECTOR_CLASSES)

    observed_distribution = validation_data.groupby("sector")["expansion_factor"].sum().reset_index(name="observed_population")
    observed_distribution["observed_share"] = observed_distribution["observed_population"] / observed_distribution["observed_population"].sum()

    hard_distribution = pd.DataFrame({"sector": predictions, "expansion_factor": sample_weights.to_numpy()}).groupby("sector")["expansion_factor"].sum().reset_index(name="hard_predicted_population")
    hard_distribution["hard_predicted_share"] = hard_distribution["hard_predicted_population"] / hard_distribution["hard_predicted_population"].sum()

    probabilistic_distribution = []

    for class_index, sector_class in enumerate(classes):
        predicted_population = (sample_weights.to_numpy() * probabilities[:, class_index]).sum()
        probabilistic_distribution.append({"sector": sector_class, "probabilistic_predicted_population": predicted_population})

    probabilistic_distribution = pd.DataFrame(probabilistic_distribution)
    probabilistic_distribution["probabilistic_predicted_share"] = probabilistic_distribution["probabilistic_predicted_population"] / probabilistic_distribution["probabilistic_predicted_population"].sum()

    distribution_comparison = pd.DataFrame({"sector": SECTOR_CLASSES})
    distribution_comparison = distribution_comparison.merge(observed_distribution, on="sector", how="left")
    distribution_comparison = distribution_comparison.merge(hard_distribution, on="sector", how="left")
    distribution_comparison = distribution_comparison.merge(probabilistic_distribution, on="sector", how="left")
    distribution_comparison = distribution_comparison.fillna(0)
    distribution_comparison["probabilistic_difference_pp"] = (distribution_comparison["probabilistic_predicted_share"] - distribution_comparison["observed_share"]) * 100

    return metrics, class_metrics, confusion, distribution_comparison

# Final model
def calculate_sector_calibration(model, validation_data, sector_features=OD_SECTOR_FEATURES, n_bins=10):
    """Calibration of the sector probabilities on held-out data, one-vs-rest per class.

    Returns ``(in_the_large, reliability)``: per-class predicted share vs observed share (weighted) with ECE, and
    the per-class reliability tables (``sector`` column) for plotting.
    """
    X = prepare_sector_features(validation_data, sector_features)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    weights = validation_data["expansion_factor"].astype(float).to_numpy()
    rows, tables = [], []
    for index, sector_class in enumerate(classes):
        outcome = (validation_data["sector"].to_numpy() == sector_class).astype(float)
        metrics = calibration_metrics(outcome, probabilities[:, index], weights, n_bins=n_bins)
        rows.append({"sector": sector_class, "observed_share": np.average(outcome, weights=weights), "predicted_share": np.average(probabilities[:, index], weights=weights), "gap_pp": metrics["calibration_gap_pp"], "ece": metrics["ece"], "calibration_slope": metrics["calibration_slope"]})
        tables.append(calculate_calibration_table(outcome, probabilities[:, index], weights, n_bins=n_bins).assign(sector=sector_class))

    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True)

# Final models
def refit_probabilistic_sector_model(model, od, sector_features=OD_SECTOR_FEATURES):
    X, y, sample_weights, groups, training_data = prepare_od_probabilistic_sector_training_data(od, sector_features=sector_features)

    final_model = clone(model)
    final_model.fit(X, y, classifier__sample_weight=sample_weights)
    attach_training_level_shares(final_model, X, sample_weights)

    return final_model, training_data

def impute_missing_sectors_hybrid(model_with_education, model_without_education, od, with_education_features=OD_SECTOR_FEATURES, without_education_features=OD_ROBUST_SECTOR_FEATURES):
    od = od.copy()
    for model in (model_with_education, model_without_education):
        assert set(model.named_steps["classifier"].classes_) == set(SECTOR_CLASSES), f"Sector model classes {list(model.named_steps['classifier'].classes_)} differ from SECTOR_CLASSES"
    known_outside = set(od.loc[~od["sector_desconocido"].astype(bool), "sector"]) - set(SECTOR_CLASSES)
    assert not known_outside, f"Known-sector rows carry classes outside SECTOR_CLASSES: {known_outside}"

    unknown_sector = od["sector_desconocido"].astype(bool)
    known_sector = ~unknown_sector

    missing_education = identify_missing_category(od["escolaridad"])
    use_with_education = unknown_sector & ~missing_education
    use_without_education = unknown_sector & missing_education

    od["sector_observado"] = od["sector"].where(known_sector, pd.NA).astype("string")
    od["sector_imputado"] = pd.Series(pd.NA, index=od.index, dtype="string")
    od["sector_final"] = od["sector_observado"].copy()
    od["sector_fue_imputado"] = unknown_sector
    od["sector_model_used"] = pd.Series("observed", index=od.index, dtype="string")
    od["sector_prediction_confidence"] = np.nan
    od["sector_marginalized_features"] = pd.Series("", index=od.index, dtype="string")

    for sector_class in SECTOR_CLASSES:
        probability_column = f"prob_sector_{sector_class}"
        od[probability_column] = 0.0
        od.loc[known_sector & od["sector"].eq(sector_class), probability_column] = 1.0

    if use_with_education.any():
        X_with_education = prepare_sector_features(od.loc[use_with_education], with_education_features)
        assert_known_levels(X_with_education)
        probabilities_with_education, marginalized_with_education = predict_proba_marginalizing(model_with_education, X_with_education)
        classes_with_education = model_with_education.named_steps["classifier"].classes_
        predictions_with_education = classes_with_education[probabilities_with_education.argmax(axis=1)]
        od.loc[use_with_education, "sector_marginalized_features"] = marginalized_with_education.to_numpy()

        od.loc[use_with_education, "sector_imputado"] = predictions_with_education
        od.loc[use_with_education, "sector_final"] = predictions_with_education
        od.loc[use_with_education, "sector_model_used"] = "with_education"
        od.loc[use_with_education, "sector_prediction_confidence"] = probabilities_with_education.max(axis=1)

        for class_index, sector_class in enumerate(classes_with_education):
            od.loc[use_with_education, f"prob_sector_{sector_class}"] = probabilities_with_education[:, class_index]

    if use_without_education.any():
        X_without_education = prepare_sector_features(od.loc[use_without_education], without_education_features)
        assert_known_levels(X_without_education)
        probabilities_without_education, marginalized_without_education = predict_proba_marginalizing(model_without_education, X_without_education)
        classes_without_education = model_without_education.named_steps["classifier"].classes_
        predictions_without_education = classes_without_education[probabilities_without_education.argmax(axis=1)]
        od.loc[use_without_education, "sector_marginalized_features"] = marginalized_without_education.to_numpy()

        od.loc[use_without_education, "sector_imputado"] = predictions_without_education
        od.loc[use_without_education, "sector_final"] = predictions_without_education
        od.loc[use_without_education, "sector_model_used"] = "without_education"
        od.loc[use_without_education, "sector_prediction_confidence"] = probabilities_without_education.max(axis=1)

        for class_index, sector_class in enumerate(classes_without_education):
            od.loc[use_without_education, f"prob_sector_{sector_class}"] = probabilities_without_education[:, class_index]

    return od

def calculate_sector_model_usage(dataframe, weight_column="expansion_factor"):
    imputed = dataframe[dataframe["sector_fue_imputado"]].copy()

    usage = imputed.groupby("sector_model_used").agg(sample_workers=("sector_model_used", "size"), weighted_population=(weight_column, "sum")).reset_index()
    usage["sample_share"] = usage["sample_workers"] / usage["sample_workers"].sum()
    usage["weighted_share"] = usage["weighted_population"] / usage["weighted_population"].sum()

    return usage

def calculate_sector_confidence_summary(dataframe):
    imputed = dataframe[dataframe["sector_fue_imputado"]].copy()

    summary = imputed.groupby("sector_model_used")["sector_prediction_confidence"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()

    return summary

def validate_sector_probability_rows(dataframe, tolerance=1e-8):
    probability_columns = [f"prob_sector_{sector_class}" for sector_class in SECTOR_CLASSES]

    probability_sums = dataframe[probability_columns].sum(axis=1)
    maximum_error = np.max(np.abs(probability_sums - 1.0))

    if maximum_error > tolerance:
        raise ValueError(f"Sector probabilities do not sum to one. Maximum error: {maximum_error:.3e}")

    return maximum_error


# Final distributions
def calculate_sector_distribution(dataframe, sector_column="sector_final", weight_column="expansion_factor"):
    distribution = dataframe.groupby(sector_column, dropna=False)[weight_column].sum().reset_index(name="weighted_population")
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution


def calculate_probabilistic_sector_distribution(dataframe, weight_column="expansion_factor"):
    results = []

    for sector_class in SECTOR_CLASSES:
        probability_column = f"prob_sector_{sector_class}"
        weighted_population = (dataframe[weight_column] * dataframe[probability_column]).sum()
        results.append({"sector": sector_class, "weighted_population": weighted_population})

    distribution = pd.DataFrame(results)
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution

# Sensitivity of the imputation to the covariate shift between known- and unknown-sector workers (review item 2.5)
SECTOR_SHIFT_PROFILE_FEATURES = ["genero", "edad_num", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "ocupacion_raw", "trabajo_semana_pasada"]

def impute_sectors_under_covariate_shift(model_with_education, model_without_education, od, with_education_features=OD_SECTOR_FEATURES, without_education_features=OD_ROBUST_SECTOR_FEATURES, profile_features=SECTOR_SHIFT_PROFILE_FEATURES):
    """Re-impute after refitting the selected models on known-sector rows **reweighted to the unknown-sector profile**.

    The validation in this stage only sees known-sector households; the imputation target is a different population
    (more educated, item non-response elsewhere in the questionnaire). Under MAR-given-x the two fits should agree;
    the difference in the imputed sector distribution is the sensitivity of the result to the shift. Education is not
    in the profile features (it is what differs most and is largely missing in the target). Returns
    ``(od_imputed_under_shift, diagnostics)``.
    """
    known = od[~od["sector_desconocido"]]
    unknown = od[od["sector_desconocido"]]
    reweighted_known, diagnostics = reweight_to_target_profile(known, unknown, profile_features, "expansion_factor", "expansion_factor")
    shifted = pd.concat([reweighted_known, unknown]).sort_index()
    model_with_education_shift, _ = refit_probabilistic_sector_model(model_with_education, shifted, sector_features=with_education_features)
    model_without_education_shift, _ = refit_probabilistic_sector_model(model_without_education, shifted, sector_features=without_education_features)
    imputed = impute_missing_sectors_hybrid(model_with_education_shift, model_without_education_shift, od, with_education_features=with_education_features, without_education_features=without_education_features)

    return imputed, diagnostics

def adjust_imputed_sector_share(od_imputed, sector_class, target_share=None):
    """Delta adjustment: scale the imputed probability of ``sector_class`` so the weighted imputed share equals
    ``target_share`` (default: the observed share among known-sector workers), renormalizing the other classes.
    Returns the adjusted frame (``sector_final`` recomputed for imputed rows) and the scaling factor."""
    od_imputed = od_imputed.copy()
    imputed = od_imputed["sector_fue_imputado"].to_numpy()
    weights = od_imputed["expansion_factor"].astype(float)
    column = f"prob_sector_{sector_class}"
    if target_share is None:
        target_share = (weights[~imputed] * (od_imputed.loc[~imputed, "sector_final"] == sector_class).to_numpy(dtype=float)).sum() / weights[~imputed].sum()
    current_share = (weights[imputed] * od_imputed.loc[imputed, column]).sum() / weights[imputed].sum()
    factor = target_share / current_share
    other_columns = [f"prob_sector_{other}" for other in SECTOR_CLASSES if other != sector_class]
    adjusted = (od_imputed.loc[imputed, column] * factor).clip(upper=1.0)
    remaining = 1.0 - adjusted
    other_total = od_imputed.loc[imputed, other_columns].sum(axis=1).replace(0, np.nan)
    for other in other_columns:
        od_imputed.loc[imputed, other] = (od_imputed.loc[imputed, other] / other_total * remaining).fillna(0.0)
    od_imputed.loc[imputed, column] = adjusted
    probability_columns = [f"prob_sector_{name}" for name in SECTOR_CLASSES]
    od_imputed.loc[imputed, "sector_final"] = np.array(SECTOR_CLASSES)[od_imputed.loc[imputed, probability_columns].to_numpy().argmax(axis=1)]
    od_imputed.loc[imputed, "sector_imputado"] = od_imputed.loc[imputed, "sector_final"]
    validate_sector_probability_rows(od_imputed)

    return od_imputed, float(factor)


def sector_test_metrics_with_uncertainty(model, test_data, sector_features=OD_SECTOR_FEATURES, n_bootstrap=500, random_state=42):
    """Held-out log loss / accuracy / macro-F1 with household-bootstrap intervals, plus the weighted-marginal baseline."""
    from sklearn.metrics import log_loss

    X = prepare_sector_features(test_data, sector_features)
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = list(model.named_steps["classifier"].classes_)
    frame = pd.DataFrame(probabilities, columns=[f"p_{c}" for c in classes])
    frame["y"] = test_data["sector"].to_numpy(); frame["w"] = test_data["expansion_factor"].astype(float).to_numpy(); frame["household"] = test_data["folio_vivienda"].astype(str).to_numpy()
    probability_columns = [f"p_{c}" for c in classes]

    def metrics(data):
        p = data[probability_columns].to_numpy(); predicted = np.array(classes)[p.argmax(axis=1)]
        return {
            "weighted_log_loss": log_loss(data["y"], p, labels=classes, sample_weight=data["w"]),
            "marginal_log_loss": marginal_log_loss(data["y"], data["w"], classes),
            "weighted_accuracy": accuracy_score(data["y"], predicted, sample_weight=data["w"]),
            "weighted_f1_macro": f1_score(data["y"], predicted, average="macro", sample_weight=data["w"], zero_division=0),
        }

    summary = bootstrap_by_group(frame, "household", metrics, n_bootstrap=n_bootstrap, random_state=random_state)
    summary.loc["relative_improvement_over_marginal", "estimate"] = 1 - summary.loc["weighted_log_loss", "estimate"] / summary.loc["marginal_log_loss", "estimate"]

    return summary
