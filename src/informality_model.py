import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .diagnose_enoe_od_dataframes import filter_common_geography
from .common import NO_ESPECIFICADO, SECTOR_CLASSES, assert_known_levels, bootstrap_by_group, cross_validate_grouped, marginal_log_loss, reweight_to_target_profile, calculate_calibration_table, calibration_metrics, fit_isotonic_calibrator, select_one_se, attach_training_level_shares, predict_proba_marginalizing, build_category_levels, count_levels_without_training_support, identify_missing_category, normalize_predicted_probabilities, normalize_sample_weights, prepare_model_features, split_feature_types


INFORMALITY_FEATURES = ["genero", "ocupacion", "edad_num", "escolaridad", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector", "lugar_trabajo"]
INFORMALITY_ROBUST_FEATURES = ["genero", "ocupacion", "edad_num", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector", "lugar_trabajo"]
from .generate_enoe_od_dataframes import ENOE_GROUP_KEYS as ENOE_HOUSEHOLD_COLUMNS  # cross-quarter household key: CV groups and bootstrap clusters

# General helpers

def prepare_informality_features(dataframe, features):
    return prepare_model_features(dataframe, features)

def predict_informal_probability(model, X, marginalize_unsupported=False, level_subsets=None):
    """P(informal) from a fitted pipeline; with ``marginalize_unsupported`` levels without training support are
    marginalized (see ``common.predict_proba_marginalizing``), otherwise ``predict_proba`` is used as is."""
    if marginalize_unsupported:
        probabilities, _ = predict_proba_marginalizing(model, X, level_subsets=level_subsets)
    else:
        probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = model.named_steps["classifier"].classes_
    informal_index = np.where(classes == 1)[0]

    if len(informal_index) != 1:
        raise ValueError("The classifier does not contain a unique informal class coded as 1.")

    return probabilities[:, informal_index[0]], probabilities

# Geography and diagnostics
def select_common_informality_population(enoe, od):
    """Training and benchmark populations for the informality model.

    Training uses **all** ENOE workers in the state with a valid label: municipalities outside the metro area enter
    as ``municipio = "otro"`` and the model separates the two regimes through that feature (review item 1.13; on the
    metro held-out fold this lowers log loss slightly and leaves the OD estimate unchanged). The benchmark the OD
    estimate is compared with is restricted to the metro municipalities sampled by both surveys (same rule as stage
    3). OD workers in metro municipalities ENOE did not sample are scored by averaging over the sampled metro
    municipalities (never as ``otro``, which here means non-metro Jalisco).

    Returns ``(training_population, benchmark_population, training_municipalities, geography_summary)``.
    """
    enoe_common, _, geography_summary = filter_common_geography(enoe, od)
    training_municipalities = set(geography_summary.loc[geography_summary["enoe"], "municipio"])

    training_population = enoe[enoe["informal"].notna()].reset_index(drop=True).copy()
    benchmark_population = enoe_common[enoe_common["informal"].notna()].reset_index(drop=True).copy()

    od_weights = od.groupby("municipio")["expansion_factor"].sum()
    geography_summary["od_workers"] = geography_summary["municipio"].map(od.groupby("municipio").size()).fillna(0).astype(int)
    geography_summary["od_weighted_share"] = geography_summary["municipio"].map(od_weights / od_weights.sum()).fillna(0.0)

    return training_population, benchmark_population, training_municipalities, geography_summary

def compare_informality_feature_missingness(enoe, od, columns):
    results = []
    for column in columns:
        enoe_missing = identify_missing_category(enoe[column])
        od_missing = identify_missing_category(od[column])

        enoe_missing_population = enoe.loc[enoe_missing, "survey_weight"].sum()
        od_missing_population = od.loc[od_missing, "expansion_factor"].sum()

        results.append({
            "variable": column,
            "enoe_missing_share": enoe_missing_population / enoe["survey_weight"].sum(),
            "od_missing_share": od_missing_population / od["expansion_factor"].sum()
        })

    return pd.DataFrame(results)

def calculate_enoe_informality_benchmark(enoe):
    sample_weights = enoe["survey_weight"].astype(float)
    informal = enoe["informal"].astype(float)

    weighted_population = sample_weights.sum()
    weighted_informal_population = (sample_weights * informal).sum()
    weighted_rate = weighted_informal_population / weighted_population

    unspecified_sector = enoe["sector"].eq(NO_ESPECIFICADO)
    benchmark = pd.DataFrame({
        "sample_workers": [len(enoe)],
        "weighted_population": [weighted_population],
        "weighted_informal_population": [weighted_informal_population],
        "unweighted_informality_rate": [informal.mean()],
        "weighted_informality_rate": [weighted_rate],
        "unspecified_sector_workers": [int(unspecified_sector.sum())],
        "unspecified_sector_weighted_share": [sample_weights[unspecified_sector].sum() / weighted_population],
        "unspecified_sector_informality_rate": [(sample_weights * informal)[unspecified_sector].sum() / max(sample_weights[unspecified_sector].sum(), 1)]
    })

    return benchmark

def calculate_enoe_informality_by_period(enoe):
    """Weighted informality rate per pooled ENOE quarter (weights were divided by the number of quarters, so the
    per-quarter weighted population is the quarter's population divided by that number; rates are unaffected)."""
    rows = []
    for period, frame in enoe.groupby("period") if "period" in enoe else [("all", enoe)]:
        weights = frame["survey_weight"].astype(float); informal = frame["informal"].astype(float)
        rows.append({"period": period, "sample_workers": len(frame), "weighted_informality_rate": (weights * informal).sum() / weights.sum()})

    return pd.DataFrame(rows)

# Train-test split
def split_enoe_informality_data(enoe, n_splits=5, test_fold=0, random_state=42):
    enoe = enoe.reset_index(drop=True)

    y = enoe["informal"].astype(int)
    groups = enoe[ENOE_HOUSEHOLD_COLUMNS].astype("string").agg("_".join, axis=1)

    cross_validation = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cross_validation.split(enoe, y, groups=groups))
    train_index, test_index = splits[test_fold]

    training_data = enoe.iloc[train_index].copy()
    test_data = enoe.iloc[test_index].copy()

    return training_data, test_data

# Training data
def prepare_enoe_informality_training_data(enoe, features=INFORMALITY_FEATURES):
    training_data = enoe[enoe["informal"].notna()].copy()
    training_data = training_data.reset_index(drop=True)

    X = prepare_informality_features(training_data, features)
    y = training_data["informal"].astype(int).reset_index(drop=True)
    sample_weights = normalize_sample_weights(training_data["survey_weight"]).reset_index(drop=True)
    groups = training_data[ENOE_HOUSEHOLD_COLUMNS].astype("string").agg("_".join, axis=1).reset_index(drop=True)

    return X, y, sample_weights, groups, training_data

# Models
def build_informality_models(features=INFORMALITY_FEATURES, random_state=42):
    numerical_features, categorical_features = split_feature_types(features)
    category_levels = build_category_levels()
    categories = [category_levels[column] for column in categorical_features]

    linear_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    linear_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error"))])

    tree_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    tree_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False))])

    # The pipelines are self-contained: the first step selects the features and cleans them (numeric coercion,
    # categorical strings with the missing label), so a pickled bundle applies to a raw harmonized frame.
    prepare = FunctionTransformer(prepare_model_features, kw_args={"features": list(features)})
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
            "model": Pipeline([("prepare", prepare), ("preprocessor", tree_preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, class_weight=None, random_state=random_state, n_jobs=-1))]),
            "params": {
                "classifier__max_leaf_nodes": [25, 50, 100],
                "classifier__max_features": ["sqrt", 0.7],
                "classifier__min_samples_leaf": [1, 5]
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



# Weighted cross-validation
def tune_informality_models(X, y, sample_weights, groups, features=INFORMALITY_FEATURES, cv_splits=5, random_state=42, verbose=True):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sample_weights = sample_weights.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    models = build_informality_models(features=features, random_state=random_state)
    cross_validation = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    splits = list(cross_validation.split(X, y, groups=groups))

    model_results = []
    grid_results = []
    best_models = {}

    for model_name, model_config in models.items():
        parameter_grid = list(ParameterGrid(model_config["params"]))
        family_results = []

        if verbose:
            print(f"Tuning {model_name}: {len(parameter_grid)} parameter combinations")

        for params in parameter_grid:
            fold_log_loss = []
            fold_brier = []
            fold_balanced_accuracy = []
            fold_accuracy = []
            fold_f1 = []
            fold_auc = []

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
                informal_probability, probabilities = predict_informal_probability(model, X_validation)
                classes = model.named_steps["classifier"].classes_

                fold_log_loss.append(log_loss(y_validation, probabilities, labels=classes, sample_weight=weights_validation))
                fold_brier.append(np.average((informal_probability - y_validation.to_numpy()) ** 2, weights=weights_validation))
                fold_balanced_accuracy.append(balanced_accuracy_score(y_validation, predictions, sample_weight=weights_validation))
                fold_accuracy.append(accuracy_score(y_validation, predictions, sample_weight=weights_validation))
                fold_f1.append(f1_score(y_validation, predictions, sample_weight=weights_validation, zero_division=0))
                fold_auc.append(roc_auc_score(y_validation, informal_probability, sample_weight=weights_validation))

            result = {
                "model": model_name,
                "weighted_log_loss": np.mean(fold_log_loss),
                "weighted_log_loss_std": np.std(fold_log_loss),
                "weighted_brier_score": np.mean(fold_brier),
                "weighted_balanced_accuracy": np.mean(fold_balanced_accuracy),
                "weighted_accuracy": np.mean(fold_accuracy),
                "weighted_f1": np.mean(fold_f1),
                "weighted_roc_auc": np.mean(fold_auc),
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

def get_best_informality_model(model_summary, best_models):
    best_model_name = model_summary.loc[model_summary["selected"], "model"].iloc[0]
    best_model = best_models[best_model_name]

    return best_model_name, best_model



# Calibration
# Masked evaluation of the without-education model on the OD non-respondent profile (review item 2.4)
OD_PROFILE_FEATURES = ["genero", "ocupacion", "edad_num", "municipio", "estado_civil", "parentesco", "tamano_viv_cat"]

def reweight_to_od_profile(enoe_rows, od_target_rows, features=OD_PROFILE_FEATURES, weight_column="survey_weight", random_state=42):
    """ENOE rows reweighted to the covariate profile of a target OD sub-population (see ``common.reweight_to_target_profile``)."""
    return reweight_to_target_profile(enoe_rows, od_target_rows, features, weight_column, "expansion_factor", random_state=random_state)

def evaluate_on_od_profile(models, enoe_test, od_target_rows, features_by_model, profile_features=OD_PROFILE_FEATURES):
    """Held-out metrics for ``{label: model}`` on the same ENOE rows under survey weights and under weights matched to
    ``od_target_rows`` (e.g. the OD workers with missing education). ``features_by_model`` maps label -> feature list."""
    reweighted, diagnostics = reweight_to_od_profile(enoe_test, od_target_rows, features=profile_features)
    rows = []
    for weighting, data in (("survey weights", enoe_test), ("OD non-respondent profile", reweighted)):
        for label, model in models.items():
            metrics, _, _ = evaluate_informality_model(model, data, features=features_by_model[label])
            rows.append(metrics.iloc[0].rename((weighting, label)))
    table = pd.DataFrame(rows)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["weighting", "model"])

    return table, diagnostics

# Held-out evaluation
def evaluate_informality_model(model, validation_data, features=INFORMALITY_FEATURES, n_calibration_bins=10):
    X = prepare_informality_features(validation_data, features)
    y_true = validation_data["informal"].astype(int)
    sample_weights = validation_data["survey_weight"].astype(float)

    predictions = model.predict(X)
    informal_probability, probabilities = predict_informal_probability(model, X)
    classes = model.named_steps["classifier"].classes_

    observed_rate = np.average(y_true, weights=sample_weights)
    predicted_rate = np.average(informal_probability, weights=sample_weights)

    metrics = pd.DataFrame({
        "accuracy": [accuracy_score(y_true, predictions)],
        "balanced_accuracy": [balanced_accuracy_score(y_true, predictions)],
        "f1": [f1_score(y_true, predictions, zero_division=0)],
        "weighted_accuracy": [accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_balanced_accuracy": [balanced_accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_f1": [f1_score(y_true, predictions, sample_weight=sample_weights, zero_division=0)],
        "weighted_roc_auc": [roc_auc_score(y_true, informal_probability, sample_weight=sample_weights)],
        "weighted_log_loss": [log_loss(y_true, probabilities, labels=classes, sample_weight=sample_weights)],
        "weighted_brier_score": [np.average((informal_probability - y_true.to_numpy()) ** 2, weights=sample_weights)],
        "observed_informality_rate": [observed_rate],
        "predicted_informality_rate": [predicted_rate],
        "calibration_gap_pp": [(predicted_rate - observed_rate) * 100]
    })
    for name, value in calibration_metrics(y_true, informal_probability, sample_weights, n_bins=n_calibration_bins).items():
        if name != "calibration_gap_pp":
            metrics[name] = [value]

    weighted_confusion = confusion_matrix(y_true, predictions, labels=[0, 1], sample_weight=sample_weights)
    confusion = pd.DataFrame(weighted_confusion, index=["formal", "informal"], columns=["formal", "informal"])

    calibration_table = calculate_calibration_table(y_true, informal_probability, sample_weights, n_bins=n_calibration_bins)

    return metrics, confusion, calibration_table



# Final ENOE models
def refit_informality_model(model, enoe, features=INFORMALITY_FEATURES, calibrate=False, cv_splits=5, random_state=42):
    """Refit the selected pipeline on ``enoe``; with ``calibrate=True`` wrap it in an isotonic map learned from
    household-grouped out-of-fold predictions (see ``common.fit_isotonic_calibrator``)."""
    X, y, sample_weights, groups, training_data = prepare_enoe_informality_training_data(enoe, features=features)

    final_model = clone(model)
    final_model.fit(X, y, classifier__sample_weight=sample_weights)
    attach_training_level_shares(final_model, X, sample_weights)
    if calibrate:
        final_model = fit_isotonic_calibrator(final_model, X, y, sample_weights, groups, cv_splits=cv_splits, random_state=random_state)

    return final_model, training_data

def calibrate_informality_model(model, enoe, features=INFORMALITY_FEATURES, cv_splits=5, random_state=42):
    """Isotonic-calibrated version of an already fitted pipeline, using out-of-fold predictions on ``enoe``."""
    X, y, sample_weights, groups, _ = prepare_enoe_informality_training_data(enoe, features=features)

    return fit_isotonic_calibrator(model, X, y, sample_weights, groups, cv_splits=cv_splits, random_state=random_state)

def compare_informality_models(models, validation_data, features=INFORMALITY_FEATURES, n_calibration_bins=10):
    """Held-out metrics side by side for a ``{label: model}`` dict (e.g. uncalibrated vs isotonic)."""
    rows, tables = [], {}
    for label, model in models.items():
        metrics, _, table = evaluate_informality_model(model, validation_data, features=features, n_calibration_bins=n_calibration_bins)
        rows.append(metrics.iloc[0].rename(label))
        tables[label] = table

    return pd.DataFrame(rows), tables



# OD sector-probability validation
def validate_od_sector_probabilities(od, tolerance=1e-8):
    probability_columns = [f"prob_sector_{sector_class}" for sector_class in SECTOR_CLASSES]

    if not set(probability_columns).issubset(od.columns):
        missing_columns = sorted(set(probability_columns) - set(od.columns))
        raise ValueError(f"Missing sector-probability columns: {missing_columns}")

    probability_sums = od[probability_columns].sum(axis=1)
    maximum_error = np.max(np.abs(probability_sums - 1.0))

    if maximum_error > tolerance:
        raise ValueError(f"Sector probabilities do not sum to one. Maximum error: {maximum_error:.3e}")

    return maximum_error



# Apply informality models to OD
def predict_od_informality(model_with_education, model_without_education, od, with_education_features=INFORMALITY_FEATURES, without_education_features=INFORMALITY_ROBUST_FEATURES, threshold=0.5, training_municipalities=None, level_subsets=None):
    od = od.copy()

    validate_od_sector_probabilities(od)

    # Metro municipalities the ENOE sample does not cover are scored by marginalizing over the sampled metro
    # municipalities (their value is set to no_especificado, which has no training support, and the marginalization
    # is restricted to ``training_municipalities``). "otro" would mean non-metro Jalisco and is not used for them.
    od["municipio_scored"] = od["municipio"].astype("string")
    level_subsets = dict(level_subsets or {})  # e.g. {"lugar_trabajo": ["otro_o_sin_local"]} to score workers without a work trip as home/mobile work
    if training_municipalities is not None:
        unsampled = ~od["municipio"].isin(set(training_municipalities) | {"otro", NO_ESPECIFICADO})
        od.loc[unsampled, "municipio_scored"] = NO_ESPECIFICADO
        level_subsets["municipio"] = sorted(training_municipalities)
    level_subsets = level_subsets or None
    scoring_data = od.drop(columns=["municipio"]).rename(columns={"municipio_scored": "municipio"})

    missing_education = identify_missing_category(od["escolaridad"])
    use_with_education = ~missing_education
    use_without_education = missing_education

    od["informality_model_used"] = pd.Series(pd.NA, index=od.index, dtype="string")
    od.loc[use_with_education, "informality_model_used"] = "with_education"
    od.loc[use_without_education, "informality_model_used"] = "without_education"

    od["informality_marginalized_features"] = pd.Series("", index=od.index, dtype="string")
    for use_mask, model, features in ((use_with_education, model_with_education, with_education_features), (use_without_education, model_without_education, without_education_features)):
        if use_mask.any():
            scenario_features = [column for column in features if column != "sector"]
            _, marginalized = predict_proba_marginalizing(model, prepare_informality_features(scoring_data.loc[use_mask].assign(sector=SECTOR_CLASSES[0]), features)[scenario_features + ["sector"]], level_subsets=level_subsets)
            od.loc[use_mask, "informality_marginalized_features"] = marginalized.to_numpy()

    informal_joint_columns = []
    formal_joint_columns = []

    assert_known_levels(prepare_informality_features(scoring_data, [column for column in with_education_features if column != "sector"]))

    scenario_columns = list(dict.fromkeys(list(with_education_features) + list(without_education_features)))
    scenario_base = scoring_data[scenario_columns].copy()
    for sector_class in SECTOR_CLASSES:
        scenario_data = scenario_base.assign(sector=sector_class)

        conditional_probability = np.full(len(od), np.nan)

        if use_with_education.any():
            X_with_education = prepare_informality_features(scenario_data.loc[use_with_education], with_education_features)
            probability_with_education, _ = predict_informal_probability(model_with_education, X_with_education, marginalize_unsupported=True, level_subsets=level_subsets)
            conditional_probability[np.flatnonzero(use_with_education.to_numpy())] = probability_with_education

        if use_without_education.any():
            X_without_education = prepare_informality_features(scenario_data.loc[use_without_education], without_education_features)
            probability_without_education, _ = predict_informal_probability(model_without_education, X_without_education, marginalize_unsupported=True, level_subsets=level_subsets)
            conditional_probability[np.flatnonzero(use_without_education.to_numpy())] = probability_without_education

        if np.isnan(conditional_probability).any():
            raise ValueError(f"Missing conditional informality probabilities for sector {sector_class}.")

        conditional_column = f"prob_informal_given_sector_{sector_class}"
        informal_joint_column = f"prob_informal_sector_{sector_class}"
        formal_joint_column = f"prob_formal_sector_{sector_class}"
        sector_probability_column = f"prob_sector_{sector_class}"

        od[conditional_column] = conditional_probability
        od[informal_joint_column] = od[sector_probability_column].astype(float) * od[conditional_column]
        od[formal_joint_column] = od[sector_probability_column].astype(float) * (1 - od[conditional_column])

        informal_joint_columns.append(informal_joint_column)
        formal_joint_columns.append(formal_joint_column)

    od["prob_informal"] = od[informal_joint_columns].sum(axis=1)
    od["prob_formal"] = od[formal_joint_columns].sum(axis=1)

    probability_error = np.max(np.abs(od["prob_informal"] + od["prob_formal"] - 1.0))

    if probability_error > 1e-8:
        raise ValueError(f"Formal and informal probabilities do not sum to one. Maximum error: {probability_error:.3e}")

    od["informal_predicted"] = (od["prob_informal"] >= threshold).astype("Int64")
    od["informality_prediction_confidence"] = np.maximum(od["prob_informal"], od["prob_formal"])

    od["prob_informal_hard_sector"] = np.nan

    for sector_class in SECTOR_CLASSES:
        sector_mask = od["sector_final"].eq(sector_class)
        od.loc[sector_mask, "prob_informal_hard_sector"] = od.loc[sector_mask, f"prob_informal_given_sector_{sector_class}"]

    od["expected_informal_population"] = od["expansion_factor"] * od["prob_informal"]
    od["expected_formal_population"] = od["expansion_factor"] * od["prob_formal"]

    return od



# Final summaries
def calculate_od_informality_summary(od):
    total_population = od["expansion_factor"].sum()
    expected_informal_population = od["expected_informal_population"].sum()
    expected_formal_population = od["expected_formal_population"].sum()

    summary = pd.DataFrame({
        "sample_workers": [len(od)],
        "weighted_population": [total_population],
        "expected_informal_population": [expected_informal_population],
        "expected_formal_population": [expected_formal_population],
        "expected_informality_rate": [expected_informal_population / total_population],
        "hard_informality_rate": [np.average(od["informal_predicted"], weights=od["expansion_factor"])],
        "hard_sector_expected_rate": [np.average(od["prob_informal_hard_sector"], weights=od["expansion_factor"])]
    })

    return summary

def calculate_informality_by_sector(od):
    results = []

    for sector_class in SECTOR_CLASSES:
        sector_probability = od[f"prob_sector_{sector_class}"]
        informal_joint_probability = od[f"prob_informal_sector_{sector_class}"]
        formal_joint_probability = od[f"prob_formal_sector_{sector_class}"]

        weighted_sector_population = (od["expansion_factor"] * sector_probability).sum()
        weighted_informal_population = (od["expansion_factor"] * informal_joint_probability).sum()
        weighted_formal_population = (od["expansion_factor"] * formal_joint_probability).sum()

        results.append({
            "sector": sector_class,
            "weighted_population": weighted_sector_population,
            "expected_informal_population": weighted_informal_population,
            "expected_formal_population": weighted_formal_population,
            "expected_informality_rate": weighted_informal_population / weighted_sector_population
        })

    return pd.DataFrame(results)

def calculate_informality_model_usage(od):
    usage = od.groupby("informality_model_used").agg(sample_workers=("informality_model_used", "size"), weighted_population=("expansion_factor", "sum")).reset_index()

    usage["sample_share"] = usage["sample_workers"] / usage["sample_workers"].sum()
    usage["weighted_share"] = usage["weighted_population"] / usage["weighted_population"].sum()

    return usage

def calculate_expected_informality_by_variable(od, column):
    grouped = od.groupby(column, dropna=False).agg(weighted_population=("expansion_factor", "sum"), expected_informal_population=("expected_informal_population", "sum"), expected_formal_population=("expected_formal_population", "sum")).reset_index()

    grouped["expected_informality_rate"] = grouped["expected_informal_population"] / grouped["weighted_population"]

    return grouped

def informality_test_metrics_with_uncertainty(model, test_data, features=INFORMALITY_FEATURES, n_bootstrap=500, random_state=42):
    """Held-out log loss / AUC / aggregate gap with household-bootstrap intervals, plus the weighted-marginal baseline."""
    X = prepare_informality_features(test_data, features)
    informal_probability, _ = predict_informal_probability(model, X)
    frame = pd.DataFrame({"y": test_data["informal"].astype(int).to_numpy(), "p": informal_probability, "w": test_data["survey_weight"].astype(float).to_numpy(), "household": test_data[ENOE_HOUSEHOLD_COLUMNS].astype("string").agg("_".join, axis=1).to_numpy()})

    def metrics(data):
        p = np.column_stack([1 - data["p"], data["p"]])
        return {
            "weighted_log_loss": log_loss(data["y"], p, labels=[0, 1], sample_weight=data["w"]),
            "marginal_log_loss": marginal_log_loss(data["y"], data["w"], [0, 1]),
            "weighted_roc_auc": roc_auc_score(data["y"], data["p"], sample_weight=data["w"]),
            "calibration_gap_pp": (np.average(data["p"], weights=data["w"]) - np.average(data["y"], weights=data["w"])) * 100,
        }

    summary = bootstrap_by_group(frame, "household", metrics, n_bootstrap=n_bootstrap, random_state=random_state)
    summary.loc["relative_improvement_over_marginal", "estimate"] = 1 - summary.loc["weighted_log_loss", "estimate"] / summary.loc["marginal_log_loss", "estimate"]

    return summary


# Sampled informality status and decomposition of the ENOE -> OD gap (review item 2.7)
def sample_informality(od, probability_column="prob_informal", random_state=42):
    """One Bernoulli draw per worker, I ~ Bernoulli(prob_informal): a discrete status that is unbiased for every
    weighted aggregate (unlike the 0.5 threshold, which shrinks toward the majority class and, for workers whose
    sector was marginalized, toward the middle)."""
    rng = np.random.default_rng(random_state)

    return (rng.random(len(od)) < od[probability_column].to_numpy()).astype(int)

GAP_DECOMPOSITION_FEATURES = ["genero", "ocupacion", "edad_num", "escolaridad", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector"]

def decompose_enoe_od_gap(model_with_education, model_without_education, enoe_benchmark, od_informality, with_education_features=INFORMALITY_FEATURES, without_education_features=INFORMALITY_ROBUST_FEATURES, profile_features=GAP_DECOMPOSITION_FEATURES):
    """Direct standardization of the ENOE benchmark to the OD covariate profile.

    Steps: ENOE observed rate → ENOE rate predicted by the shipped model (model bias on ENOE) → the same two rates after
    reweighting the ENOE rows to the OD profile on ``profile_features`` (composition effect; OD's sector is
    ``sector_final``) → OD expected rate. The last difference is what composition and model bias do not explain
    (sector imputation, unsupported levels, non-response). Returns ``(table, reweighting_diagnostics)``.
    """
    enoe = enoe_benchmark.reset_index(drop=True).copy()
    od_profile = od_informality.copy()
    od_profile["sector"] = od_profile["sector_final"]
    od_profile.loc[~od_profile["municipio"].isin(set(enoe["municipio"])), "municipio"] = NO_ESPECIFICADO
    reweighted, diagnostics = reweight_to_target_profile(enoe, od_profile, profile_features, "survey_weight", "expansion_factor")

    def rates(frame):
        missing_education = identify_missing_category(frame["escolaridad"])
        probability = np.empty(len(frame))
        if (~missing_education).any():
            probability[np.flatnonzero(~missing_education.to_numpy())], _ = predict_informal_probability(model_with_education, prepare_informality_features(frame[~missing_education], with_education_features), marginalize_unsupported=True)
        if missing_education.any():
            probability[np.flatnonzero(missing_education.to_numpy())], _ = predict_informal_probability(model_without_education, prepare_informality_features(frame[missing_education], without_education_features), marginalize_unsupported=True)
        weights = frame["survey_weight"].astype(float).to_numpy()

        return np.average(frame["informal"].astype(float), weights=weights), np.average(probability, weights=weights)

    observed, predicted = rates(enoe)
    observed_reweighted, predicted_reweighted = rates(reweighted)
    od_weights = od_informality["expansion_factor"].astype(float)
    od_expected = float((od_weights * od_informality["prob_informal"]).sum() / od_weights.sum())
    steps = [
        ("ENOE observed (metro benchmark)", observed, "—"),
        ("ENOE predicted by the model", predicted, "model bias on ENOE"),
        ("ENOE observed, reweighted to the OD profile", observed_reweighted, "composition (observed)"),
        ("ENOE predicted, reweighted to the OD profile", predicted_reweighted, "composition (predicted)"),
        ("OD expected informality", od_expected, "residual: sector imputation, unsupported levels, non-response"),
    ]
    table = pd.DataFrame(steps, columns=["step", "rate", "interpretation"])
    table["difference_pp"] = table["rate"].diff() * 100
    table.loc[3, "difference_pp"] = (predicted_reweighted - predicted) * 100  # composition on the predicted scale vs the unweighted prediction
    table.loc[2, "difference_pp"] = (observed_reweighted - observed) * 100
    table.loc[4, "difference_pp"] = (od_expected - predicted_reweighted) * 100

    return table, diagnostics
