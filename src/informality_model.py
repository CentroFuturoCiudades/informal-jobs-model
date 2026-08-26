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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SECTOR_CLASSES = ["comercio", "gobierno_otro_agricultura", "manufactura_construccion", "servicios_transporte"]
INFORMALITY_FEATURES = ["genero", "ocupacion", "edad_num", "escolaridad", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector"]
INFORMALITY_ROBUST_FEATURES = ["genero", "ocupacion", "edad_num", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector"]
ENOE_HOUSEHOLD_COLUMNS = ["tipo", "mes_cal", "cd_a", "ent", "con", "v_sel", "n_hog", "h_mud"]

# General helpers
def identify_missing_category(series):
    text = series.astype("string").str.strip().str.lower()

    return series.isna() | text.eq("").fillna(False) | text.eq("no_especificado").fillna(False)

def normalize_sample_weights(sample_weights):
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

def prepare_informality_features(dataframe, features):
    X = dataframe[features].copy()

    numerical_features = [column for column in features if column == "edad_num"]
    categorical_features = [column for column in features if column not in numerical_features]

    for column in numerical_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_features:
        X[column] = X[column].astype("string").str.strip().replace("", "no_especificado").fillna("no_especificado").astype(object)

    return X

def predict_informal_probability(model, X):
    probabilities = normalize_predicted_probabilities(model.predict_proba(X))
    classes = model.named_steps["classifier"].classes_
    informal_index = np.where(classes == 1)[0]

    if len(informal_index) != 1:
        raise ValueError("The classifier does not contain a unique informal class coded as 1.")

    return probabilities[:, informal_index[0]], probabilities

# Geography and diagnostics
def select_common_informality_population(enoe, od):
    excluded_municipalities = {"otro", "no_especificado"}

    enoe_municipalities = set(enoe["municipio"].dropna().astype(str)) - excluded_municipalities
    od_municipalities = set(od["municipio"].dropna().astype(str)) - excluded_municipalities
    common_municipalities = sorted(enoe_municipalities & od_municipalities)
    all_municipalities = sorted(enoe_municipalities | od_municipalities)

    training_population = enoe[enoe["municipio"].isin(common_municipalities)].copy()
    training_population = training_population[training_population["informal"].notna()].copy()
    training_population = training_population[training_population["sector"].isin(SECTOR_CLASSES)].copy()
    training_population = training_population.reset_index(drop=True)

    geography_summary = pd.DataFrame({
        "municipio": all_municipalities,
        "enoe": [municipality in enoe_municipalities for municipality in all_municipalities],
        "od": [municipality in od_municipalities for municipality in all_municipalities],
        "common": [municipality in common_municipalities for municipality in all_municipalities]
    })

    return training_population, geography_summary

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

    benchmark = pd.DataFrame({
        "sample_workers": [len(enoe)],
        "weighted_population": [weighted_population],
        "weighted_informal_population": [weighted_informal_population],
        "unweighted_informality_rate": [informal.mean()],
        "weighted_informality_rate": [weighted_rate]
    })

    return benchmark

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
    training_data = training_data[training_data["sector"].isin(SECTOR_CLASSES)].copy()
    training_data = training_data.reset_index(drop=True)

    X = prepare_informality_features(training_data, features)
    y = training_data["informal"].astype(int).reset_index(drop=True)
    sample_weights = normalize_sample_weights(training_data["survey_weight"]).reset_index(drop=True)
    groups = training_data[ENOE_HOUSEHOLD_COLUMNS].astype("string").agg("_".join, axis=1).reset_index(drop=True)

    return X, y, sample_weights, groups, training_data

# Models
def build_informality_models(features=INFORMALITY_FEATURES, random_state=42):
    numerical_features = [column for column in features if column == "edad_num"]
    categorical_features = [column for column in features if column not in numerical_features]

    linear_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    linear_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    tree_numerical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    tree_categorical_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="no_especificado")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    linear_preprocessor = ColumnTransformer([("numerical", linear_numerical_preprocessor, numerical_features), ("categorical", linear_categorical_preprocessor, categorical_features)])
    tree_preprocessor = ColumnTransformer([("numerical", tree_numerical_preprocessor, numerical_features), ("categorical", tree_categorical_preprocessor, categorical_features)])

    models = {
        "LogisticRegression": {
            "model": Pipeline([("preprocessor", linear_preprocessor), ("classifier", LogisticRegression(max_iter=2000, random_state=random_state))]),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "model": Pipeline([("preprocessor", tree_preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, class_weight=None, random_state=random_state, n_jobs=-1))]),
            "params": {
                "classifier__max_leaf_nodes": [25, 50, 100],
                "classifier__max_features": ["sqrt", 0.7],
                "classifier__min_samples_leaf": [1, 5]
            }
        },
        "GradientBoosting": {
            "model": Pipeline([("preprocessor", tree_preprocessor), ("classifier", HistGradientBoostingClassifier(max_iter=500, early_stopping=True, class_weight=None, random_state=random_state))]),
            "params": {
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
    best_models = {}

    for model_name, model_config in models.items():
        parameter_grid = list(ParameterGrid(model_config["params"]))
        best_result = None

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
                fold_f1.append(f1_score(y_validation, predictions, sample_weight=weights_validation))
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
                "best_params": params
            }

            if best_result is None or result["weighted_log_loss"] < best_result["weighted_log_loss"]:
                best_result = result

        best_model = clone(model_config["model"])
        best_model.set_params(**best_result["best_params"])
        best_model.fit(X, y, classifier__sample_weight=sample_weights)

        model_results.append(best_result)
        best_models[model_name] = best_model

    model_summary = pd.DataFrame(model_results).sort_values("weighted_log_loss").reset_index(drop=True)

    return model_summary, best_models

def get_best_informality_model(model_summary, best_models):
    best_model_name = model_summary.iloc[0]["model"]
    best_model = best_models[best_model_name]

    return best_model_name, best_model



# Calibration
def calculate_calibration_table(y_true, probabilities, sample_weights, n_bins=10):
    calibration = pd.DataFrame({
        "informal": np.asarray(y_true, dtype=float),
        "probability": np.asarray(probabilities, dtype=float),
        "weight": np.asarray(sample_weights, dtype=float)
    })

    calibration["bin"] = pd.cut(calibration["probability"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    calibration["weighted_informal"] = calibration["weight"] * calibration["informal"]
    calibration["weighted_probability"] = calibration["weight"] * calibration["probability"]

    calibration_table = calibration.groupby("bin", observed=True).agg(sample_workers=("informal", "size"), weighted_population=("weight", "sum"), weighted_informal=("weighted_informal", "sum"), weighted_probability=("weighted_probability", "sum")).reset_index()

    calibration_table["observed_rate"] = calibration_table["weighted_informal"] / calibration_table["weighted_population"]
    calibration_table["predicted_probability"] = calibration_table["weighted_probability"] / calibration_table["weighted_population"]

    return calibration_table



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
        "f1": [f1_score(y_true, predictions)],
        "weighted_accuracy": [accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_balanced_accuracy": [balanced_accuracy_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_f1": [f1_score(y_true, predictions, sample_weight=sample_weights)],
        "weighted_roc_auc": [roc_auc_score(y_true, informal_probability, sample_weight=sample_weights)],
        "weighted_log_loss": [log_loss(y_true, probabilities, labels=classes, sample_weight=sample_weights)],
        "weighted_brier_score": [np.average((informal_probability - y_true.to_numpy()) ** 2, weights=sample_weights)],
        "observed_informality_rate": [observed_rate],
        "predicted_informality_rate": [predicted_rate],
        "calibration_gap_pp": [(predicted_rate - observed_rate) * 100]
    })

    weighted_confusion = confusion_matrix(y_true, predictions, labels=[0, 1], sample_weight=sample_weights)
    confusion = pd.DataFrame(weighted_confusion, index=["formal", "informal"], columns=["formal", "informal"])

    calibration_table = calculate_calibration_table(y_true, informal_probability, sample_weights, n_bins=n_calibration_bins)

    return metrics, confusion, calibration_table



# Final ENOE models
def refit_informality_model(model, enoe, features=INFORMALITY_FEATURES):
    X, y, sample_weights, groups, training_data = prepare_enoe_informality_training_data(enoe, features=features)

    final_model = clone(model)
    final_model.fit(X, y, classifier__sample_weight=sample_weights)

    return final_model, training_data



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
def predict_od_informality(model_with_education, model_without_education, od, with_education_features=INFORMALITY_FEATURES, without_education_features=INFORMALITY_ROBUST_FEATURES, threshold=0.5):
    od = od.copy()

    validate_od_sector_probabilities(od)

    missing_education = identify_missing_category(od["escolaridad"])
    use_with_education = ~missing_education
    use_without_education = missing_education

    od["informality_model_used"] = pd.Series(pd.NA, index=od.index, dtype="string")
    od.loc[use_with_education, "informality_model_used"] = "with_education"
    od.loc[use_without_education, "informality_model_used"] = "without_education"

    informal_joint_columns = []
    formal_joint_columns = []

    for sector_class in SECTOR_CLASSES:
        scenario_data = od.copy()
        scenario_data["sector"] = sector_class

        conditional_probability = np.full(len(od), np.nan)

        if use_with_education.any():
            X_with_education = prepare_informality_features(scenario_data.loc[use_with_education], with_education_features)
            probability_with_education, _ = predict_informal_probability(model_with_education, X_with_education)
            conditional_probability[np.flatnonzero(use_with_education.to_numpy())] = probability_with_education

        if use_without_education.any():
            X_without_education = prepare_informality_features(scenario_data.loc[use_without_education], without_education_features)
            probability_without_education, _ = predict_informal_probability(model_without_education, X_without_education)
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