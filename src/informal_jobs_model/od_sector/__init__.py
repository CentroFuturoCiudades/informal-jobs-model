"""Imputation of the economic activity (``giro_empresa``) of OD workers who did not report it.

Self-contained: depends on eodgdl (survey, zone system), mxcensus (DENUE, Marco Geoestadístico) and scikit-learn
only, and never on ENOE or on the harmonized variables of the informality pipeline. Public entry points:

- :func:`build_worker_features` — the worker frame from ``eodgdl.load_eod()`` tables;
- the training/evaluation functions of :mod:`model` (see the stage-4 notebook);
- :func:`impute_giro` — score a worker frame with a fitted bundle.
"""

from ._config import (
    DENUE_RELEASE, DENUE_STATE_CODE, DESTINATION_AMBITO_LEVELS, DESTINATION_FEATURES, EMPLOYED_CATEGORIES, GIRO_CLASSES,
    GIRO_LABELS, GIRO_SLUGS, KEYS, MOBILITY_FEATURES, NO_ESPECIFICADO, NUMERIC_FEATURES, ROBUST_SECTOR_FEATURES,
    SECTOR_FEATURES, SHIFT_PROFILE_FEATURES, build_category_levels, load_config,
)
from ._ml import (
    count_levels_without_training_support, fold_table, identify_missing_category, prepare_model_features, select_one_se,
)
from .features import add_destination_features, build_worker_features, compute_work_trip_destination
from .model import (
    PROBABILITY_COLUMNS, adjust_imputed_share, build_models, calculate_calibration, calculate_confidence_summary,
    calculate_distribution, calculate_feature_missingness, calculate_model_usage, calculate_probabilistic_distribution,
    compare_known_unknown_profiles, evaluate_model, fit_destination_models, get_best_model, impute_giro,
    impute_under_covariate_shift, missing_education, prepare_training_data, refit_model, split_known_data,
    test_metrics_with_uncertainty, tune_models, validate_probability_rows,
)
