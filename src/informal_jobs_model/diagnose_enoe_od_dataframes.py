import pandas as pd

from .common import AMG_MUNICIPALITIES, HARMONIZED_FEATURES, identify_missing_category

def filter_common_geography(enoe, od, target_municipalities=AMG_MUNICIPALITIES):
    enoe = enoe.copy()
    od = od.copy()
    
    enoe_municipalities = set(enoe.loc[enoe["municipio"].isin(target_municipalities), "municipio"].unique())
    od_municipalities = set(od.loc[od["municipio"].isin(target_municipalities), "municipio"].unique())
    common_municipalities = sorted(enoe_municipalities & od_municipalities)

    enoe_common = enoe[enoe["municipio"].isin(common_municipalities)].copy()
    od_common = od[od["municipio"].isin(common_municipalities)].copy()

    geography_summary = pd.DataFrame({
        "municipio": target_municipalities,
        "enoe": [municipality in enoe_municipalities for municipality in target_municipalities],
        "od": [municipality in od_municipalities for municipality in target_municipalities],
        "common": [municipality in common_municipalities for municipality in target_municipalities]
    })

    return enoe_common, od_common, geography_summary

def calculate_weighted_distribution(dataframe, column, weight_column):
    distribution = dataframe.groupby(column, dropna=False)[weight_column].sum().reset_index(name="weighted_population")
    distribution["weighted_share"] = distribution["weighted_population"] / distribution["weighted_population"].sum()

    return distribution

def compare_weighted_distributions(enoe, od, columns):
    comparisons = []

    for column in columns:
        enoe_distribution = calculate_weighted_distribution(enoe, column, "survey_weight").rename(columns={"weighted_population": "enoe_population", "weighted_share": "enoe_share"})
        od_distribution = calculate_weighted_distribution(od, column, "expansion_factor").rename(columns={"weighted_population": "od_population", "weighted_share": "od_share"})
        comparison = enoe_distribution.merge(od_distribution, on=column, how="outer")
        comparison["variable"] = column
        comparison["category"] = comparison[column].astype("string")
        comparison["enoe_share"] = comparison["enoe_share"].fillna(0)
        comparison["od_share"] = comparison["od_share"].fillna(0)
        comparison["difference_pp"] = (comparison["od_share"] - comparison["enoe_share"]) * 100
        comparisons.append(comparison[["variable", "category", "enoe_population", "od_population", "enoe_share", "od_share", "difference_pp"]])

    return pd.concat(comparisons, ignore_index=True)

def calculate_weighted_informality_rate(enoe):
    valid = enoe["informal"].notna() & enoe["survey_weight"].notna()
    enoe_valid = enoe.loc[valid].copy()

    sample_workers = len(enoe_valid)
    sample_informal_workers = enoe_valid["informal"].sum()
    unweighted_rate = enoe_valid["informal"].mean()

    weighted_workers = enoe_valid["survey_weight"].sum()
    weighted_informal_workers = (enoe_valid["survey_weight"] * enoe_valid["informal"]).sum()
    weighted_rate = weighted_informal_workers / weighted_workers

    benchmark = pd.DataFrame({
        "sample_workers": [sample_workers],
        "sample_informal_workers": [sample_informal_workers],
        "unweighted_informality_rate": [unweighted_rate],
        "weighted_population": [weighted_workers],
        "weighted_informal_population": [weighted_informal_workers],
        "weighted_informality_rate": [weighted_rate]
    })

    return benchmark

def calculate_informality_by_variable(enoe, column):
    valid = enoe["informal"].notna() & enoe["survey_weight"].notna()
    enoe_valid = enoe.loc[valid].copy()
    enoe_valid["weighted_informal"] = enoe_valid["survey_weight"] * enoe_valid["informal"]

    informality = enoe_valid.groupby(column, dropna=False).agg(sample_workers=("informal", "size"), weighted_population=("survey_weight", "sum"), weighted_informal_population=("weighted_informal", "sum")).reset_index()
    informality["informality_rate"] = informality["weighted_informal_population"] / informality["weighted_population"]

    return informality

def calculate_informality_profiles(enoe, columns):
    profiles = []

    for column in columns:
        profile = calculate_informality_by_variable(enoe, column)
        profile["variable"] = column
        profile["category"] = profile[column].astype("string")
        profiles.append(profile[["variable", "category", "sample_workers", "weighted_population", "weighted_informal_population", "informality_rate"]])

    return pd.concat(profiles, ignore_index=True)

def calculate_missingness(dataframe, columns, weight_column, missing_label="no_especificado"):
    results = []

    total_rows = len(dataframe)
    total_weight = dataframe[weight_column].sum()

    for column in columns:
        missing = identify_missing_category(dataframe[column])  # same rule as the model stages (strip + lower)
        missing_rows = missing.sum()
        missing_weight = dataframe.loc[missing, weight_column].sum()

        results.append({
            "variable": column,
            "missing_rows": missing_rows,
            "sample_missing_share": missing_rows / total_rows,
            "weighted_missing_population": missing_weight,
            "weighted_missing_share": missing_weight / total_weight
        })

    return pd.DataFrame(results)

def compare_known_sector_distributions(enoe, od):
    enoe_known = enoe[~enoe["sector_desconocido"]].copy()
    od_known = od[~od["sector_desconocido"]].copy()

    enoe_distribution = calculate_weighted_distribution(enoe_known, "sector", "survey_weight").rename(columns={"weighted_population": "enoe_population", "weighted_share": "enoe_share"})
    od_distribution = calculate_weighted_distribution(od_known, "sector", "expansion_factor").rename(columns={"weighted_population": "od_population", "weighted_share": "od_share"})

    comparison = enoe_distribution.merge(od_distribution, on="sector", how="outer")
    comparison["enoe_share"] = comparison["enoe_share"].fillna(0)
    comparison["od_share"] = comparison["od_share"].fillna(0)
    comparison["difference_pp"] = (comparison["od_share"] - comparison["enoe_share"]) * 100

    return comparison