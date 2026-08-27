"""Compare pipeline outputs against a reference run (e.g. ``outputs_baseline/``).

Usage: ``uv run python -m src.compare_outputs outputs_baseline outputs [--stage 1|2|all]``
Every function tolerates missing files so partial re-runs can be compared.
"""
import argparse
from pathlib import Path

import pandas as pd

from .diagnose_enoe_od_dataframes import calculate_weighted_distribution, calculate_weighted_informality_rate
from .generate_enoe_od_dataframes import ENOE_PERSON_KEYS

STAGE_FILES = {
    1: ["enoe_workers", "od_workers"],
    2: ["enoe_harmonized", "od_harmonized"],
    3: ["od_sector_imputed", "od_informality_imputed"],
}
HARMONIZED_COLUMNS = ["genero", "ocupacion", "edad_cat", "escolaridad", "municipio", "estado_civil", "parentesco", "tamano_viv_cat", "sector"]
# The baseline used the raw IMEPLAN header names for the OD keys.
OD_KEY_ALIASES = {"Folio Vivienda": "folio_vivienda", "Folio Habitante": "folio_habitante"}
OD_KEYS = ["folio_vivienda", "folio_habitante"]


def _weight_column(name):
    return "survey_weight" if name.startswith("enoe") else "expansion_factor"

def _read(directory, name):
    path = Path(directory) / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).rename(columns=OD_KEY_ALIASES)

def _files_for_stage(stage):
    stages = [1, 2, 3] if stage == "all" else [int(stage)]
    return [name for s in stages for name in STAGE_FILES[s]]

def _abbreviate(items, limit=6):
    return items if len(items) <= limit else items[:limit] + [f"... (+{len(items) - limit})"]

def compare_row_counts(baseline_dir, new_dir, files):
    rows = []
    for name in files:
        base, new = _read(baseline_dir, name), _read(new_dir, name)
        if base is None or new is None:
            continue
        rows.append({
            "file": name, "rows_base": len(base), "rows_new": len(new), "cols_base": base.shape[1], "cols_new": new.shape[1],
            "n_cols_only_base": len(set(base.columns) - set(new.columns)), "n_cols_only_new": len(set(new.columns) - set(base.columns)),
            "cols_only_base": _abbreviate(sorted(set(base.columns) - set(new.columns))), "cols_only_new": _abbreviate(sorted(set(new.columns) - set(base.columns))),
        })
    return pd.DataFrame(rows)

def compare_weighted_totals(baseline_dir, new_dir, files):
    rows = []
    for name in files:
        base, new = _read(baseline_dir, name), _read(new_dir, name)
        weight = _weight_column(name)
        if base is None or new is None or weight not in base or weight not in new:
            continue
        total_base, total_new = base[weight].sum(), new[weight].sum()
        rows.append({"file": name, "weight": weight, "total_base": total_base, "total_new": total_new, "ratio": total_new / total_base})
    return pd.DataFrame(rows)

def compare_key_overlap(baseline_dir, new_dir):
    """Row-identity overlap on survey keys, before any harmonization."""
    rows = []
    for name, keys in (("enoe_workers", ENOE_PERSON_KEYS), ("od_workers", OD_KEYS)):
        base, new = _read(baseline_dir, name), _read(new_dir, name)
        if base is None or new is None or not set(keys) <= set(base.columns) or not set(keys) <= set(new.columns):
            continue
        base_keys = set(map(tuple, base[keys].astype("Int64").itertuples(index=False)))
        new_keys = set(map(tuple, new[keys].astype("Int64").itertuples(index=False)))
        rows.append({"file": name, "in_both": len(base_keys & new_keys), "only_base": len(base_keys - new_keys), "only_new": len(new_keys - base_keys)})
    return pd.DataFrame(rows)

def compare_category_distributions(base, new, columns, weight_column):
    comparisons = []
    for column in columns:
        if column not in base or column not in new:
            continue
        base_distribution = calculate_weighted_distribution(base, column, weight_column).rename(columns={"weighted_population": "population_base", "weighted_share": "share_base"})
        new_distribution = calculate_weighted_distribution(new, column, weight_column).rename(columns={"weighted_population": "population_new", "weighted_share": "share_new"})
        comparison = base_distribution.merge(new_distribution, on=column, how="outer")
        comparison["variable"] = column
        comparison["category"] = comparison[column].astype("string")
        comparison[["share_base", "share_new"]] = comparison[["share_base", "share_new"]].fillna(0)
        comparison["difference_pp"] = (comparison["share_new"] - comparison["share_base"]) * 100
        comparisons.append(comparison[["variable", "category", "population_base", "population_new", "share_base", "share_new", "difference_pp"]])
    if not comparisons:
        return pd.DataFrame(columns=["variable", "category", "population_base", "population_new", "share_base", "share_new", "difference_pp"])
    return pd.concat(comparisons, ignore_index=True)

def compare_harmonized_distributions(baseline_dir, new_dir, columns=HARMONIZED_COLUMNS):
    results = {}
    for name in STAGE_FILES[2]:
        base, new = _read(baseline_dir, name), _read(new_dir, name)
        if base is None or new is None:
            continue
        results[name] = compare_category_distributions(base, new, columns, _weight_column(name))
    return results

def compare_enoe_informality(baseline_dir, new_dir):
    base, new = _read(baseline_dir, "enoe_harmonized"), _read(new_dir, "enoe_harmonized")
    if base is None or new is None:
        return None
    return pd.DataFrame({"baseline": calculate_weighted_informality_rate(base).iloc[0], "new": calculate_weighted_informality_rate(new).iloc[0]})

def _od_informality_summary(od):
    weighted_mean = lambda frame: (frame["prob_informal"] * frame["expansion_factor"]).sum() / frame["expansion_factor"].sum()
    summary = {"overall": weighted_mean(od)}
    summary["hard_rate"] = (od["informal_predicted"] * od["expansion_factor"]).sum() / od["expansion_factor"].sum()
    for group_column in ("sector_final", "municipio"):
        if group_column in od:
            for category, frame in od.groupby(group_column):
                summary[f"{group_column}={category}"] = weighted_mean(frame)
    return pd.Series(summary)

def compare_od_informality(baseline_dir, new_dir):
    base, new = _read(baseline_dir, "od_informality_imputed"), _read(new_dir, "od_informality_imputed")
    if base is None or new is None:
        return None
    comparison = pd.DataFrame({"baseline": _od_informality_summary(base), "new": _od_informality_summary(new)})
    comparison["difference_pp"] = (comparison["new"] - comparison["baseline"]) * 100
    return comparison

def compare_all(baseline_dir, new_dir, stage="all"):
    files = _files_for_stage(stage)
    return {
        "row_counts": compare_row_counts(baseline_dir, new_dir, files),
        "weighted_totals": compare_weighted_totals(baseline_dir, new_dir, files),
        "key_overlap": compare_key_overlap(baseline_dir, new_dir),
        "harmonized_distributions": compare_harmonized_distributions(baseline_dir, new_dir) if stage in ("2", "all", 2) else {},
        "enoe_informality": compare_enoe_informality(baseline_dir, new_dir) if stage in ("2", "all", 2) else None,
        "od_informality": compare_od_informality(baseline_dir, new_dir) if stage in ("3", "all", 3) else None,
    }

def print_comparison(results, max_difference_pp=0.5):
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:,.4f}".format):
        for title in ("row_counts", "weighted_totals", "key_overlap"):
            print(f"== {title}\n{results[title].to_string(index=False) if len(results[title]) else '(nothing to compare)'}\n")
        for name, table in results["harmonized_distributions"].items():
            flagged = table[table["difference_pp"].abs() > max_difference_pp]
            print(f"== {name}: harmonized distributions, max |diff| = {table['difference_pp'].abs().max():.3f} pp; {len(flagged)} categories above {max_difference_pp} pp")
            if len(flagged):
                print(flagged.to_string(index=False))
            print()
        for title in ("enoe_informality", "od_informality"):
            if results[title] is not None:
                print(f"== {title}\n{results[title].to_string()}\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir")
    parser.add_argument("new_dir")
    parser.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--max-difference-pp", type=float, default=0.5)
    args = parser.parse_args()
    print_comparison(compare_all(args.baseline_dir, args.new_dir, args.stage), args.max_difference_pp)

if __name__ == "__main__":
    main()
