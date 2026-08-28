"""OD-only features of the work-trip destination: DENUE establishment mix at the destination AGEB or locality, and the
kind of destination (urban AGEB, rural locality, airport, outside the metro zone, unknown). Review item 4.4."""

import numpy as np
import pandas as pd

from .common import SECTOR_CLASSES, load_config
from .harmonize_enoe_od_dataframes import load_mapping

_DESTINATION = load_config("od")["destination"]  # see src/config/od.yaml
DENUE_RELEASE = str(_DESTINATION["denue_release"])
DENUE_STATE_CODE = _DESTINATION["state_code"]
DESTINATION_AMBITO_LEVELS = list(_DESTINATION["ambito_levels"])
SMALL_ESTABLISHMENT_LEVELS = tuple(_DESTINATION["small_establishment_levels"])
DESTINATION_NUMERIC_FEATURES = ["dest_establecimientos_log", "dest_share_grandes"] + [
    f"dest_share_{sector}" for sector in SECTOR_CLASSES
]


def _denue_aggregates(state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Per-AGEB (13-character CVEGEO) and per-locality (9-character) establishment mix from DENUE."""
    import mxcensus

    denue = mxcensus.load_denue(state=state, release=release)
    scian_map = {str(k): v for k, v in load_mapping("sector")["denue_scian2"].items()}
    frame = pd.DataFrame(
        {
            "ageb": denue["cve_ent"].astype(str).str.zfill(2)
            + denue["cve_mun"].astype(str).str.zfill(3)
            + denue["cve_loc"].astype(str).str.zfill(4)
            + denue["ageb"].astype(str).str.zfill(4),
            "sector": denue["codigo_act"].astype(str).str[:2].map(scian_map),
            "large": ~denue["per_ocu"]
            .astype(str)
            .str.startswith(SMALL_ESTABLISHMENT_LEVELS),
        }
    )
    unmapped = frame["sector"].isna().sum()
    assert unmapped == 0, (
        f"{unmapped} DENUE establishments with a SCIAN sector missing from sector.yaml denue_scian2"
    )
    frame["localidad"] = frame["ageb"].str[:9]

    def aggregate(key):
        grouped = frame.groupby(key)
        table = pd.DataFrame(
            {
                "dest_establecimientos_log": np.log1p(grouped.size()),
                "dest_share_grandes": grouped["large"].mean(),
            }
        )
        for sector in SECTOR_CLASSES:
            table[f"dest_share_{sector}"] = grouped["sector"].apply(
                lambda values: (values == sector).mean()
            )
        return table

    return aggregate("ageb"), aggregate("localidad")


def _destination_crosswalk():
    """IMEPLAN destination code -> INEGI CVEGEO and ámbito (urban AGEB / rural locality) via eodgdl and the marco."""
    import eodgdl
    import mxcensus

    agebs = eodgdl.load_imeplan_agebs(eodgdl.load_taz())
    crosswalk = (
        agebs[["CVEGEO_EOD", "CVEGEO"]]
        .astype(str)
        .drop_duplicates("CVEGEO_EOD")
        .set_index("CVEGEO_EOD")["CVEGEO"]
    )
    _, mg_loc_ageb = mxcensus.load_mg_census(state=DENUE_STATE_CODE)
    marco = mg_loc_ageb.reset_index()
    rural = set(marco.loc[marco["AMBITO"].astype(str) == "Rural", "CVEGEO"].astype(str))

    return crosswalk, rural


def add_destination_features(od, state=DENUE_STATE_CODE, release=DENUE_RELEASE):
    """Attach the destination ámbito and DENUE establishment mix to OD workers (columns ``destino_cvegeo`` and
    ``destino_zona`` must come from the work trips). Unknown, airport and out-of-metro destinations keep NaN in the
    DENUE columns (imputed by the model pipelines) and carry the information in ``destino_ambito``."""
    od = od.copy()
    crosswalk, rural = _destination_crosswalk()
    by_ageb, by_localidad = _denue_aggregates(state=state, release=release)

    code = od["destino_cvegeo"].astype("string")
    zona = od["destino_zona"].astype("string")
    resolved = code.map(crosswalk).fillna(code.where(code.str.len() == 13))
    ambito = pd.Series("desconocido", index=od.index, dtype=object)
    ambito[resolved.notna() & (resolved.str.len() == 13)] = "ageb_urbana"
    ambito[resolved.isin(rural)] = "localidad_rural"
    ambito[zona.fillna("").str.startswith("Acceso")] = "fuera_zm"
    ambito[zona.eq("Aeropuerto")] = "aeropuerto"
    ambito[code.isna()] = "desconocido"
    od["destino_ambito"] = ambito

    features = by_ageb.reindex(resolved.astype(object))
    fallback = by_localidad.reindex(
        resolved.astype(object).str[:9] if hasattr(resolved, "str") else resolved
    )
    features = features.where(features.notna(), fallback.to_numpy())
    features.index = od.index
    for column in DESTINATION_NUMERIC_FEATURES:
        od[column] = features[column].astype(float)
    od.loc[
        ~od["destino_ambito"].isin(["ageb_urbana", "localidad_rural"]),
        DESTINATION_NUMERIC_FEATURES,
    ] = np.nan

    return od
