import numpy as np
import pandas as pd


NO_ESPECIFICADO = "no_especificado"

AGE_BINS = [0, 3, 5, 6, 8, 12, 15, 18, 25, 50, 60, 65, np.inf]
AGE_LABELS = ["0_2", "3_4", "5", "6_7", "8_11", "12_14", "15_17", "18_24", "25_49", "50_59", "60_64", "65_y_mas"]

# DATA TYPES
def prepare_enoe_data_types(enoe):
    enoe = enoe.copy()
    numeric_columns = ["mun", "sex", "pos_ocu", "scian", "eda", "cs_p13_1", "emp_ppal", "e_con", "par_c", "dwelling_size", "survey_weight", "survey_stratum", "survey_psu"]
    for column in numeric_columns:
        enoe[column] = pd.to_numeric(enoe[column], errors="coerce").astype("Int64")

    return enoe

def prepare_od_data_types(od):
    od = od.copy()
    numeric_columns = ["Edad", "expansion_factor"]
    for column in numeric_columns:
        od[column] = pd.to_numeric(od[column], errors="coerce").astype("Int64")
    text_columns = ["Sexo de nacimiento", "Ocupación:", "Escolaridad", "Municipio", "Estado civil", "Parentesco", "Giro de la empresa donde trabaja:", "dwelling_size"]
    for column in text_columns:
        od[column] = od[column].astype("string").str.strip()

    return od

# GENDER
def harmonize_enoe_gender(enoe):
    enoe = enoe.copy()
    enoe["genero"] = enoe["sex"].map({1: "H", 2: "F"}).fillna(NO_ESPECIFICADO)

    return enoe

def harmonize_od_gender(od):
    od = od.copy()
    od["genero"] = od["Sexo de nacimiento"].map({"Hombres": "H", "Mujeres": "F"}).fillna(NO_ESPECIFICADO)

    return od

# OCCUPATION
def harmonize_enoe_occupation(enoe):
    enoe = enoe.copy()
    occupation_mapping = {
        1: "trabajador",
        2: "trabajador",
        3: "independiente",
        4: "otro"
    }
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
        "Hogar": "otro",
        "Estudiante": "otro",
        "Jubilado o pensionado": "otro",
        "Desempleado": "otro"
    }
    od["ocupacion"] = od["Ocupación:"].map(occupation_mapping).fillna(NO_ESPECIFICADO)

    return od

# AGE
def harmonize_enoe_age(enoe):
    enoe = enoe.copy()
    enoe["edad_num"] = enoe["eda"].copy()
    enoe["edad_cat"] = pd.cut(enoe["edad_num"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("string").fillna(NO_ESPECIFICADO)

    return enoe

def harmonize_od_age(od):
    od = od.copy()
    od["edad_num"] = od["Edad"].copy()
    od["edad_cat"] = pd.cut(od["edad_num"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("string").fillna(NO_ESPECIFICADO)

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
        99: NO_ESPECIFICADO
    }
    enoe["escolaridad"] = enoe["cs_p13_1"].map(education_mapping).fillna(NO_ESPECIFICADO)

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
        "No sabe": NO_ESPECIFICADO
    }
    od["escolaridad"] = od["Escolaridad"].map(education_mapping).fillna(NO_ESPECIFICADO)

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
        83: "tala"
    }
    enoe["municipio"] = enoe["mun"].map(municipality_mapping).fillna("otro")

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
        "Tala": "tala"
    }
    od["municipio"] = od["Municipio"].map(municipality_mapping).fillna("otro")

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
        6: "soltero"
    }
    enoe["estado_civil"] = enoe["e_con"].map(marital_status_mapping).fillna(NO_ESPECIFICADO)

    return enoe

def harmonize_od_marital_status(od):
    od = od.copy()
    marital_status_mapping = {
        "Soltero": "soltero",
        "Casado": "casado",
        "Unión libre": "union_libre",
        "Viudo": "viudo",
        "Separado": "separado",
        "Divorciado": "divorciado"
    }
    od["estado_civil"] = od["Estado civil"].map(marital_status_mapping).fillna(NO_ESPECIFICADO)

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
        6: "sin_parentesco"
    }
    relationship_code = enoe["par_c"] // 100
    enoe["parentesco"] = relationship_code.map(relationship_mapping).fillna(NO_ESPECIFICADO)

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
        "Sin parentesco": "sin_parentesco"
    }

    od["parentesco"] = od["Parentesco"].map(relationship_mapping).fillna(NO_ESPECIFICADO)

    return od

# HOUSEHOLD SIZE
def harmonize_enoe_household_size(enoe):
    enoe = enoe.copy()
    enoe["tamano_viv_num"] = enoe["dwelling_size"].copy()
    enoe["tamano_viv_cat"] = pd.cut(enoe["tamano_viv_num"], bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf], labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10_y_mas"], right=False).astype("string").fillna(NO_ESPECIFICADO)

    return enoe

def harmonize_od_household_size(od):
    od = od.copy()
    household_size = od["dwelling_size"].replace({"10 y +": "10_y_mas", "10 y más": "10_y_mas"})
    valid_categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10_y_mas"]
    od["tamano_viv_cat"] = household_size.where(household_size.isin(valid_categories), NO_ESPECIFICADO)
    od["tamano_viv_num"] = pd.to_numeric(od["tamano_viv_cat"].replace({"10_y_mas": "10"}), errors="coerce").astype("Int64")

    return od

# ECONOMIC SECTOR
def harmonize_enoe_sector(enoe):
    enoe = enoe.copy()
    sector_mapping = {
        1: "gobierno_otro_agricultura",
        2: "gobierno_otro_agricultura",
        3: "gobierno_otro_agricultura",
        4: "manufactura_construccion",
        5: "manufactura_construccion",
        6: "comercio",
        7: "comercio",
        8: "servicios_transporte",
        9: "gobierno_otro_agricultura",
        10: "servicios_transporte",
        11: "servicios_transporte",
        12: "servicios_transporte",
        13: "servicios_transporte",
        14: "servicios_transporte",
        15: "servicios_transporte",
        16: "servicios_transporte",
        17: "servicios_transporte",
        18: "servicios_transporte",
        19: "servicios_transporte",
        20: "gobierno_otro_agricultura",
        21: NO_ESPECIFICADO
    }
    enoe["sector"] = enoe["scian"].map(sector_mapping).fillna(NO_ESPECIFICADO)
    enoe["sector_desconocido"] = enoe["sector"].eq(NO_ESPECIFICADO)

    return enoe

def harmonize_od_sector(od):
    od = od.copy()
    sector_mapping = {
        "Comercio": "comercio",
        "Servicio": "servicios_transporte",
        "Educación": "servicios_transporte",
        "Industria": "manufactura_construccion",
        "Gobierno/sector público": "gobierno_otro_agricultura"
    }
    od["sector"] = od["Giro de la empresa donde trabaja:"].map(sector_mapping).fillna(NO_ESPECIFICADO)
    od["sector_desconocido"] = od["sector"].eq(NO_ESPECIFICADO)

    return od

def generate_enoe_informal_label(enoe):
    enoe = enoe.copy()
    enoe["informal"] = enoe["emp_ppal"].map({1: 1, 2: 0}).astype("Int64")

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

    return od