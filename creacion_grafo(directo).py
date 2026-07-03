"""
Script reproducible (desde el notebook prueba_difDatos2023.ipynb) para generar:
1) secciones_con_datos.shp
2) red_sociodemo_secciones.graphml

Objetivos principales:
- Respetar al maximo el flujo original del notebook.
- Dejar todo el proceso comentado paso a paso.
- Permitir comprobacion de igualdad con los resultados actuales.

Uso recomendado (sin sobreescribir):
    python generar_datos_finales.py

Uso para sobreescribir outputs finales:
    python generar_datos_finales.py --suffix "" --write-final
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _configure_gdal_proj_env() -> None:
    """
    Configura GDAL_DATA y PROJ_LIB automaticamente si no estan definidos.

    En algunos entornos de Windows (sobre todo conda), GDAL funciona pero emite:
    "Cannot find gdalvrt.xsd (GDAL_DATA is not defined)".
    Este bloque busca rutas tipicas dentro del entorno activo y las exporta.
    """
    candidates_gdal = [
        Path(sys.prefix) / "Library" / "share" / "gdal",  # conda Windows
        Path(sys.prefix) / "share" / "gdal",               # conda/Linux/macOS
    ]
    candidates_proj = [
        Path(sys.prefix) / "Library" / "share" / "proj",  # conda Windows
        Path(sys.prefix) / "share" / "proj",               # conda/Linux/macOS
    ]

    if not os.environ.get("GDAL_DATA"):
        for p in candidates_gdal:
            if p.exists():
                os.environ["GDAL_DATA"] = str(p)
                break

    if not os.environ.get("PROJ_LIB"):
        for p in candidates_proj:
            if p.exists():
                os.environ["PROJ_LIB"] = str(p)
                break


_configure_gdal_proj_env()

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

import funciones2


# -----------------------------------------------------------------------------
# Configuracion base de rutas (mismo esquema del notebook)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DADES_DIR = ROOT / "dades"
GENERATED_DIR = ROOT / "datos_generados"


# -----------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Devuelve hash SHA256 de un fichero para comparacion exacta por bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zfill_sc(series: pd.Series, width: int = 5) -> pd.Series:
    """Normaliza codigos de seccion censal a string con ceros a la izquierda."""
    return series.astype(str).str.strip().str.zfill(width)


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# -----------------------------------------------------------------------------
# Bloque A: Precio alquiler + paso 2011 -> 2023 (overlay ponderado)
# -----------------------------------------------------------------------------
def cargar_precio_alquiler(carpeta: Path) -> pd.DataFrame:
    """Replica el tratamiento del notebook para precio alquiler por m2 (2023)."""
    precio_alquiler = pd.read_csv(carpeta / "precio_alquiler_PORm2.csv")
    precio_alquiler = precio_alquiler[
        (precio_alquiler["Tipo de territorio"] == "Secci\u00f3 censal")
        & (precio_alquiler["Inmueble"] == "Vivienda colectiva")
    ].copy()

    precio_alquiler["SC"] = zfill_sc(precio_alquiler["Territorio"], width=5)
    precio_alquiler["IQR_precio_alquiler"] = (
        precio_alquiler["2023.2"].astype(float) - precio_alquiler["2023.1"].astype(float)
    )

    mediana_iqr = precio_alquiler[["SC", "2023", "IQR_precio_alquiler"]].rename(
        columns={"2023": "mediana_precio_alquiler"}
    )
    mediana_iqr["mediana_precio_alquiler"] = mediana_iqr["mediana_precio_alquiler"].astype(
        "float32"
    )
    return mediana_iqr


def cargar_shapefiles_2011_2023(carpeta: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Carga los shapefiles igual que en el notebook y crea claves de seccion."""
    sec_2011 = gpd.read_file(
        carpeta / "seccionado2011_INE" / "SECC_CE_20210101_INE_WM.shp",
        engine="fiona",
    ).to_crs(epsg=25831)

    sec_2023 = gpd.read_file(
        carpeta / "secciones2023" / "bseccenv10sh1f1_20230101_0.shp",
        engine="fiona",
    ).to_crs(epsg=25831)

    sec_2011 = sec_2011[(sec_2011["CMUN"] == "019") & (sec_2011["CPRO"] == "08")].copy()
    sec_2011["CUSEC_2011"] = zfill_sc(sec_2011["CDIS"].astype(str) + sec_2011["CSEC"].astype(str))

    sec_2023 = sec_2023[sec_2023["MUNDISSEC"].str.startswith("08019")].copy()
    sec_2023["CUSEC_2023"] = zfill_sc(sec_2023["DISTRICTE"].astype(str) + sec_2023["SECCIO"].astype(str))

    return sec_2011, sec_2023


def pasar_precio_2011_a_2023(
    sec_2011: gpd.GeoDataFrame,
    sec_2023: gpd.GeoDataFrame,
    mediana_iqr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica del notebook:
    1) join precio en 2011
    2) overlay 2011-2023
    3) peso corregido por area con datos
    4) agregacion final por CUSEC_2023
    """
    sec_2011 = sec_2011.copy()
    sec_2011["CUSEC_2011"] = zfill_sc(sec_2011["CUSEC_2011"])

    mediana_iqr = mediana_iqr.copy()
    mediana_iqr["SC"] = zfill_sc(mediana_iqr["SC"])

    sec_2011_datos = sec_2011.merge(mediana_iqr, left_on="CUSEC_2011", right_on="SC", how="left")
    sec_2011_datos["area_original"] = sec_2011_datos.geometry.area

    interseccion = gpd.overlay(sec_2011_datos, sec_2023, how="intersection")
    interseccion["area_trozo"] = interseccion.geometry.area
    interseccion["tiene_datos"] = interseccion["mediana_precio_alquiler"].notna()

    interseccion["area_con_datos_2023"] = interseccion.groupby("CUSEC_2023")["area_trozo"].transform(
        lambda x: x[interseccion.loc[x.index, "tiene_datos"]].sum()
    )

    interseccion["peso_corregido"] = np.where(
        interseccion["tiene_datos"],
        interseccion["area_trozo"] / interseccion["area_con_datos_2023"],
        np.nan,
    )

    interseccion["precio_ponderado"] = (
        interseccion["mediana_precio_alquiler"].astype(float) * interseccion["peso_corregido"]
    )
    interseccion["IQR_ponderada"] = (
        interseccion["IQR_precio_alquiler"].astype(float) * interseccion["peso_corregido"]
    )

    mediana_alquiler = (
        interseccion.groupby("CUSEC_2023")
        .agg({"precio_ponderado": "sum", "IQR_ponderada": "sum"})
        .reset_index()
    )
    mediana_alquiler.rename(
        columns={
            "CUSEC_2023": "SC",
            "precio_ponderado": "median_prA",
            "IQR_ponderada": "IQR_prA",
        },
        inplace=True,
    )

    # Filtro de cobertura de area (mismo criterio del notebook)
    ratio_area = interseccion["area_con_datos_2023"] / interseccion.groupby("CUSEC_2023")[
        "area_trozo"
    ].transform("sum")
    secciones_con_relleno = interseccion[(ratio_area < 0.40) & (interseccion["mediana_precio_alquiler"].notna())]
    valores_filtro = secciones_con_relleno["CUSEC_2023"].unique()
    mediana_alquiler.loc[mediana_alquiler["SC"].isin(valores_filtro), ["median_prA", "IQR_prA"]] = np.nan

    return mediana_alquiler


# -----------------------------------------------------------------------------
# Bloque B: Inmuebles, renta, educacion, origen, ocupacion, etc.
# -----------------------------------------------------------------------------
def crear_info_locales2(carpeta: Path) -> pd.DataFrame:
    info_locales = pd.read_csv(carpeta / "2023_locals_us_desti.csv")
    columnas_geograficas = [
        "Any",
        "Codi_districte",
        "Nom_districte",
        "Codi_barri",
        "Nom_barri",
        "Seccio_censal",
    ]

    tabla_resumen = info_locales.pivot_table(
        index=columnas_geograficas,
        columns=["Concepte", "Desc_us_desti_principal"],
        values="Nombre",
        aggfunc="sum",
        fill_value=0,
    )

    m2_total = tabla_resumen["Superf\u00edcie_m2"].sum(axis=1)
    if "S\u00f2l sense edificar" in tabla_resumen["Superf\u00edcie_m2"].columns:
        m2_total = m2_total - tabla_resumen[("Superf\u00edcie_m2", "S\u00f2l sense edificar")]

    info_locales2 = pd.DataFrame(index=tabla_resumen.index)
    info_locales2["Num_Oficinas"] = tabla_resumen.get(("Nombre", "Oficines"), 0)
    info_locales2["Num_Viviendas"] = tabla_resumen.get(("Nombre", "Habitatge"), 0)
    info_locales2["Num_Comercios"] = tabla_resumen.get(("Nombre", "Comer\u00e7"), 0)
    info_locales2["Num_Ensenyament_Cultura"] = tabla_resumen.get(("Nombre", "Ensenyament i cultura"), 0)

    m2_vivienda = tabla_resumen.get(("Superf\u00edcie_m2", "Habitatge"), 0)
    info_locales2["Porcentaje_m2_Vivienda"] = np.where(m2_total > 0, (m2_vivienda / m2_total) * 100, 0)
    info_locales2 = info_locales2.reset_index()
    return info_locales2


def crear_df_inmueble_renda(carpeta: Path) -> pd.DataFrame:
    renda_persona = pd.read_csv(carpeta / "2023_atles_renda_bruta_persona.csv")
    indice_gini = pd.read_csv(carpeta / "2023_atles_renda_index_gini.csv")
    edad_media = pd.read_csv(carpeta / "2023_loc_hab_edat_mitjana.csv")
    m2_medio = pd.read_csv(carpeta / "2023_loc_hab_sup_mitjana.csv")
    info_locales2 = crear_info_locales2(carpeta)

    dfs: Dict[str, pd.DataFrame] = {
        "renda_persona": renda_persona,
        "indice_gini": indice_gini,
        "edad_media": edad_media,
        "m2_medio": m2_medio,
        "info_locales2": info_locales2,
    }

    # Igual que en notebook: minusculas y eliminar columna any cuando exista
    for name, df in list(dfs.items()):
        df = df.copy()
        df.columns = df.columns.str.lower()
        if "any" in df.columns:
            df = df.drop(columns="any")
        dfs[name] = df

    # Merge secuencial respetando tu estilo
    values = list(dfs.values())
    df_inmueble_renda = values[0]
    for df in values[1:]:
        df_inmueble_renda = df_inmueble_renda.merge(df)

    df_inmueble_renda["SC"] = (
        df_inmueble_renda["codi_districte"].astype(str).str.zfill(2)
        + df_inmueble_renda["seccio_censal"].astype(str).str.zfill(3)
    )
    return df_inmueble_renda


def crear_umbral_renda_unir(carpeta: Path) -> pd.DataFrame:
    umbral_renda = pd.read_csv(carpeta / "2023umbrales_relativos_renta.csv", encoding="Latin1", sep=";")
    umbral_renda["SC"] = umbral_renda["Secciones"].str.extract(r"(\d{5})$")

    prefijo = r"^Poblaci\u00f3n con ingresos por unidad de consumo por\s*"
    umbral_renda["tipo_renta"] = (
        umbral_renda["Distribuci\u00f3n de la renta por unidad de consumo"].astype(str).str.replace(prefijo, "", regex=True)
    )

    umbral_renda["Total"] = pd.to_numeric(
        umbral_renda["Total"].str.replace(",", ".", regex=False), errors="coerce"
    )
    umbral_renda_pivot = (
        umbral_renda[["SC", "Total", "tipo_renta"]]
        .pivot_table(index="SC", columns="tipo_renta", values="Total")
        .reset_index()
    )
    umbral_renda_pivot.columns.name = None

    umbral_renda_unir = umbral_renda_pivot[
        ["SC", "debajo 40% de la mediana", "encima 160% de la mediana"]
    ].copy()
    umbral_renda_unir.rename(
        columns={
            "debajo 40% de la mediana": "renda< 40% mediana",
            "encima 160% de la mediana": "renda> 160% mediana",
        },
        inplace=True,
    )
    return umbral_renda_unir


def crear_educacion_unir(carpeta: Path) -> pd.DataFrame:
    educacion = pd.read_csv(carpeta / "2023educacion.csv", encoding="Latin1", sep=";", dtype=str)
    educacion["SC"] = educacion["Secciones"].str.extract(r"(\d{5})$")
    educacion["Total"] = pd.to_numeric(
        educacion["Total"].str.replace(".", "", regex=False), errors="coerce"
    )

    educacion_pivot = (
        educacion.pivot_table(
            index="SC",
            columns="Nivel de formaci\u00f3n alcanzado",
            values="Total",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    educacion_pivot.columns.name = None

    cols_educacion = educacion_pivot.columns.difference(["SC", "Total"])
    educacion_pivot[cols_educacion] = educacion_pivot[cols_educacion].div(educacion_pivot["Total"], axis=0)

    educaccion_unir = educacion_pivot[["SC", "Educaci\u00f3n primaria e inferior", "Educaci\u00f3n superior"]].copy()
    return educaccion_unir


def crear_pais_nacimiento_unir(carpeta: Path) -> pd.DataFrame:
    pais_nacimiento = pd.read_csv(carpeta / "2023pais_nacimiento.csv", encoding="Latin1", sep=";", dtype=str)
    pais_nacimiento["SC"] = pais_nacimiento["Secciones"].str.extract(r"(\d{5})$")
    pais_nacimiento["Total"] = pd.to_numeric(
        pais_nacimiento["Total"].str.replace(".", "", regex=False), errors="coerce"
    )

    pais_nacimiento = (
        pais_nacimiento.pivot_table(index="SC", columns="Lugar de nacimiento", values="Total")
        .reset_index()
    )
    pais_nacimiento.columns.name = None

    cols_pais = pais_nacimiento.columns.difference(["SC", "Total"])
    pais_nacimiento[cols_pais] = pais_nacimiento[cols_pais].div(pais_nacimiento["Total"], axis=0)

    pais_nacimiento_unir = pais_nacimiento[["SC", "Extranjero"]].copy()
    return pais_nacimiento_unir


def crear_origen_esp_unir(carpeta: Path) -> pd.DataFrame:
    origen_esp = pd.read_csv(carpeta / "2023origen_espa\u00f1a.csv", encoding="Latin1", sep=";", dtype=str)
    origen_esp["SC"] = origen_esp["Secciones"].str.extract(r"(\d{5})$")
    origen_esp["Total"] = pd.to_numeric(origen_esp["Total"].str.replace(".", "", regex=False))

    origen_esp = (
        origen_esp.pivot_table(
            index="SC",
            columns="Relaci\u00f3n entre lugar de nacimiento y lugar de residencia",
            values="Total",
        )
        .reset_index()
    )
    origen_esp.columns.name = None

    cols_origen = origen_esp.columns.difference(["SC", "Total"])
    origen_esp[cols_origen] = origen_esp[cols_origen].div(origen_esp["Total"], axis=0)

    origen_esp_unir = origen_esp[["SC", "Mismo municipio al de residencia"]].copy()
    origen_esp_unir.rename(columns={"Mismo municipio al de residencia": "nacido_en_Barcelona"}, inplace=True)
    return origen_esp_unir


def crear_act_laboral_unir(carpeta: Path) -> pd.DataFrame:
    act_laboral = pd.read_csv(carpeta / "2023ocupacion.csv", encoding="Latin1", sep=";", dtype=str)
    act_laboral["SC"] = act_laboral["Secciones"].str.extract(r"(\d{5})$")
    act_laboral["Total"] = pd.to_numeric(act_laboral["Total"].str.replace(".", "", regex=False))

    act_laboral = (
        act_laboral.pivot_table(index="SC", columns="Relaci\u00f3n con la actividad", values="Total")
        .reset_index()
    )
    act_laboral.columns.name = None
    act_laboral["tasa_paro"] = act_laboral["Parado/a"] / (act_laboral["Parado/a"] + act_laboral["Ocupado/a"])

    act_laboral_unir = act_laboral[["SC", "tasa_paro"]].copy()
    return act_laboral_unir


def unir_bloques_socioeconomicos(carpeta: Path, mediana_alquiler: pd.DataFrame) -> pd.DataFrame:
    """Construye df_socioeconomico como en notebook (merge final por SC)."""
    df_inmueble_renda = crear_df_inmueble_renda(carpeta)

    dfs_unir = {
        "umbral_renda_UNIR": crear_umbral_renda_unir(carpeta),
        "educaccion_UNIR": crear_educacion_unir(carpeta),
        "pais_nacimiento_UNIR": crear_pais_nacimiento_unir(carpeta),
        "origen_esp_UNIR": crear_origen_esp_unir(carpeta),
        "act_laboral_UNIR": crear_act_laboral_unir(carpeta),
    }

    for name, df in list(dfs_unir.items()):
        dfs_unir[name] = df.assign(SC=zfill_sc(df["SC"]))

    for _, df in dfs_unir.items():
        df_inmueble_renda = df_inmueble_renda.merge(df, on="SC", how="left")

    df_socioeconomico = df_inmueble_renda.merge(mediana_alquiler, how="left")
    return df_socioeconomico


# -----------------------------------------------------------------------------
# Bloque C: Union final con shapefile + guardado secciones_con_datos
# -----------------------------------------------------------------------------
def crear_secciones_con_datos(
    carpeta: Path,
    df_socioeconomico: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Replica union final del notebook y renombrado de columnas para exportacion."""
    sec_2023 = gpd.read_file(
        carpeta / "secciones2023" / "bseccenv10sh1f1_20230101_0.shp",
        engine="fiona",
    ).to_crs(epsg=25831)
    sec_2023 = sec_2023[sec_2023["MUNDISSEC"].str.startswith("08019")].copy()
    sec_2023["CUSEC_2023"] = zfill_sc(sec_2023["DISTRICTE"].astype(str) + sec_2023["SECCIO"].astype(str))

    sec_2023["SC_area"] = sec_2023.geometry.area
    sec_2023["SC_perim"] = sec_2023.geometry.length

    datos_con_shp = sec_2023.merge(df_socioeconomico, left_on="CUSEC_2023", right_on="SC", how="left")

    # Limpieza tal cual notebook
    datos_con_shp.loc[datos_con_shp["median_prA"] < 0.01, ["median_prA", "IQR_prA"]] = np.nan

    # Renombrado para evitar limites y conflictos de shapefile
    datos_con_shp = datos_con_shp.rename(
        columns={
            "codi_districte": "codi_dist",
            "nom_districte": "nom_dist",
            "codi_barri": "codi_bar",
            "nom_barri": "nom_barri",
            "seccio_censal": "sec_cens",
            "import_renda_bruta_\u20ac": "rend_brut",
            "index_gini": "gini_idx",
            "edat_mitjana": "edat_mitj",
            "sup_mitjana_m2": "supVIV_m2",
            "num_oficinas": "n_ofic",
            "num_viviendas": "n_viv",
            "num_comercios": "n_com",
            "num_ensenyament_cultura": "n_ens_cul",
            "porcentaje_m2_vivienda": "pct_m2_v",
            "renda< 40% mediana": "renta_40",
            "renda> 160% mediana": "renta_160",
            "Educaci\u00f3n primaria e inferior": "edu_prim",
            "Educaci\u00f3n superior": "edu_super",
            "Extranjero": "prop_extr",
            "nacido_en_Barcelona": "naci_bcn",
            "tasa_paro": "tasa_paro",
            "median_prA": "med_prA",
            "IQR_prA": "iqr_prA",
        }
    )

    return datos_con_shp


def guardar_secciones_con_datos(datos_con_shp: gpd.GeoDataFrame, output_base_shp: Path) -> None:
    """Guarda shapefile final (y componentes asociados .dbf/.shx/.prj)."""
    output_base_shp.parent.mkdir(parents=True, exist_ok=True)
    datos_con_shp.to_file(output_base_shp, engine="fiona")


# -----------------------------------------------------------------------------
# Bloque D: Construccion grafo de secciones + export graphml
# -----------------------------------------------------------------------------
def construir_grafo_secciones(
    datos_con_shp: gpd.GeoDataFrame,
    carpeta_dades: Path,
) -> nx.MultiDiGraph:
    """
    Replica del notebook usando funciones2.calcular_dist_centorides,
    con representative points y DEM para pendiente/desnivel.
    """
    ox.settings.use_cache = True
    ox.settings.log_console = False

    # Misma descarga de red base
    G = ox.graph_from_place("Barcelona, Espa\u00f1a", network_type="drive")

    secciones = datos_con_shp.copy()
    secc2 = secciones.rename(columns={"CUSEC_2023": "ID_seccion"})

    dem_path = carpeta_dades / "MDT_Barcelona_AWS.tif"
    if not dem_path.exists():
        raise FileNotFoundError(
            f"No se encuentra DEM en {dem_path}. Genera primero el raster con alturas_y_barreras.py"
        )

    graf_dist = funciones2.calcular_dist_centorides(
        G,
        secc2,
        dem_path=str(dem_path),
        use_representative_point=True,
        include_centroid_leg=True,
    )

    # Purga de diagonales (vecinos de esquina), como en notebook
    secc_nodes, secc_edges = ox.graph_to_gdfs(graf_dist)
    secc_edges = secc_edges[secc_edges["shared_boundary_length"] >= 1.1].copy()

    # Join de atributos de datos_sencillo sobre nodos
    datos_sencillo = datos_con_shp[
        [
            "CUSEC_2023",
            "codi_dist",
            "nom_dist",
            "codi_bar",
            "nom_barri",
            "SC_area",
            "SC_perim",
            "sec_cens",
            "rend_brut",
            "gini_idx",
            "edat_mitj",
            "supVIV_m2",
            "n_ofic",
            "n_viv",
            "n_com",
            "n_ens_cul",
            "pct_m2_v",
            "SC",
            "renta_40",
            "renta_160",
            "edu_prim",
            "edu_super",
            "prop_extr",
            "naci_bcn",
            "tasa_paro",
            "med_prA",
            "iqr_prA",
        ]
    ].copy()
    datos_sencillo = datos_sencillo.set_index("CUSEC_2023")
    secc_nodes = secc_nodes.join(datos_sencillo)

    # Convertir objetos a string (igual que notebook antes de exportar GraphML)
    for col in secc_nodes.columns:
        if secc_nodes[col].dtype == "object":
            secc_nodes[col] = secc_nodes[col].astype(str)

    G_secciones = ox.graph_from_gdfs(secc_nodes, secc_edges)
    return G_secciones


def guardar_grafo_graphml(G: nx.MultiDiGraph, output_graphml: Path) -> None:
    output_graphml.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=str(output_graphml))


# -----------------------------------------------------------------------------
# Bloque E: Comprobaciones y comparativa con outputs actuales
# -----------------------------------------------------------------------------
def _normalize_for_compare(gdf: gpd.GeoDataFrame, key_col: str = "CUSEC_2023") -> pd.DataFrame:
    """Normaliza orden para comparacion estable entre ejecuciones."""
    df = gdf.copy()

    if key_col in df.columns:
        df[key_col] = zfill_sc(df[key_col])
        df = df.sort_values(key_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Geometria a WKB para comparar exactitud geometrica byte a byte
    df["__geometry_wkb__"] = df.geometry.to_wkb(hex=True)
    df = df.drop(columns="geometry")

    # Ordenar columnas para evitar diferencias por orden de campos
    ordered_cols = sorted(df.columns)
    return df[ordered_cols]


def compare_shapefile_semantic(path_a: Path, path_b: Path) -> Tuple[bool, str]:
    """Compara shapefiles por contenido (no por bytes de ficheros auxiliares)."""
    if not path_a.exists() or not path_b.exists():
        return False, "Alguno de los shapefiles no existe."

    gdf_a = gpd.read_file(path_a, engine="fiona")
    gdf_b = gpd.read_file(path_b, engine="fiona")

    norm_a = _normalize_for_compare(gdf_a)
    norm_b = _normalize_for_compare(gdf_b)

    if list(norm_a.columns) != list(norm_b.columns):
        cols_a = set(norm_a.columns)
        cols_b = set(norm_b.columns)
        return (
            False,
            f"Columnas distintas. Solo A: {sorted(cols_a - cols_b)} | Solo B: {sorted(cols_b - cols_a)}",
        )

    if norm_a.shape != norm_b.shape:
        return False, f"Shape distinto. A={norm_a.shape}, B={norm_b.shape}"

    equal = norm_a.equals(norm_b)
    if equal:
        return True, "Contenido semantico del shapefile: IGUAL"
    return False, "Contenido semantico del shapefile: DISTINTO"


def compare_graph_semantic(path_a: Path, path_b: Path) -> Tuple[bool, str]:
    """Compara dos graphml por estructura y atributos cargados con OSMnx/NetworkX."""
    if not path_a.exists() or not path_b.exists():
        return False, "Alguno de los graphml no existe."

    g1 = ox.load_graphml(path_a)
    g2 = ox.load_graphml(path_b)

    # Chequeo basico rapido
    if g1.number_of_nodes() != g2.number_of_nodes() or g1.number_of_edges() != g2.number_of_edges():
        return (
            False,
            f"Nodos/aristas distintos. A=({g1.number_of_nodes()}, {g1.number_of_edges()}) "
            f"B=({g2.number_of_nodes()}, {g2.number_of_edges()})",
        )

    # Nodo set exacto
    if set(g1.nodes) != set(g2.nodes):
        return False, "Conjunto de nodos distinto"

    # Aristas (multigrafo) exactas por (u,v,key)
    edges1 = set(g1.edges(keys=True))
    edges2 = set(g2.edges(keys=True))
    if edges1 != edges2:
        return False, "Conjunto de aristas distinto"

    # Atributos de nodos
    for n in g1.nodes:
        if g1.nodes[n] != g2.nodes[n]:
            return False, f"Atributos de nodo distintos en {n}"

    # Atributos de aristas
    for u, v, k in edges1:
        if g1[u][v][k] != g2[u][v][k]:
            return False, f"Atributos de arista distintos en ({u}, {v}, {k})"

    return True, "Contenido semantico del graphml: IGUAL"


def print_network_checks(graphml_path: Path) -> None:
    """Imprime resumen de la red final para validar visualmente/estructuralmente."""
    G = ox.load_graphml(graphml_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G.to_undirected())
    isolated = [n for n, d in G.degree() if d == 0]

    print_header("RESUMEN RED FINAL")
    print(f"Nodos: {n_nodes}")
    print(f"Aristas: {n_edges}")
    print(f"Componentes conectadas: {n_components}")
    print(f"Nodos aislados: {len(isolated)}")

    # Muestra estadisticos de algunas variables clave de arista
    dist_diff_vals: List[float] = []
    slope_vals: List[float] = []
    shared_vals: List[float] = []
    for _, _, data in G.edges(data=True):
        try:
            dist_diff_vals.append(float(data.get("dist_diff", np.nan)))
        except Exception:
            pass
        try:
            slope_vals.append(float(data.get("pendiente_media_pct", np.nan)))
        except Exception:
            pass
        try:
            shared_vals.append(float(data.get("shared_boundary_length", np.nan)))
        except Exception:
            pass

    def _safe_stats(values: Iterable[float], name: str) -> None:
        arr = np.array(list(values), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"{name}: sin datos")
            return
        print(
            f"{name}: min={arr.min():.4f} p50={np.percentile(arr, 50):.4f} "
            f"p95={np.percentile(arr, 95):.4f} max={arr.max():.4f}"
        )

    _safe_stats(dist_diff_vals, "dist_diff")
    _safe_stats(slope_vals, "pendiente_media_pct")
    _safe_stats(shared_vals, "shared_boundary_length")


def remove_shapefile_family(base_shp_path: Path) -> None:
    """Elimina todos los sidecars de un shapefile base, si existen."""
    suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd", ".sbn", ".sbx"]
    for ext in suffixes:
        p = base_shp_path.with_suffix(ext)
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera secciones_con_datos + red_sociodemo_secciones")
    parser.add_argument(
        "--suffix",
        default="_repro",
        help="Sufijo para outputs (default: _repro). Usa '' para nombres finales.",
    )
    parser.add_argument(
        "--write-final",
        action="store_true",
        help="Si se activa, copia/reescribe los outputs finales sin sufijo.",
    )
    args = parser.parse_args()

    suffix = args.suffix

    output_shp = GENERATED_DIR / f"secciones_con_datos{suffix}.shp"
    output_graphml = GENERATED_DIR / f"red_sociodemo_secciones{suffix}.graphml"

    ref_shp = GENERATED_DIR / "secciones_con_datos.shp"
    ref_graphml = GENERATED_DIR / "red_sociodemo_secciones.graphml"

    print_header("1) BLOQUE PRECIO ALQUILER + PASO 2011 A 2023")
    mediana_iqr = cargar_precio_alquiler(DADES_DIR)
    sec_2011, sec_2023 = cargar_shapefiles_2011_2023(DADES_DIR)
    mediana_alquiler = pasar_precio_2011_a_2023(sec_2011, sec_2023, mediana_iqr)
    print(f"mediana_alquiler filas: {len(mediana_alquiler)}")

    print_header("2) BLOQUE SOCIOECONOMICO")
    df_socioeconomico = unir_bloques_socioeconomicos(DADES_DIR, mediana_alquiler)
    print(f"df_socioeconomico shape: {df_socioeconomico.shape}")

    print_header("3) UNION FINAL + EXPORT SHP")
    datos_con_shp = crear_secciones_con_datos(DADES_DIR, df_socioeconomico)
    print(f"datos_con_shp shape: {datos_con_shp.shape}")
    print("NaNs por columna (top 15):")
    print(datos_con_shp.isna().sum().sort_values(ascending=False).head(15))

    # Limpieza previa del destino para evitar mezclar sidecars viejos
    remove_shapefile_family(output_shp)
    guardar_secciones_con_datos(datos_con_shp, output_shp)
    print(f"Guardado shapefile: {output_shp}")

    print_header("4) CONSTRUCCION Y EXPORT DEL GRAFO FINAL")
    G_secciones = construir_grafo_secciones(datos_con_shp, DADES_DIR)
    guardar_grafo_graphml(G_secciones, output_graphml)
    print(f"Guardado graphml: {output_graphml}")

    print_network_checks(output_graphml)

    print_header("5) COMPARACION CON OUTPUTS ACTUALES")
    if ref_shp.exists() and ref_graphml.exists():
        # Comparacion estricta por hash de fichero principal
        hash_shp_new = sha256_file(output_shp)
        hash_shp_ref = sha256_file(ref_shp)
        print(f"SHA256 SHP nuevo: {hash_shp_new}")
        print(f"SHA256 SHP ref  : {hash_shp_ref}")
        print(f"SHP bytes exactos: {hash_shp_new == hash_shp_ref}")

        hash_g_new = sha256_file(output_graphml)
        hash_g_ref = sha256_file(ref_graphml)
        print(f"SHA256 GRAPHML nuevo: {hash_g_new}")
        print(f"SHA256 GRAPHML ref  : {hash_g_ref}")
        print(f"GRAPHML bytes exactos: {hash_g_new == hash_g_ref}")

        # Comparacion semantica (estructura + atributos)
        ok_shp, msg_shp = compare_shapefile_semantic(output_shp, ref_shp)
        ok_graph, msg_graph = compare_graph_semantic(output_graphml, ref_graphml)
        print(msg_shp)
        print(msg_graph)

        iguales_semanticos = ok_shp and ok_graph
        print(f"\nRESULTADO GLOBAL (semantico): {iguales_semanticos}")
    else:
        print("No hay archivos de referencia actuales para comparar.")

    # Opcional: copiar como final sin sufijo
    if args.write_final and suffix != "":
        print_header("6) ESCRITURA FINAL (SIN SUFIJO)")
        remove_shapefile_family(ref_shp)
        datos_con_shp.to_file(ref_shp, engine="fiona")
        ox.save_graphml(G_secciones, filepath=str(ref_graphml))
        print(f"Sobrescrito: {ref_shp}")
        print(f"Sobrescrito: {ref_graphml}")


if __name__ == "__main__":
    main()