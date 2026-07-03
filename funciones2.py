import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.strtree import STRtree
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def preprocesar_datos(graf, shp_censo, epsg=25830, use_representative_point=True):
    secciones = shp_censo.to_crs(epsg=epsg).copy()
    G_calles_proj = ox.project_graph(graf, to_crs=f"EPSG:{epsg}")
    try:
        G_calles_proj = ox.utils_graph.get_largest_component(G_calles_proj, strongly=False)
    except Exception:
        pass
    if use_representative_point:
        secciones['centroid'] = secciones.geometry.representative_point()
    else:
        secciones['centroid'] = secciones.geometry.centroid
    secciones['x'] = secciones['centroid'].x
    secciones['y'] = secciones['centroid'].y
    return G_calles_proj, secciones

def mapear_nodos_red(G, secciones):
    print("Mapeando centroides a nodos de la red vial...")
    secciones['nodo_red'] = ox.distance.nearest_nodes(G, X=secciones['x'], Y=secciones['y'])
    return secciones

def obtener_pares_vecinos(secciones, id_col='ID_seccion', buffer_m=0.25):
    print("Buscando vecindades espaciales...")
    df_geo = secciones[[id_col, 'geometry']].copy()
    df_geo['geometry'] = df_geo['geometry'].buffer(0)

    if buffer_m and buffer_m > 0:
        df_buf = df_geo.copy()
        df_buf['geometry'] = df_buf['geometry'].buffer(buffer_m)
        vecinos = gpd.sjoin(df_buf, df_buf, how='inner', predicate='intersects')
    else:
        vecinos = gpd.sjoin(df_geo, df_geo, how='inner', predicate='touches')

    vecinos = vecinos[vecinos[id_col + '_left'] != vecinos[id_col + '_right']]

    if buffer_m and buffer_m > 0:
        left_geom = df_geo.loc[vecinos.index].geometry.values
        right_geom = df_geo.loc[vecinos['index_right']].geometry.values
        dist_ok = np.fromiter(
            (lg.distance(rg) <= buffer_m for lg, rg in zip(left_geom, right_geom)),
            dtype=bool,
            count=len(vecinos)
        )
        vecinos = vecinos.loc[dist_ok]

    return vecinos

# --- Funciones auxiliares ---
def _path_to_line(G, path):
    if not path or len(path) < 2: return None
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
    return LineString(coords)

def _get_shared_boundary(geom1, geom2, tolerance=0.5):
    """
    Calcula la longitud del borde compartido tolerando huecos milimétricos, 
    polígonos solapados y contactos en diagonal.
    """
    try:
        # 1. INTENTO ESTRICTO (Si el censo está perfectamente dibujado)
        shared = geom1.intersection(geom2)
        
        if shared.geom_type in ('LineString', 'MultiLineString') and shared.length > 0:
            return shared.length, shared
            
        if shared.geom_type == 'GeometryCollection':
            lines = [g for g in shared.geoms if g.geom_type in ('LineString', 'MultiLineString')]
            if lines:
                if len(lines) == 1: return lines[0].length, lines[0]
                return sum(l.length for l in lines), MultiLineString(lines)

        # 2. INTENTO TOLERANTE (Si hay solapamiento, hueco o toque de esquinas)
        # Extraemos solo el contorno (las líneas) de los polígonos
        borde1 = geom1.boundary
        borde2 = geom2.boundary
        
        # Engordamos el contorno del vecino medio metro
        borde2_engordado = borde2.buffer(tolerance)
        
        # Cortamos el contorno 1 con ese "rotulador grueso"
        shared_aprox = borde1.intersection(borde2_engordado)
        
        # Si la intersección tiene longitud, hemos capturado la frontera real
        if shared_aprox.length > 0:
            return shared_aprox.length, shared_aprox
            
        return 0.0, None
        
    except Exception:
        # Si Shapely lanza un error topológico crítico (geometría inválida), devolvemos 0
        return 0.0, None

def _calc_shared_ratio(shared_len, perimeter_orig, perimeter_dest):
    """Calcula la proporción de perímetro compartido para ambas secciones."""
    # Evitar división por cero en secciones degeneradas
    p_orig = max(perimeter_orig, 1e-6)
    p_dest = max(perimeter_dest, 1e-6)
    
    ratio_orig = shared_len / p_orig  # % del perímetro de origen que toca con destino
    ratio_dest = shared_len / p_dest  # % del perímetro de destino que toca con origen
    
    ratio_orig_seguro = max(round(ratio_orig, 4), 0.0001)
    ratio_dest_seguro = max(round(ratio_dest, 4), 0.0001)
    
    return ratio_orig_seguro, ratio_dest_seguro

def _sample_raster_points(dem_path, points, points_crs):
    """Sample raster values at point coordinates. Returns list of floats or NaN."""
    try:
        import rasterio
    except Exception:
        warnings.warn("rasterio not available; centroid elevations will be NaN")
        return [np.nan] * len(points)

    with rasterio.open(dem_path) as src:
        pts = points
        if src.crs and points_crs and src.crs != points_crs:
            pts = gpd.GeoSeries(points, crs=points_crs).to_crs(src.crs)

        coords = [(p.x, p.y) for p in pts]
        values = []
        for v in src.sample(coords):
            val = float(v[0]) if v is not None else np.nan
            if src.nodata is not None and val == src.nodata:
                val = np.nan
            values.append(val)

    return values


def _calc_elevation_stats(G, path):
    """
    Devuelve:
            - desnivel_abs_m: suma de |Δz| (sube + baja)
            - pendiente_media_abs_pct: pendiente media absoluta ponderada por longitud (%)
    """
    if not path or len(path) < 2:
        return 0.0, 0.0

    elevs = [G.nodes[n].get('elevation', np.nan) for n in path]

    desnivel_abs_m = 0.0
    sum_len = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        eu, ev = elevs[i], elevs[i+1]

        if np.isnan(eu) or np.isnan(ev):
            continue

        edata = G.get_edge_data(u, v)
        if not edata:
            continue

        # elegir una arista representativa (la más corta)
        k = min(edata, key=lambda kk: edata[kk].get('length', float('inf')))
        length = edata[k].get('length', None)
        if length is None or length <= 0:
            continue

        dz = ev - eu

        # ascenso acumulado (solo subidas)
        #if dz > 0:
        #    ascenso_m += dz

        # desnivel absoluto acumulado (sube + baja)
        desnivel_abs_m += abs(dz)

        # pendiente media ponderada
        sum_len += length

    pendiente_media_abs_pct = (desnivel_abs_m / sum_len) * 100 if sum_len > 0 else 0.0

    return float(desnivel_abs_m), float(pendiente_media_abs_pct)

def calcular_distancias_y_grafo(G_calles, secciones, vecinos_df, id_col='ID_seccion', 
                                barreras_gdf=None, dem_path=None,
                                include_centroid_leg=True):
    print("Calculando rutas y construyendo grafo...")
    G_censo = nx.MultiDiGraph()
    G_routing = G_calles.to_undirected()

    def _edge_weight(u, v, attrs):
        """Devuelve un peso de ruta robusto para Graph y MultiGraph."""
        def _safe_float(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(value) or value <= 0:
                return None
            return value

        if isinstance(attrs, dict):
            direct_length = _safe_float(attrs.get('length'))
            if direct_length is not None:
                return direct_length

            edge_lengths = []
            for edge_data in attrs.values():
                if not isinstance(edge_data, dict):
                    continue

                length = _safe_float(edge_data.get('length'))
                if length is not None:
                    edge_lengths.append(length)
                    continue

                geom = edge_data.get('geometry')
                if geom is not None:
                    try:
                        geom_length = _safe_float(geom.length)
                    except Exception:
                        geom_length = None
                    if geom_length is not None:
                        edge_lengths.append(geom_length)

            if edge_lengths:
                return min(edge_lengths)

        # Fallback final: distancia geométrica entre nodos
        return float(np.hypot(
            G_routing.nodes[u]['x'] - G_routing.nodes[v]['x'],
            G_routing.nodes[u]['y'] - G_routing.nodes[v]['y']
        ))

    # Opcional: Elevación
    if dem_path:
        print("→ Añadiendo elevación desde raster...")
        G_calles = ox.add_node_elevations_raster(G_calles, filepath=dem_path)
        if include_centroid_leg:
            secciones['elev_centroid'] = _sample_raster_points(
                dem_path,
                secciones['centroid'],
                secciones.crs
            )

    # Diccionario rápido de geometrías
    geom_dict = secciones.set_index(id_col)['geometry'].to_dict()

    # Diccionario rápido de perímetros
    secciones['perimetro'] = secciones.geometry.length
    perimetros_dict = secciones.set_index(id_col)['perimetro'].to_dict()


    # Indexar barreras SOLO si existen
    barreras_tree = None
    barreras_geoms = None
    usar_barreras = False
    if barreras_gdf is not None and not barreras_gdf.empty:
        barreras_gdf_proj = barreras_gdf.to_crs(secciones.crs)
        barreras_tree = STRtree(barreras_gdf_proj.geometry.values)
        barreras_geoms = barreras_gdf_proj.geometry.values
        usar_barreras = True

    # 1. Nodos
    for idx, row in secciones.iterrows():
        G_censo.add_node(row[id_col], x=row['x'], y=row['y'], geometry=row['centroid'])

    info_cols = ['nodo_red', 'centroid']
    if 'elev_centroid' in secciones.columns:
        info_cols.append('elev_centroid')
    info_secciones = secciones.set_index(id_col)[info_cols].to_dict('index')
    edges_to_add = []

    # 2. Iterar vecinos
    for i, (idx, row) in enumerate(vecinos_df.iterrows()):
        id_origen = row[f'{id_col}_left']
        id_destino = row[f'{id_col}_right']
        d_orig, d_dest = info_secciones[id_origen], info_secciones[id_destino]

        dist_euc = d_orig['centroid'].distance(d_dest['centroid'])
        n_orig, n_dest = d_orig['nodo_red'], d_dest['nodo_red']

        # Optional: extra leg distance from centroid to nearest network node
        if include_centroid_leg:
            extra_leg = (
                np.hypot(d_orig['centroid'].x - G_calles.nodes[n_orig]['x'],
                          d_orig['centroid'].y - G_calles.nodes[n_orig]['y'])
                + np.hypot(d_dest['centroid'].x - G_calles.nodes[n_dest]['x'],
                           d_dest['centroid'].y - G_calles.nodes[n_dest]['y'])
            )
        else:
            extra_leg = 0.0

        if n_orig == n_dest:
            dist_ruta_red, path = 0, []
        else:
            try:
                path = nx.shortest_path(G_routing, n_orig, n_dest, weight=_edge_weight)
                dist_ruta_red = nx.shortest_path_length(G_routing, n_orig, n_dest, weight=_edge_weight)
            except nx.NetworkXNoPath:
                dist_ruta_red, path = None, []

        if dist_ruta_red is None:
            continue

        if include_centroid_leg:
            dist_ruta = dist_ruta_red + extra_leg
        else:
            dist_ruta = dist_ruta_red

        # DICCIONARIO BASE (siempre presente)
        edge_data = {
            'dist_euclidiana': dist_euc,
            'dist_ruta': dist_ruta,
            'dist_ruta_red': dist_ruta_red,
            'extra_leg_m': extra_leg if include_centroid_leg else 0.0,
            'dist_diff': dist_ruta - dist_euc
        }

        # 🟢 FRONTERA COMPARTIDA (SIEMPRE se calcula)        
        # Calcular perímetro compartido
        shared_len, shared_geom = _get_shared_boundary(geom_dict[id_origen], geom_dict[id_destino])

        # Calcular ratios de perímetro compartido
        p_orig = perimetros_dict[id_origen]
        p_dest = perimetros_dict[id_destino]
        ratio_orig, ratio_dest = _calc_shared_ratio(shared_len, p_orig, p_dest)

        edge_data['shared_boundary_length'] = shared_len  # Opcional: mantener absoluto si lo necesitas
        edge_data['shared_ratio_orig'] = ratio_orig    # Proporción del perímetro de ORIGEN compartida
        edge_data['shared_ratio_dest'] = ratio_dest    # Proporción del perímetro de DESTINO compartida
        edge_data['shared_ratio_avg'] = round((ratio_orig + ratio_dest) / 2, 4)  # Promedio simétrico

        # 🟡 BARRERAS (solo si se pasan)
        if usar_barreras and shared_len > 0.5 and shared_geom:
            crosses_barrier = False
            pct_barrera = 0.0

            if path:
                path_line = _path_to_line(G_calles, path)
                if path_line:
                    idxs = barreras_tree.query(path_line)
                    if len(idxs) > 0:
                        for ib in idxs:
                            if path_line.intersects(barreras_geoms[ib]):
                                crosses_barrier = True
                                break

            # Buffer de 0.5m para compensar desfases topológicos entre capas
            shared_buffered = shared_geom.buffer(0.5)
            idxs_b = barreras_tree.query(shared_buffered)
            if len(idxs_b) > 0:
                inter_len = sum(barreras_geoms[i].intersection(shared_buffered).length for i in idxs_b)
                pct_barrera = min((inter_len / shared_len) * 100, 100.0)

            edge_data['crosses_barrier'] = crosses_barrier
            edge_data['pct_barrera_frontera'] = round(pct_barrera, 2)

        # 🔵 ELEVACIÓN (solo si se pasa DEM)
        if dem_path:
            desnivel_abs_m, p_med = _calc_elevation_stats(G_calles, path)
            extra_dz = 0.0

            if include_centroid_leg:
                # Add centroid-to-node elevation change when available
                z_centroid_orig = d_orig.get('elev_centroid', np.nan)
                z_centroid_dest = d_dest.get('elev_centroid', np.nan)
                z_node_orig = G_calles.nodes[n_orig].get('elevation', np.nan)
                z_node_dest = G_calles.nodes[n_dest].get('elevation', np.nan)

                if not np.isnan(z_centroid_orig) and not np.isnan(z_node_orig):
                    extra_dz += abs(z_centroid_orig - z_node_orig)
                if not np.isnan(z_centroid_dest) and not np.isnan(z_node_dest):
                    extra_dz += abs(z_centroid_dest - z_node_dest)

                desnivel_abs_m = desnivel_abs_m + extra_dz
                if dist_ruta and dist_ruta > 0:
                    p_med = (desnivel_abs_m / dist_ruta) * 100
                else:
                    p_med = 0.0

            edge_data['desnivel_acumulado_m'] = round(desnivel_abs_m, 2)
            edge_data['pendiente_media_pct'] = round(p_med, 2)

        edges_to_add.append((id_origen, id_destino, edge_data))

    # 3. Añadir aristas
    for u, v, data in edges_to_add:
        G_censo.add_edge(u, v, **data)

    aislados = [n for n, deg in G_censo.degree() if deg == 0]
    if aislados:
        try:
            muestra = sorted(aislados)[:20]
        except TypeError:
            muestra = aislados[:20]
        print(f"Aviso: secciones sin vecinos: {len(aislados)}")
        print(f"Ejemplo: {muestra}")

    return G_censo

def calcular_dist_centorides(graf, shp_censo, barreras_gdf=None, dem_path=None,
                             include_centroid_leg=True, use_representative_point=True):
    ID_COL = 'ID_seccion'
    ZONA_EPSG = 25831
    G_proj, secciones_proc = preprocesar_datos(
        graf,
        shp_censo,
        epsg=ZONA_EPSG,
        use_representative_point=use_representative_point
    )
    secciones_proc = mapear_nodos_red(G_proj, secciones_proc)
    pares_vecinos = obtener_pares_vecinos(secciones_proc, id_col=ID_COL)
    G_final = calcular_distancias_y_grafo(G_proj, secciones_proc, pares_vecinos, id_col=ID_COL,
                                          barreras_gdf=barreras_gdf,
                                          dem_path=dem_path,
                                          include_centroid_leg=include_centroid_leg)
    G_final.graph['crs'] = f"EPSG:{ZONA_EPSG}"
    return G_final

if __name__ == "__main__":
    print("El módulo funciones.py se está ejecutando directamente.")
else:
    print("El módulo funciones.py está siendo importado.")