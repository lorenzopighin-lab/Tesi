# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:33:36 2025

@author: loren
"""
import os
import math
from pysheds.grid import Grid
import rasterio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from scipy.signal import convolve2d

#%%Data

data_folder = 'C:/Users/loren/Desktop/Tesi_magi/codes/data'

# Load the DEM files

grid = Grid.from_raster(os.path.join(data_folder,'DEMfel.tif'), data_name='grid data')
dem = grid.read_raster(os.path.join(data_folder,'DEMfel.tif'), data_name='dem')

flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)
fdir = grid.flowdir(inflated_dem)




# Compute accumulation
acc = grid.accumulation(fdir)

# Snap pour point to high accumulation cell (find the main outlet)
xy = 1.6825e6, 5.065e6
x_snap, y_snap = grid.snap_to_mask(acc > 100000,(xy))


#plot densità di drenaggio e sezione di chiusura
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=3)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
ax.scatter([x_snap], [y_snap], s=80, facecolors='none', edgecolors='red',
           linewidth=1.8, zorder=4, label='Punto')

plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')

n_pixels = int(np.count_nonzero(catch))   # equivalente a int(catch.sum())
print("Pixel nel catchment:", n_pixels)

#%%%plots
# DEM 
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=3)
im = ax.imshow(inflated_dem, extent=grid.extent, zorder=2, cmap='terrain')
ax.scatter([x_snap], [y_snap], s=80, facecolors='none', edgecolors='red',
           linewidth=1.8, zorder=4, label='Punto')
plt.colorbar(im, ax=ax, label='Elevation')
plt.title('DEM', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

#CATCH
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(catch, extent=grid.extent, zorder=3)
ax.scatter([x_snap], [y_snap], s=80, facecolors='none', edgecolors='red',
           linewidth=1.8, zorder=4, label='Punto')
ax.set_title('DEM', size=14)
plt.tight_layout()
plt.show()
#%%branches
grid.clip_to(catch)
branches = grid.extract_river_network(fdir, acc>900)

#%%%plot branches 

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
_ = plt.title('Channel network (>900 accumulation)', size=14)

#%% Selecting pixels

transform_view = grid.affine       # affine della view corrente (dopo il clip)
shape_view = grid.view(fdir).shape 

shapes = ((feat["geometry"], 1) for feat in branches["features"])
branches_raster = rasterio.features.rasterize(shapes=shapes, out_shape=shape_view, 
                                              transform=transform_view, fill=0,
                                              all_touched=False, dtype="uint8",)
#aim for pixels on the raster
binary = (branches_raster == 1).astype(np.uint8)
# --- identificazione delle confluenze ---
# Calcolo del numero di pixel della rete adiacenti (8-neighborhood)
K = np.ones((3, 3), dtype=np.uint8)
count_3x3 = convolve2d(binary, K, mode='same', boundary='fill', fillvalue=0)
neighbor_count = count_3x3 - binary
# Un nodo di confluenza ha grado >= 3 nella rete; sfruttiamo neighbor_count per identificarli
confluences_mask = (binary == 1) & (neighbor_count >= 3)
pixels_raster = confluences_mask.astype(np.uint8)

#%%%CHAT

# --- 0) Accumulation nella view (stessa shape di pixels_raster) ---
acc_view = grid.view(acc)              # acc calcolato prima: acc = grid.accumulation(fdir)
H, W = acc_view.shape

# --- 1) Soglia area contribuente ---
# in km^2:

cell_area_m2 = abs(grid.affine.a * grid.affine.e)
thr_km2 = 0.5
thr_cells = math.ceil((thr_km2 * 1e6) / cell_area_m2)

mask_acc = acc_view >= thr_cells

def build_stream_graph(stream_mask, pixel_dx, pixel_dy):
    """Costruisce un grafo non orientato dei pixel appartenenti alla rete di drenaggio."""
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    graph = nx.Graph()
    rows, cols = np.where(stream_mask)
    nodes = set(zip(rows, cols))
    graph.add_nodes_from(nodes)

    for r, c in nodes:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in nodes:
                continue
            # distanza euclidea in metri fra pixel adiacenti
            dist = math.hypot(dc * pixel_dx, dr * pixel_dy)
            graph.add_edge((r, c), (nr, nc), weight=dist)

    return graph


def greedy_stream_sampling(graph, candidates, acc_array, min_distance):
    """Seleziona nodi sul grafo mantenendo una distanza minima lungo la rete."""
    if not candidates:
        return []

    # ordina per accumulation decrescente
    candidates_sorted = sorted(candidates,
                               key=lambda rc: acc_array[rc],
                               reverse=True)

    selected = []
    for node in candidates_sorted:
        keep = True
        for chosen in selected:
            try:
                dist = nx.shortest_path_length(graph, node, chosen, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if dist < min_distance:
                keep = False
                break
        if keep:
            selected.append(node)
    return selected


# --- 2) Candidati iniziali: confluenze con soglia di accumulation ---
candidates_mask = confluences_mask & mask_acc

# Parametro di distanza minima lungo la rete (metri)
pixel_dx = abs(grid.affine.a)
pixel_dy = abs(grid.affine.e)
min_stream_distance_m = 500  # personalizzabile

stream_graph = build_stream_graph(binary.astype(bool), pixel_dx, pixel_dy)
candidate_nodes = list(zip(*np.where(candidates_mask)))

selected_nodes = greedy_stream_sampling(stream_graph, candidate_nodes, acc_view,
                                        min_stream_distance_m)

selected_mask = np.zeros_like(candidates_mask, dtype=bool)
for r, c in selected_nodes:
    selected_mask[r, c] = True

# --- 4) Risultati ---
# selected_mask: booleano con i pixel scelti, distanziati lungo la rete
pixels_selected = selected_mask.astype(np.uint8)

# coordinate (x,y) dei pixel scelti

r_sel, c_sel = np.where(selected_mask)
xsel, ysel = rasterio.transform.xy(grid.affine, r_sel, c_sel, offset='center')

# --- Snap dei pixel selezionati alla cella di accumulo più vicina sulla rete ---
channel_mask = branches_raster.astype(bool)
snapped_indices = []   # (row, col) snappati
snapped_coords = []    # (x, y) snappati

def _local_max_index(r, c, acc_arr, mask=None, max_radius=5):
    """Trova l'indice (row, col) della cella con accumulo massimo vicina."""
    h, w = acc_arr.shape
    if mask is not None:
        mask = mask.astype(bool)
    r = int(r)
    c = int(c)
    best_rc = (r, c)
    if 0 <= r < h and 0 <= c < w:
        best_val = acc_arr[r, c]
    else:
        best_val = -np.inf
    for radius in range(max_radius + 1):
        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(w, c + radius + 1)
        window = acc_arr[r0:r1, c0:c1]
        if window.size == 0:
            continue
        if mask is not None:
            mask_window = mask[r0:r1, c0:c1]
            if not mask_window.any():
                continue
            masked_vals = np.where(mask_window, window, -np.inf)
            idx = np.argmax(masked_vals)
            if np.isneginf(masked_vals.flat[idx]):
                continue
        else:
            idx = np.argmax(window)
        wr, wc = np.unravel_index(idx, window.shape)
        candidate_val = window[wr, wc]
        if candidate_val > best_val:
            best_val = candidate_val
            best_rc = (r0 + wr, c0 + wc)
        if mask is not None and mask_window[wr, wc]:
            return best_rc
    return best_rc

for rr, cc, x, y in zip(r_sel, c_sel, xsel, ysel):
    snapped_xy = None
    if channel_mask.any():
        try:
            snapped_xy = grid.snap_to_mask(channel_mask, (x, y))
        except Exception:
            snapped_xy = None
    if snapped_xy is not None:
        xsnap, ysnap = snapped_xy
        try:
            rsnap, csnap = grid.index((xsnap, ysnap))
        except Exception:
            rsnap, csnap = rr, cc
    else:
        rsnap, csnap = rr, cc
        xsnap, ysnap = x, y

    # in caso di snap fuori canale, cerca localmente il massimo di accumulo
    if not (0 <= rsnap < H and 0 <= csnap < W) or not channel_mask[rsnap, csnap]:
        rsnap, csnap = _local_max_index(rr, cc, acc_view, mask=channel_mask)
        xsnap, ysnap = rasterio.transform.xy(grid.affine, rsnap, csnap, offset='center')

    snapped_indices.append((int(rsnap), int(csnap)))
    snapped_coords.append((float(xsnap), float(ysnap)))
#%%%plot pixel sulla rete
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
zorder=2

sc = ax.scatter(xsel, ysel,
                    s=18, c='red',
                    edgecolors='k', linewidths=0.3,
                    zorder=5, label='Pixel selezionati') 
_ = plt.title('Channel network (>5000 accumulation)', size=14)

#%%%controllo pixels
count_ones = np.sum(branches_raster)
confluences = pixels_selected.sum()
print("Pixel canali", count_ones)
print("Numero confluences:", confluences)

check = pixels_selected * branches_raster
check_ones = np.sum(check)
print("confluences sulla rete", check_ones)

if check_ones == confluences: 
    print("Pixel Trovati")
else: print("Discrepanze pixel rete")
#%% Area contribuente dei pixel estratti (catchments)
rows, cols = np.where(selected_mask)                         # indici dei pixel selezionati
xs, ys = rasterio.transform.xy(transform_view, rows, cols, offset='center')
coords = np.column_stack([xs, ys])



fdir_view = grid.view(fdir)         # shape della view, es. (82, 52)        # transform corrente (della view)

xmin, ymin, xmax, ymax = grid.bbox

catchments = []   # lista di maschere boolean (una per punto)
areas = []        # area m² (o unità del tuo CRS) di ogni catchment
labels = np.zeros_like(fdir_view, dtype=np.int32)  # raster etichettato (facoltativo)
expected_areas = []
valid_xs = []
valid_ys = []

os.makedirs("catchments_tif", exist_ok=True)

for (rsnap, csnap), (x_snap, y_snap) in zip(snapped_indices, snapped_coords):
    if not (0 <= rsnap < H and 0 <= csnap < W):
        continue
    expected_area = acc_view[rsnap, csnap] * cell_area_m2
    if expected_area <= 0:
        continue

    try:
        mask = grid.catchment(x=x_snap, y=y_snap, fdir=fdir_view,
                              xytype='coordinate')
    except Exception:
        mask = None
    if mask is None or not np.any(mask):
        continue

    actual_area = mask.sum() * cell_area_m2
    deviation = abs(actual_area - expected_area) / expected_area if expected_area else np.inf

    if deviation > 0.2:
        # prova un nuovo snap basato sul massimo di accumulo locale
        alt_r, alt_c = _local_max_index(rsnap, csnap, acc_view, mask=channel_mask, max_radius=10)
        if (alt_r, alt_c) != (rsnap, csnap):
            x_snap, y_snap = rasterio.transform.xy(grid.affine, alt_r, alt_c, offset='center')
            expected_area = acc_view[alt_r, alt_c] * cell_area_m2
            if expected_area <= 0:
                continue
            try:
                mask = grid.catchment(x=x_snap, y=y_snap, fdir=fdir_view,
                                      xytype='coordinate')
            except Exception:
                mask = None
            if mask is None or not np.any(mask):
                continue
            actual_area = mask.sum() * cell_area_m2
            deviation = abs(actual_area - expected_area) / expected_area if expected_area else np.inf
            rsnap, csnap = alt_r, alt_c

    if deviation > 0.2:
        continue

    catchments.append(mask)
    actual_area = mask.sum() * cell_area_m2
    areas.append(actual_area)
    expected_areas.append(expected_area)
    label_id = len(catchments)
    labels[mask & (labels == 0)] = label_id
    valid_xs.append(x_snap)
    valid_ys.append(y_snap)

xs = np.asarray(valid_xs)
ys = np.asarray(valid_ys)

print(f"Catchment estratti: {len(catchments)}")
if expected_areas:
    areas_arr = np.asarray(areas)
    expected_arr = np.asarray(expected_areas)
    deviation_pct = np.where(expected_arr > 0,
                             np.abs(areas_arr - expected_arr) / expected_arr * 100,
                             np.nan)
    print(f"Deviazione media area rispetto ad accumulation: {np.nanmean(deviation_pct):.2f}%")
#%%% Plot Aree contribuenti

def plot_catchment_i(i,
                     catchments,        # lista di mask boolean (come creato prima)
                     xs, ys,      # coordinate dei seed (ordinate come catchments)
                     catch,             # mask booleana del bacino principale (stessa view)
                     transform_view,    # grid.affine della view clippata
                     dem_view=None,     # opzionale: DEM nella view (stessa shape)
                     main_seed=None     # opzionale: (x0, y0) seed del bacino principale
                     ):
    """
    Visualizza il sottobacino i-esimo sovrapposto al bacino principale, con i relativi punti seed.
    """
    mask_i = catchments[i]          # boolean array (H, W)
    h, w = mask_i.shape
    xmin, ymin, xmax, ymax = rasterio.transform.array_bounds(h, w, 
                                                             transform_view)

    fig, ax = plt.subplots(figsize=(7, 7))

    # 1) base DEM (opzionale)
    if dem_view is not None and dem_view.shape == mask_i.shape:
        ax.imshow(dem_view, extent=(xmin, xmax, ymin, ymax), origin="upper", 
                  cmap="gray")

    # 2) bacino principale in grigio trasparente
    ax.imshow(np.where(catch, 1, np.nan), extent=(xmin, xmax, ymin, ymax),
              origin="upper", alpha=0.25)
    
    # 4) sottobacino i-esimo più marcato
    ax.imshow(np.where(mask_i, 1, np.nan), extent=(xmin, xmax, ymin, ymax),
              origin="upper")
    
    # 3) rete di drenaggio
    for branch in branches['features']:
            line = np.asarray(branch['geometry']['coordinates'])
            plt.plot(line[:, 0], line[:, 1], alpha=0.5)
 

    # ) punti seed
    # seed del sottobacino i-esimo
    ax.scatter([xs[i]], [ys[i]], s=30, marker='o', edgecolor='k', 
               linewidth=0.8, label=f"seed #{i}")
    # seed del bacino principale (se fornito)
    
    
    ax.scatter([x_snap], [y_snap], s=40, marker='^', edgecolor='k', 
                   linewidth=0.8, label="seed principale")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Area contribuente #{i} su bacino principale")
    ax.legend(loc="best")
    plt.show()
    
    
i = 0# indice del sottobacino (0-based)


plot_catchment_i(i, catchments=catchments, xs=xs, ys=ys, catch=grid.view(catch),
                 transform_view=grid.affine)

#%% Stats piogge
# --- Parametri simulazione pioggia ---
T = 200                   # numero di timestep (es. 200 giorni)
p_rain = 0.30             # probabilità che un pixel piova (metti quello che vuoi)
rain_value = 10.0         # valore quando piove (es. 10 mm)
value_unit = "mm"         # solo informativo per l'output/etichetta

# --- Griglia: usiamo la view clippata (coerente coi catchments) ---
fdir_view = grid.view(fdir)         
H, W = fdir_view.shape
cell_area = abs(transform_view.a * transform_view.e)  # area cella (tip. m²)

# --- 1) Genera serie di pioggia 0/10 di shape (T, H, W) ---
rng = np.random.default_rng(seed=42)  # fissiamo il seme per riproducibilità
rain = rng.choice([0.0, rain_value], size=(T, H, W), p=[1-p_rain, p_rain])


#Precipitazioni medie
rain_time_mean = rain.mean(axis=0)  # media temporale per pixel 


means_mm = np.array([
    float(rain_time_mean[mask].mean()) if mask.any() else np.nan
    for mask in catchments])  #precipitazioni medie sul sottobacino


#Frequenze precipitazioni
rain_events = (rain > 0)                 # True/False: piove?
freq_pixel = rain_events.mean(axis=0)    # media su T -> frequenza

# 2) frequenza media per sottobacino -> vettore (N_catchments,)
freq_catch = np.array([
    float(freq_pixel[mask].mean()) if mask.any() else np.nan
    for mask in catchments])


