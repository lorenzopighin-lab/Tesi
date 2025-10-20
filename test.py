
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:28:45 2025

@author: loren
"""

import os
import math

import array 
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from pysheds.grid import Grid
from rasterio.transform import rowcol



# --- Configurazione generale ---

dem_path  = "C:/Users/loren/Desktop/Tesi_magi/codes/data/DE210960.tif"
POUR_POINT = array.array('f', [11.501329, 48.728112	]) 
SNAP_ACCUMULATION_THRESHOLD = 100000
BRANCH_ACCUMULATION_THRESHOLD = 5000
MIN_ACCUMULATION_KM2 = 10

# --- Parametri dimensionali ---
# Il DEM è espresso in coordinate geografiche ma sappiamo che le celle
# rappresentano quadrati da 30 m di lato.  Per le operazioni che richiedono
# distanze o superfici in metri utilizziamo quindi questo fattore di
# conversione esplicito.
PIXEL_SIZE_METERS = 30.0
PIXEL_AREA_M2 = PIXEL_SIZE_METERS ** 2

# --- Parametri selezione pixel ---
# "equidistant": distribuisce i pixel intermedi in modo equidistante lungo il ramo
# "spacing": posiziona i pixel intermedi ogni INTERMEDIATE_SPACING_METERS lungo il ramo
PIXEL_PLACEMENT_MODE = "equidistant"
MAX_INTERMEDIATE_PIXELS = 1
INTERMEDIATE_SPACING_METERS = 600.0

with rasterio.open(dem_path) as src:
    dem = src.read(1, out_dtype='float32')
    profile = src.profile

# --- Lettura dei dati di input ---
grid = Grid.from_raster(os.path.join(dem_path), data_name="grid data")
dem = grid.read_raster(os.path.join(dem_path), data_name="dem")

flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)
fdir = grid.flowdir(inflated_dem)

# Calcola l'accumulo di flusso a valle di ogni cella.
acc = grid.accumulation(fdir)

# Aggancia il punto di chiusura sul pixel con accumulo sufficiente.
x_snap, y_snap = grid.snap_to_mask(
    acc > SNAP_ACCUMULATION_THRESHOLD, POUR_POINT,)


# --- Visualizzazione accumulo di flusso e punto di chiusura ---
fig, ax = plt.subplots(figsize=(8, 6))
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
inflated_dem_masked = np.ma.masked_invalid(inflated_dem)

fig, ax = plt.subplots(figsize=(8, 6))
dem_im = ax.imshow(
    inflated_dem_masked,
    extent=grid.extent,
    cmap='terrain',
    origin='upper',
)
ax.scatter(
    [x_snap],
    [y_snap],
    s=80,
    facecolors='none',
    edgecolors='red',
    linewidth=1.8,
    label='Punto',
)
ax.set_title('DEM', size=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(dem_im, ax=ax, label='Elevation')
plt.tight_layout()

# Catchment outline over DEM
fig, ax = plt.subplots(figsize=(8, 6))
dem_im = ax.imshow(
    inflated_dem_masked,
    extent=grid.extent,
    cmap='terrain',
    origin='upper',
)

# coordinate arrays for contour plotting
x_coords = np.linspace(grid.extent[0], grid.extent[1], catch.shape[1])
y_coords = np.linspace(grid.extent[3], grid.extent[2], catch.shape[0])

catch_bool = catch.astype(bool)
ax.contour(
    x_coords,
    y_coords,
    catch_bool,
    levels=[0.5],
    colors='dodgerblue',
    linewidths=1.8,
)

# optional filled overlay of the catchment (transparent red)
catch_overlay = np.ma.masked_where(~catch_bool, catch_bool)
ax.imshow(
    catch_overlay,
    extent=grid.extent,
    origin='upper',
    cmap=colors.ListedColormap([(1.0, 0.0, 0.0, 0.35)]),
    zorder=4,
)
ax.scatter(
    [x_snap],
    [y_snap],
    s=80,
    facecolors='none',
    edgecolors='red',
    linewidth=1.8,
    label='Punto',
)
ax.set_title('Catchment Outline on DEM', size=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(dem_im, ax=ax, label='Elevation')
plt.tight_layout()
plt.show()
#%%branches
# Limita la griglia al bacino e ricava la rete drenante principale.
grid.clip_to(catch)
catch_view = grid.view(catch).astype(bool)
branches = grid.extract_river_network(
    fdir, acc > BRANCH_ACCUMULATION_THRESHOLD
)
                                      
# coordinate del seed principale nella view clippata
main_seed_row, main_seed_col = map(int, rowcol(grid.affine, x_snap, y_snap))
main_seed_x, main_seed_y = rasterio.transform.xy(
    grid.affine, main_seed_row, main_seed_col, offset="center")

#%%%plot branches 

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
_ = plt.title(
    f"Channel network (>{BRANCH_ACCUMULATION_THRESHOLD} accumulation)",
    size=14,
)
#%% Impostazioni distribuzione pixel lungo i rami
# I parametri sono definiti nella sezione di configurazione iniziale.
# Rasterizza la rete per identificare i pixel appartenenti a ciascun ramo.

fdir_view = grid.view(fdir)
transform_view = grid.affine       # affine della view corrente (dopo il clip)
shape_view = fdir_view.shape

shapes = (
    (feat["geometry"], idx + 1) for idx, feat in enumerate(branches["features"])
)
branches_raster = rasterio.features.rasterize(
    shapes=shapes,
    out_shape=shape_view,
    transform=transform_view,
    fill=0,
    all_touched=False,
    dtype="uint16",
)


# --- identificazione delle confluenze e dei pixel a monte ---
acc_view = grid.view(acc)              # acc calcolato prima: acc = grid.accumulation(fdir)
H, W = acc_view.shape

cell_area_m2 = PIXEL_AREA_M2
cell_size_x = PIXEL_SIZE_METERS
cell_size_y = PIXEL_SIZE_METERS
thr_cells = math.ceil((MIN_ACCUMULATION_KM2 * 1e6) / cell_area_m2)

# Seleziona i pixel con accumulo sufficiente per essere considerati candidati.
mask_acc = acc_view >= thr_cells

# Offsets per analizzare i vicini secondo lo schema D8.
neighbor_offsets = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Mapping tra offset e codici di direzione del flusso D8.
d8_from_offset = {
    (-1, -1): 32,
    (-1, 0): 64,
    (-1, 1): 128,
    (0, -1): 16,
    (0, 1): 1,
    (1, -1): 8,
    (1, 0): 4,
    (1, 1): 2,
}

# Mapping inverso dai codici D8 all'offset.
d8_to_offset = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}


selected_coords_set = set()
selected_coords_list = []
# Mappa branch_id -> lista di pixel selezionati (per analisi e debug).
branch_selections = {}

def _add_selected_coord(rc):
    """Aggiunge una cella alla lista dei punti selezionati se valida."""
    if rc is None:
        return False
    r, c = map(int, rc)
    if not (0 <= r < H and 0 <= c < W):
        return False
    if catch_view is not None and not catch_view[r, c]:
        return False
    if not mask_acc[r, c]:
        return False

    key = (r, c)
    if key in selected_coords_set:
        return False

    selected_coords_set.add(key)
    selected_coords_list.append(key)
    return True

unique_branch_ids = np.unique(branches_raster)
unique_branch_ids = unique_branch_ids[unique_branch_ids > 0]

# Per ciascun ramo seleziona pixel significativi lungo la direzione del flusso.
for branch_id in unique_branch_ids:
    branch_mask = branches_raster == branch_id
    candidate_indices = np.argwhere(branch_mask)
    if candidate_indices.size == 0:
        continue
    
# Filtra i pixel del ramo che soddisfano le condizioni sul bacino e sull'accumulo.
    filtered_coords = []
    for r, c in candidate_indices:
        if catch_view is not None and not catch_view[r, c]:
            continue
        if not mask_acc[r, c]:
            continue
        filtered_coords.append((int(r), int(c)))

    if not filtered_coords:
        continue

    filtered_coords.sort(key=lambda rc: acc_view[rc])
    n_coords = len(filtered_coords)

    # Distanze cumulative lungo il ramo (ordinate da monte a valle)
    cumulative_distances = [0.0]
    for idx in range(1, n_coords):
        r0, c0 = filtered_coords[idx - 1]
        r1, c1 = filtered_coords[idx]
        dr = (r1 - r0) * cell_size_y
        dc = (c1 - c0) * cell_size_x
        step_distance = math.hypot(dr, dc)
        cumulative_distances.append(cumulative_distances[-1] + step_distance)

    branch_selected = []
    used_indices = set()

# Seleziona sempre l'estremo a monte del ramo.
    if _add_selected_coord(filtered_coords[0]):
        branch_selected.append(filtered_coords[0])
        used_indices.add(0)

# Seleziona sempre l'estremo a valle del ramo.
    if n_coords > 1 and _add_selected_coord(filtered_coords[-1]):
        branch_selected.append(filtered_coords[-1])
        used_indices.add(n_coords - 1)

# Numero massimo di slot disponibili per eventuali punti intermedi.
    available_slots = max(0, n_coords - 2)

    if n_coords > 2 and available_slots > 0:
        mode = (PIXEL_PLACEMENT_MODE or "").strip().lower()
        # Target distances esprime le posizioni desiderate lungo il ramo.
        target_distances = []

        if mode == "spacing":
            try:
                spacing_m = float(INTERMEDIATE_SPACING_METERS)
            except (TypeError, ValueError):
                spacing_m = 0.0
            if spacing_m > 0 and cumulative_distances[-1] > 0:
                n_targets = int(cumulative_distances[-1] // spacing_m)
                max_intermediate = None
                if MAX_INTERMEDIATE_PIXELS is not None:
                    try:
                        max_intermediate = int(MAX_INTERMEDIATE_PIXELS)
                    except (TypeError, ValueError):
                        max_intermediate = None
                if max_intermediate is not None:
                    n_targets = min(n_targets, max_intermediate)
                n_targets = min(n_targets, available_slots)
                target_distances = [spacing_m * i for i in range(1, n_targets + 1)]
        else:
            max_intermediate = None
            if MAX_INTERMEDIATE_PIXELS is not None:
                try:
                    max_intermediate = int(MAX_INTERMEDIATE_PIXELS)
                except (TypeError, ValueError):
                    max_intermediate = None
            if max_intermediate is not None:
                n_intermediate = min(max_intermediate, available_slots)
            else:
                n_intermediate = max(1, math.ceil(n_coords / 30))
                n_intermediate = min(n_intermediate, available_slots)

            if n_intermediate > 0 and cumulative_distances[-1] > 0:
                target_distances = np.linspace(
                    0, cumulative_distances[-1], n_intermediate + 2
                )[1:-1]

        for target_distance in target_distances:
            available_indices = [
                i for i in range(1, n_coords - 1) if i not in used_indices
            ]
            if not available_indices:
                break

            closest_idx = min(
                available_indices,
                key=lambda i: abs(cumulative_distances[i] - target_distance),
            )

            coord = filtered_coords[closest_idx]
            if _add_selected_coord(coord):
                branch_selected.append(coord)
            used_indices.add(closest_idx)

    if branch_selected:
        branch_selected.sort(key=lambda rc: acc_view[rc])
        branch_selections[int(branch_id)] = branch_selected

total_selected = sum(len(v) for v in branch_selections.values())
print(f"Selezionati {total_selected} pixel su {len(branch_selections)} rami.")

selected_coords_list.sort(key=lambda rc: (-acc_view[rc], rc[0], rc[1])) 
# Ordina i pixel per importanza (accumulo decrescente e coordinate stabili).


# --- 4) Risultati ---
# selected_mask: booleano con i pixel scelti, distanziati lungo la rete

# Mask dei canali, utile per gli snap e per vincolare la ricerca dei massimi.
channel_mask = branches_raster.astype(bool)
# Lista di dizionari con le informazioni principali di ciascun pixel selezionato.
selected_points = []

def _build_selected_mask(points):
    """Costruisce una mask booleana partendo dalla lista di punti selezionati."""
    mask = np.zeros_like(branches_raster, dtype=bool)
    for pt in points:
        r, c = pt.get("snapped_index", (None, None))
        if r is None or c is None:
            continue
        if 0 <= r < H and 0 <= c < W:
            mask[r, c] = True
    return mask


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

# Costruisce il set di punti selezionati con coordinate e informazioni di snap.
for rr, cc in selected_coords_list:
    point = {
        "row": int(rr),
        "col": int(cc),
        "original_index": (int(rr), int(cc)),
    }
    x, y = rasterio.transform.xy(grid.affine, rr, cc, offset="center")
    point["initial_coord"] = (float(x), float(y))

    snapped_xy = None
    if channel_mask.any():
        try:
            snapped_xy = grid.snap_to_mask(channel_mask, (x, y))
        except Exception:
            snapped_xy = None
    if snapped_xy is not None:
        xsnap, ysnap = snapped_xy
        try:
            rsnap_float, csnap_float = rowcol(grid.affine, xsnap, ysnap)
            rsnap, csnap = int(rsnap_float), int(csnap_float)
        except Exception:
            rsnap, csnap = int(rr), int(cc)
    else:
        rsnap, csnap = int(rr), int(cc)
        xsnap, ysnap = x, y


    if not (0 <= rsnap < H and 0 <= csnap < W) or not channel_mask[rsnap, csnap]:
        rsnap, csnap = _local_max_index(rr, cc, acc_view, mask=channel_mask)
        xsnap, ysnap = rasterio.transform.xy(
            grid.affine, rsnap, csnap, offset="center"
        )

    point["snapped_index"] = (int(rsnap), int(csnap))
    point["snapped_coord"] = (float(xsnap), float(ysnap))
    selected_points.append(point)

#%% Area contribuente dei pixel estratti (catchments)

xmin, ymin, xmax, ymax = grid.bbox

catchments = []   # lista di maschere boolean (una per punto)
areas = []        # area m² (o unità del tuo CRS) di ogni catchment
labels = np.zeros_like(fdir_view, dtype=np.int32)  # raster etichettato (facoltativo)
expected_areas = []
catchment_points = []
failed_points = []

os.makedirs("catchments_tif", exist_ok=True)

for point in selected_points:
    rsnap, csnap = point["snapped_index"]
    x_snap, y_snap = point["snapped_coord"]

    if not (0 <= rsnap < H and 0 <= csnap < W):
        failed_points.append({"point": point, "reason": "indice fuori dai limiti"})
        continue

    expected_area = acc_view[rsnap, csnap] * cell_area_m2
    if expected_area <= 0:
        failed_points.append({"point": point, "reason": "accumulation nulla"})
        continue

    def _compute_catchment(row, col):
        """Estrae l'area contribuente a partire da una cella della rete."""
        row = int(row)
        col = int(col)
        if not (0 <= row < H and 0 <= col < W):
            return None
        if catch_view is not None and not catch_view[row, col]:
            return None

        mask_local = np.zeros_like(fdir_view, dtype=bool)
        queue = [(row, col)]
        mask_local[row, col] = True
        head = 0

        while head < len(queue):
            cr, cc = queue[head]
            head += 1

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if mask_local[nr, nc]:
                        continue
                    if catch_view is not None and not catch_view[nr, nc]:
                        continue

                    delta_row = cr - nr
                    delta_col = cc - nc
                    flow_code = d8_from_offset.get((delta_row, delta_col))
                    if flow_code is None:
                        continue

                    if fdir_view[nr, nc] == 0 or fdir_view[nr, nc] != flow_code:
                        continue

                    mask_local[nr, nc] = True
                    queue.append((nr, nc))

        if not mask_local.any():
            return None

        return mask_local

    mask = _compute_catchment(rsnap, csnap)

    # prova un nuovo snap basato sul massimo di accumulo locale se necessario
    if mask is None:
        alt_r, alt_c = _local_max_index(
            rsnap, csnap, acc_view, mask=channel_mask, max_radius=10)
        
        if (alt_r, alt_c) != (rsnap, csnap):
            rsnap, csnap = int(alt_r), int(alt_c)
            x_snap, y_snap = rasterio.transform.xy(
                grid.affine, rsnap, csnap, offset='center')
            
            expected_area = acc_view[rsnap, csnap] * cell_area_m2
            if expected_area > 0:
                mask = _compute_catchment(rsnap, csnap)

    if mask is None:
        x_orig, y_orig = point.get("initial_coord", (None, None))
        r_orig, c_orig = point["row"], point["col"]
        if (
            x_orig is not None
            and y_orig is not None
            and 0 <= r_orig < H
            and 0 <= c_orig < W
        ):
            expected_area_orig = acc_view[r_orig, c_orig] * cell_area_m2
            if expected_area_orig > 0:
                candidate_mask = _compute_catchment(r_orig, c_orig)
                if candidate_mask is not None:
                    mask = candidate_mask
                    rsnap, csnap = int(r_orig), int(c_orig)
                    x_snap, y_snap = float(x_orig), float(y_orig)
                    expected_area = expected_area_orig

    if mask is None:
        point["catchment_found"] = False
        failed_points.append({"point": point, "reason": "catchment non trovato"})
        continue
    
    point["catchment_found"] = True
    point["snapped_index"] = (int(rsnap), int(csnap))
    point["snapped_coord"] = (float(x_snap), float(y_snap))

    actual_area = mask.sum() * cell_area_m2
    deviation = (
        abs(actual_area - expected_area) / expected_area if expected_area else np.inf
    )
    if deviation > 0.2:
        print(
            f"Avviso: deviazione area {deviation*100:.1f}% per il seed in {x_snap:.2f}, {y_snap:.2f}"
        )

    catchments.append(mask)
    areas.append(actual_area)
    expected_areas.append(expected_area)

    point["expected_area"] = float(expected_area)
    point["catchment_area"] = float(actual_area)
    point["deviation"] = float(deviation)
    catchment_points.append(point.copy())

    label_id = len(catchments)
    labels[mask & (labels == 0)] = label_id

selected_mask = _build_selected_mask(catchment_points)
pixels_selected = selected_mask.astype(np.uint8)

#%%%plot pixel sulla rete
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5, 6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

sc = ax.scatter(
                    [pt["snapped_coord"][0] for pt in catchment_points],
                    [pt["snapped_coord"][1] for pt in catchment_points],
                    s=18, c='red',
                    edgecolors='k', linewidths=0.3,
                    zorder=5, label='Pixel selezionati')
_ = plt.title(
    f"Channel network (>{BRANCH_ACCUMULATION_THRESHOLD} accumulation)",
    size=14,
)

#%%%controllo pixels
count_ones = np.sum(branches_raster > 0)
selected_count = pixels_selected.sum()
print("Pixel canali", count_ones)
print("Numero confluences:", selected_count)

check = pixels_selected.astype(bool) & (branches_raster > 0)
check_ones = np.sum(check)
print("confluences sulla rete", check_ones)

if check_ones == selected_count:
    print("Pixel Trovati")
else:
    print("Discrepanze pixel rete")

if failed_points:
    print(f"Catchment non estratti: {len(failed_points)}")
    for item in failed_points:
        coord = item["point"].get("initial_coord", (np.nan, np.nan))
        print(f" - seed {coord} -> {item['reason']}")

xs = np.asarray([pt["snapped_coord"][0] for pt in catchment_points])
ys = np.asarray([pt["snapped_coord"][1] for pt in catchment_points])

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
    if not catchments:
        raise ValueError("Nessun catchment disponibile per il plotting")
    if i < 0 or i >= len(catchments):
        raise IndexError(f"Indice {i} fuori range per {len(catchments)} catchments disponibili")

    mask_i = catchments[i]          # boolean array (H, W)         # boolean array (H, W)
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
    if main_seed is not None:
        ax.scatter(
            [main_seed[0]],
            [main_seed[1]],
            s=40,
            marker='^',
            edgecolor='k',
            linewidth=0.8,
            label="seed principale",
        )

    # punti seed dei sottobacini
    ax.scatter(
        xs,
        ys,
        s=20,
        c='red',
        edgecolor='k',
        linewidths=0.3,
        label='seed sottobacini',
    )

    # evidenzia il seed del sottobacino corrente
    ax.scatter(
        [xs[i]],
        [ys[i]],
        s=45,
        marker='o',
        facecolor='yellow',
        edgecolor='k',
        linewidth=0.6,
        label=f'seed catchment #{i}',
    )

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Area contribuente #{i} su bacino principale")
    ax.legend(loc="best")
    plt.show()
    
    
i = 120  # indice del sottobacino (0-based)

if catchments:
    plot_catchment_i(
        i,
        catchments=catchments,
        xs=xs,
        ys=ys,
        catch=grid.view(catch),
        transform_view=grid.affine,
        main_seed=(main_seed_x, main_seed_y),
    )
else:
    print("Nessun catchment valido da plottare.")

#%% Stats piogge
# --- Parametri simulazione pioggia ---
T = 200                   # numero di timestep (es. 200 giorni)
p_rain = 0.30             # probabilità che un pixel piova (metti quello che vuoi)
rain_value = 10.0         # valore quando piove (es. 10 mm)
value_unit = "mm"         # solo informativo per l'output/etichetta

# --- Griglia: usiamo la view clippata (coerente coi catchments) ---
H, W = fdir_view.shape
cell_area = PIXEL_AREA_M2  # area cella in metri quadrati
# --- 1) Genera serie di pioggia 0/10 di shape (T, H, W) ---
rng = np.random.default_rng(seed=42)  # fissiamo il seme per riproducibilità
rain = rng.choice([0.0, rain_value], size=(T, H, W), p=[1-p_rain, p_rain])

# --- Serie temporali di precipitazione mediate per catchment ---
rain_flat = rain.reshape(T, -1)
catchment_series = []
for mask in catchments:
    if mask is None or not mask.any():
        catchment_series.append(np.full(T, np.nan, dtype=float))
        continue

    mask_flat = mask.reshape(-1)
    ts_mean = rain_flat[:, mask_flat].mean(axis=1)
    catchment_series.append(ts_mean)

if catchment_series:
    rain_timeseries_catchments = np.vstack(catchment_series)
    # Precipitazioni medie nel periodo simulato (media sui timestep)
    means_mm = rain_timeseries_catchments.mean(axis=1)
else:
    rain_timeseries_catchments = np.empty((0, T), dtype=float)
    means_mm = np.array([], dtype=float)


#Frequenze precipitazioni
rain_events = (rain > 0)                 # True/False: piove?
freq_pixel = rain_events.mean(axis=0)    # media su T -> frequenza

# 2) frequenza media per sottobacino -> vettore (N_catchments,)
freq_catch = np.array([
    float(freq_pixel[mask].mean()) if mask.any() else np.nan
    for mask in catchments])


