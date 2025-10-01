# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:28:38 2025

@author: loren
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:18:38 2025

@author: loren
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:19:03 2025

@author: loren
"""
import os
import math
from pysheds.grid import Grid
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from scipy.signal import convolve2d

#%%Data

data_folder = 'C:/Users/loren/Desktop/Tesi_magi/codes/data'

# Load the DEM files

grid = Grid.from_raster(os.path.join(data_folder,'DEMfel.tif'), data_name='grid data')
dem = grid.read_raster(os.path.join(data_folder,'DEMfel.tif'), data_name='dem')


inflated_dem = grid.resolve_flats(dem)
fdir = grid.flowdir(inflated_dem)


         #N-3  NE-2 E-1  SE-8  S-7  SW-6   W-5  NW-4
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

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
im = ax.imshow(dem, extent=grid.extent, zorder=2, cmap='terrain')
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
    
_ = plt.title('Channel network (>5000 accumulation)', size=14)

#%% Selecting pixels

transform_view = grid.affine       # affine della view corrente (dopo il clip)
shape_view = grid.view(fdir).shape 

shapes = ((feat["geometry"], 1) for feat in branches["features"])
branches_raster = rasterio.features.rasterize(shapes=shapes, out_shape=shape_view, 
                                              transform=transform_view, fill=0,
                                              all_touched=False, dtype="uint8",)
#aim for pixels on the raster
binary = (branches_raster == 1).astype(np.uint8)
K = np.ones((3, 3), dtype=np.uint8)
count_3x3 = convolve2d(binary, K, mode='same', boundary='fill', fillvalue=0)
neighbor_count = count_3x3 - binary
pixels = (binary == 1) & (neighbor_count >= 4)
pixels_raster = pixels.astype(np.uint8)

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

# --- 2) Candidati iniziali: pixel-canale con >=3 vicini + soglia di accumulation ---
candidates = (pixels_raster.astype(bool)) & mask_acc

# --- 3) Selezione "non adiacente" (8-vicini), privilegiando acc più alto ---
rows, cols = np.where(candidates)
if rows.size == 0:
    selected_mask = np.zeros_like(candidates, dtype=bool)
else:
    # ordina i candidati per accumulation decrescente
    acc_vals = acc_view[rows, cols]
    order = np.argsort(acc_vals)[::-1]          # indici ordinati per valore decrescente
    rows, cols = rows[order], cols[order]

    selected_mask = np.zeros_like(candidates, dtype=bool)
    blocked = np.zeros_like(candidates, dtype=bool)   # celle vietate (selezionati + adiacenti)

    for r, c in zip(rows, cols):
        if blocked[r, c]:
            continue
        # seleziona (r,c)
        selected_mask[r, c] = True
        # "blocca" la sua 8-neighborhood (3x3) per evitare adiacenti
        r0 = max(0, r-1); r1 = min(H, r+2)
        c0 = max(0, c-1); c1 = min(W, c+2)
        blocked[r0:r1, c0:c1] = True

# --- 4) Risultati ---
# selected_mask: booleano con i pixel scelti (no adiacenze, sopra soglia)
pixels_selected = selected_mask.astype(np.uint8)

# (opzionale) coordinate (x,y) dei pixel scelti

r_sel, c_sel = np.where(selected_mask)
xsel, ysel = rasterio.transform.xy(grid.affine, r_sel, c_sel, offset='center')

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
rows, cols = np.where(pixels)                         # indici dei pixel selezionati
xs, ys = rasterio.transform.xy(transform_view, rows, cols, offset='center') 
coords = np.column_stack([xs, ys])



fdir_view = grid.view(fdir)         # shape della view, es. (82, 52)        # transform corrente (della view)
              
xmin, ymin, xmax, ymax = grid.bbox 

xs = np.asarray(xsel); ys = np.asarray(ysel)


catchments = []   # lista di maschere boolean (una per punto)
areas = []        # area m² (o unità del tuo CRS) di ogni catchment
labels = np.zeros_like(fdir_view, dtype=np.int32)  # raster etichettato (facoltativo)


os.makedirs("catchments_tif", exist_ok=True)

for i, (x, y) in enumerate(zip(xs, ys), start=1):
    try:
        mask = grid.catchment(x=x, y=y, fdir=fdir_view, dirmap=dirmap, 
                              xytype='coordinate')
    except Exception:
        # punto fuori canale o su NoData: salta
        continue
    if mask is None or not np.any(mask):
        continue

    catchments.append(mask)                   # salva la maschera del sottobacino
    areas.append(mask.sum() * cell_area_m2)      # area contribuente del punto
    labels[mask & (labels == 0)] = i          # (opzionale) raster etichettato
    
print(f"Catchment estratti: {len(catchments)}")
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
    
    
i = 54 # indice del sottobacino (0-based)


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


