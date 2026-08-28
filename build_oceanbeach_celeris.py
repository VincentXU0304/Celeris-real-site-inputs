#!/usr/bin/env python3
"""Build a COMPLETE, REAL Celeris dataset for Ocean Beach SF - a FAITHFUL Python port of the
official create_Celeris_inputs.m + spectrum_2D.m. Produces the exact minimal 22-field config.json
(NO incident_wave_type cruft), a spectrum_2D-style waves.txt, and an ASCII bathy.txt.

Bathymetry = NOAA CUDEM 1/9 arc-sec real topobathy (offshore -> SOUTH edge, so 'from south'=90 works).
Waves = real Open-Meteo Hs/Tp. No fabrication.
Spectrum: JONSWAP (gamma 3.3) x frequency-dependent cos^2s directional spreading, energy
renormalised to the target Hmo, 1% truncation, then periodic angle snapping. NO TMA finite-depth
factor is applied here; depth effects enter through the Boussinesq solver over the real bathymetry.
See README.md.

Ported from the Celeris input-generation logic in Celeris WebGPU
(https://github.com/plynett/plynett.github.io), MIT License, Copyright (c) 2023 plynett.
Celeris was developed by Sasan Tavakkol, building on an earlier demo by Stephen Thompson,
advised by Patrick Lynett (USC). See THIRD_PARTY_LICENSES/celeris-webgpu-MIT.txt and the
citations in README.md.

"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import cv2
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds

ROOT = Path(__file__).resolve().parent          # outputs go to ./data/ next to this script
# ===== pick a REAL Open-Meteo event -> SEPARATE output folder per event (nothing overwritten) =====
EVENTS = {                                        # name: (Hs m, Tp s, Thetap deg = dir-270)  -- WAVE-GRADE set:
    "2025-01-19_low":      (1.34, 16.9, 10.0),    # LOW       small clean NW groundswell 2025-01-19 (dir 280)
    "2025-03-01_moderate": (2.52, 15.2, 12.0),    # MODERATE  clean mid swell 2025-03-01 (dir 282)
    "2024-12-23_wavemax":  (3.98, 16.6, 6.0),     # HIGH      2024-12-23 (dir 276)  [folder name kept from earlier build]
    "2023-01-05_flood":    (5.34, 14.4, -19.0),   # VERY HIGH 2023-01-05 CA bomb cyclone (dir 251); 5.34m < 5.4m cap -> full
}
EVENT = sys.argv[1] if len(sys.argv) > 1 else "2023-01-05_flood"
Hmo, Tp, Thetap = EVENTS[EVENT]
OUT = ROOT / "data" / f"celeris_oceanbeach_{EVENT}"
OUT.mkdir(parents=True, exist_ok=True)
print(f"=== EVENT {EVENT}: Hs={Hmo} m, Tp={Tp} s, Thetap={Thetap} deg -> data/{OUT.name} ===")
VRT = ("/vsicurl/https://coast.noaa.gov/htdata/raster2/elevation/"
       "NCEI_ninth_Topobathy_2014_8483/NCEI_ninth_Topobathy_2014_m8483_EPSG-4269.vrt")
BBOX = [-122.524, 37.734, -122.508, 37.754]     # W,S,E,N Ocean Beach SF (landscape, long coast)
DX, DST_CRS, G = 2.0, "EPSG:32610", 9.81         # 2 m (was 4 m): finer -> resolves breaking like the example
LAND_LIM = 3.0                                    # crop off dry land above +3 m (never wet even at max run-up)

# ===== create_Celeris_inputs.m parameters =====
NLSW_or_Bous, Courant_num, isManning, friction = 1, 0.2, 0, 0.001
# Hmo, Tp, Thetap come from EVENTS above (per-event REAL Open-Meteo forcing, selected by CLI arg)
Theta, dissipation_threshold, whiteWaterDecayRate, timeScheme = 2.0, 0.15, 0.02, 2   # breaking slope 0.15 = example
seaLevel, Bcoef, tridiag_solve = 0.0, 0.06666667, 2
west_bt, east_bt, south_bt, north_bt = 2, 0, 3, 3     # EXAMPLE topology (0=wall 1=sponge 2=incoming 3=PERIODIC):
#                                                       W=2 incoming (offshore), E=0 wall (land=dry +6.5m),
#                                                       N/S=3 periodic (along-shore). Periodic along-shore is the
#                                                       whole reason spectrum_2D_periodic snaps the angles.

# --- real CUDEM, reproject to UTM, reorient so OFFSHORE = SOUTH edge (row 0 = south) ---
with rasterio.open(VRT) as src:
    win = from_bounds(*BBOX, src.transform)
    data = src.read(1, window=win).astype(np.float32)
    src_transform = src.window_transform(win)
data[data < -1e4] = -99999.0
ub = transform_bounds("EPSG:4326", DST_CRS, *BBOX)
Wg, Hg = int((ub[2] - ub[0]) / DX), int((ub[3] - ub[1]) / DX)
b0 = np.full((Hg, Wg), np.nan, np.float32)
reproject(data, b0, src_transform=src_transform, src_crs="EPSG:4269",
          dst_transform=from_origin(ub[0], ub[3], DX, DX), dst_crs=DST_CRS,
          src_nodata=-99999.0, dst_nodata=np.nan, resampling=Resampling.bilinear)
if np.isnan(b0).any():
    b0 = cv2.inpaint(np.nan_to_num(b0), np.isnan(b0).astype(np.uint8), 3, cv2.INPAINT_NS)
bathy = b0                                       # north-up: cols=x=W..E (col0=WEST=offshore), rows=y=N..S
# --- crop off high dry land on the EAST (cross-shore) so the domain is offshore->surf->run-up only ---
col_mean = bathy.mean(axis=0)                    # mean elevation per column (increases W->E toward land)
above = np.where(col_mean > LAND_LIM)[0]
keep_frac = 1.0
if len(above):
    orig_W = bathy.shape[1]
    east_cut = min(above[0] + 1, orig_W)
    keep_frac = east_cut / orig_W
    bathy = bathy[:, :east_cut]                  # keep offshore .. +3 m run-up; drop the dead dune/cliff
    print(f"east crop: kept cross-shore to +{LAND_LIM} m, dropped {(orig_W-east_cut)*DX:.0f} m of high dry land")
# --- make bathy PERIODIC along-shore (N-S) so the type-3 periodic wrap has no depth seam ---
seam = 0.5 * (bathy[0, :] + bathy[-1, :])
mism_before = float(np.abs(bathy[0, :] - bathy[-1, :]).max())
mB = int(120 / DX)                                # blend band (rows) at each along-shore edge (~120 m)
for i in range(mB):
    w = i / mB
    bathy[i, :] = (1 - w) * seam + w * bathy[i, :]
    bathy[-1 - i, :] = (1 - w) * seam + w * bathy[-1 - i, :]
print(f"periodic N-S blend: edge mismatch {mism_before:.2f} m -> {np.abs(bathy[0,:]-bathy[-1,:]).max():.4f} m (band {mB} rows, interior real)")
HEIGHT, WIDTH = bathy.shape                       # HEIGHT=rows(along-shore N-S), WIDTH=cols(cross-shore W-E)
base_depth = float(-bathy.min())
np.savetxt(OUT / "bathy.txt", bathy, fmt="%.4f")   # like MATLAB save bathy.txt -ascii
bathy.astype("<f4").tofile(OUT / "bathy.bin")
print(f"bathy {WIDTH}x{HEIGHT} base_depth={base_depth:.1f}  (offshore WEST col0={bathy[:,0].mean():.1f} m, land EAST col-1={bathy[:,-1].mean():.1f} m)")

# --- overlay: NAIP 0.6 m aerial (MS Planetary Computer), reprojected to the cropped bathy grid, north-up ---
try:
    import rasterio.merge
    feats = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", timeout=90, json={
        "collections": ["naip"], "bbox": BBOX,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}], "limit": 4}).json()["features"]
    year0 = feats[0]["properties"]["datetime"][:4]                 # newest year; merge its tile(s) over the bbox
    srcs = []
    for f in feats:
        if f["properties"]["datetime"][:4] != year0:
            continue
        signed = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign",
                              params={"href": f["assets"]["image"]["href"]}, timeout=60).json()["href"]
        srcs.append(rasterio.open("/vsicurl/" + signed))
    naip_crs = srcs[0].crs
    mb = transform_bounds("EPSG:4326", naip_crs, *BBOX)
    mos, mos_tr = rasterio.merge.merge(srcs, bounds=mb, res=0.6)
    for s in srcs:
        s.close()
    mos = mos[:3].astype(np.float32)                              # R,G,B (drop NIR)
    OVDX = 1.0                                                    # overlay pixel size (m): crisp, not huge
    ovW, ovH = int(WIDTH * DX / OVDX), int(HEIGHT * DX / OVDX)    # cropped bathy extent (WIDTH already = east_cut)
    ov3 = np.zeros((3, ovH, ovW), np.float32)
    reproject(mos, ov3, src_transform=mos_tr, src_crs=naip_crs,
              dst_transform=from_origin(ub[0], ub[3], OVDX, OVDX), dst_crs=DST_CRS,
              resampling=Resampling.bilinear)
    rgb = np.moveaxis(ov3, 0, -1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) ** 0.9
    ov = (rgb[..., ::-1] * 255).astype(np.uint8)                  # BGR for cv2
    ov = ov[::-1]                                                 # V-flip: Celeris renders the overlay texture upside-down vs the grid
    cv2.imwrite(str(OUT / "overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"overlay.jpg = NAIP {year0} 0.6m ({len(srcs)} tile(s)) -> {ov.shape[1]}x{ov.shape[0]} px")
except Exception as e:
    print("overlay skipped:", e)

# ===== spectrum_2D_periodic.m port -> waves.txt (JONSWAP + directional spread + periodic angle snapping) =====
gamma_s, spread_o, del_f, del_t = 3.3, 50.0, 0.001, 5.0
h_gen = float(-bathy[:, 0].mean())              # depth at the offshore (WEST) generation boundary
boundary_angle = 0.0                            # west-boundary normal = +x (shore-normal); theta measured from it
boundary_length = HEIGHT * DX - 2 * DX          # along-shore length of the generation line (minus 2 cells, per .m)
Hs = min(Hmo, h_gen * 0.5)
f_peak = 1.0 / Tp
beta = 0.0624 / (0.23 + 0.033 * gamma_s - 0.185 * (1.9 + gamma_s) ** -1)
f = np.arange(max(del_f, 1 / 25), 1 / 5 + del_f, del_f)
E = np.zeros(len(f))
for i, fi in enumerate(f):
    sigma = 0.07 if fi <= f_peak else 0.09
    frat = fi / f_peak
    E[i] = beta * Hs ** 2 / (fi * frat ** 4) * math.exp(-1.25 / frat ** 4) * gamma_s ** (math.exp(-(frat - 1) ** 2 / (2 * sigma ** 2)))
theta = np.arange(-20 + Thetap, 20 + Thetap + del_t, del_t)     # deg, 9 dirs about Thetap
D = np.zeros((len(f), len(theta)))
for i, fi in enumerate(f):
    fr = fi / f_peak
    spread = spread_o * fr ** 5 if fr < 1 else spread_o * fr ** -2.5
    beta_s = 2 ** (2 * spread - 1) / math.pi * math.gamma(spread + 1) ** 2 / math.gamma(2 * spread + 1)
    for j, th in enumerate(theta):
        D[i, j] = beta_s * math.cos(0.5 * ((th - Thetap) * math.pi / 180)) ** (2 * spread)
D /= D.sum(axis=1, keepdims=True)
E_D = E[:, None] * D
def dfl_full(i):
    if i == 0: return f[1] - f[0]
    if i == len(f) - 1: return f[-1] - f[-2]
    return (f[i + 1] - f[i]) / 2 + (f[i] - f[i - 1]) / 2
Hmo_calc = sum(E_D[i, :].sum() * dfl_full(i) for i in range(len(f)))
E_D *= (Hmo / (math.sqrt(Hmo_calc) * 4.004)) ** 2
jmid = round(len(theta) / 2) - 1
keep = np.where(E_D[:, jmid] > 0.01 * E_D[:, jmid].max())[0]     # 1% truncation on peak-dir energy
fk = f[keep]                                                     # kept freqs (contiguous block after truncation)
def dfl_keep(c):
    if c == 0: return fk[1] - fk[0]
    if c == len(fk) - 1: return fk[-1] - fk[-2]
    return (fk[c + 1] - fk[c]) / 2 + (fk[c] - fk[c - 1]) / 2

def theta_c_periodic(theta_o_deg, f_c):
    """spectrum_2D_periodic snapping: pick the angle so an integer # of crests fits along the boundary."""
    rel = theta_o_deg - boundary_angle
    omega = 2 * math.pi * f_c
    k_c = omega * omega / (G * math.sqrt(math.tanh(omega * omega * h_gen / G)))
    L_c = 2 * math.pi / k_c
    k_y = math.sin(math.radians(rel)) * k_c
    L_y = 2 * math.pi / (1e-6 + k_y)
    n_wav = abs(boundary_length / L_y)
    if n_wav < 0.5:                                              # near shore-normal -> just use normal
        return boundary_angle
    n = round(n_wav)
    L_y_near = boundary_length / n
    if L_y_near < L_c:                                           # first guess impossible -> squeeze one less
        n -= 1
        if n == 0:
            return boundary_angle
        L_y_near = boundary_length / n
    k_y_near = 2 * math.pi / L_y_near
    k_ratio = math.copysign(1.0, k_y) * k_y_near / k_c
    k_ratio = max(-1.0, min(1.0, k_ratio))
    return math.degrees(math.asin(k_ratio)) + boundary_angle

rng = np.random.default_rng(0)
with (OUT / "waves.txt").open("w") as fh:
    fh.write("\n[NumberOfWaves] {}\n=================================\n".format(len(keep) * len(theta)))
    for c, i in enumerate(keep):
        dfc = dfl_keep(c)
        for j in range(len(theta)):
            amp = math.sqrt(2 * E_D[i, j] * dfc)
            tc = theta_c_periodic(theta[j], f[i])
            fh.write(f"{amp:.6g} {1/f[i]:.6g} {math.radians(tc):.6g} {rng.uniform(0,2*math.pi):.6g}\n")
print(f"waves.txt {len(keep)*len(theta)} comps ({len(keep)}f x {len(theta)}dir) spectrum_2D_PERIODIC port; "
      f"h_gen={h_gen:.1f}m boundary_len={boundary_length:.0f}m")

(OUT / "models.json").write_text(json.dumps({"models": []}))

# ===== config.json: EXACT 22 fields from create_Celeris_inputs.m (no cruft) =====
cfg = {
    "WIDTH": WIDTH, "HEIGHT": HEIGHT, "dx": DX, "dy": DX, "Courant_num": Courant_num,
    "NLSW_or_Bous": NLSW_or_Bous, "base_depth": base_depth, "g": G, "Theta": Theta,
    "friction": friction, "isManning": isManning, "dissipation_threshold": dissipation_threshold,
    "whiteWaterDecayRate": whiteWaterDecayRate, "timeScheme": timeScheme, "seaLevel": 0.0,
    "Bcoef": Bcoef, "tridiag_solve": tridiag_solve, "west_boundary_type": west_bt,
    "east_boundary_type": east_bt, "south_boundary_type": south_bt,
    "north_boundary_type": north_bt, "significant_wave_height": Hmo,
}
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
print("config.json = exact 22-field minimal (no incident_wave_type). Files:",
      sorted(p.name for p in OUT.glob("*")))
