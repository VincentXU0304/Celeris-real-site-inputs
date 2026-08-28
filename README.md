# celeris-real-site-inputs

Build a complete, real [Celeris](https://github.com/plynett/plynett.github.io) input set (bathymetry, spectral wavemaker, config) for an actual coastal site from public data. No fabricated depths, no synthetic sea states.

Celeris is a phase-resolving Boussinesq nearshore wave model. Running it on a real site means producing three things in exactly the format it expects: an ASCII bathymetry grid, a `waves.txt` list of spectral components, and a `config.json`. This repository does that for Ocean Beach, San Francisco, driven by NOAA CUDEM topobathymetry and Open-Meteo Marine sea states.

It was written for an MSc thesis (2026) on training-free coastal video generation, where the Celeris free-surface field is used as a structure-control signal for a frozen video generator. The bathymetry and wave front end is reusable on its own, so it is released separately here.

## What it produces

| Output | Contents |
|---|---|
| `bathy.txt` | ASCII elevation grid on the simulation mesh, negative underwater, positive on land |
| `waves.txt` | Spectral components as `amplitude period direction_rad phase`, one per line, with the count in a `[NumberOfWaves]` header |
| `config.json` | The minimal 22 fields Celeris reads: grid, Courant number, solver mode, base depth, friction, breaking parameters, boundary types |
| `overlay.jpg` | Optional Sentinel-2 RGB backdrop for the domain (`make_overlay.py`) |

## Data sources

All public, none require an account or API key.

- **Bathymetry**: NOAA CUDEM 1/9 arc-second topobathymetry, read directly over HTTP with GDAL `/vsicurl`, no download step.
- **Wave forcing**: Open-Meteo Marine hindcast significant wave height, peak period and direction. Four real Ocean Beach events ship as presets, spanning roughly a fourfold range in wave height.
- **Satellite backdrop**: Sentinel-2 L2A from the public AWS Earth Search STAC.

## The wave spectrum, stated precisely

JONSWAP with gamma 3.3, multiplied by a frequency-dependent cos^2s directional distribution over nine directions at five degree spacing about the peak. Total energy is then renormalised to the target Hmo, components below one percent of the peak-direction maximum are truncated, and each remaining component's direction is snapped so that an integer number of crests fits along the boundary. That last step is required because the along-shore boundaries are periodic.

**No TMA finite-depth factor is applied.** The Kitaigorodskii depth function is not multiplied into the spectrum here. Depth effects enter through the Boussinesq solver running over the real bathymetry, as shoaling, refraction and depth-limited breaking, rather than through a pre-corrected input spectrum. If you want a TMA input spectrum instead, multiply `E[i]` by phi(omega_h) before the renormalisation step, keeping in mind that renormalising to a target Hmo afterwards preserves total variance and changes only the spectral shape.

## Usage

```bash
pip install numpy opencv-python requests rasterio
python build_oceanbeach_celeris.py 2023-01-05_flood
```

The event name is optional and selects one of the four presets defined at the top of the script. Output lands in `./data/celeris_oceanbeach_<event>/`. Point Celeris at that folder.

To fetch the satellite backdrop for a domain:

```bash
python make_overlay.py data/celeris_oceanbeach_2023-01-05_flood
```

## Known issues

- `make_overlay.py` uses a wider bounding box than `build_oceanbeach_celeris.py`. Its own comment says the two must match, and they currently do not. Set them equal before relying on the overlay for spatial alignment.
- The site is Ocean Beach specific: the bounding box, the CUDEM tile and the destination CRS are constants near the top of the script. Moving to another site means editing those three values and checking that the offshore boundary still lands on the intended edge.

## Downstream use in the thesis

Included for context, not required by this repository. The Celeris grayscale free-surface elevation video was used as a structure control for a frozen video generator: Wan2.2 TI2V-5B with the `alibaba-pai/Wan2.2-Fun-5B-Control` control model, 45 frames at 832x480, on a single 24 GB consumer GPU.

One practical note that cost some time: use the **UniPC** sampler (`uni_pc` in ComfyUI). The stochastic DPM++ SDE sampler is a poor fit for flow-matching Wan models and produced oversaturated artefacts.

## Credit

Celeris was developed by Sasan Tavakkol, building on an earlier demo project by Stephen Thompson, and advised by Patrick Lynett at the University of Southern California.

The input-generation logic here is a Python port of the official Celeris WebGPU implementation, which is MIT licensed, Copyright (c) 2023 plynett. Its full licence text is kept in [`THIRD_PARTY_LICENSES/celeris-webgpu-MIT.txt`](THIRD_PARTY_LICENSES/celeris-webgpu-MIT.txt).

## Citation

If you use Celeris itself, cite the solver:

> Tavakkol, S. and Lynett, P. (2017) 'Celeris: a GPU-accelerated open source software with a Boussinesq-type wave solver for real-time, interactive simulation and visualization', *Computer Physics Communications*, 217, pp. 117-127. doi:10.1016/j.cpc.2017.03.002

The spectral irregular wavemaker that this repository ports is described in a different paper, which is easy to miss:

> Tavakkol, S. and Lynett, P. (2017) 'Opportunities for interactive, physics-driven wave simulation using the Boussinesq-type model, Celeris', *Coastal Engineering Proceedings*, 1(35), waves.11

For the Unity rewrite and the improved sponge layer:

> Tavakkol, S. and Lynett, P. (2020) 'Celeris Base: an interactive and immersive Boussinesq-type nearshore wave simulation software', *Computer Physics Communications*, 248, 106966

## Licence

MIT. See [LICENSE](LICENSE).
