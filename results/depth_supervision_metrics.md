# Depth Supervision Metrics — workdir_dense_312 (589 images, indoor 360)

## v2: Aligned L1 + opacity freeze (commit 28ece53)
Config: aligned L1, opacity fully frozen, no normals, cull=0.005, max_gs=500K, 69K steps

### Tensorboard (per-sample, not averaged)
| Step | Total Loss | RGB Loss | PSNR |
|---|---|---|---|
| 5K | 0.073 | 0.072 | 23.0 |
| 10K | 0.047 | 0.045 | 24.3 |
| 20K | 0.029 | 0.027 | 29.3 |
| 30K | 0.030 | 0.028 | 30.4 |
| 40K | 0.021 | 0.020 | 33.1 |
| 50K | 0.030 | 0.029 | 28.9 |
| 60K | 0.024 | 0.022 | 30.5 |
| 69K | 0.027 | 0.025 | 30.4 |

### DIAG progression
| Step | ms/step | alive/500K | scale_mean | isect |
|---|---|---|---|---|
| 5K | ~78 | 182K (36%) | 0.0140 | 6M |
| 10K | ~78 | 303K (61%) | 0.0096 | 15M |
| 20K | ~73 | 312K (62%) | 0.0090 | 14M |
| 30K | ~73 | 290K (58%) | 0.0081 | 15M |
| 40K | ~67 | 277K (55%) | 0.0074 | 15M |
| 50K | ~66 | 255K (51%) | 0.0054 | 12M |
| 60K | ~64 | 253K (51%) | 0.0051 | 12M |

## v3: Dual opacity + normals (commit f26a503)
Config: aligned L1, dual opacity (start 15K), normals (0.02), cull=0.005, max_gs=500K, 60K steps

### Tensorboard (per-sample, not averaged)
| Step | Total Loss | RGB Loss | PSNR |
|---|---|---|---|
| 5K | 0.074 | 0.073 | 22.7 |
| 10K | 0.048 | 0.047 | 24.1 |
| 15K | 0.036 | 0.035 | 27.4 |
| 20K | 0.028 | 0.026 | 29.5 |
| 30K | 0.027 | 0.026 | 31.4* |

*30K single-sample; averaged 500-sample windows below

### Averaged PSNR comparison (500 samples per window)
| Window | v2 avg PSNR | v3 avg PSNR | Delta | v2 avg Loss | v3 avg Loss |
|---|---|---|---|---|---|
| 15-20K | 27.54 (±1.83) | 27.55 (±1.87) | +0.01 | 0.0374 | 0.0373 |
| 20-25K | 28.23 (±1.78) | 28.44 (±1.77) | +0.21 | 0.0348 | 0.0339 |
| 25-30K | 29.05 (±1.75) | 29.26 (±1.78) | +0.20 | 0.0317 | 0.0307 |

### DIAG progression
| Step | ms/step | alive/500K | scale_mean | isect |
|---|---|---|---|---|
| 5K | ~103 | ~182K | 0.0133 | 5M |
| 10K | ~103 | 248K (50%) | 0.0116 | 18M |
| 15K | ~101 | 281K (56%) | 0.0100 | 14M |
| 20K | ~100 | 292K (58%) | 0.0096 | 16M |
| 25K | ~100 | 299K (60%) | 0.0092 | 20M |
| 30K | ~96 | 283K (57%) | 0.0083 | 16M |

### Notes
- v3 ~30% slower per step (103ms vs 73ms) due to normal loss extra backward pass
- Dual opacity activated at step 15K — no population collapse (alive 281K stable)
- +0.2 dB consistent improvement after 20K (validated over 500 samples)
- Normal loss effectively zero (0.0000-0.0001) — finite-diff normals too smooth
- Depth L1 dipped to 0.04 right after dual_opac activated (step 18.8K)
- Dead-zone Gaussians (0.005-0.05 opacity) persist — gradient vanishing problem

## v4: 1M gs + cull=0.01 (killed at step 6.5K)
Config: aligned L1, no dual opacity, normals 0.02, cull=0.01, max_gs=1M, killed manually
- Excessive relocation churn at 1M + cull=0.01 → isect=61M (489MB), VRAM 2.0G/5.5G
- PSNR 21.9 at 5K (worst of all runs — churn disrupting training)
- 190ms/step (3× slower than 500K runs)
- Conclusion: cull=0.01 is too aggressive at 1M Gaussians

## v5: Clean baseline — L1 + normals, no dual opacity (commit TBD)
Config: aligned L1, normals 0.02, opacity frozen, cull=0.005, max_gs=500K, 30K steps
- **No floaters** — visual quality clean, confirming dual opacity caused v3's artifacts
- PLY: 105MB, 444K Gaussians (56K filtered by nerfstudio export)
- scale_max: 1.74 (stable, no rogue explosion — v3 hit 17)
- alive: 320K (64%) at step 25K — highest of any run

### Tensorboard (per-sample)
| Step | Total Loss | RGB Loss | PSNR |
|---|---|---|---|
| 5K | 0.071 | 0.070 | 23.4 |
| 10K | 0.051 | 0.047 | 23.7 |
| 15K | 0.043 | 0.042 | 26.2 |
| 20K | 0.029 | 0.027 | 29.3 |
| 25K | 0.035 | 0.034 | 27.5 |
| 30K | — | — | — |

### Averaged comparison (500 samples/window, 20-25K)
| Run | Avg PSNR | Avg Loss | Notes |
|---|---|---|---|
| v2 (L1, 500K) | 28.23 | 0.0348 | No floaters (assumed, never checked PLY) |
| v3 (dual opac) | 28.44 | 0.0339 | +0.21 dB but floaters + rogues |
| v5 (L1+normals) | 28.14 | 0.0352 | **No floaters**, clean visuals |

### Key findings
- Dual opacity (StableGS) caused rogue Gaussians (scale_max 17) and visual floaters
- opacities_aux never learned (sigmoid saturation at logit(0.99), derivative=0.01)
- Normal loss near zero (0.0001) — finite-diff normals too smooth for meaningful signal
- Aligned L1 is the real improvement; normals and dual opacity are marginal/harmful
- 500K + cull=0.005 is the sweet spot for 24GB GPU + 589 images

### DIAG progression
| Step | ms/step | alive/500K | scale_mean | scale_max | isect |
|---|---|---|---|---|---|
| 5K | ~100 | ~180K (36%) | 0.0139 | 1.0 | 5M |
| 10K | ~100 | 248K (50%) | 0.0116 | 2.1 | 15M |
| 15K | ~97 | 280K (56%) | 0.0103 | 4.5 | 23M |
| 20K | ~95 | 304K (61%) | 0.0094 | 3.7 | 12M |
| 25K | ~90 | 320K (64%) | 0.0087 | 1.7 | 18M |

## Previous run: Pearson + 1M gs (OOM'd at step 6200)
Config: Pearson correlation, opacity frozen, cull=0.03, max_gs=1M
- OOM at step ~6200 in isect_tiles (7.85 GiB allocation)
- Pearson values erratic: -0.02 to 0.62
- VRAM: 6.7G/8.4G (vs 1.0G/2.8G in v2/v3)
- Root cause: cull_alpha_thresh=0.03 caused excessive MCMC relocation churn
