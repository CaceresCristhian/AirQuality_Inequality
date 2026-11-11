# Quantifying Socioeconomic Air Quality Disparities across Regulatory In-Situ Networks

## Overview
This repository integrates regulatory physical ground observations from the US Environmental Protection Agency (EPA) Air Quality System (AQS) with official county socioeconomic data from the US Census Bureau Small Area Income and Poverty Estimates (SAIPE).

## Data Provenance
- **EPA AQS Network**: Parameter 88101 ($\mathrm{PM}_{2.5}$ Federal Reference Method / Federal Equivalent Method mass).
  - Continuous 2023 calendar year daily physical ground monitor records (847,162 observations).
  - 996 active regulatory monitoring stations (filtered for $\ge 50$ valid operational sampling days).
  - Spans **623 distinct counties** across 50 states.
- **US Census Bureau SAIPE**: 2022 Small Area Income and Poverty Estimates (`est22all.txt`).
  - County median household income, poverty percentage, and population estimates.
  - Linked to EPA stations via State Code + County Code FIPS identifiers.

## Empirical Econometric Inequality Metrics
All confidence intervals computed via 2,000-iteration county-clustered bootstrap:
- **Lowest Household Income Decile Mean $\mathrm{PM}_{2.5}$**: $8.55\ \mu\mathrm{g}/\mathrm{m}^3$
- **Highest Household Income Decile Mean $\mathrm{PM}_{2.5}$**: $7.81\ \mu\mathrm{g}/\mathrm{m}^3$
- **Socioeconomic Disparity Ratio**: $1.09\times$ (95% CI: $[1.00, 1.21]$)
- **Population-Weighted Atkinson Inequality Index ($\epsilon = 0.75$)**: $0.017$
- **Theil Inequality Index**: $0.021$
- **Gini Inequality Index ($G$)**: $0.113$ (95% CI: $[0.101, 0.125]$)
- **Environmental Concentration Index ($C$)**: $-0.028$ (95% CI: $[-0.042, -0.013]$)
- **Relative Concentration Ratio ($C/G$)**: $-0.245$ (95% CI: $[-0.361, -0.117]$)

## Spatially Blocked Cross-Validation (8 Regional Clusters)
To evaluate spatial generalization without spatial autocorrelation leakage, the 996 stations are partitioned into 8 contiguous geographic clusters via $K$-means clustering:

### Panel A: Overall Benchmark
| Spatial Mapping Method | MAE ($\mu\mathrm{g}/\mathrm{m}^3$) | RMSE | Pearson $r$ | 80% Nominal Interval Coverage |
| :--- | :---: | :---: | :---: | :---: |
| Network Mean Baseline | 1.92 | 2.42 | 0.000 | -- |
| **Inverse Distance Weighting (IDW)** | **1.70** | **2.19** | **0.329** | -- |
| Spatial Ridge Regression | 1.84 | 2.36 | 0.231 | -- |
| Quantile HistGradientBoosting | 1.93 | 2.57 | 0.234 | **56.0%** |

### Panel B: Regional Holdout Breakdown (8 Spatial Blocks)
| Block ID (Representative States) | Station Count | Mean $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$) | IDW MAE ($\mu\mathrm{g}/\mathrm{m}^3$) | HistGBM MAE ($\mu\mathrm{g}/\mathrm{m}^3$) |
| :--- | :---: | :---: | :---: | :---: |
| Block 0 (AR, KS, LA, MS) | 114 | 9.20 | 1.14 | 1.57 |
| Block 1 (AZ, CA, ID, MT) | 191 | 7.58 | 2.18 | 2.76 |
| Block 2 (DE, DC, ME, MD) | 208 | 8.66 | 1.59 | 1.84 |
| Block 3 (AL, FL, GA, KY) | 133 | 8.76 | 1.14 | 1.10 |
| Block 4 (HI) | 14 | 3.19 | 4.91 | 6.67 |
| Block 5 (AZ, CO, ID, KS) | 123 | 6.42 | 2.17 | 1.72 |
| Block 6 (AK) | 6 | 6.97 | 2.40 | 2.56 |
| Block 7 (IL, IN, IA, KY) | 207 | 9.93 | 1.53 | 1.79 |

*Key Findings*:
1. **Distance-Decay Superiority**: In geostatistical spatial interpolation across entirely held-out unmonitored clusters, IDW provides lower error than HistGBM in 6 of 8 regional blocks (overall MAE 1.70 vs 1.93 $\mu\mathrm{g}/\mathrm{m}^3$, $p < 10^{-9}$, paired Wilcoxon test; 95% bootstrap CI $[-0.30, -0.16]$).
2. **Interval Degradation**: Quantile regression intervals exhibit substantial coverage degradation under regional domain shift (56.0% empirical coverage vs 80% nominal), reaching 0.0% coverage in isolated island domains (Hawaii).

## Execution
```bash
# Core dataset construction and spatial inequality benchmark
python src/inequality_mapping_engine.py

# Run test suite
pytest tests/test_pipeline.py

# Compile publication paper
tectonic paper/main.tex --outdir paper/
```
