# Reproducibility Log: Air Quality Inequality Mapping Benchmark

**Manuscript Title**: Quantifying Socioeconomic Air Quality Disparities and Spatial Generalization Limits across Regulatory In-Situ Networks  
**Authors**: Cristhian David Cáceres Mateus, Talha Ali Khan  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  
**Target Year**: 2023  
**Log Updated**: 22 October 2026  

---

## 1. Reproduction Steps

To execute the econometric inequality calculation, spatially blocked cross-validation, and artifact generation:

```bash
# 1. Environment requirements
pip install -r requirements.txt

# 2. Run the unit test suite (9 tests passing)
pytest -v tests/test_pipeline.py

# 3. Execute the analytical pipeline
python src/inequality_mapping_engine.py

# 4. Compile the manuscript PDF
tectonic paper/main.tex --outdir paper/
```

---

## 2. Data Sources and Provenance

- **Regulatory Ground Monitoring**: US EPA Air Quality System (AQS), parameter code 88101 (PM$_{2.5}$ FRM/FEM mass), annual summary observations for calendar year 2023.
- **Completeness Criteria**: Active monitoring stations with $\ge 50$ valid operational sampling days in 2023.
- **Socioeconomic Indicators**: US Census Bureau Small Area Income and Poverty Estimates (SAIPE) county-level release for 2022 (`est22all.txt`), joined via FIPS state and county identifiers.
- **Population Weighting**: Total county population approximated from SAIPE poverty counts and poverty rate percentages.
- **Cohort Volume**: 996 physical monitoring stations across 623 distinct counties in 50 states.

---

## 3. Methodological and Evaluation Protocols

1. **Econometric Inequality Modeling**:
   - Atkinson Inequality Index $A_\epsilon$ evaluated across parameter range $\epsilon \in [0.25, 2.0]$.
   - Theil Generalized Entropy Index $T$.
   - Gini Inequality Index $G$ and Kakwani/Wagstaff Environmental Concentration Index $C$ relative to county household income.
   - Uncertainty estimation: 2,000-replicate county-clustered bootstrap generating 95% confidence intervals for $G$, $C$, $C/G$, and the decile disparity ratio.
2. **Spatially Blocked Cross-Validation**:
   - $K$-means clustering partitions all 996 monitoring stations into 8 contiguous geographic regional blocks.
   - 8-fold cross-validation evaluates reconstruction error when transferring to completely unmonitored spatial blocks.
   - Statistical comparisons: paired Wilcoxon signed-rank test and 1,000-replicate bootstrap confidence intervals on error differences.

---

## 4. Current Status and Open Items

- **Pipeline Execution**: `src/inequality_mapping_engine.py` generates `paper/tables/macros.tex`, `paper/tables/tab_spatial_cv_results.tex`, and `paper/figures/fig1_us_airquality_inequality.png`.
- **Test Suite**: 9 unit tests passing (`pytest -v tests/test_pipeline.py`), covering index properties, fold disjointness, dynamic macro consistency, citation integrity, and source secrets scanning.
- **Prose Quality**: 0 errors, 0 warnings from `vale.exe paper/main.tex`.
- **LaTeX Compilation**: Compiles via `tectonic.exe` with 0 fatal errors.
- **Open Items for Future Work**:
  - Incorporating multi-year longitudinal monitor panels (2018–2023) to assess persistence across non-wildfire years.
  - Sub-county exposure mapping via Census tract or block-group spatial disaggregation.
  - Gaussian Process / Kriging with Matérn kernel fitting as an additional geostatistical baseline.
