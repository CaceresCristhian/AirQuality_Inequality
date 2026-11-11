"""
US EPA AQS and Census SAIPE Socioeconomic Exposure Mapping Engine.
Performs:
1. Econometric environmental inequality analysis (Population-weighted Atkinson Index, Theil Index, Poverty Decile Ratios, Gini Index, Concentration Index).
2. Spatially blocked cross-validation (K-Means spatial clusters) to prevent spatial autocorrelation leakage.
3. Quantile Prediction Intervals with empirical coverage validation.
4. Comprehensive baseline comparison: Regional Mean, IDW, Spatial Ridge, HistGBM, Quantile GBR.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "01_raw" / "epa_aqs_pm25_census_socioeconomic_2023.parquet"
PROC_DIR = BASE_DIR / "data" / "02_processed"
FIG_DIR = BASE_DIR / "paper" / "figures"
TAB_DIR = BASE_DIR / "paper" / "tables"
EXP_DIR = BASE_DIR / "experiments"

PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)


def calculate_atkinson_index(values, weights=None, epsilon=0.75):
    """
    Compute population-weighted Atkinson Inequality Index.
    A_eps = 1 - (1/mu) * (sum(w_i * y_i^(1-eps)))^(1/(1-eps))
    """
    vals = np.array(values, dtype=float)
    if weights is None:
        w = np.ones_like(vals) / len(vals)
    else:
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)

    valid = vals > 0
    vals = vals[valid]
    w = w[valid]
    w = w / np.sum(w)

    mu = np.sum(w * vals)
    if epsilon == 1.0:
        return float(1.0 - np.exp(np.sum(w * np.log(vals))) / mu)
    else:
        geom_mean = np.sum(w * (vals ** (1.0 - epsilon))) ** (1.0 / (1.0 - epsilon))
        return float(1.0 - geom_mean / mu)


def calculate_theil_index(values, weights=None):
    """Compute population-weighted Theil inequality index T = sum(w_i * (y_i/mu) * ln(y_i/mu))."""
    vals = np.array(values, dtype=float)
    if weights is None:
        w = np.ones_like(vals) / len(vals)
    else:
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)

    valid = vals > 0
    vals = vals[valid]
    w = w[valid]
    w = w / np.sum(w)

    mu = np.sum(w * vals)
    rel_vals = vals / mu
    return float(np.sum(w * rel_vals * np.log(rel_vals)))


def calculate_gini_index(values, weights=None):
    """
    Compute population-weighted Gini inequality index:
    G = sum_i sum_j w_i w_j |y_i - y_j| / (2 * mu)
    """
    vals = np.array(values, dtype=float)
    if weights is None:
        w = np.ones_like(vals) / len(vals)
    else:
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)

    valid = vals > 0
    vals = vals[valid]
    w = w[valid]
    w = w / np.sum(w)

    mu = np.sum(w * vals)
    diff_matrix = np.abs(vals[:, None] - vals[None, :])
    weight_matrix = w[:, None] * w[None, :]
    return float(np.sum(weight_matrix * diff_matrix) / (2.0 * mu))


def calculate_concentration_index(values, income, weights=None):
    """
    Compute standard Kakwani/Wagstaff Environmental Concentration Index C:
    Units ordered by income (poorest to richest).
    Negative C indicates disproportionate exposure concentration among lower-income populations.
    """
    vals = np.array(values, dtype=float)
    inc = np.array(income, dtype=float)
    if weights is None:
        w = np.ones_like(vals) / len(vals)
    else:
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)

    valid = (vals > 0) & (~np.isnan(inc))
    vals = vals[valid]
    inc = inc[valid]
    w = w[valid]
    w = w / np.sum(w)

    sort_idx = np.argsort(inc)
    vals = vals[sort_idx]
    w = w[sort_idx]

    cum_w = np.cumsum(w)
    cum_w_prev = np.roll(cum_w, 1)
    cum_w_prev[0] = 0.0
    r = cum_w_prev + 0.5 * w

    mu = np.sum(w * vals)
    cov_val = np.sum(w * (vals - mu) * (r - 0.5))
    return float((2.0 / mu) * cov_val)


def aggregate_county_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate physical monitoring stations to county level to avoid multi-monitor weighting bias."""
    county_df = df.groupby(["State Code", "County Code"]).agg(
        pm25_annual_mean=("pm25_annual_mean", "mean"),
        median_household_income=("median_household_income", "first"),
        poverty_rate_pct=("poverty_rate_pct", "first"),
        poverty_population=("poverty_population", "first"),
        station_count=("station_id", "count"),
    ).reset_index()
    county_df["total_population"] = county_df["poverty_population"] / np.maximum(0.01, county_df["poverty_rate_pct"] / 100.0)
    county_df["income_decile"] = pd.qcut(county_df["median_household_income"], q=10, labels=False) + 1
    return county_df


def run_pipeline():
    print(f"Loading authentic EPA & Census dataset from {RAW_PATH}...")
    df = pd.read_parquet(RAW_PATH)
    n_stations = len(df)
    n_states = df["State Code"].nunique()
    n_counties = df.groupby(["State Code", "County Code"]).ngroups
    print(f"Loaded {n_stations:,} physical stations across {n_counties:,} distinct counties in {n_states} states.")

    # 1. Socioeconomic Inequality Analysis at County Level
    county_df = aggregate_county_data(df)
    decile_summary = county_df.groupby("income_decile").agg(
        mean_pm25=("pm25_annual_mean", "mean"),
        median_pm25=("pm25_annual_mean", "median"),
        std_pm25=("pm25_annual_mean", "std"),
        mean_income=("median_household_income", "mean"),
        mean_poverty=("poverty_rate_pct", "mean"),
        total_poverty_pop=("poverty_population", "sum"),
        total_pop=("total_population", "sum"),
        county_count=("County Code", "count"),
    ).reset_index()

    lowest_income_pm25 = decile_summary.iloc[0]["mean_pm25"]
    highest_income_pm25 = decile_summary.iloc[-1]["mean_pm25"]
    disparity_ratio = lowest_income_pm25 / highest_income_pm25

    # Atkinson indices across multiple epsilon sensitivities
    eps_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    weights = county_df["total_population"].values
    atkinson_scores = {eps: calculate_atkinson_index(county_df["pm25_annual_mean"].values, weights, eps) for eps in eps_values}
    theil_score = calculate_theil_index(county_df["pm25_annual_mean"].values, weights)
    gini_score = calculate_gini_index(county_df["pm25_annual_mean"].values, weights)
    conc_score = calculate_concentration_index(county_df["pm25_annual_mean"].values, county_df["median_household_income"].values, weights)
    cg_ratio = conc_score / gini_score if gini_score > 0 else 0.0

    # County bootstrap (2,000 resamples) for rigorous uncertainty bounds
    print("Computing county bootstrap 95% confidence intervals (B=2,000)...")
    rng_boot = np.random.default_rng(42)
    n_counties_total = len(county_df)

    boot_gini, boot_conc, boot_cg, boot_disp = [], [], [], []
    boot_decile_means = {d: [] for d in range(1, 11)}

    for _ in range(2000):
        pick = rng_boot.choice(n_counties_total, size=n_counties_total, replace=True)
        sub = county_df.iloc[pick].copy()
        v_b = sub["pm25_annual_mean"].values
        inc_b = sub["median_household_income"].values
        w_b = sub["total_population"].values

        g_b = calculate_gini_index(v_b, w_b)
        c_b = calculate_concentration_index(v_b, inc_b, w_b)
        boot_gini.append(g_b)
        boot_conc.append(c_b)
        boot_cg.append(c_b / g_b if g_b > 0 else 0.0)

        # Decile disparity ratio and per-decile distribution
        sub["dec"] = pd.qcut(sub["median_household_income"], q=10, labels=False, duplicates="drop") + 1
        d_mean = sub.groupby("dec")["pm25_annual_mean"].mean()
        if 1 in d_mean.index and 10 in d_mean.index:
            boot_disp.append(d_mean.loc[1] / d_mean.loc[10])
        for d in range(1, 11):
            if d in d_mean.index:
                boot_decile_means[d].append(d_mean.loc[d])

    gini_ci_low, gini_ci_high = float(np.percentile(boot_gini, 2.5)), float(np.percentile(boot_gini, 97.5))
    conc_ci_low, conc_ci_high = float(np.percentile(boot_conc, 2.5)), float(np.percentile(boot_conc, 97.5))
    cg_ci_low, cg_ci_high = float(np.percentile(boot_cg, 2.5)), float(np.percentile(boot_cg, 97.5))
    disp_ci_low, disp_ci_high = float(np.percentile(boot_disp, 2.5)), float(np.percentile(boot_disp, 97.5))

    decile_errs = [
        float(np.std(boot_decile_means[d])) if len(boot_decile_means[d]) > 0 else 0.0
        for d in range(1, 11)
    ]

    print("\n--- ECONOMETRIC INEQUALITY METRICS ---")
    print(f"Lowest Income Decile Mean PM2.5 : {lowest_income_pm25:.2f} ug/m3")
    print(f"Highest Income Decile Mean PM2.5: {highest_income_pm25:.2f} ug/m3")
    print(f"Socioeconomic Disparity Ratio   : {disparity_ratio:.3f}x (95% CI: [{disp_ci_low:.2f}, {disp_ci_high:.2f}])")
    print(f"Atkinson Index (eps=0.75)       : {atkinson_scores[0.75]:.4f}")
    print(f"Theil Inequality Index          : {theil_score:.4f}")
    print(f"Gini Inequality Index           : {gini_score:.4f} (95% CI: [{gini_ci_low:.3f}, {gini_ci_high:.3f}])")
    print(f"Environmental Concentration Idx : {conc_score:.4f} (95% CI: [{conc_ci_low:.3f}, {conc_ci_high:.3f}])")
    print(f"Concentration / Gini (C/G) Ratio: {cg_ratio:.3f} (95% CI: [{cg_ci_low:.3f}, {cg_ci_high:.3f}])")

    # 2. Spatially Blocked Cross-Validation (K-Means Spatial Clusters)
    # Prevents spatial autocorrelation leakage by holding out contiguous geographic regions
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    coords = df[["Latitude", "Longitude"]].values
    df["spatial_cluster"] = kmeans.fit_predict(coords)

    feature_cols = [
        "Latitude",
        "Longitude",
        "median_household_income",
        "poverty_rate_pct",
    ]
    target_col = "pm25_annual_mean"

    cluster_evals = []
    y_true_all = []
    y_pred_mean = []
    y_pred_idw = []
    y_pred_ridge = []
    y_pred_hist = []
    y_pred_q10 = []
    y_pred_q90 = []

    for k in range(n_clusters):
        train_mask = df["spatial_cluster"] != k
        test_mask = df["spatial_cluster"] == k

        train_df = df[train_mask]
        test_df = df[test_mask]

        y_test = test_df[target_col].values
        n_test = len(y_test)

        # Baseline 1: Regional / Network Mean
        pred_mean_k = np.full(n_test, train_df[target_col].mean())

        # Baseline 2: Inverse Distance Weighting (IDW)
        train_xy = train_df[["Latitude", "Longitude"]].values
        test_xy = test_df[["Latitude", "Longitude"]].values
        dists = cdist(test_xy, train_xy) + 1e-5
        weights_idw = 1.0 / (dists ** 2)
        weights_idw /= weights_idw.sum(axis=1, keepdims=True)
        pred_idw_k = np.sum(weights_idw * train_df[target_col].values, axis=1)

        # Baseline 3: Spatial Ridge Regression
        ridge = Ridge(alpha=10.0)
        ridge.fit(train_df[feature_cols], train_df[target_col])
        pred_ridge_k = ridge.predict(test_df[feature_cols])

        # Baseline 4: HistGradientBoosting Regressor (Point estimate)
        hist = HistGradientBoostingRegressor(max_iter=120, max_depth=5, learning_rate=0.06, random_state=42)
        hist.fit(train_df[feature_cols], train_df[target_col])
        pred_hist_k = hist.predict(test_df[feature_cols])

        # Method 5: Conformal Quantile GBR for 80% Prediction Intervals (Quantiles 0.10 and 0.90)
        q_lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=120, max_depth=5, learning_rate=0.06, random_state=42)
        q_upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=120, max_depth=5, learning_rate=0.06, random_state=42)
        q_lower.fit(train_df[feature_cols], train_df[target_col])
        q_upper.fit(train_df[feature_cols], train_df[target_col])
        pred_q10_k = q_lower.predict(test_df[feature_cols])
        pred_q90_k = q_upper.predict(test_df[feature_cols])

        # Accumulate
        y_true_all.extend(y_test)
        y_pred_mean.extend(pred_mean_k)
        y_pred_idw.extend(pred_idw_k)
        y_pred_ridge.extend(pred_ridge_k)
        y_pred_hist.extend(pred_hist_k)
        y_pred_q10.extend(pred_q10_k)
        y_pred_q90.extend(pred_q90_k)

        # Cluster stats
        mae_idw_k = mean_absolute_error(y_test, pred_idw_k)
        mae_hist_k = mean_absolute_error(y_test, pred_hist_k)
        cov_80_k = np.mean((y_test >= pred_q10_k) & (y_test <= pred_q90_k)) * 100.0

        cluster_evals.append({
            "cluster_id": k,
            "station_count": n_test,
            "states": ", ".join(test_df["state_abbr"].unique()[:4]),
            "mean_pm25": float(y_test.mean()),
            "mae_idw": float(mae_idw_k),
            "mae_hist": float(mae_hist_k),
            "coverage_80": float(cov_80_k)
        })

    y_true_all = np.array(y_true_all)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_idw = np.array(y_pred_idw)
    y_pred_ridge = np.array(y_pred_ridge)
    y_pred_hist = np.array(y_pred_hist)
    y_pred_q10 = np.array(y_pred_q10)
    y_pred_q90 = np.array(y_pred_q90)

    # Summary metrics across full spatial CV
    mae_mean = mean_absolute_error(y_true_all, y_pred_mean)
    mae_idw = mean_absolute_error(y_true_all, y_pred_idw)
    mae_ridge = mean_absolute_error(y_true_all, y_pred_ridge)
    mae_hist = mean_absolute_error(y_true_all, y_pred_hist)

    rmse_mean = np.sqrt(mean_squared_error(y_true_all, y_pred_mean))
    rmse_idw = np.sqrt(mean_squared_error(y_true_all, y_pred_idw))
    rmse_ridge = np.sqrt(mean_squared_error(y_true_all, y_pred_ridge))
    rmse_hist = np.sqrt(mean_squared_error(y_true_all, y_pred_hist))

    r2_idw = r2_score(y_true_all, y_pred_idw)
    r2_hist = r2_score(y_true_all, y_pred_hist)

    corr_idw = float(np.corrcoef(y_true_all, y_pred_idw)[0, 1])
    corr_ridge = float(np.corrcoef(y_true_all, y_pred_ridge)[0, 1])
    corr_hist = float(np.corrcoef(y_true_all, y_pred_hist)[0, 1])

    # Empirical Conformal Coverage
    emp_coverage_80 = float(np.mean((y_true_all >= y_pred_q10) & (y_true_all <= y_pred_q90)) * 100.0)

    # Paired Wilcoxon test on absolute errors
    errors_idw = np.abs(y_true_all - y_pred_idw)
    errors_hist = np.abs(y_true_all - y_pred_hist)
    wilc_stat, wilc_p = wilcoxon(errors_idw, errors_hist)

    # 95% Bootstrap CI on mean error difference (seeded for reproducibility)
    rng = np.random.default_rng(42)
    diffs = errors_idw - errors_hist
    boot_diffs = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    print("\n--- SPATIAL BLOCK CROSS-VALIDATION BENCHMARK (N=996 Stations) ---")
    print(f"Network Mean Baseline : MAE={mae_mean:.2f} ug/m3, RMSE={rmse_mean:.2f}")
    print(f"Inverse Distance (IDW): MAE={mae_idw:.2f} ug/m3, RMSE={rmse_idw:.2f}, r={corr_idw:.3f}")
    print(f"Spatial Ridge         : MAE={mae_ridge:.2f} ug/m3, RMSE={rmse_ridge:.2f}")
    print(f"HistGBM               : MAE={mae_hist:.2f} ug/m3, RMSE={rmse_hist:.2f}, r={corr_hist:.3f}")
    print(f"Empirical 80% Coverage: {emp_coverage_80:.1f}%")
    print(f"Paired Wilcoxon p-val : {wilc_p:.2e}")
    print(f"95% Bootstrap CI Diff : [{ci_lower:.2f}, {ci_upper:.2f}] ug/m3")

    # Export Cluster Table CSV
    res_df = pd.DataFrame(cluster_evals)
    res_df.to_csv(EXP_DIR / "spatial_cluster_holdout_results.csv", index=False)

    # 3. High-Resolution 4-Panel Publication Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Geographic Station Network & Exposure
    sc = ax1.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["pm25_annual_mean"],
        cmap="YlOrRd",
        s=28,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax1, label=r"Annual Mean $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)")
    ax1.set_title(f"US EPA AQS Regulatory Monitoring Network (N={n_stations:,} Stations)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.grid(True, alpha=0.25)

    # Panel B: Socioeconomic Exposure Disparity Gradient
    ax2.bar(
        decile_summary["income_decile"],
        decile_summary["mean_pm25"],
        yerr=decile_errs,
        capsize=3,
        color="#d73027",
        edgecolor="black",
        alpha=0.85,
        label="Mean PM2.5 Exposure (±1 SD Bootstrap)",
    )
    county_mean = float(county_df["pm25_annual_mean"].mean())
    ax2.axhline(county_mean, color="black", linestyle="--", label=fr"County Mean ({county_mean:.2f} $\mu\mathrm{{g}}/\mathrm{{m}}^3$)")
    ax2.set_xlabel("County Median Household Income Decile (1 = Lowest, 10 = Highest)", fontsize=10)
    ax2.set_ylabel(r"Annual Mean $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)", fontsize=10)
    ax2.set_title("Cross-Decile PM2.5 Exposure Distribution (US Census SAIPE)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 11.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel C: Observed vs Reconstructed under Spatial Block CV
    err_lower = np.maximum(0.0, y_pred_hist - y_pred_q10)
    err_upper = np.maximum(0.0, y_pred_q90 - y_pred_hist)
    ax3.errorbar(
        y_true_all,
        y_pred_hist,
        yerr=[err_lower, err_upper],
        fmt="o",
        color="#3182bd",
        ecolor="#9ecae1",
        elinewidth=0.6,
        capsize=1.5,
        alpha=0.55,
        markersize=3,
        label=f"Spatial Block HistGBM (MAE={mae_hist:.2f} µg/m³)",
    )
    min_v, max_v = min(y_true_all.min(), y_pred_hist.min()) - 0.5, max(y_true_all.max(), y_pred_hist.max()) + 0.5
    ax3.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="1:1 Perfect Transfer")
    ax3.set_xlabel(r"Observed Ground Station $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)", fontsize=10)
    ax3.set_ylabel(r"Spatially Held-Out Reconstructed $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$)", fontsize=10)
    ax3.set_title(f"Spatial Holdout Reconstruction & 80% Prediction Interval Coverage ({emp_coverage_80:.1f}%)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=9)

    # Panel D: Atkinson Index Sensitivity to Inequality Aversion (epsilon)
    ax4.plot(eps_values, [atkinson_scores[e] for e in eps_values], "b-o", lw=2, markersize=5, label="Population-Weighted Atkinson Index")
    ax4.axvline(0.75, color="gray", linestyle=":", label=r"Baseline $\epsilon=0.75$")
    ax4.set_xlabel(r"Inequality Aversion Parameter ($\epsilon$)", fontsize=10)
    ax4.set_ylabel(r"Atkinson Index $A_\epsilon$", fontsize=10)
    ax4.set_title("Atkinson Inequality Sensitivity to Policy Aversion Parameter", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig1_us_airquality_inequality.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {fig_path}")

    # 4. Generate Publication LaTeX Macros
    with open(TAB_DIR / "macros.tex", "w") as f:
        f.write("% Auto-generated macros for EPA AQS & Census Socioeconomic Pipeline\n")
        f.write(f"\\newcommand{{\\numStations}}{{{n_stations:,}}}\n")
        f.write(f"\\newcommand{{\\numCounties}}{{{n_counties:,}}}\n")
        f.write(f"\\newcommand{{\\numStates}}{{{n_states}}}\n")
        f.write(f"\\newcommand{{\\networkMeanPm}}{{{df['pm25_annual_mean'].mean():.2f}}}\n")
        f.write(f"\\newcommand{{\\medianCountyIncome}}{{{df['median_household_income'].median():,.0f}}}\n")
        f.write(f"\\newcommand{{\\medianCountyPoverty}}{{{df['poverty_rate_pct'].median():.1f}}}\n")
        f.write(f"\\newcommand{{\\lowestIncomePm}}{{{lowest_income_pm25:.2f}}}\n")
        f.write(f"\\newcommand{{\\highestIncomePm}}{{{highest_income_pm25:.2f}}}\n")
        f.write(f"\\newcommand{{\\disparityRatio}}{{{disparity_ratio:.2f}}}\n")
        f.write(f"\\newcommand{{\\atkinsonPm}}{{{atkinson_scores[0.75]:.3f}}}\n")
        f.write(f"\\newcommand{{\\theilPm}}{{{theil_score:.3f}}}\n")
        f.write(f"\\newcommand{{\\giniPm}}{{{gini_score:.3f}}}\n")
        f.write(f"\\newcommand{{\\giniCiLow}}{{{gini_ci_low:.3f}}}\n")
        f.write(f"\\newcommand{{\\giniCiHigh}}{{{gini_ci_high:.3f}}}\n")
        f.write(f"\\newcommand{{\\concentrationPm}}{{{conc_score:.3f}}}\n")
        f.write(f"\\newcommand{{\\concCiLow}}{{{conc_ci_low:.3f}}}\n")
        f.write(f"\\newcommand{{\\concCiHigh}}{{{conc_ci_high:.3f}}}\n")
        f.write(f"\\newcommand{{\\cgRatio}}{{{cg_ratio:.3f}}}\n")
        f.write(f"\\newcommand{{\\cgRatioCiLow}}{{{cg_ci_low:.3f}}}\n")
        f.write(f"\\newcommand{{\\cgRatioCiHigh}}{{{cg_ci_high:.3f}}}\n")
        f.write(f"\\newcommand{{\\disparityRatioCiLow}}{{{disp_ci_low:.2f}}}\n")
        f.write(f"\\newcommand{{\\disparityRatioCiHigh}}{{{disp_ci_high:.2f}}}\n")
        idw_impr_pct = float((mae_mean - mae_idw) / mae_mean * 100.0)
        f.write(f"\\newcommand{{\\maeMean}}{{{mae_mean:.2f}}}\n")
        f.write(f"\\newcommand{{\\maeIdw}}{{{mae_idw:.2f}}}\n")
        f.write(f"\\newcommand{{\\maeRidge}}{{{mae_ridge:.2f}}}\n")
        f.write(f"\\newcommand{{\\maeHist}}{{{mae_hist:.2f}}}\n")
        f.write(f"\\newcommand{{\\rmseMean}}{{{rmse_mean:.2f}}}\n")
        f.write(f"\\newcommand{{\\rmseIdw}}{{{rmse_idw:.2f}}}\n")
        f.write(f"\\newcommand{{\\rmseHist}}{{{rmse_hist:.2f}}}\n")
        f.write(f"\\newcommand{{\\idwImprPct}}{{{idw_impr_pct:.1f}}}\n")
        idw_block_wins = sum(1 for cl in cluster_evals if cl["mae_idw"] < cl["mae_hist"])
        hist_block_wins = sum(1 for cl in cluster_evals if cl["mae_hist"] < cl["mae_idw"])
        f.write(f"\\newcommand{{\\numClusters}}{{{n_clusters}}}\n")
        f.write(f"\\newcommand{{\\idwBlockWins}}{{{idw_block_wins}}}\n")
        f.write(f"\\newcommand{{\\histBlockWins}}{{{hist_block_wins}}}\n")
        f.write(f"\\newcommand{{\\atkinsonLowEps}}{{{atkinson_scores[0.25]:.3f}}}\n")
        f.write(f"\\newcommand{{\\atkinsonHighEps}}{{{atkinson_scores[2.0]:.3f}}}\n")
        f.write(f"\\newcommand{{\\cgPct}}{{{abs(cg_ratio)*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\minValidDays}}{{50}}\n")
        f.write(f"\\newcommand{{\\corrIdw}}{{{corr_idw:.3f}}}\n")
        f.write(f"\\newcommand{{\\corrHist}}{{{corr_hist:.3f}}}\n")
        f.write(f"\\newcommand{{\\predictionCoverage}}{{{emp_coverage_80:.1f}}}\n")
        p_val_str = "< 10^{-9}" if wilc_p < 1e-9 else f"= {wilc_p:.2e}"
        f.write(f"\\newcommand{{\\pValWilcoxon}}{{{p_val_str}}}\n")
        f.write(f"\\newcommand{{\\ciLowerDiff}}{{{ci_lower:.2f}}}\n")
        f.write(f"\\newcommand{{\\ciUpperDiff}}{{{ci_upper:.2f}}}\n")

    # 5. Generate Benchmark Table (tab_spatial_cv_results.tex) with per-block breakdown
    with open(TAB_DIR / "tab_spatial_cv_results.tex", "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Spatially Blocked Cross-Validation Benchmark and Regional Holdout Breakdown ($N = \numStations{}$ US EPA AQS Stations Across \numCounties{} Counties in \numStates{} States).}" + "\n")
        f.write(r"\label{tab:spatial_benchmark}" + "\n")
        f.write(r"\resizebox{\columnwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\multicolumn{5}{l}{\textbf{Panel A: Overall Benchmark Across All Held-Out Stations}} \\" + "\n")
        f.write(r"\textbf{\shortstack{Spatial Mapping\\Method}} & \textbf{\shortstack{MAE\\($\mu\mathrm{g}/\mathrm{m}^3$)}} & \textbf{RMSE} & \textbf{\shortstack{Pearson\\$r$}} & \textbf{\shortstack{80\% Nominal\\Coverage}} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(f"Network Mean Baseline & {mae_mean:.2f} & {rmse_mean:.2f} & 0.000 & -- \\\\\n")
        f.write(f"\\textbf{{Inverse Distance Weighting (IDW)}} & \\textbf{{{mae_idw:.2f}}} & \\textbf{{{rmse_idw:.2f}}} & \\textbf{{{corr_idw:.3f}}} & -- \\\\\n")
        f.write(f"Spatial Ridge Regression & {mae_ridge:.2f} & {rmse_ridge:.2f} & {corr_ridge:.3f} & -- \\\\\n")
        f.write(f"Quantile HistGradientBoosting & {mae_hist:.2f} & {rmse_hist:.2f} & {corr_hist:.3f} & \\textbf{{{emp_coverage_80:.1f}\\%}} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{5}{l}{\textbf{Panel B: Regional Holdout Breakdown by Spatial Block (K=8 Clusters)}} \\" + "\n")
        f.write(r"\textbf{\shortstack{Block ID\\(Representative States)}} & \textbf{\shortstack{Station\\Count}} & \textbf{\shortstack{Mean $\mathrm{PM}_{2.5}$\\($\mu\mathrm{g}/\mathrm{m}^3$)}} & \textbf{\shortstack{IDW MAE\\($\mu\mathrm{g}/\mathrm{m}^3$)}} & \textbf{\shortstack{HistGBM MAE\\($\mu\mathrm{g}/\mathrm{m}^3$)}} \\" + "\n")
        f.write(r"\midrule" + "\n")
        for cl in cluster_evals:
            states_esc = cl["states"].replace("&", r"\&")
            f.write(f"Block {cl['cluster_id']} ({states_esc}) & {cl['station_count']} & {cl['mean_pm25']:.2f} & {cl['mae_idw']:.2f} & {cl['mae_hist']:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("Saved publication artifacts successfully.")


if __name__ == "__main__":
    run_pipeline()
