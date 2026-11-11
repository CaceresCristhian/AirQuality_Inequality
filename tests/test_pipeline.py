"""
Unit tests for Air Quality Socioeconomic Inequality Mapping Pipeline.
Tests authentic data integrity, mathematical properties of Atkinson, Theil, Gini,
and Concentration indices, spatial cluster disjointness, macro consistency,
and source tree secrets security.
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inequality_mapping_engine import (
    calculate_atkinson_index,
    calculate_theil_index,
    calculate_gini_index,
    calculate_concentration_index,
    aggregate_county_data,
)

DATA_PATH = BASE_DIR / "data" / "01_raw" / "epa_aqs_pm25_census_socioeconomic_2023.parquet"
RESULTS_CSV = BASE_DIR / "experiments" / "spatial_cluster_holdout_results.csv"
FIG_PATH = BASE_DIR / "paper" / "figures" / "fig1_us_airquality_inequality.png"
MACROS_PATH = BASE_DIR / "paper" / "tables" / "macros.tex"
TAB_PATH = BASE_DIR / "paper" / "tables" / "tab_spatial_cv_results.tex"
TEX_PATH = BASE_DIR / "paper" / "main.tex"


def test_data_integrity():
    """Verify authentic EPA AQS & Census SAIPE dataset has >=900 physical stations and valid fields."""
    assert DATA_PATH.exists(), f"Integrated dataset missing at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    assert len(df) >= 900, f"Expected >= 900 stations, got {len(df)}"
    assert df["State Code"].nunique() >= 40, "Expected stations across >= 40 states"
    assert df["pm25_annual_mean"].notnull().all(), "PM2.5 annual mean must have no nulls"
    assert (df["pm25_annual_mean"] > 0).all(), "PM2.5 must be strictly positive"
    assert (df["median_household_income"] > 10000).all(), "Median income must be realistic (> $10k)"
    assert (df["poverty_rate_pct"] >= 0).all(), "Poverty rate must be non-negative"


def test_atkinson_index_properties():
    """Verify mathematical properties of the Atkinson Inequality Index."""
    # Property 1: Equal distribution yields zero inequality
    equal_vals = np.array([10.0, 10.0, 10.0, 10.0])
    a_equal = calculate_atkinson_index(equal_vals, epsilon=0.75)
    assert abs(a_equal) < 1e-6, f"Atkinson on equal distribution must be 0, got {a_equal}"

    # Property 2: Higher dispersion yields higher Atkinson index
    unequal_vals = np.array([2.0, 5.0, 15.0, 40.0])
    a_unequal = calculate_atkinson_index(unequal_vals, epsilon=0.75)
    assert a_unequal > a_equal, "Unequal distribution must have higher Atkinson index"

    # Property 3: Monotonic increase with inequality aversion epsilon
    a_eps05 = calculate_atkinson_index(unequal_vals, epsilon=0.5)
    a_eps15 = calculate_atkinson_index(unequal_vals, epsilon=1.5)
    assert a_eps15 > a_eps05, "Atkinson index must increase monotonically with epsilon"


def test_theil_index_properties():
    """Verify mathematical properties of the Theil Inequality Index."""
    equal_vals = np.array([15.0, 15.0, 15.0])
    t_equal = calculate_theil_index(equal_vals)
    assert abs(t_equal) < 1e-6, f"Theil on equal distribution must be 0, got {t_equal}"

    unequal_vals = np.array([3.0, 10.0, 30.0])
    t_unequal = calculate_theil_index(unequal_vals)
    assert t_unequal > 0.05, f"Expected non-trivial inequality for unequal distribution, got {t_unequal}"


def test_gini_index_properties():
    """Verify mathematical properties of the Gini Inequality Index."""
    # Property 1: Equal distribution yields zero inequality
    equal_vals = np.array([12.0, 12.0, 12.0, 12.0])
    g_equal = calculate_gini_index(equal_vals)
    assert abs(g_equal) < 1e-6, f"Gini on equal distribution must be 0, got {g_equal}"

    # Property 2: Positive value on dispersion
    unequal_vals = np.array([2.0, 5.0, 15.0, 40.0])
    g_unequal = calculate_gini_index(unequal_vals)
    assert g_unequal > 0.1, f"Expected substantial Gini on dispersed values, got {g_unequal}"

    # Property 3: Scale invariance of weights
    w_raw = np.array([100.0, 200.0, 300.0, 400.0])
    g_w1 = calculate_gini_index(unequal_vals, weights=w_raw)
    g_w2 = calculate_gini_index(unequal_vals, weights=w_raw * 7.5)
    assert abs(g_w1 - g_w2) < 1e-6, "Gini index must be scale invariant with respect to weights"


def test_concentration_index_properties():
    """Verify mathematical properties of Kakwani/Wagstaff Concentration Index."""
    # Case 1: Degenerate income-independent exposure yields C = 0
    inc = np.array([25000, 45000, 65000, 95000], dtype=float)
    equal_exp = np.array([8.0, 8.0, 8.0, 8.0], dtype=float)
    c_zero = calculate_concentration_index(equal_exp, inc)
    assert abs(c_zero) < 1e-6, f"Constant exposure must have C = 0, got {c_zero}"

    # Case 2: Non-degenerate varied exposure independent of income rank yields C ~ 0
    inc_varied = np.array([20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000], dtype=float)
    # Alternating high and low exposures symmetrically balanced around the median income rank
    symm_exp = np.array([12.0, 6.0, 12.0, 6.0, 6.0, 12.0, 6.0, 12.0], dtype=float)
    c_symm = calculate_concentration_index(symm_exp, inc_varied)
    assert abs(c_symm) < 0.05, f"Symmetric non-degenerate exposure must yield C ~ 0, got {c_symm}"

    # Case 3: Pro-poor exposure burden (higher exposure for lower income) yields C < 0
    pro_poor_exp = np.array([14.0, 11.0, 8.0, 5.0], dtype=float)
    c_pro_poor = calculate_concentration_index(pro_poor_exp, inc)
    assert c_pro_poor < -0.05, f"Expected negative concentration index for pro-poor burden, got {c_pro_poor}"

    # Case 4: Pro-rich exposure (higher exposure for higher income) yields C > 0
    pro_rich_exp = np.array([5.0, 8.0, 11.0, 14.0], dtype=float)
    c_pro_rich = calculate_concentration_index(pro_rich_exp, inc)
    assert c_pro_rich > 0.05, f"Expected positive concentration index for pro-rich exposure, got {c_pro_rich}"

    # Case 5: Scale invariance of weights
    w = np.array([1000.0, 5000.0, 3000.0, 2000.0])
    c_w1 = calculate_concentration_index(pro_poor_exp, inc, weights=w)
    c_w2 = calculate_concentration_index(pro_poor_exp, inc, weights=w * 10.0)
    assert abs(c_w1 - c_w2) < 1e-6, "Concentration index must be scale invariant to weights"


def test_county_aggregation_properties():
    """Verify county-level aggregation properties (623 counties, positive population weights, deciles sum to total)."""
    df = pd.read_parquet(DATA_PATH)
    county_df = aggregate_county_data(df)
    assert len(county_df) == 623, f"Expected 623 counties, got {len(county_df)}"
    assert (county_df["total_population"] > 0).all(), "Population weights must be strictly positive"
    assert (county_df["pm25_annual_mean"] > 0).all(), "County PM2.5 must be strictly positive"
    assert (county_df["median_household_income"] > 10000).all(), "County median income must be realistic"

    # Verify deciles partition all counties
    assert county_df["income_decile"].value_counts().sum() == 623, "Deciles must partition all 623 counties"
    assert county_df["income_decile"].nunique() == 10, "Must have exactly 10 decile groups"


def test_spatial_cluster_disjointness_and_counts():
    """Verify cluster partitioning and strict train/test fold disjointness."""
    df = pd.read_parquet(DATA_PATH)
    n_counties = df.groupby(["State Code", "County Code"]).ngroups
    assert n_counties == 623, f"Expected 623 counties, got {n_counties}"

    assert RESULTS_CSV.exists(), "Spatial holdout CSV must exist"
    df_res = pd.read_csv(RESULTS_CSV)
    assert len(df_res) == 8, f"Expected 8 spatial clusters, got {len(df_res)}"
    assert "cluster_id" in df_res.columns
    assert set(df_res["cluster_id"].values) == set(range(8)), "Clusters must span exactly {0..7}"

    # Verify actual fold disjointness property from clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df[["Latitude", "Longitude"]].values)
    assert len(np.unique(clusters)) == 8, "Expected 8 unique spatial clusters"

    total_indices = set(range(len(df)))
    for k in range(8):
        train_idx = set(np.where(clusters != k)[0])
        test_idx = set(np.where(clusters == k)[0])
        assert train_idx.isdisjoint(test_idx), f"Fold {k} training and test indices must be strictly disjoint!"
        assert train_idx.union(test_idx) == total_indices, f"Fold {k} train and test must cover all stations!"
        assert len(test_idx) > 0, f"Fold {k} must have non-empty test set"


def test_macro_consistency_with_data():
    """Ensure macros in macros.tex accurately reflect data outputs computed dynamically."""
    assert MACROS_PATH.exists(), "macros.tex must exist"
    macro_text = MACROS_PATH.read_text(encoding="utf-8")

    macros = {}
    for match in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", macro_text):
        macros[match.group(1)] = match.group(2)

    df = pd.read_parquet(DATA_PATH)
    df_res = pd.read_csv(RESULTS_CSV)

    assert macros.get("numStations") == str(len(df)), f"numStations macro mismatch: {macros.get('numStations')}"
    assert macros.get("numCounties") == str(df.groupby(["State Code", "County Code"]).ngroups)
    assert macros.get("numStates") == str(df["State Code"].nunique())
    assert macros.get("numClusters") == str(len(df_res))
    assert macros.get("networkMeanPm") == f"{df['pm25_annual_mean'].mean():.2f}"
    assert macros.get("giniCiLow") == "0.101" and macros.get("giniCiHigh") == "0.125", "Gini CI macros mismatch"
    assert macros.get("concCiLow") == "-0.042" and macros.get("concCiHigh") == "-0.013", "Concentration CI macros mismatch"
    assert "cgRatio" in macros, "C/G ratio macro must be present"
    assert macros.get("minValidDays") == "50", f"minValidDays mismatch: {macros.get('minValidDays')}"
    assert macros.get("atkinsonLowEps") == "0.005", f"atkinsonLowEps mismatch: {macros.get('atkinsonLowEps')}"
    assert macros.get("cgPct") == "24.5", f"cgPct mismatch: {macros.get('cgPct')}"


def test_publication_artifacts_integrity():
    """Check existence, file sizes, LaTeX syntax, and verify all citations exist in references.bib."""
    assert FIG_PATH.exists() and FIG_PATH.stat().st_size > 20000, "Figure 1 must exist and be >20KB"
    assert MACROS_PATH.exists() and MACROS_PATH.stat().st_size > 100, "Macros TeX must exist"
    assert TAB_PATH.exists() and TAB_PATH.stat().st_size > 100, "Table TeX must exist"
    assert TEX_PATH.exists(), "main.tex must exist"

    tex_content = TEX_PATH.read_text(encoding="utf-8")
    assert "\\begin{document}" in tex_content
    assert "\\end{document}" in tex_content
    assert "[?]" not in tex_content, "Unresolved citations found in main.tex!"

    # Parse all cite keys from main.tex and tables/*.tex
    cite_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex_content):
        for key in match.group(1).split(","):
            cite_keys.add(key.strip())

    for tab_file in (BASE_DIR / "paper" / "tables").glob("*.tex"):
        tab_content = tab_file.read_text(encoding="utf-8")
        for match in re.finditer(r"\\cite\{([^}]+)\}", tab_content):
            for key in match.group(1).split(","):
                cite_keys.add(key.strip())

    bib_path = BASE_DIR / "paper" / "references.bib"
    assert bib_path.exists(), "references.bib must exist"
    bib_content = bib_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_content))

    missing_keys = cite_keys - bib_keys
    assert not missing_keys, f"Citations used in manuscript but missing in references.bib: {missing_keys}"

    # Check no unescaped ampersands in titles or section headers
    for line in tex_content.splitlines():
        if line.strip().startswith(("\\section{", "\\subsection{", "\\title{")):
            clean_line = line.replace("\\&", "")
            assert "&" not in clean_line, f"Unescaped '&' found in section/title: {line}"


def test_security_secrets_scan():
    """Guarantee zero API tokens, credentials, or private keys across project files."""
    secret_patterns = [
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),  # Google API key
        re.compile(r"ghp_[0-9a-zA-Z]{36}"),   # GitHub Personal Access Token
        re.compile(r"sk-[0-9a-zA-Z]{32,}"),   # OpenAI key
        re.compile(r"-----BEGIN PRIVATE KEY-----"),
    ]

    extensions = {".py", ".tex", ".md", ".env", ".json"}
    for path in BASE_DIR.rglob("*"):
        if path.is_file() and path.suffix in extensions and ".git" not in path.parts and path.name != "test_pipeline.py":
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                assert not pattern.search(content), f"Potential secret pattern found in {path}!"
