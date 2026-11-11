"""
Download authentic regulatory PM2.5 observations from US EPA AQS (2023)
and merge with official US Census Bureau county-level socioeconomic data (SAIPE).
Guarantees 100% empirical physical observations and genuine socioeconomic inequality indicators.
"""

from pathlib import Path
import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "01_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

EPA_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip"
SAIPE_URL = "https://www2.census.gov/programs-surveys/saipe/datasets/2022/2022-state-and-county/est22all.txt"
OUTPUT_PATH = RAW_DIR / "epa_aqs_pm25_census_socioeconomic_2023.parquet"


def fetch_epa_aqs_pm25():
    print(f"Downloading authentic US EPA AQS Daily PM2.5 for 2023 from {EPA_URL}...")
    req = urllib.request.Request(EPA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_filename = z.namelist()[0]
        print(f"Extracting {csv_filename}...")
        with z.open(csv_filename) as f:
            df_epa = pd.read_csv(
                f,
                usecols=[
                    "State Code",
                    "County Code",
                    "Site Num",
                    "Latitude",
                    "Longitude",
                    "Date Local",
                    "Arithmetic Mean",
                    "State Name",
                    "County Name",
                    "City Name",
                ],
                dtype={"State Code": int, "County Code": int, "Site Num": int},
                low_memory=False,
            )

    print(f"Loaded {len(df_epa):,} raw daily EPA records.")
    # Filter valid measurements >= 0
    df_epa = df_epa[df_epa["Arithmetic Mean"] >= 0.0]

    # Aggregate by physical monitoring station
    site_agg = df_epa.groupby(
        ["State Code", "County Code", "Site Num", "Latitude", "Longitude", "State Name", "County Name"]
    ).agg(
        pm25_annual_mean=("Arithmetic Mean", "mean"),
        pm25_median=("Arithmetic Mean", "median"),
        pm25_98th=("Arithmetic Mean", lambda s: np.percentile(s, 98)),
        pm25_max=("Arithmetic Mean", "max"),
        obs_days=("Date Local", "count"),
    ).reset_index()

    # Retain regulatory stations with at least 50 valid sampling days in 2023
    site_agg = site_agg[site_agg["obs_days"] >= 50].reset_index(drop=True)
    site_agg["station_id"] = (
        site_agg["State Code"].astype(str).str.zfill(2)
        + "-"
        + site_agg["County Code"].astype(str).str.zfill(3)
        + "-"
        + site_agg["Site Num"].astype(str).str.zfill(4)
    )
    print(f"Active regulatory PM2.5 monitoring stations (>=50 days): {len(site_agg):,}")
    return site_agg


def fetch_census_saipe():
    print(f"Downloading US Census Bureau SAIPE county socioeconomic data from {SAIPE_URL}...")
    req = urllib.request.Request(SAIPE_URL, headers={"User-Agent": "Mozilla/5.0"})
    records = []
    with urllib.request.urlopen(req, timeout=45) as resp:
        for line in resp:
            l = line.decode("latin1")
            if len(l) < 200:
                continue
            try:
                st_fips = int(l[0:2])
                cty_fips = int(l[3:7])
                if cty_fips == 0:
                    continue  # Skip national and state-level aggregate lines
                poverty_count = int(l[7:15].strip())
                poverty_rate = float(l[34:39].strip())
                median_income = float(l[133:140].strip())
                county_name = l[193:238].strip()
                state_abbr = l[239:242].strip()
                records.append(
                    {
                        "State Code": st_fips,
                        "County Code": cty_fips,
                        "poverty_rate_pct": poverty_rate,
                        "poverty_population": poverty_count,
                        "median_household_income": median_income,
                        "census_county_name": county_name,
                        "state_abbr": state_abbr,
                    }
                )
            except Exception:
                continue

    df_saipe = pd.DataFrame(records)
    print(f"Parsed {len(df_saipe):,} US counties from Census SAIPE.")
    return df_saipe


def build_integrated_dataset():
    site_agg = fetch_epa_aqs_pm25()
    df_saipe = fetch_census_saipe()

    # Merge EPA sites with Census socioeconomics
    merged = pd.merge(site_agg, df_saipe, on=["State Code", "County Code"], how="inner")
    print(
        f"Successfully synchronized {len(merged):,} authentic EPA stations across "
        f"{merged.groupby(['State Code', 'County Code']).ngroups} distinct counties in {merged['State Code'].nunique()} states."
    )
    print(f"Median County Household Income: ${merged['median_household_income'].median():,.0f}")
    print(f"Median County Poverty Rate: {merged['poverty_rate_pct'].median():.1f}%")
    print(f"Network Mean Annual PM2.5: {merged['pm25_annual_mean'].mean():.2f} ug/m3")

    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved authentic integrated dataset to {OUTPUT_PATH}")
    return merged


if __name__ == "__main__":
    build_integrated_dataset()
