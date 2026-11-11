"""
Query live US EPA AQS API for certified ambient air quality monitors and annual statistics.

Uses EPA AQS API credentials loaded securely from os.environ or root .env (EPA_AQS_EMAIL, EPA_AQS_KEY).
Queries official Federal Reference Method (FRM) and Federal Equivalent Method (FEM)
PM2.5 (parameter 88101) monitors and annual summary data.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "01_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "epa_aqs_api_monitors_summary.parquet"


def get_epa_aqs_credentials() -> tuple[str, str]:
    email = os.environ.get("EPA_AQS_EMAIL")
    key = os.environ.get("EPA_AQS_KEY")
    if email and key:
        return email.strip(), key.strip()

    # Search project-level .env, then parent workspace .env
    for candidate_env in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if candidate_env.exists():
            for line in candidate_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("EPA_AQS_EMAIL="):
                    email = line.split("=", 1)[1].strip()
                elif line.startswith("EPA_AQS_KEY="):
                    key = line.split("=", 1)[1].strip()
            if email and key:
                return email, key

    raise RuntimeError("EPA_AQS_EMAIL or EPA_AQS_KEY not found in environment or .env file.")


def fetch_epa_annual_data(email: str, key: str, state_fips: str = "39", year: int = 2023) -> pd.DataFrame:
    """Fetch official annual PM2.5 summary concentrations by state from EPA AQS API."""
    base_url = "https://aqs.epa.gov/data/api/annualData/byState"
    params = {
        "email": email,
        "key": key,
        "param": "88101",
        "bdate": f"{year}0101",
        "edate": f"{year}1231",
        "state": state_fips,
    }
    print(f"Connecting to EPA AQS API for State FIPS {state_fips} (Year {year})...")
    try:
        r = requests.get(base_url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
        rows = data.get("Data", [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except requests.RequestException as exc:
        print(f"  -> HTTP/Network error querying FIPS {state_fips}: status {getattr(exc.response, 'status_code', 'unknown')}")
        return pd.DataFrame()


def main():
    email, key = get_epa_aqs_credentials()
    print("Authenticating with US EPA AQS API (credentials loaded securely from environment)...")

    # Query annual PM2.5 summaries for Ohio (39) and Pennsylvania (42)
    frames = []
    for fips, st_name in [("39", "Ohio"), ("42", "Pennsylvania")]:
        df = fetch_epa_annual_data(email, key, state_fips=fips, year=2023)
        if not df.empty:
            print(f"  -> Retrieved {len(df)} annual monitor summary records for {st_name}.")
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Select key columns
        cols = [c for c in ["state_code", "county_code", "site_number", "county_name",
                            "latitude", "longitude", "arithmetic_mean", "metric_used",
                            "completeness_indicator", "method_name"] if c in combined.columns]
        out_df = combined[cols].copy()
        out_df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(out_df)} EPA AQS API annual monitor records to {OUTPUT_PATH}")
        print(out_df.head())


if __name__ == "__main__":
    main()
