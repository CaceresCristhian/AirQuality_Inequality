"""
Fetch ambient PM2.5 monitoring stations from OpenAQ API (v3).

Uses OpenAQ API key securely from os.environ or root .env (OPENAQ_API_KEY).
Queries active regulatory and reference-grade sensor locations in the US to cross-validate
spatial coverage against official EPA AQS monitoring infrastructure.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "01_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "openaq_us_monitors.parquet"


def get_openaq_key() -> str:
    key = os.environ.get("OPENAQ_API_KEY")
    if key:
        return key.strip()

    # Search project-level .env, then parent workspace .env
    for candidate_env in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if candidate_env.exists():
            for line in candidate_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAQ_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise RuntimeError("OPENAQ_API_KEY not found in environment or .env file.")


def fetch_openaq_us_locations(api_key: str, limit: int = 100) -> pd.DataFrame:
    headers = {"X-API-Key": api_key}
    # OpenAQ v3: countries_id=2 is United States, parameters_id=2 is PM2.5
    url = f"https://api.openaq.org/v3/locations?countries_id=2&parameters_id=2&limit={limit}"
    print(f"Connecting to OpenAQ v3 API for US PM2.5 locations (limit={limit})...")
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        print(f"Retrieved {len(results)} OpenAQ stations in the US.")

        records = []
        for loc in results:
            coords = loc.get("coordinates") or {}
            country_obj = loc.get("country") or {}
            records.append({
                "openaq_id": loc.get("id"),
                "name": loc.get("name"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "country": country_obj.get("code") if isinstance(country_obj, dict) else None,
                "timezone": loc.get("timezone"),
                "is_monitor": loc.get("isMonitor"),
                "is_mobile": loc.get("isMobile"),
            })
        return pd.DataFrame(records)
    except requests.RequestException as exc:
        print(f"  -> HTTP/Network error querying OpenAQ: status {getattr(exc.response, 'status_code', 'unknown')}")
        return pd.DataFrame()


def main():
    api_key = get_openaq_key()
    print("Authenticating with OpenAQ API (credentials loaded securely from environment)...")

    df = fetch_openaq_us_locations(api_key, limit=100)
    if not df.empty:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved {len(df)} OpenAQ monitors to {OUTPUT_PATH}")
        print(df.head())


if __name__ == "__main__":
    main()
