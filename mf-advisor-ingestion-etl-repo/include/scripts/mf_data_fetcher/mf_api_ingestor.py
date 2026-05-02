import asyncio
import httpx
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "https://api.mfapi.in/mf"
CONCURRENT_REQUESTS = 12
ROWS_TO_FETCH = 2   # latest 2 NAV entries


# -----------------------------
# STEP 1: Fetch all schemes
# -----------------------------
def fetch_all_schemes():
    response = httpx.get(BASE_URL, timeout=20)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    return df[['schemeCode', 'schemeName']]


# -----------------------------
# STEP 2: Async NAV fetch
# -----------------------------
async def fetch_nav(client, scheme_code):
    url = f"{BASE_URL}/{scheme_code}"
    try:
        res = await client.get(url, timeout=20)
        res.raise_for_status()
        return scheme_code, res.json()
    except Exception as e:
        print(f"Error fetching {scheme_code}: {e}")
        return scheme_code, None


# -----------------------------
# STEP 3: Rate-limited wrapper
# -----------------------------
async def fetch_with_limit(semaphore, client, scheme_code):
    async with semaphore:
        return await fetch_nav(client, scheme_code)


# -----------------------------
# STEP 4A: Flatten NAV + metadata
# -----------------------------
def process_full_response(scheme_code, raw_json):
    if raw_json is None:
        return None

    try:
        nav_df = pd.DataFrame(raw_json['data'])

        nav_df['date'] = pd.to_datetime(
            nav_df['date'],
            format='%d-%m-%Y'
        )
        nav_df['nav'] = nav_df['nav'].astype(float)

        # Sort DESC to get latest first
        nav_df = nav_df.sort_values('date', ascending=False)

        # Take latest 2 records
        nav_df = nav_df.head(ROWS_TO_FETCH)

        if nav_df.empty:
            return None

        # Sort ASC for readability
        nav_df = nav_df.sort_values('date')

        # Extract metadata
        meta = raw_json.get('meta', {})

        nav_df['scheme_code'] = scheme_code
        nav_df['scheme_name'] = meta.get('scheme_name')
        nav_df['fund_house'] = meta.get('fund_house')
        nav_df['scheme_type'] = meta.get('scheme_type')
        nav_df['scheme_category'] = meta.get('scheme_category')

        return nav_df[
            [
                "scheme_code",
                "scheme_name",
                "fund_house",
                "scheme_type",
                "scheme_category",
                "date",
                "nav",
            ]
        ]

    except Exception as e:
        print(f"Processing error for {scheme_code}: {e}")
        return None


# -----------------------------
# STEP 4B: Store raw JSON
# -----------------------------
def process_raw_json(scheme_code, raw_json):
    return {
        "scheme_code": scheme_code,
        "full_response": raw_json
    }


# -----------------------------
# STEP 5: Orchestrator
# -----------------------------
async def run_pipeline():
    print("Fetching all schemes...")
    schemes_df = fetch_all_schemes()

    scheme_codes = schemes_df['schemeCode'].tolist()
    print(f"Total schemes: {len(scheme_codes)}")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    nav_results = []
    raw_results = []

    async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50)
    ) as client:

        tasks = [
            fetch_with_limit(semaphore, client, code)
            for code in scheme_codes
        ]

        for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Fetching NAV data"
        ):
            scheme_code, raw_json = await future

            if raw_json is None:
                continue

            # Store raw JSON
            raw_results.append(
                process_raw_json(scheme_code, raw_json)
            )

            # Process NAV + metadata
            df = process_full_response(scheme_code, raw_json)
            if df is not None:
                nav_results.append(df)

    # Combine NAV data
    if nav_results:
        nav_df = pd.concat(nav_results, ignore_index=True)
    else:
        nav_df = pd.DataFrame()

    # Raw JSON DataFrame
    raw_df = pd.DataFrame(raw_results)

    return nav_df, raw_df


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    nav_df, raw_df = asyncio.run(run_pipeline())

    print("\nNAV Data Sample:")
    print(nav_df.head())

    print("\nRaw Data Sample:")
    print(raw_df.head())

    # Save outputs
    nav_df.to_csv("mf_nav_last_2_entries.csv", index=False)
    raw_df.to_parquet("mf_raw_full_response.parquet", index=False)

    print("\nFiles saved:")
    print("✔ mf_nav_last_2_entries.csv")
    print("✔ mf_raw_full_response.parquet")