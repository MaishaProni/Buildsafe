

import pandas as pd
import re
from pathlib import Path
import numpy as np

# paths
BASE = Path(__file__).resolve().parent
INPUT = BASE / "soil_topography.csv"
OUTPUT = BASE / "soil_topography_clean.csv"

def find_header_row(path: Path):
    """Detect the header row that contains Latitude and Longitude."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if "Latitude" in line and "Longitude" in line:
                return i
    return 1  

def to_float(value):
   
    s = str(value).replace(",", ".")
    s = re.sub(r"[^\d\.\-]+", "", s)
    try:
        return float(s)
    except:
        return np.nan

def canonical_div(name: str) -> str:
    m = {
        "chittagong": "Chittagong",
        "chattogram": "Chittagong",
        "barisal": "Barisal",
        "barishal": "Barisal",
        "rajshahi": "Rajshahi",
        "rangpur": "Rangpur",
        "khulna": "Khulna",
        "sylhet": "Sylhet",
        "dhaka": "Dhaka",
        "mymensingh": "Mymensingh",
    }
    return m.get(str(name).strip().lower(), str(name).strip())

def main():
    if not INPUT.exists():
        print(f"Missing file: {INPUT}")
        return

    
    hdr = find_header_row(INPUT)
    print(f"Detected header row: {hdr}")

    # read from that row
    df = pd.read_csv(INPUT, header=hdr, encoding="utf-8-sig")

    # normalize columns
    df.rename(columns=lambda x: x.strip(), inplace=True)
    # rename key columns
    rename_map = {
        "lat": "Latitude", "latitude": "Latitude",
        "lon": "Longitude", "lng": "Longitude", "long": "Longitude", "longitude": "Longitude",
        "soil type": "Soil_Type", "soiltype": "Soil_Type",
        "bearing": "Bearing_Capacity", "erosion": "Erosion_Risk",
        "building_suitability": "Building_Suitability", "place_name": "Place", "division_name": "Division"
    }
    for col in list(df.columns):
        key = col.lower().replace(" ", "_")
        if key in rename_map:
            df.rename(columns={col: rename_map[key]}, inplace=True)

    # clean coords
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise SystemExit(f"❌ Could not find Latitude/Longitude columns. Found: {df.columns.tolist()}")

    df["Latitude"] = df["Latitude"].map(to_float)
    df["Longitude"] = df["Longitude"].map(to_float)

    # drop invalids
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[df["Latitude"].between(20.5, 27) & df["Longitude"].between(87.5, 93)]
    print(f"✓ Filtered: {before} → {len(df)} valid rows")

    # tidy divisions
    if "Division" in df.columns:
        df["Division"] = df["Division"].map(canonical_div)

    # reorder
    cols_order = [
        "Place", "Division", "Latitude", "Longitude",
        "Soil_Type", "Bearing_Capacity", "Erosion_Risk",
        "Building_Suitability", "Source_Datasets"
    ]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df.loc[:, cols]

    # save clean version
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f" Cleaned file saved to: {OUTPUT}")

    # preview
    print("\nSample:")
    print(df.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
