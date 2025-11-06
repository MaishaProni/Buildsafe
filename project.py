# console_predict.py
# Run from terminal:  python console_predict.py
# Prompts for Latitude/Longitude, loads data + model from this folder, and prints a result.

import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ----------------------------
# Paths (relative to this file)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR

SOIL_CSV = DATA_PATH / "soil_topography_clean.csv"

# Pick the first flood CSV that matches, or fall back to default name
cand = list(DATA_PATH.glob("Bangladesh_Cities_Flood_Frequency_with_*.csv"))
FLOOD_CSV = cand[0] if cand else DATA_PATH / "Bangladesh_Cities_Flood_Frequency_with_Coordinates.csv"

POPULATION_CSV = DATA_PATH / "bangladesh_population_density.csv"

MODEL_PATH = DATA_PATH / "vulnerability_model.pkl"
SCALER_PATH = DATA_PATH / "scaler.pkl"

# ----------------------------
# Helpers
# ----------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize header names and map common aliases to canonical names used below."""
    # Trim whitespace first
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Build a lower-key map
    alias_map = {
        "lat": "Latitude",
        "latitude": "Latitude",
        "lon": "Longitude",
        "lng": "Longitude",
        "longitude": "Longitude",
        "place": "Place",
        "place_name": "Place",
        "division": "Division",
        "division_name": "Division",
        "soiltype": "Soil_Type",
        "soil type": "Soil_Type",
        "bearing": "Bearing_Capacity",
        "bearing_capacity": "Bearing_Capacity",
        "erosion": "Erosion_Risk",
        "erosion_risk": "Erosion_Risk",
        "building_suitability": "Building_Suitability",
        "flood_frequency": "Flood_Frequency",
        "population_density": "Population_Density_Per_Sq_Km",
        "population_density_per_sq_km": "Population_Density_Per_Sq_Km",
    }
    lower_map = {k.lower(): v for k, v in alias_map.items()}
    for c in list(df.columns):
        key = c.lower().replace(" ", "_")
        if key in lower_map:
            df.rename(columns={c: lower_map[key]}, inplace=True)
    return df

def calculate_distance(lat1, lng1, lat2, lng2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def get_nearest_locations(lat, lng, soil_df, flood_df, count=5):
    """Find nearest documented locations from soil_df and merge flood freq if available."""
    locations = []
    if soil_df is not None and not soil_df.empty:
        for _, row in soil_df.iterrows():
            if pd.notna(row.get("Latitude")) and pd.notna(row.get("Longitude")):
                distance = calculate_distance(lat, lng, row["Latitude"], row["Longitude"])
                locations.append({
                    "name": row.get("Place", "Unknown"),
                    "lat": float(row["Latitude"]),
                    "lng": float(row["Longitude"]),
                    "division": row.get("Division", ""),
                    "soil": row.get("Soil_Type", "N/A"),
                    "bearing": row.get("Bearing_Capacity", "N/A"),
                    "erosion": row.get("Erosion_Risk", "N/A"),
                    "suitability": row.get("Building_Suitability", "N/A"),
                    "distance": distance
                })

    # Merge flood frequency by place name if present
    if flood_df is not None and not flood_df.empty and locations:
        # Normalize flood columns too (in case)
        flood_df = _normalize_cols(flood_df)
        for _, row in flood_df.iterrows():
            if pd.notna(row.get("Latitude")) and pd.notna(row.get("Longitude")):
                existing = next((x for x in locations if x["name"] == row.get("Place")), None)
                if existing:
                    existing["flood"] = row.get("Flood_Frequency", "N/A")

    locations.sort(key=lambda x: x["distance"])
    return locations[:count]

def predict_vulnerability(lat, lng, pop_density, model, scaler, soil_df, flood_df):
    """Build features consistent with app.py and predict a 0–100 vulnerability score."""
    # Find closest reference
    nearest = get_nearest_locations(lat, lng, soil_df, flood_df, 1)
    if not nearest:
        return None, None, None  # no data

    closest = nearest[0]

    # Text → scores (same mappings as app.py)
    bearing_map = {"Poor": 4, "Variable (often poor)": 3.5, "Variable": 3, "Moderate": 2, "Good": 1}
    erosion_map = {
        "Low to Medium": 1, "Medium": 2, "Medium-High": 3, "High": 4,
        "High (flash floods)": 4.5, "High (landslides/flash)": 4.5,
        "High (hills/urban floods)": 4, "High (tidal/sea level)": 4,
        "High (coastal)": 4, "High (river/coastal)": 4, "High (urban & river)": 4
    }
    flood_map = {
        "Low to Medium": 1, "Medium": 2, "Medium-High": 3, "High": 4,
        "High (coastal/flash)": 4.5, "High (flash/landslide)": 4.5, "High (urban)": 3.5
    }

    bearing_score = bearing_map.get(closest.get("bearing"), 2)
    erosion_score = erosion_map.get(closest.get("erosion"), 2)
    flood_score = flood_map.get(closest.get("flood", "Medium"), 2)

    # Population density normalization (match app.py)
    max_pop = 4532  # Dhaka's density reference
    pop_score = (float(pop_density) / max_pop) * 4.0

    features = np.array([[bearing_score, erosion_score, flood_score, pop_score, lat, lng]], dtype=float)
    features_scaled = scaler.transform(features)
    pred = float(model.predict(features_scaled)[0])
    pred = max(0.0, min(100.0, pred))  # clamp
    return pred, closest, nearest

# ----------------------------
# Main
# ----------------------------
def main():
    # Prompt for input
    try:
        lat = float(input("Enter Latitude  (20.5 to 27): ").strip())
        lng = float(input("Enter Longitude (87.5 to 93): ").strip())
    except Exception:
        print("Invalid input. Please enter numeric values.")
        return

    # Validate Bangladesh bounds (same as app.py)
    if not (20.5 <= lat <= 27 and 87.5 <= lng <= 93):
        print(json.dumps({"error": "Coordinates outside Bangladesh", "status": "invalid"}, indent=2))
        return

    # Load CSVs
    if not SOIL_CSV.exists():
        print(f"Missing file: {SOIL_CSV.name}")
        return
    if not POPULATION_CSV.exists():
        print(f"Missing file: {POPULATION_CSV.name}")
        return

    soil_df = pd.read_csv(SOIL_CSV)
    flood_df = pd.read_csv(FLOOD_CSV) if FLOOD_CSV.exists() else pd.DataFrame()
    pop_df = pd.read_csv(POPULATION_CSV)

    # Normalize headers for safety
    soil_df = _normalize_cols(soil_df)
    flood_df = _normalize_cols(flood_df) if not flood_df.empty else flood_df
    pop_df = _normalize_cols(pop_df)

    # Load model + scaler (optional: fall back to neutral score)
    model = scaler = None
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    # Get division + population density from nearest location (default 1000)
    nearest = get_nearest_locations(lat, lng, soil_df, flood_df, count=5)
    if not nearest:
        print(json.dumps({"error": "No data available", "status": "no_data"}, indent=2))
        return
    closest = nearest[0]
    division = closest.get("division", "")
    pop_density = 1000.0
    if not pop_df.empty and "Division" in pop_df.columns and division:
        match = pop_df[pop_df["Division"] == division]
        if not match.empty and "Population_Density_Per_Sq_Km" in match.columns:
            pop_density = float(match["Population_Density_Per_Sq_Km"].values[0])

    # Predict
    if model is None or scaler is None:
        # If model/scaler missing, fallback to 50 and still show context
        vulnerability = 50.0
        model_loaded = False
    else:
        vulnerability, closest, nearest = predict_vulnerability(
            lat, lng, pop_density, model, scaler, soil_df, flood_df
        )
        model_loaded = True
        if vulnerability is None:
            print(json.dumps({"error": "No data available", "status": "no_data"}, indent=2))
            return

    # Status bucketing like app.py
    distance_km = float(closest["distance"])
    suitability = str(closest.get("suitability", ""))
    if distance_km < 2 and "Not recommended" in suitability:
        status = "not_suitable"
        message = f"Not Suitable - Too close to {closest['name']} which is not recommended"
    elif vulnerability > 70:
        status = "high_risk"
        message = f"High Risk Area - Vulnerability Score: {vulnerability:.1f}/100"
    elif vulnerability > 50:
        status = "caution"
        message = f"Requires Caution - Vulnerability Score: {vulnerability:.1f}/100"
    else:
        status = "suitable"
        message = f"Potentially Suitable - Vulnerability Score: {vulnerability:.1f}/100"

    # Build output
    result = {
        "status": status,
        "message": message,
        "vulnerability_score": round(float(vulnerability), 2),
        "model_loaded": model_loaded,
        "coordinates": {"lat": lat, "lng": lng},
        "nearest_location": {
            "name": closest.get("name"),
            "distance_km": round(distance_km, 2),
            "division": closest.get("division", ""),
            "soil_type": closest.get("soil", "N/A"),
            "bearing_capacity": closest.get("bearing", "N/A"),
            "suitability": closest.get("suitability", "N/A"),
            "flood_frequency": closest.get("flood", "N/A"),
            "erosion_risk": closest.get("erosion", "N/A"),
            "population_density": pop_density,
        },
        "nearby_locations": [
            {
                "name": loc.get("name"),
                "distance_km": round(float(loc["distance"]), 2),
                "division": loc.get("division", ""),
            }
            for loc in nearest[1:4]
        ],
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
