import pandas as pd

# Load
quake = pd.read_csv("raw_data/earthquake_data.csv")
coords = pd.read_csv("processed_data/Updated_Coordinates.csv")
tect  = pd.read_csv("raw_data/tectonic_plates.csv")

# Standardize headers
for df in (quake, coords, tect):
    df.columns = df.columns.str.strip()

# Align key name to "district"
if "City" in quake.columns: quake = quake.rename(columns={"City": "district"})
if "Place" in coords.columns and "district" not in coords.columns:
    coords = coords.rename(columns={"Place": "district"})
if "District" in tect.columns and "district" not in tect.columns:
    tect = tect.rename(columns={"District": "district"})

# Keys (case-insensitive)
quake["_key"] = quake["district"].astype(str).str.strip().str.casefold()
coords["_key"] = coords["district"].astype(str).str.strip().str.casefold()
tect["_key"]   = tect["district"].astype(str).str.strip().str.casefold()

# Merge quake + coords
merged = quake.merge(coords[["_key","Latitude","Longitude"]], on="_key", how="left")

# Merge tectonic (keep tectonic division separate)
tect = tect.rename(columns={"Division":"Tectonic_Division"})
merged = merged.merge(
    tect[["_key","Tectonic_Zone","Approx_Distance_to_Plate_km","Nearest_Plate",
          "Last_Major_Earthquake_Year","Last_Major_Earthquake_Magnitude",
          "Notable_Fault_or_Structure","Seismic_Hazard_Level","Tectonic_Division"]],
    on="_key", how="left"
)
merged = merged.dropna(subset=["Latitude", "Longitude", "All_Earthquakes", "M3_plus", "M4_plus","Approx_Distance_to_Plate_km","Seismic_Hazard_Level"])

# Finalize
merged = merged.drop(columns=["_key"])
merged.to_csv("processed_data/Updated_earthquake_data.csv", index=False, encoding="utf-8-sig")
print("Saved as Updated_earthquake_data.csv")
