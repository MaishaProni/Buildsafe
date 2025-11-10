import pandas as pd

# Load the datasets
rain = pd.read_csv("raw_data/rainfall_data.csv")
coords = pd.read_csv("processed_data/Updated_Coordinates.csv")

# Clean column names
rain.columns = rain.columns.str.strip()
coords.columns = coords.columns.str.strip()

rain = rain.rename(columns={"location": "district"})

# Rename 'Place' to 'district' if needed
if "Place" in coords.columns and "district" not in coords.columns:
    coords = coords.rename(columns={"Place": "district"})

# Merge rainfall with coordinates on 'district'
combined = rain.merge(
    coords[["district", "Latitude", "Longitude"]],
    on="district",
    how="left"
)

# Drop rows missing essential fields (adjust columns as needed)
cleaned = combined.dropna(
    subset=["annual_precip_mm", "max_24h_precip_mm", "Latitude", "Longitude"]
)

# Save
combined.to_csv("processed_data/Updated_rainfall_data.csv", index=False)
# cleaned.to_csv("Updated_rainfall_data_cleaned.csv", index=False)  # optional

print("Saved as Updated_rainfall_data.csv")
