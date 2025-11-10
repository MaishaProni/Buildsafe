import pandas as pd

# Load data
fdi = pd.read_csv("flood_data.csv")
coords = pd.read_csv("Updated_coordinates.csv")

# Clean column names
fdi.columns = fdi.columns.str.strip()
coords.columns = coords.columns.str.strip()

# Rename coordinate column for joining
coords = coords.rename(columns={"Place": "district"})

# Merge on district
combined = fdi.merge(coords[["district", "Latitude", "Longitude"]],
                     on="district", how="left")

# Drop rows missing important data
cleaned = combined.dropna(subset=["FDI_proxy_0_1", "Latitude", "Longitude"])

# Save cleaned dataset
cleaned.to_csv("Updated_flood_data.csv", index=False)

print("Saved as Updated_flood_data.csv")
