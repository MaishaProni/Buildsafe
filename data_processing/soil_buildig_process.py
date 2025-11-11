import pandas as pd

# Load the datasets
soil_building = pd.read_csv("raw_data/soil_building_data.csv")
upazila = pd.read_csv("processed_data/Updated_upazila_distance_coast.csv")

# Clean column names
soil_building.columns = soil_building.columns.str.strip()
upazila.columns = upazila.columns.str.strip()

# Rename if necessary to ensure matching merge key
upazila = upazila.rename(columns={"Upazila": "upazila"})

# Merge by 'upazila' to bring in Latitude and Longitude
combined = soil_building.merge(upazila[["upazila", "Latitude", "Longitude"]],
                               on="upazila", how="left")

# Drop rows missing key info if needed
cleaned = combined.dropna(subset=["Latitude", "Longitude"])

# Save the updated file
combined.to_csv("processed_data/Updated_soil_building_data.csv", index=False)

print("Saved as Updated_soil_building_data.csv")
