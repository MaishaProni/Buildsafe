import pandas as pd


urban= pd.read_csv("raw_data/urban_infastructure_data.csv")            
coords = pd.read_csv("processed_data/Updated_Coordinates.csv")       


urban.columns = urban.columns.str.strip()
coords.columns = coords.columns.str.strip()




urban= urban.merge(coords[["Place", "Latitude", "Longitude"]],
                         on="Place", how="left")

cleaned = urban.dropna(subset=["population_density", "landuse_water_pct", "building_footprint_area_km2", "Latitude", "Longitude"])


urban.to_csv("processed_data/Updated_urban_infastructure_data.csv", index=False)

print("Saved as Updated_urban_infastructure_data.csv")
