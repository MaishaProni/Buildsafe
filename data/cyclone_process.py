import pandas as pd


cyclone = pd.read_csv("cyclone_data.csv")            
coords = pd.read_csv("Updated_Coordinates.csv")       


cyclone.columns = cyclone.columns.str.strip()
coords.columns = coords.columns.str.strip()


coords = coords.rename(columns={"Place": "district"})

combined = cyclone.merge(coords[["district", "Latitude", "Longitude"]],
                         on="district", how="left")

cleaned = combined.dropna(subset=["cyclone_count", "avg_max_wind_kph", "strongest_category", "Latitude", "Longitude"])


combined.to_csv("Updated_cyclone_data.csv", index=False)

print("Saved as Updated_cyclone_data.csv")
