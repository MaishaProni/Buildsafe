import pandas as pd


fdi = pd.read_csv("flood_data.csv")
coords = pd.read_csv("Updated_coordinates.csv")


fdi.columns = fdi.columns.str.strip()
coords.columns = coords.columns.str.strip()


coords = coords.rename(columns={"Place": "district"})


combined = fdi.merge(coords[["district", "Latitude", "Longitude"]],
                     on="district", how="left")


cleaned = combined.dropna(subset=["FDI_proxy_0_1", "Latitude", "Longitude"])


cleaned.to_csv("Updated_flood_data.csv", index=False)

print("Saved as Updated_flood_data.csv")
