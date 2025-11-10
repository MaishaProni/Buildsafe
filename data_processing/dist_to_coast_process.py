import pandas as pd


df = pd.read_csv("raw_data/upazila_distance_to_coast.csv", skiprows=1)

df.columns = ["Latitude", "Longitude", "dist_coast", "upazila"]


df = df.dropna()


df.to_csv("processed_data/Updated_upazila_distance_cleaned.csv", index=False)

print("Cleaned data saved as 'bd_upazila_distance_cleaned.csv'")
