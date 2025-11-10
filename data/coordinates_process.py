import pandas as pd


df1 = pd.read_csv("Coordinates.csv")
df2 = pd.read_csv("Bangladesh_District_Coordinates.csv")

df1.columns = df1.columns.str.strip()
df2 = df2.rename(columns={
    "district": "Place",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "division": "Division"
})
df2.columns = df2.columns.str.strip()


df1 = df1.dropna(subset=["Latitude", "Longitude"])
df2 = df2.dropna(subset=["Latitude", "Longitude"])


df1["_key"] = df1["Place"].astype(str).str.strip().str.casefold()
df2["_key"] = df2["Place"].astype(str).str.strip().str.casefold()


merged = df1.merge(df2[["_key", "Latitude", "Longitude", "Division"]], on="_key", how="left", suffixes=("", "_2"))


for col in ["Latitude", "Longitude", "Division"]:
    merged[col] = merged[col].where(~merged[col].isna() & (merged[col] != ""), merged[f"{col}_2"])
    merged.drop(columns=[f"{col}_2"], inplace=True, errors="ignore")


missing = df2.loc[~df2["_key"].isin(merged["_key"])]
combined = pd.concat([merged, missing], ignore_index=True)


combined = combined.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"])


combined.to_csv("Updated_Coordinates.csv", index=False, encoding="utf-8-sig")

print(" Cleaned, merged, and saved as Updated_Coordinates.csv")
