
import pandas as pd, os
print("CWD:", os.getcwd())
try:
    df = pd.read_csv("soil_topography.csv")
except Exception as e:
    print("Read error:", e); raise

print("Columns:", list(df.columns))

# Try to detect latitude/longitude columns
lat_col = next((c for c in df.columns if c.lower().strip().startswith(("lat","latitude"))), None)
lon_col = next((c for c in df.columns if c.lower().strip().startswith(("lon","lng","long","longitude"))), None)
print("Guessed:", lat_col, lon_col)

# Coerce to numeric (strings like '23.78°N' become NaN)
for c in (lat_col, lon_col):
    if c:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("°","", regex=False)
                                           .str.replace("N","", regex=False)
                                           .str.replace("S","", regex=False)
                                           .str.replace("E","", regex=False)
                                           .str.replace("W","", regex=False)
                                           .str.replace(",","."), errors="coerce")
ok = df[lat_col].between(20.5,27) & df[lon_col].between(87.5,93)
print("Usable rows (numeric & in Bangladesh bounds):", int(ok.sum()))
print(df.loc[ok, [lat_col, lon_col]].head(5))

