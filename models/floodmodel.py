# flood_vulnerability_model.py
# Build Flood Vulnerability Index (FVI) + train spatial model + single Seaborn-style plot.

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, Any

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config (tune later if you want)
# -------------------------
# All inputs are assumed to be 0..1 proxies already (as in your screenshot).
WEIGHTS_FLOOD = {
    "national_severity_weight":        0.15,
    "basin_exposure_weight":           0.15,
    "erosion_weight":                  0.10,
    "proxy_peak_exceedance_score":     0.20,
    "proxy_days_above_danger_score":   0.15,
    "proxy_percent_area_flooded_score":0.15,
    "FDI_proxy_0_1":                   0.10,   # optional proxy, keep small to avoid circularity
}

NAME_CANDIDATES = ("district", "District", "Place", "place", "Upazila", "City", "Name")

# -------------------------
# Utils
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in KM for arrays/Series/scalars."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def _mode_or_nan(s: pd.Series):
    s = s.dropna().astype(str).str.strip()
    return s.mode().iloc[0] if not s.mode().empty else np.nan

def _derive_place_per_location(raw: pd.DataFrame) -> pd.DataFrame:
    keep = ["Latitude", "Longitude"] + [c for c in NAME_CANDIDATES if c in raw.columns]
    tmp = raw[keep].copy()
    g = tmp.groupby(["Latitude","Longitude"], as_index=False).agg(_mode_or_nan)

    place = pd.Series(index=g.index, dtype=object)
    for c in NAME_CANDIDATES:
        if c in g.columns:
            place = place.fillna(g[c])
    place = place.fillna("Unknown")
    return pd.DataFrame({"Latitude": g["Latitude"], "Longitude": g["Longitude"], "Place": place})

def clamp01(s: pd.Series) -> pd.Series:
    """Ensure 0..1 numeric range (handles strings gracefully)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.clip(lower=0, upper=1).fillna(0)

# -------------------------
# Build FVI dataset (one row per lat/lon across years)
# -------------------------
def build_fvi_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Required
    for c in ["Latitude", "Longitude", "year"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Columns used for FVI
    cols_needed = [
        "national_severity_weight",
        "basin_exposure_weight",
        "erosion_weight",
        "proxy_peak_exceedance_score",
        "proxy_days_above_danger_score",
        "proxy_percent_area_flooded_score",
        "FDI_proxy_0_1"
    ]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column for FVI: {c}")

    # Aggregate by coordinate across years -> mean (robust & simple)
    agg_map = {c: "mean" for c in cols_needed}
    grp = df.groupby(["Latitude","Longitude"], as_index=False).agg(agg_map)

    # Attach a human-readable place name
    places = _derive_place_per_location(df)
    out = grp.merge(places, on=["Latitude","Longitude"], how="left")

    # Clamp inputs to 0..1 just in case
    for c in cols_needed:
        out[c] = clamp01(out[c])

    # Composite FVI 0..1, then scale to 0..100
    out["FVI_0_1"] = sum(WEIGHTS_FLOOD[c] * out[c] for c in cols_needed)
    out["FVI"] = (out["FVI_0_1"] * 100).round(2)
    out["FVI_Class"] = pd.cut(out["FVI"], [0,25,50,75,100],
                              labels=["Low","Medium","High","Very High"], include_lowest=True)
    return out

# -------------------------
# Train spatial model on (lat, lon) → FVI
# -------------------------
def train_spatial_model(df: pd.DataFrame, n_neighbors: int = 8, model_path: str | None = "flood_knn.pkl"):
    X = df[["Latitude","Longitude"]].astype(float).values
    y = df["FVI"].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(Xs, y)

    if model_path:
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": knn}, f)

    return knn, scaler

def load_spatial_model(model_path: str = "flood_knn.pkl"):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]

def predict_fvi(lat: float, lon: float, model, scaler) -> float:
    Xq = scaler.transform(np.array([[lat, lon]], dtype=float))
    pred = float(model.predict(Xq)[0])
    return max(0.0, min(100.0, pred))

# -------------------------
# Nearest + safer
# -------------------------
def nearest_observed(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    d = haversine_km(lat, lon, df["Latitude"].values, df["Longitude"].values)
    return df.iloc[int(np.argmin(d))]

def nearest_safer(df: pd.DataFrame, base_row: pd.Series, lat: float, lon: float, k: int = 3) -> pd.DataFrame:
    safer = df[df["FVI"] < float(base_row["FVI"])].copy()
    if safer.empty:
        safer = df[(df["Latitude"] != base_row["Latitude"]) | (df["Longitude"] != base_row["Longitude"])].copy()
    safer["distance_km"] = haversine_km(lat, lon, safer["Latitude"].values, safer["Longitude"].values)
    return safer.sort_values(["distance_km","FVI"], ascending=[True, True]).head(k)

# -------------------------
# One Seaborn-style plot (matplotlib bars) + pretty console
# -------------------------
def _zone_label_and_color_flood(fvi: float) -> tuple[str, str]:
    if fvi < 25:   return "Low",    "#2ecc71"   # green
    if fvi < 50:   return "Medium", "#f1c40f"   # yellow
    if fvi < 75:   return "High",   "#e67e22"   # orange
    return "Very High", "#e74c3c"                # red

def visualize_oneplot_for_coordinate(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    model,
    scaler,
    k: int = 3,
    out_path: str | None = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    One chart with Safety Index bars:
      - your point (predicted FVI),
      - dataset average,
      - k nearest safer spots (with distance)
    Background shows safety zones mapped from FVI bands.
    """
    # --- compute ---
    pred_fvi = predict_fvi(lat, lon, model, scaler)
    base = nearest_observed(df, lat, lon)
    recs = nearest_safer(df, base, lat, lon, k=k)

    avg_fvi = float(df["FVI"].mean())
    avg_safety = 100 - avg_fvi
    chosen_place = str(base.get("Place", "Unknown"))
    chosen_safety = 100 - pred_fvi

    # plotting table
    rows = [
        {"label": f"Your point ({chosen_place})", "group": "Chosen",  "FVI": float(pred_fvi), "Safety": float(chosen_safety)},
        {"label": "Dataset Average",               "group": "Average", "FVI": float(avg_fvi),  "Safety": float(avg_safety)},
    ]
    for _, r in recs.iterrows():
        rows.append({
            "label": f"{str(r.get('Place','Unknown'))}  ({r['distance_km']:.1f} km)",
            "group": "Safer",
            "FVI": float(r["FVI"]),
            "Safety": float(100 - r["FVI"]),
        })
    plot_df = pd.DataFrame(rows)

    # --- style ---
    sns.set_theme(style="whitegrid", context="talk")

    # Safety zones for Safety Index (100 - FVI)
    bands = [
        (75, 100, "Low flood risk (FVI 0–25)",     "#e8f4fd"),  # light blue
        (50, 75,  "Medium (25–50)",                 "#d7ecfb"),
        (25, 50,  "High (50–75)",                   "#cfe3f8"),
        (0, 25,   "Very High (75–100)",             "#c7d9f4"),
    ]

    def _bar_color(group: str) -> str:
        return {"Chosen": "#1f4e79", "Average": "#7f8c8d", "Safer": "#2e86de"}.get(group, "#95a5a6")

    # order rows: chosen, average, then safer by Safety desc
    safer_part = plot_df[plot_df["group"] == "Safer"].sort_values("Safety", ascending=False)
    ordered = pd.concat(
        [plot_df[plot_df["group"] == "Chosen"],
         plot_df[plot_df["group"] == "Average"],
         safer_part],
        ignore_index=True
    )

    # --- draw (matplotlib so we can color each bar) ---
    fig, ax = plt.subplots(figsize=(12, 7))

    for lo, hi, label, color in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.7, lw=0)

    y_pos = np.arange(len(ordered))
    colors = [_bar_color(g) for g in ordered["group"]]
    ax.barh(y_pos, ordered["Safety"].values, edgecolor="black", color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["label"].tolist())

    # annotate FVI
    for i, (safety, fvi) in enumerate(zip(ordered["Safety"], ordered["FVI"])):
        ax.text(safety + 1, i, f"FVI {fvi:.1f}", va="center", fontsize=11)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Safety Index (100 − FVI) — higher is safer")
    ax.set_ylabel("")
    ax.set_title("BuildSafe • Flood Vulnerability — One-View Summary")

    from matplotlib.patches import Patch
    handles = [
        Patch(color="#1f4e79", label="Your point"),
        Patch(color="#7f8c8d", label="Dataset average"),
        Patch(color="#2e86de", label=f"Nearest safer spots (k={k})"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
    elif show:
        plt.show()

    # Console summary
    zone, _ = _zone_label_and_color_flood(pred_fvi)
    avg_zone, _ = _zone_label_and_color_flood(avg_fvi)

    print("\n" + "═" * 70)
    print("🌊 BuildSafe — Flood Vulnerability Summary")
    print("─" * 70)
    print(f"📍 Input:  lat {lat:.4f}, lon {lon:.4f}")
    print(f"🪧 Nearest observed place: {chosen_place}")
    print(f"📈 Predicted FVI: {pred_fvi:.2f}  |  Safety Index: {100 - pred_fvi:.2f}  |  Zone: {zone}")
    print(f"📊 Dataset Avg FVI: {avg_fvi:.2f}  |  Avg Safety: {avg_safety:.2f}  |  Zone: {avg_zone}")
    if not recs.empty:
        print("✅ Nearest safer spots:")
        for _, r in recs.iterrows():
            rz, _ = _zone_label_and_color_flood(float(r['FVI']))
            print(f"   • {str(r.get('Place','Unknown')):<18}  "
                  f"FVI {float(r['FVI']):>5.1f}  |  Safety {100 - float(r['FVI']):>5.1f}  "
                  f"|  {float(r['distance_km']):>6.1f} km  |  {rz}")
    else:
        print("ℹ️  No strictly safer locations found nearby — plot shows closest alternatives.")
    if out_path:
        print(f"🖼️  Figure saved → {Path(out_path).resolve()}")
    print("═" * 70 + "\n")

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "predicted_FVI": float(round(pred_fvi, 2)),
        "predicted_Safety_Index": float(round(100 - pred_fvi, 2)),
        "nearest_observed_place": chosen_place,
        "avg_FVI": float(round(avg_fvi, 2)),
        "avg_Safety_Index": float(round(avg_safety, 2)),
        "recommendations_count": int(len(recs))
    }

# -------------------------
# Demo run
# -------------------------
# if __name__ == "__main__":
#     # Change path to your flood CSV
#     CSV = r"./Updated_flood_data.csv"   # e.g., the file with the columns shown in your screenshot
#     df = build_fvi_dataset(CSV)
#     model, scaler = train_spatial_model(df, n_neighbors=8, model_path="flood_knn.pkl")

#     # Example coordinate (adjust as you like)
#     info = visualize_oneplot_for_coordinate(
#         df, lat=23.86, lon=90.00, model=model, scaler=scaler, k=3,
#         out_path=None,   # or "outputs/flood_summary.png"
#         show=True
#     )
#     print(info)
