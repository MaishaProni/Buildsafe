# cyclone_vulnerability_model.py
# Build Cyclone Vulnerability Index (CVI) + train spatial model + single Seaborn-style plot (matplotlib).

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, Any, Optional

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------
# Config (tune later if you want)
# -------------------------
# Inputs are mixed scales; we normalize internally before weighting.
WEIGHTS_CYCLONE = {
    "w_cyclone_count":      0.15,
    "w_wind":               0.15,
    "w_damage":             0.15,
    "w_fatalities":         0.15,
    "w_pop_pct":            0.10,
    "w_exposure":           0.15,
    "w_coastal_flag":       0.05,  # small bump if flagged coastal
    "w_proximity":          0.10,  # proximity to coast (closer = riskier)
}

NAME_CANDIDATES = ("district", "District", "Place", "place", "Upazila", "City", "Name")

# -------------------------
# Utils
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
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
    s = pd.to_numeric(s, errors="coerce")
    return s.clip(lower=0, upper=1).fillna(0)

# -------------------------
# Coastal distance KNN (Latitude, Longitude) -> dist_coast (km)
# -------------------------
def train_or_load_coast_model(
    upazila_csv: str,
    model_path: Optional[str] = "coast_knn.pkl",
    n_neighbors: int = 5
):
    up = pd.read_csv(upazila_csv)
    # Expected: Latitude, Longitude, dist_coast, upazila (or similar)
    cols = list(up.columns)
    if len(cols) >= 4:
        up = up.rename(columns={cols[0]:"Latitude", cols[1]:"Longitude", cols[2]:"dist_coast"})
    if not {"Latitude","Longitude","dist_coast"}.issubset(up.columns):
        raise ValueError("Upazila file must have Latitude, Longitude, dist_coast columns.")

    up = up.dropna(subset=["Latitude","Longitude","dist_coast"]).copy()

    scaler = StandardScaler()
    X = up[["Latitude","Longitude"]].astype(float).values
    y = up["dist_coast"].astype(float).values
    Xs = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(Xs, y)

    if model_path:
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": knn}, f)

    return knn, scaler

def load_coast_model(model_path: str = "coast_knn.pkl"):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]

def predict_coast_distance(lat: float, lon: float, model, scaler) -> float:
    Xq = scaler.transform(np.array([[lat, lon]], dtype=float))
    return float(model.predict(Xq)[0])

# -------------------------
# Build CVI dataset (one row per lat/lon — or per district row)
# -------------------------
def build_cvi_dataset(
    cyclone_csv: str,
    upazila_csv_for_coast: Optional[str] = None,  # if provided, we compute predicted_dist_coast
    coast_model_path: Optional[str] = "coast_knn.pkl"
) -> pd.DataFrame:
    df = pd.read_csv(cyclone_csv)

    # Required columns (from your sample)
    required = [
        "district","decade","cyclone_count","avg_max_wind_kph","strongest_category",
        "avg_damage_usd_million","avg_fatalities","population_affected_percent",
        "exposure_index","coastal_flag","Latitude","Longitude"
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # If we don't already have distance, compute it via KNN using upazila file
    if "predicted_dist_coast" not in df.columns:
        if upazila_csv_for_coast is None and (coast_model_path is None):
            raise ValueError("Provide upazila_csv_for_coast or precomputed 'predicted_dist_coast' column.")
        try:
            # Prefer loading an existing model if present; else train from the CSV
            if coast_model_path and Path(coast_model_path).exists():
                coast_model, coast_scaler = load_coast_model(coast_model_path)
            else:
                coast_model, coast_scaler = train_or_load_coast_model(upazila_csv_for_coast, model_path=coast_model_path)

            df["predicted_dist_coast"] = df.apply(
                lambda r: predict_coast_distance(float(r["Latitude"]), float(r["Longitude"]), coast_model, coast_scaler),
                axis=1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute coastal distance: {e}")

    # Build normalized components
    out = df.copy()

    # Normalize numeric drivers
    mm = MinMaxScaler()
    out["norm_cyclone_count"] = mm.fit_transform(out[["cyclone_count"]])
    out["norm_wind"]          = mm.fit_transform(out[["avg_max_wind_kph"]])
    out["norm_damage"]        = mm.fit_transform(out[["avg_damage_usd_million"]])
    out["norm_fatalities"]    = mm.fit_transform(out[["avg_fatalities"]])

    # population_affected_percent is 0..100 → 0..1
    out["norm_pop_pct"] = clamp01(out["population_affected_percent"] / 100.0)

    # exposure_index ~ 0..1, coastal_flag 0/1
    out["norm_exposure"]   = clamp01(out["exposure_index"])
    out["norm_coastal_fl"] = clamp01(out["coastal_flag"])

    # Proximity to coast (closer = riskier). Cap by the 95th percentile to avoid long-tail.
    cap = float(np.percentile(out["predicted_dist_coast"], 95)) if out["predicted_dist_coast"].notna().any() else 350.0
    cap = max(cap, 1.0)
    out["proximity_score"] = 1.0 - (out["predicted_dist_coast"].astype(float).clip(0, cap) / cap)

    # Composite CVI (0..1), then 0..100
    out["CVI_0_1"] = (
        WEIGHTS_CYCLONE["w_cyclone_count"] * out["norm_cyclone_count"] +
        WEIGHTS_CYCLONE["w_wind"]          * out["norm_wind"] +
        WEIGHTS_CYCLONE["w_damage"]        * out["norm_damage"] +
        WEIGHTS_CYCLONE["w_fatalities"]    * out["norm_fatalities"] +
        WEIGHTS_CYCLONE["w_pop_pct"]       * out["norm_pop_pct"] +
        WEIGHTS_CYCLONE["w_exposure"]      * out["norm_exposure"] +
        WEIGHTS_CYCLONE["w_coastal_flag"]  * out["norm_coastal_fl"] +
        WEIGHTS_CYCLONE["w_proximity"]     * out["proximity_score"]
    )
    out["CVI"] = (out["CVI_0_1"] * 100).round(2)
    out["CVI_Class"] = pd.cut(out["CVI"], [0,25,50,75,100],
                              labels=["Low","Medium","High","Very High"], include_lowest=True)

    # Attach readable place
    places = _derive_place_per_location(out.rename(columns={"district":"Place"}))
    out = out.merge(places, on=["Latitude","Longitude"], how="left", suffixes=("",""))

    return out

# -------------------------
# Train spatial model on (lat, lon) → CVI
# -------------------------
def train_spatial_model(df: pd.DataFrame, n_neighbors: int = 8, model_path: str | None = "cyclone_knn.pkl"):
    X = df[["Latitude","Longitude"]].astype(float).values
    y = df["CVI"].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(Xs, y)

    if model_path:
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": knn}, f)

    return knn, scaler

def load_spatial_model(model_path: str = "cyclone_knn.pkl"):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]

def predict_cvi(lat: float, lon: float, model, scaler) -> float:
    Xq = scaler.transform(np.array([[lat, lon]], dtype=float))
    pred = float(model.predict(Xq)[0])
    return max(0.0, min(100.0, pred))

# -------------------------
# Nearest + safer (same UX as flood)
# -------------------------
def nearest_observed(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    d = haversine_km(lat, lon, df["Latitude"].values, df["Longitude"].values)
    return df.iloc[int(np.argmin(d))]

def nearest_safer(df: pd.DataFrame, base_row: pd.Series, lat: float, lon: float, k: int = 3) -> pd.DataFrame:
    """Find k nearest safer (lower CVI) unique locations, excluding the same one."""
    # Exclude the same point
    df_filtered = df[
        ~((df["Latitude"] == base_row["Latitude"]) & (df["Longitude"] == base_row["Longitude"]))
    ].copy()

    # Filter safer rows (lower CVI)
    safer = df_filtered[df_filtered["CVI"] < float(base_row["CVI"])].copy()

    # Fallback if none are safer
    if safer.empty:
        safer = df_filtered.copy()

    # Compute distances
    safer["distance_km"] = haversine_km(lat, lon, safer["Latitude"].values, safer["Longitude"].values)

    # Deduplicate by place/district (keeps nearest instance)
    if "Place" in safer.columns:
        safer = safer.sort_values(["distance_km", "CVI"]).drop_duplicates("Place", keep="first")
    elif "district" in safer.columns:
        safer = safer.sort_values(["distance_km", "CVI"]).drop_duplicates("district", keep="first")

    # Sort final list
    return safer.sort_values(["distance_km", "CVI"], ascending=[True, True]).head(k)

# -------------------------
# Plot + console summary (cyclone flavor)
# -------------------------
def _zone_label_and_color_cyclone(cvi: float) -> tuple[str, str]:
    if cvi < 25:   return "Low",    "#2ecc71"
    if cvi < 50:   return "Medium", "#f1c40f"
    if cvi < 75:   return "High",   "#e67e22"
    return "Very High", "#e74c3c"

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
      - your point (predicted CVI),
      - dataset average,
      - k nearest safer spots (with distance)
    Background shows safety zones mapped from CVI bands.
    """
    pred_cvi = predict_cvi(lat, lon, model, scaler)
    base = nearest_observed(df, lat, lon)
    recs = nearest_safer(df, base, lat, lon, k=k)

    avg_cvi = float(df["CVI"].mean())
    avg_safety = 100 - avg_cvi
    chosen_place = str(base.get("Place", base.get("district","Unknown")))
    chosen_safety = 100 - pred_cvi

    rows = [
        {"label": f"Your point ({chosen_place})", "group": "Chosen",  "CVI": float(pred_cvi), "Safety": float(chosen_safety)},
        {"label": "Dataset Average",               "group": "Average", "CVI": float(avg_cvi),  "Safety": float(avg_safety)},
    ]
    for _, r in recs.iterrows():
        rows.append({
            "label": f"{str(r.get('Place','Unknown'))}  ({r['distance_km']:.1f} km)",
            "group": "Safer",
            "CVI": float(r["CVI"]),
            "Safety": float(100 - r["CVI"]),
        })
    plot_df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="talk")

    bands = [
        (75, 100, "Low cyclone risk (CVI 0–25)", "#e8f4fd"),
        (50, 75,  "Medium (25–50)",               "#d7ecfb"),
        (25, 50,  "High (50–75)",                 "#cfe3f8"),
        (0, 25,   "Very High (75–100)",           "#c7d9f4"),
    ]

    def _bar_color(group: str) -> str:
        return {"Chosen": "#1f4e79", "Average": "#7f8c8d", "Safer": "#2e86de"}.get(group, "#95a5a6")

    safer_part = plot_df[plot_df["group"] == "Safer"].sort_values("Safety", ascending=False)
    ordered = pd.concat(
        [plot_df[plot_df["group"] == "Chosen"],
         plot_df[plot_df["group"] == "Average"],
         safer_part],
        ignore_index=True
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    for lo, hi, label, color in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.7, lw=0)

    y_pos = np.arange(len(ordered))
    colors = [_bar_color(g) for g in ordered["group"]]
    ax.barh(y_pos, ordered["Safety"].values, edgecolor="black", color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["label"].tolist())

    for i, (safety, cvi) in enumerate(zip(ordered["Safety"], ordered["CVI"])):
        ax.text(safety + 1, i, f"CVI {cvi:.1f}", va="center", fontsize=11)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Safety Index (100 − CVI) — higher is safer")
    ax.set_ylabel("")
    ax.set_title("BuildSafe • Cyclone Vulnerability — One-View Summary")

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

    zone, _ = _zone_label_and_color_cyclone(pred_cvi)
    avg_zone, _ = _zone_label_and_color_cyclone(avg_cvi)

    print("\n" + "═" * 70)
    print("🌪️ BuildSafe — Cyclone Vulnerability Summary")
    print("─" * 70)
    print(f"📍 Input:  lat {lat:.4f}, lon {lon:.4f}")
    print(f"🪧 Nearest observed place: {chosen_place}")
    print(f"📈 Predicted CVI: {pred_cvi:.2f}  |  Safety Index: {100 - pred_cvi:.2f}  |  Zone: {zone}")
    print(f"📊 Dataset Avg CVI: {avg_cvi:.2f}  |  Avg Safety: {avg_safety:.2f}  |  Zone: {avg_zone}")
    if not recs.empty:
        print("✅ Nearest safer spots:")
        for _, r in recs.iterrows():
            print(f"   • {str(r.get('Place','Unknown')):<18}  "
                  f"CVI {float(r['CVI']):>5.1f}  |  Safety {100 - float(r['CVI']):>5.1f}  "
                  f"|  {float(r['distance_km']):>6.1f} km")
    else:
        print("ℹ️  No strictly safer locations found nearby — plot shows closest alternatives.")
    if out_path:
        print(f"🖼️  Figure saved → {Path(out_path).resolve()}")
    print("═" * 70 + "\n")

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "predicted_CVI": float(round(pred_cvi, 2)),
        "predicted_Safety_Index": float(round(100 - pred_cvi, 2)),
        "nearest_observed_place": chosen_place,
        "avg_CVI": float(round(avg_cvi, 2)),
        "avg_Safety_Index": float(round(avg_safety, 2)),
        "recommendations_count": int(len(recs))
    }

# -------------------------
# Demo / usage
# -------------------------
# if __name__ == "__main__":
#     CYCLONE_CSV = "./Updated_cyclone_data.csv"
#     UPAZILA_CSV = "./bd_upazila_distance_to_coast.csv"
#
#     # Build dataset (adds predicted_dist_coast if missing) + CVI
#     df = build_cvi_dataset(CYCLONE_CSV, upazila_csv_for_coast=UPAZILA_CSV, coast_model_path="coast_knn.pkl")
#     df.to_csv("Updated_cyclone_with_CVI.csv", index=False)
#
#     # Train spatial model
#     model, scaler = train_spatial_model(df, n_neighbors=8, model_path="cyclone_knn.pkl")
#
#     # Visualize a coordinate (e.g., Bagerhat)
#     info = visualize_oneplot_for_coordinate(
#         df, lat=22.65, lon=89.7667, model=model, scaler=scaler, k=3,
#         out_path=None,  # or "outputs/cyclone_summary.png"
#         show=True
#     )
#     print(info)
