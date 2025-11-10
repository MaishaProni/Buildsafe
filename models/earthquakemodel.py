# eq_vulnerability_model.py
# End-to-end: build EVI dataset + train spatial model + Seaborn one-plot viz (no maps/HTML).

import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config
# -------------------------
WEIGHTS = {
    "freq_mag":   0.40,
    "distance":   0.20,
    "recency":    0.15,
    "maj_mag":    0.10,
    "hazard":     0.10,
    "exposure":   0.05,
}
HAZARD_MAP = {"very low": 0.0, "low": 0.33, "medium": 0.66, "moderate": 0.66, "high": 1.0}
PLACE_CANDIDATES = ("Place","place","district","District","City","Upazila","Thana","Name")

# -------------------------
# Utils
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in KM for arrays/Series or scalars."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def _mode_or_nan(s: pd.Series):
    s = s.dropna().astype(str).str.strip()
    return s.mode().iloc[0] if not s.mode().empty else np.nan

def _derive_place_per_location(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a 'Place' label per (lat,lon) from most frequent name-like columns."""
    keep_cols = ["Latitude","Longitude"] + [c for c in PLACE_CANDIDATES if c in raw.columns]
    if "Division" in raw.columns: keep_cols.append("Division")
    if "Tectonic_Zone" in raw.columns: keep_cols.append("Tectonic_Zone")
    tmp = raw[keep_cols].copy()
    g = tmp.groupby(["Latitude","Longitude"], as_index=False).agg(_mode_or_nan)

    place = pd.Series(index=g.index, dtype=object)
    for c in PLACE_CANDIDATES:
        if c in g.columns:
            place = place.fillna(g[c])
    if "Division" in g.columns:
        place = place.fillna(g["Division"])
    if "Tectonic_Zone" in g.columns:
        place = place.fillna(g["Tectonic_Zone"])
    place = place.fillna("Unknown")

    return pd.DataFrame({"Latitude": g["Latitude"], "Longitude": g["Longitude"], "Place": place})

# -------------------------
# Build EVI dataset
# -------------------------
def aggregate_locations(df: pd.DataFrame) -> pd.DataFrame:
    mag_cols = [c for c in ["M3_plus","M4_plus","M5_plus","M6_plus","M7_plus","M8_plus","M9_plus"] if c in df.columns]
    years = df["Year"].nunique() if "Year" in df.columns else 1

    grp = df.groupby(["Latitude", "Longitude"], as_index=False).agg({
        **{m: "sum" for m in mag_cols},
        **({"All_Earthquakes": "sum"} if "All_Earthquakes" in df.columns else {}),
        **({"Population": "mean"} if "Population" in df.columns else {}),
        **({"Approx_Distance_to_Plate_km": "mean"} if "Approx_Distance_to_Plate_km" in df.columns else {}),
        **({"Last_Major_Earthquake_Year": "max"} if "Last_Major_Earthquake_Year" in df.columns else {}),
        **({"Last_Major_Earthquake_Magnitude": "max"} if "Last_Major_Earthquake_Magnitude" in df.columns else {}),
    })

    if "Seismic_Hazard_Level" in df.columns:
        hz_modes = (
            df.assign(_hz=df["Seismic_Hazard_Level"].astype(str).str.lower().str.strip())
              .groupby(["Latitude","Longitude"])["_hz"]
              .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
              .reset_index()
        )
        grp = grp.merge(hz_modes, on=["Latitude","Longitude"], how="left")
        grp.rename(columns={"_hz": "Seismic_Hazard_Level"}, inplace=True)

    grp["years_observed"] = years
    return grp

def compute_evi(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    out = df.copy()

    # Frequency severity (weighted by magnitude, per year)
    mag_weights = {"M3_plus":0.5,"M4_plus":1.0,"M5_plus":1.75,"M6_plus":2.5,"M7_plus":3.5,"M8_plus":5.0,"M9_plus":7.0}
    present = [m for m in mag_weights if m in out.columns]
    out["freq_score_raw"] = 0.0
    for m in present:
        per_year = out[m] / out["years_observed"].replace(0, np.nan)
        out["freq_score_raw"] += per_year.fillna(0) * mag_weights[m]
    out["freq_mag"] = minmax(out["freq_score_raw"])

    # Distance to plate (closer => higher risk)
    if "Approx_Distance_to_Plate_km" in out.columns:
        inv_dist = 1 / pd.to_numeric(out["Approx_Distance_to_Plate_km"], errors="coerce").replace(0, np.nan)
        out["distance"] = minmax(inv_dist.fillna(inv_dist.max()))
    else:
        out["distance"] = 0.0

    # Recency of last major
    if "Last_Major_Earthquake_Year" in out.columns:
        yrs_since = current_year - pd.to_numeric(out["Last_Major_Earthquake_Year"], errors="coerce")
        yrs_since = yrs_since.clip(lower=0)
        out["recency"] = 1.0 - minmax(yrs_since.fillna(yrs_since.max()))
    else:
        out["recency"] = 0.0

    # Magnitude of last major
    if "Last_Major_Earthquake_Magnitude" in out.columns:
        mag = pd.to_numeric(out["Last_Major_Earthquake_Magnitude"], errors="coerce")
        out["maj_mag"] = ((mag - 4.5) / 4.0).clip(0, 1).fillna(0)
    else:
        out["maj_mag"] = 0.0

    # Hazard level label → 0..1
    if "Seismic_Hazard_Level" in out.columns:
        hz = out["Seismic_Hazard_Level"].astype(str).str.lower().str.strip()
        out["hazard"] = hz.map(HAZARD_MAP).fillna(0.33)
    else:
        out["hazard"] = 0.33

    # Exposure (population)
    if "Population" in out.columns:
        pop = pd.to_numeric(out["Population"], errors="coerce")
        out["exposure"] = minmax(np.log1p(pop.fillna(pop.median())))
    else:
        out["exposure"] = 0.0

    # Final EVI
    out["EVI_0_1"] = (
        WEIGHTS["freq_mag"] * out["freq_mag"] +
        WEIGHTS["distance"] * out["distance"] +
        WEIGHTS["recency"]  * out["recency"] +
        WEIGHTS["maj_mag"]  * out["maj_mag"] +
        WEIGHTS["hazard"]   * out["hazard"] +
        WEIGHTS["exposure"] * out["exposure"]
    ).clip(0, 1)
    out["EVI"] = (out["EVI_0_1"] * 100).round(2)
    out["EVI_Class"] = pd.cut(out["EVI"], [0,25,50,75,100], labels=["Low","Medium","High","Very High"], include_lowest=True)
    return out

def build_evi_dataset(csv_path: str, current_year: int = 2025) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    for col in ["Latitude","Longitude","Year"]:
        if col not in raw.columns:
            raise ValueError(f"Missing required column: {col}")
    agg = aggregate_locations(raw)
    scored = compute_evi(agg, current_year=current_year)
    places = _derive_place_per_location(raw)
    return scored.merge(places, on=["Latitude","Longitude"], how="left")

# -------------------------
# Train spatial model on (lat, lon) → EVI
# -------------------------
def train_spatial_model(df: pd.DataFrame, n_neighbors: int = 8, model_path: str | None = "knn_model.pkl"):
    X = df[["Latitude","Longitude"]].astype(float).values
    y = df["EVI"].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(Xs, y)

    if model_path:
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": knn}, f)

    return knn, scaler

def load_spatial_model(model_path: str = "knn_model.pkl"):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]

def predict_evi(lat: float, lon: float, model, scaler) -> float:
    Xq = scaler.transform(np.array([[lat, lon]], dtype=float))
    pred = float(model.predict(Xq)[0])
    return max(0.0, min(100.0, pred))

# -------------------------
# Nearest + helpers
# -------------------------
def nearest_observed(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    d = haversine_km(lat, lon, df["Latitude"].values, df["Longitude"].values)
    return df.iloc[int(np.argmin(d))]

def nearest_safer(df: pd.DataFrame, base_row: pd.Series, lat: float, lon: float, k: int = 3) -> pd.DataFrame:
    safer = df[df["EVI"] < float(base_row["EVI"])].copy()
    if safer.empty:
        safer = df[(df["Latitude"] != base_row["Latitude"]) | (df["Longitude"] != base_row["Longitude"])].copy()
    safer["distance_km"] = haversine_km(lat, lon, safer["Latitude"].values, safer["Longitude"].values)
    return safer.sort_values(["distance_km","EVI"], ascending=[True, True]).head(k)

# -------------------------
# Seaborn single-plot visualization + pretty console summary
# -------------------------
def _zone_label_and_color(evi: float) -> tuple[str, str]:
    if evi < 25:   return "Low",    "#2ecc71"  # green
    if evi < 50:   return "Medium", "#f1c40f"  # yellow
    if evi < 75:   return "High",   "#e67e22"  # orange
    return "Very High", "#e74c3c"              # red

def visualize_oneplot_for_coordinate(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    model,
    scaler,
    k: int = 3,
    out_path: str | None = None,
    show: bool = True
) -> dict:
    import seaborn as sns
    from matplotlib.patches import Patch

    # --- compute pieces ---
    pred_evi = predict_evi(lat, lon, model, scaler)
    base = nearest_observed(df, lat, lon)             # nearest observed row for naming
    recs = nearest_safer(df, base, lat, lon, k=k)     # k safer spots near the input
    avg_evi = float(df["EVI"].mean())
    avg_safety = 100 - avg_evi

    chosen_place = str(base.get("Place", "Unknown"))
    chosen_safety = 100 - pred_evi

    # Build plotting table: chosen, average, and k safer
    rows = []
    rows.append({"label": f"Your point ({chosen_place})", "group": "Chosen", "EVI": float(pred_evi), "Safety": float(chosen_safety)})
    rows.append({"label": "Dataset Average", "group": "Average", "EVI": float(avg_evi), "Safety": float(avg_safety)})
    for _, r in recs.iterrows():
        rows.append({
            "label": f"{str(r.get('Place','Unknown'))}  ({r['distance_km']:.1f} km)",
            "group": "Safer",
            "EVI": float(r["EVI"]),
            "Safety": float(100 - r["EVI"]),
        })
    plot_df = pd.DataFrame(rows)

    # --- style ---
    sns.set_theme(style="whitegrid", context="talk")

    # Background safety zones (we plot Safety on X, so convert EVI bands)
    bands = [
        (75, 100, "Low risk (EVI 0–25)",     "#e8f6ef"),
        (50, 75,  "Medium risk (25–50)",     "#fff6d5"),
        (25, 50,  "High risk (50–75)",       "#fde6cf"),
        (0, 25,   "Very High risk (75–100)", "#fde2e2"),
    ]

    def _bar_color(group: str) -> str:
        return {"Chosen": "#34495e", "Average": "#7f8c8d", "Safer": "#2e86de"}.get(group, "#95a5a6")

    # Order rows: chosen, average, then safer (sorted by Safety desc)
    safer_part = plot_df[plot_df["group"] == "Safer"].sort_values("Safety", ascending=False)
    ordered = pd.concat(
        [plot_df[plot_df["group"] == "Chosen"],
         plot_df[plot_df["group"] == "Average"],
         safer_part],
        ignore_index=True
    )

    # --- Plot (use matplotlib so each bar can have its own color) ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # safety bands
    for lo, hi, label, color in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.5, lw=0)

    y_pos = np.arange(len(ordered))
    colors = [_bar_color(g) for g in ordered["group"]]
    ax.barh(y_pos, ordered["Safety"].values, edgecolor="black", color=colors)

    # y labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["label"].tolist())

    # annotate EVI values on bars
    for i, (safety, evi) in enumerate(zip(ordered["Safety"], ordered["EVI"])):
        ax.text(safety + 1, i, f"EVI {evi:.1f}", va="center", fontsize=11)

    # cosmetics
    ax.set_xlim(0, 100)
    ax.set_xlabel("Safety Index (100 − EVI) — higher is safer")
    ax.set_ylabel("")
    ax.set_title("BuildSafe • Earthquake Vulnerability — One-View Summary")

    # mini legend
    handles = [
        Patch(color="#34495e", label="Your point"),
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

    # --- pretty console summary ---
    def _zone_label_and_color(evi: float) -> tuple[str, str]:
        if evi < 25:   return "Low",    "#2ecc71"
        if evi < 50:   return "Medium", "#f1c40f"
        if evi < 75:   return "High",   "#e67e22"
        return "Very High", "#e74c3c"

    chosen_zone, _ = _zone_label_and_color(pred_evi)
    avg_zone, _ = _zone_label_and_color(avg_evi)

    print("\n" + "═" * 70)
    print("🏗️  BuildSafe — Earthquake Vulnerability Summary")
    print("─" * 70)
    print(f"📍 Input:  lat {lat:.4f}, lon {lon:.4f}")
    print(f"🪧 Nearest observed place: {chosen_place}")
    print(f"📈 Predicted EVI: {pred_evi:.2f}  |  Safety Index: {100 - pred_evi:.2f}  |  Zone: {chosen_zone}")
    print(f"📊 Dataset Avg EVI: {avg_evi:.2f}  |  Avg Safety: {avg_safety:.2f}  |  Zone: {avg_zone}")
    if not recs.empty:
        print("✅ Nearest safer spots:")
        for _, r in recs.iterrows():
            rz, _ = _zone_label_and_color(float(r['EVI']))
            print(f"   • {str(r.get('Place','Unknown')):<18}  "
                  f"EVI {float(r['EVI']):>5.1f}  |  Safety {100 - float(r['EVI']):>5.1f}  "
                  f"|  {float(r['distance_km']):>6.1f} km  |  {rz}")
    else:
        print("ℹ️  No strictly safer locations found nearby — the plot shows closest alternatives.")
    if out_path:
        print(f"🖼️  Figure saved → {Path(out_path).resolve()}")
    print("═" * 70 + "\n")

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "predicted_EVI": float(round(pred_evi, 2)),
        "predicted_Safety_Index": float(round(100 - pred_evi, 2)),
        "nearest_observed_place": chosen_place,
        "avg_EVI": float(round(avg_evi, 2)),
        "avg_Safety_Index": float(round(avg_safety, 2)),
        "recommendations_count": int(len(recs))
    }
