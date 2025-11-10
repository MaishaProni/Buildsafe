# app.py — Streamlit I/O wrapper (no file uploads; reads local CSVs like main.py)

import sys, os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we can import your local packages when running from anywhere
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# === Local data paths (EDIT IF NEEDED) ===
EQ_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_earthquake_data.csv"
FL_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_flood_data.csv"
CY_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_cyclone_data.csv"
UPAZILA_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_upazila_distance_to_coast.csv"

# === Imports from your models ===
from models.earthquakemodel import (
    build_evi_dataset as build_evi,
    train_spatial_model as train_eq_model,
    visualize_oneplot_for_coordinate as viz_eq,
)
from models.floodmodel import (
    build_fvi_dataset as build_fvi,
    train_spatial_model as train_fl_model,
    visualize_oneplot_for_coordinate as viz_fl,
)
from models.cyclone_model import (
    build_cvi_dataset as build_cvi,
    train_spatial_model as train_cy_model,
    visualize_oneplot_for_coordinate as viz_cy,
)

st.set_page_config(page_title="BuildSafe — Hazards", page_icon="🛰️", layout="wide")
st.title("🛰️ BuildSafe — Hazard Vulnerability Explorer")

with st.sidebar:
    st.header("Settings")
    lat = st.number_input("Latitude", value=22.6500, format="%.6f")
    lon = st.number_input("Longitude", value=89.7667, format="%.6f")
    k_neighbors = st.slider("K neighbors (spatial KNN)", 3, 20, 8, 1)
    st.caption("Uses local CSVs; no uploads needed.")

tabs = st.tabs(["🌋 Earthquake", "🌊 Flood", "🌪️ Cyclone", "🚀 Run All"])

# ---------------------------
# Helpers
# ---------------------------
def run_eq(lat: float, lon: float, k: int):
    with st.spinner("Building EVI dataset & training model…"):
        df = build_evi(EQ_CSV)                 # expects path
        model, scaler = train_eq_model(df, n_neighbors=k)
    with st.spinner("Predicting & rendering…"):
        viz_eq(df, lat=lat, lon=lon, model=model, scaler=scaler, k=3, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

def run_fl(lat: float, lon: float, k: int):
    with st.spinner("Building FVI dataset & training model…"):
        df = build_fvi(FL_CSV)                 # expects path
        model, scaler = train_fl_model(df, n_neighbors=k)
    with st.spinner("Predicting & rendering…"):
        viz_fl(df, lat=lat, lon=lon, model=model, scaler=scaler, k=3, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

def run_cy(lat: float, lon: float, k: int):
    with st.spinner("Building CVI dataset (with coast distance) & training model…"):
        # build_cvi adds predicted_dist_coast if missing
        df = build_cvi(CY_CSV, upazila_csv_for_coast=UPAZILA_CSV, coast_model_path="coast_knn.pkl")
        model, scaler = train_cy_model(df, n_neighbors=k, model_path="cyclone_knn.pkl")
    with st.spinner("Predicting & rendering…"):
        viz_cy(df, lat=lat, lon=lon, model=model, scaler=scaler, k=3, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

# ---------------------------
# EARTHQUAKE TAB
# ---------------------------
with tabs[0]:
    st.subheader("🌋 Earthquake Vulnerability")
    if st.button("Run Earthquake", type="primary"):
        try:
            run_eq(lat, lon, k_neighbors)
        except Exception as e:
            st.error(f"Earthquake run failed: {e}")

# ---------------------------
# FLOOD TAB
# ---------------------------
with tabs[1]:
    st.subheader("🌊 Flood Vulnerability")
    if st.button("Run Flood", type="primary"):
        try:
            run_fl(lat, lon, k_neighbors)
        except Exception as e:
            st.error(f"Flood run failed: {e}")

# ---------------------------
# CYCLONE TAB
# ---------------------------
with tabs[2]:
    st.subheader("🌪️ Cyclone Vulnerability")
    if st.button("Run Cyclone", type="primary"):
        try:
            run_cy(lat, lon, k_neighbors)
        except Exception as e:
            st.error(f"Cyclone run failed: {e}")

# ---------------------------
# RUN ALL TAB
# ---------------------------
with tabs[3]:
    st.subheader("🚀 Run All Models")
    if st.button("Run All (EQ + Flood + Cyclone)", type="primary"):
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Earthquake**")
            try:
                run_eq(lat, lon, k_neighbors)
            except Exception as e:
                st.error(f"EQ failed: {e}")
        with cols[1]:
            st.markdown("**Flood**")
            try:
                run_fl(lat, lon, k_neighbors)
            except Exception as e:
                st.error(f"Flood failed: {e}")
        with cols[2]:
            st.markdown("**Cyclone**")
            try:
                run_cy(lat, lon, k_neighbors)
            except Exception as e:
                st.error(f"Cyclone failed: {e}")
