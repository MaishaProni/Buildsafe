import streamlit as st
import pandas as pd
from models.cyclone_model import build_cvi_dataset, load_spatial_model, predict_cvi, visualize_oneplot_for_coordinate

st.title("🌪️ BuildSafe — Cyclone Vulnerability Explorer")

lat = st.number_input("Latitude", value=22.65, format="%.4f")
lon = st.number_input("Longitude", value=89.7667, format="%.4f")
show_plot = st.checkbox("Show vulnerability chart", value=True)

if st.button("Predict Vulnerability"):
    df = pd.read_csv(r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_cyclone_data.csv")
    model, scaler = load_spatial_model("cyclone_knn.pkl")
    pred = predict_cvi(lat, lon, model, scaler)
    st.success(f"Predicted CVI: {pred:.2f} | Safety Index: {100 - pred:.2f}")
    if show_plot:
        visualize_oneplot_for_coordinate(df, lat, lon, model, scaler, k=3, show=True)
