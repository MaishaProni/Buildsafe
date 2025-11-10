from models.earthquakemodel import build_evi_dataset, train_spatial_model, visualize_oneplot_for_coordinate

CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_earthquake_data.csv"
df = build_evi_dataset(CSV)
model, scaler = train_spatial_model(df, n_neighbors=8)

info = visualize_oneplot_for_coordinate(
    df, lat=22.35, lon=90.30, model=model, scaler=scaler, k=3,
    #out_path="outputs/buildsafe_summary.png",  # or None to just show
    show=True
)
print(info)
from models.floodmodel import build_fvi_dataset, train_spatial_model, visualize_oneplot_for_coordinate

CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_flood_data.csv"
df = build_fvi_dataset(CSV)
model, scaler = train_spatial_model(df, n_neighbors=8)

res = visualize_oneplot_for_coordinate(
    df, lat=23.86, lon=90.00, model=model, scaler=scaler, k=3,
#out_path="outputs/flood_summary.png",  # or None to just show
    show=True
)
print(res)
from models.cyclone_model import build_cvi_dataset, train_spatial_model, visualize_oneplot_for_coordinate

CYCLONE_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_cyclone_data.csv"
UPAZILA_CSV = r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_upazila_distance_coast.csv"

    # 1️⃣ Build dataset (adds predicted coastal distance + CVI)
df = build_cvi_dataset(CYCLONE_CSV, upazila_csv_for_coast=UPAZILA_CSV)
df.to_csv("Updated_cyclone_with_CVI.csv", index=False)
print("✅ Saved 'Updated_cyclone_with_CVI.csv'")

    # 2️⃣ Train spatial KNN model
model, scaler = train_spatial_model(df, n_neighbors=8, model_path="cyclone_knn.pkl")
print("✅ Trained and saved 'cyclone_knn.pkl'")

    # 3️⃣ Predict and visualize for one location (e.g., Bagerhat)
info = visualize_oneplot_for_coordinate(
        df,
        lat=22.65, lon=89.7667,  # change as needed
        model=model,
        scaler=scaler,
        k=3,
        out_path=None,  # or e.g. "cyclone_summary.png"
        show=True
    )
print(info)
