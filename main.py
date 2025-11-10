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
