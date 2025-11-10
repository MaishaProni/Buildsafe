import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# === Load training data ===
dist_df = pd.read_csv(r"C:\Users\Maisha\OneDrive\Desktop\Buildsafe\Buildsafe\data_processing\processed_data\Updated_upazila_distance_coast.csv")
dist_df.columns = ["Latitude", "Longitude", "dist_coast", "upazila"]
dist_df = dist_df.dropna(subset=["Latitude", "Longitude", "dist_coast"])

# === Prepare training data ===
X_train = dist_df[["Latitude", "Longitude"]]
y_train = dist_df["dist_coast"]

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === Train KNN Regressor ===
knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
knn.fit(X_train_scaled, y_train)

# === Define prediction function ===
def predict_distance(lat, lon):
    """Predict approximate distance from coast using latitude and longitude."""
    coords_scaled = scaler.transform([[lat, lon]])
    prediction = knn.predict(coords_scaled)[0]
    return round(float(prediction), 2)

# === Example: Manual input ===
lat_input = float(input("Enter Latitude: "))
lon_input = float(input("Enter Longitude: "))

predicted_dist = predict_distance(lat_input, lon_input)
print(f"🌊 Predicted distance from coast ≈ {predicted_dist} km")

