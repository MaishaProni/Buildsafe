# app.py
# Flask backend using trained ML model for predictions

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Paths
DATA_PATH = Path.home() / "Documents" / "Building"
SOIL_CSV = DATA_PATH / "soil_topography.csv"
FLOOD_CSV = DATA_PATH / "Bangladesh_Cities_Flood_Frequency_with_Coordinates.csv"
POPULATION_CSV = DATA_PATH / "bangladesh_population_density.csv"
MODEL_PATH = DATA_PATH / "vulnerability_model.pkl"
SCALER_PATH = DATA_PATH / "scaler.pkl"
ENCODERS_PATH = DATA_PATH / "label_encoders.pkl"

# Global variables
soil_data = None
flood_data = None
population_data = None
model = None
scaler = None
encoders = None

def load_data():
    """Load all data and trained model"""
    global soil_data, flood_data, population_data, model, scaler, encoders
    
    try:
        # Load CSVs
        if SOIL_CSV.exists():
            soil_data = pd.read_csv(SOIL_CSV)
            print(f"✓ Loaded soil data: {len(soil_data)} records")
        
        if FLOOD_CSV.exists():
            flood_data = pd.read_csv(FLOOD_CSV)
            print(f"✓ Loaded flood data: {len(flood_data)} records")
        
        if POPULATION_CSV.exists():
            population_data = pd.read_csv(POPULATION_CSV)
            print(f"✓ Loaded population density: {len(population_data)} records")
        
        # Load trained model
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded trained model")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")
            print(f"  Please run: python train_vulnerability_model.py")
        
        # Load scaler
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded scaler")
        
        # Load encoders
        if ENCODERS_PATH.exists():
            with open(ENCODERS_PATH, 'rb') as f:
                encoders = pickle.load(f)
            print(f"✓ Loaded label encoders")
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance using Haversine formula"""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_nearest_locations(lat, lng, count=5):
    """Find nearest documented locations"""
    locations = []
    
    if soil_data is not None:
        for _, row in soil_data.iterrows():
            if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                distance = calculate_distance(lat, lng, row['Latitude'], row['Longitude'])
                locations.append({
                    'name': row.get('Place', 'Unknown'),
                    'lat': float(row['Latitude']),
                    'lng': float(row['Longitude']),
                    'division': row.get('Division', ''),
                    'soil': row.get('Soil_Type', 'N/A'),
                    'bearing': row.get('Bearing_Capacity', 'N/A'),
                    'erosion': row.get('Erosion_Risk', 'N/A'),
                    'suitability': row.get('Building_Suitability', 'N/A'),
                    'distance': distance
                })
    
    if flood_data is not None:
        for _, row in flood_data.iterrows():
            if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                existing = next((x for x in locations if x['name'] == row.get('Place')), None)
                if existing:
                    existing['flood'] = row.get('Flood_Frequency', 'N/A')
    
    locations.sort(key=lambda x: x['distance'])
    return locations[:count]

def predict_vulnerability(lat, lng, pop_density):
    """Use trained model to predict vulnerability score"""
    if model is None or scaler is None:
        return None
    
    # Create feature vector based on nearest location
    nearest = get_nearest_locations(lat, lng, 1)
    if not nearest:
        return None
    
    closest = nearest[0]
    
    # Map bearing to score
    bearing_map = {'Poor': 4, 'Variable (often poor)': 3.5, 'Variable': 3, 'Moderate': 2, 'Good': 1}
    bearing_score = bearing_map.get(closest['bearing'], 2)
    
    # Map erosion to score
    erosion_map = {
        'Low to Medium': 1, 'Medium': 2, 'Medium-High': 3, 'High': 4,
        'High (flash floods)': 4.5, 'High (landslides/flash)': 4.5,
        'High (hills/urban floods)': 4, 'High (tidal/sea level)': 4,
        'High (coastal)': 4, 'High (river/coastal)': 4, 'High (urban & river)': 4
    }
    erosion_score = erosion_map.get(closest['erosion'], 2)
    
    # Map flood to score
    flood_map = {
        'Low to Medium': 1, 'Medium': 2, 'Medium-High': 3, 'High': 4,
        'High (coastal/flash)': 4.5, 'High (flash/landslide)': 4.5, 'High (urban)': 3.5
    }
    flood_score = flood_map.get(closest.get('flood', 'Medium'), 2)
    
    # Normalize population density score
    max_pop = 4532  # Dhaka's density
    pop_score = (pop_density / max_pop) * 4
    
    # Create feature vector
    features = np.array([[bearing_score, erosion_score, flood_score, pop_score, lat, lng]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    vulnerability_score = model.predict(features_scaled)[0]
    
    return max(0, min(100, vulnerability_score))  # Clamp between 0-100

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    try:
        data = request.json
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))
        
        # Validate coordinates
        if not (20.5 <= lat <= 27 and 87.5 <= lng <= 93):
            return jsonify({
                'error': 'Coordinates outside Bangladesh',
                'status': 'invalid'
            }), 400
        
        # Get nearest locations
        nearest = get_nearest_locations(lat, lng, 5)
        if not nearest:
            return jsonify({
                'error': 'No data available',
                'status': 'no_data'
            }), 400
        
        closest = nearest[0]
        distance = closest['distance']
        
        # Get population density for the division
        pop_density = 1000  # Default
        if population_data is not None and 'Division' in closest:
            div_data = population_data[population_data['Division'] == closest['division']]
            if not div_data.empty:
                pop_density = div_data['Population_Density_Per_Sq_Km'].values[0]
        
        # Predict vulnerability
        vulnerability = predict_vulnerability(lat, lng, pop_density)
        
        # Determine suitability status
        if vulnerability is None:
            vulnerability = 50
        
        if distance < 2 and 'Not recommended' in str(closest.get('suitability', '')):
            status = 'not_suitable'
            message = f"Not Suitable - Too close to {closest['name']} which is not recommended"
        elif vulnerability > 70:
            status = 'high_risk'
            message = f"High Risk Area - Vulnerability Score: {vulnerability:.1f}/100"
        elif vulnerability > 50:
            status = 'caution'
            message = f"Requires Caution - Vulnerability Score: {vulnerability:.1f}/100"
        else:
            status = 'suitable'
            message = f"Potentially Suitable - Vulnerability Score: {vulnerability:.1f}/100"
        
        response = {
            'status': status,
            'message': message,
            'vulnerability_score': round(vulnerability, 2),
            'coordinates': {'lat': lat, 'lng': lng},
            'nearest_location': {
                'name': closest.get('name'),
                'distance_km': round(closest['distance'], 2),
                'division': closest.get('division', ''),
                'soil_type': closest.get('soil', 'N/A'),
                'bearing_capacity': closest.get('bearing', 'N/A'),
                'suitability': closest.get('suitability', 'N/A'),
                'flood_frequency': closest.get('flood', 'N/A'),
                'erosion_risk': closest.get('erosion', 'N/A'),
                'population_density': pop_density
            },
            'nearby_locations': [
                {
                    'name': loc.get('name'),
                    'distance_km': round(loc['distance'], 2),
                    'division': loc.get('division', '')
                }
                for loc in nearest[1:4]
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'soil_data': soil_data is not None,
        'flood_data': flood_data is not None,
        'population_data': population_data is not None
    })

if __name__ == '__main__':
    print("Loading data and model...")
    load_data()
    print("\nStarting Flask server on port 5555...")
    app.run(debug=True, port=5555, host='0.0.0.0')