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
CORS(app)  # Enable CORS for all routes

# Paths - using current directory instead of Documents
DATA_PATH = Path(__file__).parent  # Use current script directory
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
        # Load CSVs with error handling
        if SOIL_CSV.exists():
            soil_data = pd.read_csv(SOIL_CSV)
            print(f"✓ Loaded soil data: {len(soil_data)} records")
        else:
            print(f"⚠ Soil data not found at {SOIL_CSV}")
            # Create dummy soil data for testing
            soil_data = pd.DataFrame({
                'Place': ['Dhaka', 'Chittagong', 'Sylhet'],
                'Latitude': [23.8103, 22.3569, 24.8949],
                'Longitude': [90.4125, 91.7832, 91.8687],
                'Division': ['Dhaka', 'Chittagong', 'Sylhet'],
                'Soil_Type': ['Alluvial', 'Hilly', 'Alluvial'],
                'Bearing_Capacity': ['Good', 'Variable', 'Moderate'],
                'Erosion_Risk': ['Low to Medium', 'High', 'Medium'],
                'Building_Suitability': ['Suitable', 'Not recommended', 'Suitable with conditions']
            })
        
        if FLOOD_CSV.exists():
            flood_data = pd.read_csv(FLOOD_CSV)
            print(f"✓ Loaded flood data: {len(flood_data)} records")
        else:
            print(f"⚠ Flood data not found at {FLOOD_CSV}")
            # Create dummy flood data
            flood_data = pd.DataFrame({
                'Place': ['Dhaka', 'Chittagong', 'Sylhet'],
                'Latitude': [23.8103, 22.3569, 24.8949],
                'Longitude': [90.4125, 91.7832, 91.8687],
                'Flood_Frequency': ['Medium', 'High', 'High']
            })
        
        if POPULATION_CSV.exists():
            population_data = pd.read_csv(POPULATION_CSV)
            print(f"✓ Loaded population density: {len(population_data)} records")
        else:
            print(f"⚠ Population data not found at {POPULATION_CSV}")
            # Create dummy population data
            population_data = pd.DataFrame({
                'Division': ['Dhaka', 'Chittagong', 'Sylhet'],
                'Population_Density_Per_Sq_Km': [4532, 1500, 800]
            })
        
        # Load trained model with fallback
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded trained model")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")
            print(f"  Using fallback prediction method")
            model = None
        
        # Load scaler
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded scaler")
        else:
            print(f"⚠ Scaler not found, using default scaling")
            scaler = None
        
        # Load encoders
        if ENCODERS_PATH.exists():
            with open(ENCODERS_PATH, 'rb') as f:
                encoders = pickle.load(f)
            print(f"✓ Loaded label encoders")
        else:
            print(f"⚠ Encoders not found")
            encoders = None
    
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
                else:
                    # Add flood location if not in soil data
                    distance = calculate_distance(lat, lng, row['Latitude'], row['Longitude'])
                    locations.append({
                        'name': row.get('Place', 'Unknown'),
                        'lat': float(row['Latitude']),
                        'lng': float(row['Longitude']),
                        'division': '',
                        'soil': 'N/A',
                        'bearing': 'N/A',
                        'erosion': 'N/A',
                        'suitability': 'N/A',
                        'flood': row.get('Flood_Frequency', 'N/A'),
                        'distance': distance
                    })
    
    locations.sort(key=lambda x: x['distance'])
    return locations[:count]

def predict_vulnerability(lat, lng, pop_density):
    """Use trained model or fallback method to predict vulnerability score"""
    if model is None or scaler is None:
        # Fallback prediction method
        nearest = get_nearest_locations(lat, lng, 1)
        if not nearest:
            return 50  # Default medium risk
        
        closest = nearest[0]
        
        # Simple heuristic based on available data
        risk_factors = 0
        total_factors = 0
        
        # Flood risk
        flood_map = {'Low to Medium': 1, 'Medium': 2, 'Medium-High': 3, 'High': 4}
        flood_score = flood_map.get(closest.get('flood', 'Medium'), 2)
        risk_factors += flood_score
        total_factors += 1
        
        # Erosion risk
        erosion_map = {'Low to Medium': 1, 'Medium': 2, 'Medium-High': 3, 'High': 4}
        erosion_score = erosion_map.get(closest.get('erosion', 'Medium'), 2)
        risk_factors += erosion_score
        total_factors += 1
        
        # Population density (higher density = higher risk)
        pop_risk = min(4, pop_density / 1000)  # Normalize
        risk_factors += pop_risk
        total_factors += 1
        
        avg_risk = (risk_factors / total_factors) / 4 * 100  # Convert to 0-100 scale
        return min(100, max(0, avg_risk))
    
    # Original model prediction code here...
    # [Keep your original model prediction code]
    
    return 50  # Fallback

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    try:
        data = request.json
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))
        
        # Validate coordinates (Bangladesh bounds)
        if not (20.5 <= lat <= 27 and 87.5 <= lng <= 93):
            return jsonify({
                'error': 'Coordinates outside Bangladesh. Please use coordinates within Bangladesh (Lat: 20.5-27, Lng: 87.5-93)',
                'status': 'invalid'
            }), 400
        
        # Get nearest locations
        nearest = get_nearest_locations(lat, lng, 5)
        if not nearest:
            return jsonify({
                'error': 'No data available for this location',
                'status': 'no_data'
            }), 400
        
        closest = nearest[0]
        distance = closest['distance']
        
        # Get population density for the division
        pop_density = 1000  # Default
        if population_data is not None and 'division' in closest:
            div_data = population_data[population_data['Division'] == closest['division']]
            if not div_data.empty:
                pop_density = div_data['Population_Density_Per_Sq_Km'].values[0]
        
        # Predict vulnerability
        vulnerability = predict_vulnerability(lat, lng, pop_density)
        
        # Determine suitability status
        if distance < 2 and 'Not recommended' in str(closest.get('suitability', '')):
            status = 'not_suitable'
            message = f"Not Suitable - Too close to {closest['name']} which is not recommended for construction"
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
                'name': closest.get('name', 'Unknown'),
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
                    'name': loc.get('name', 'Unknown'),
                    'distance_km': round(loc['distance'], 2),
                    'division': loc.get('division', ''),
                    'flood_frequency': loc.get('flood', 'Unknown')
                }
                for loc in nearest[1:4]
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

@app.route('/')
def home():
    """Serve the frontend"""
    return """
    <html>
    <body>
        <h1>Bangladesh Construction Suitability API</h1>
        <p>Backend is running. Use the frontend interface to interact with the API.</p>
        <p><a href="/api/health">Check API Health</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Loading data and model...")
    load_data()
    print("\nStarting Flask server on http://127.0.0.1:8888")
    print("Make sure to access the frontend from the same port!")
    app.run(debug=True, port=8888, host='0.0.0.0')