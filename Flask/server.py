# Import necessary libraries
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import folium
from folium.plugins import HeatMap
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB connection
client = pymongo.MongoClient(MONGO_URI)
db = client["store_database"]
collection = db["store_locations"]

# Define realistic latitude and longitude ranges for major cities
city_bounds = {
    "pune": (18.4, 18.7, 73.7, 74.0),
    "mumbai": (18.89, 19.30, 72.75, 72.95),
    "delhi": (28.40, 28.89, 76.85, 77.30),
    "bangalore": (12.85, 13.15, 77.45, 77.75),
    "chennai": (12.85, 13.20, 80.15, 80.30),
    "kolkata": (22.45, 22.75, 88.30, 88.50),
    "hyderabad": (17.20, 17.60, 78.25, 78.65),
    "ahmedabad": (23.00, 23.20, 72.50, 72.70),
    "lucknow": (26.75, 27.00, 80.80, 81.10),
    "jaipur": (26.75, 27.05, 75.75, 76.00)
}

# Perform Regression Analysis to calculate weights
def calculate_weights():
    # Fetch all data from MongoDB
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    # Clean and convert numeric columns
    numeric_cols = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                    'Population Density', 'Avg Income', 'Rating', 'Rent Cost']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Regression Analysis
    target_variable = 'Rating'
    features = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                'Population Density', 'Avg Income', 'Rent Cost']

    X = sm.add_constant(df[features])
    y = df[target_variable]
    model = sm.OLS(y, X).fit()
    weights = model.params.drop('const').div(model.params.drop('const').sum()).round(3)

    return weights

weights = calculate_weights()

@app.route('/find-store', methods=['POST'])
def find_store():
    data = request.json
    print("Received data:", data)  # Debugging print
    
    # Use case-insensitive regex search
    selected_city = data.get('City', '').strip()
    selected_store_type = data.get('Store Type', '').strip()

    print(f"Searching for: City = {selected_city}, Store Type = {selected_store_type}")

    # Fetch data from MongoDB with case-insensitive regex search
    data = list(collection.find({
        "City": {"$regex": selected_city, "$options": "i"},
        "Store Type": {"$regex": selected_store_type, "$options": "i"}
    }, {"_id": 0}))

    print(f"Found {len(data)} matching records")

    if not data:
        return jsonify({
            "error": f"No data available for Store Type '{selected_store_type}' in City {selected_city}.",
            "details": {
                "city": selected_city,
                "store_type": selected_store_type
            }
        }), 404

    df = pd.DataFrame(data)

    # Ensure output remains within city boundaries
    selected_city_lower = selected_city.lower()
    if selected_city_lower in city_bounds:
        lat_min, lat_max, lng_min, lng_max = city_bounds[selected_city_lower]
        df = df[
            (df['Latitude'].between(lat_min, lat_max)) &
            (df['Longitude'].between(lng_min, lng_max))
        ]

    # Verify we still have data after boundary filtering
    if df.empty:
        return jsonify({
            "error": f"No locations found within city boundaries for Store Type '{selected_store_type}' in City {selected_city}."
        }), 404

    # Calculate Suitability Score
    features = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                'Population Density', 'Avg Income', 'Rent Cost']
    df['Suitability Score'] = (df[features] * weights).sum(axis=1).round(3)

    # Get Best Location
    best_idx = df['Suitability Score'].idxmax()
    best = df.loc[best_idx]
    lat, lng = float(best['Latitude']), float(best['Longitude'])

    # Create Interactive Map
    m = folium.Map(location=[lat, lng], zoom_start=14, tiles='cartodbpositron')

    # Heatmap with fixed gradient
    HeatMap(
        df[['Latitude', 'Longitude', 'Suitability Score']].values.tolist(),
        radius=20,
        gradient={'0.4': 'blue', '0.6': 'yellow', '1.0': 'red'}
    ).add_to(m)

    # Star marker for best location
    folium.Marker(
        location=[lat, lng],
        tooltip="Best Location",
        popup=f"Suitability: {best['Suitability Score']}",
        icon=folium.Icon(color='gold', icon='star', prefix='fa')
    ).add_to(m)

    # Save heatmap as HTML
    heatmap_path = f"heatmaps/{selected_city}_{selected_store_type}_heatmap.html"
    os.makedirs("heatmaps", exist_ok=True)
    m.save(heatmap_path)

    # Return results
    return jsonify({
        "latitude": lat,
        "longitude": lng,
        "suitabilityScore": best['Suitability Score'],
        "heatmap_url": f"http://localhost:5000/{heatmap_path}"
    })

@app.route('/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_file(f"heatmaps/{filename}")

@app.route('/correlation_matrix', methods=['GET'])
def correlation_matrix():
    # Fetch all data from MongoDB
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    # Generate correlation matrix
    numeric_cols = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                    'Population Density', 'Avg Income', 'Rating', 'Rent Cost']
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")

    # Save image
    img_path = "heatmaps/correlation_matrix.png"
    os.makedirs("heatmaps", exist_ok=True)
    plt.savefig(img_path)
    plt.close()

    return send_file(img_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)