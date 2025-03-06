import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"
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
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    numeric_cols = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                    'Population Density', 'Avg Income', 'Rating', 'Rent Cost']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    target_variable = 'Rating'
    features = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                'Population Density', 'Avg Income', 'Rent Cost']
    X = sm.add_constant(df[features])
    y = df[target_variable]
    model = sm.OLS(y, X).fit()
    weights = model.params.drop('const').div(model.params.drop('const').sum()).round(3)
    return weights

weights = calculate_weights()

# Streamlit App
st.title("Store Locator App")

# Input fields
city = st.text_input("Enter City")
store_type = st.text_input("Enter Store Type")

if st.button("Find Best Store"):
    data = list(collection.find({
        "City": {"$regex": city, "$options": "i"},
        "Store Type": {"$regex": store_type, "$options": "i"}
    }, {"_id": 0}))

    if not data:
        st.error(f"No data available for Store Type '{store_type}' in City {city}.")
    else:
        df = pd.DataFrame(data)
        selected_city_lower = city.lower()
        if selected_city_lower in city_bounds:
            lat_min, lat_max, lng_min, lng_max = city_bounds[selected_city_lower]
            df = df[
                (df['Latitude'].between(lat_min, lat_max)) &
                (df['Longitude'].between(lng_min, lng_max))
            ]

        if df.empty:
            st.error(f"No locations found within city boundaries for Store Type '{store_type}' in City {city}.")
        else:
            features = ['Foot Traffic', 'Avg Spending Capacity', 'Competition Density',
                       'Population Density', 'Avg Income', 'Rent Cost']
            df['Suitability Score'] = (df[features] * weights).sum(axis=1).round(3)
            best_idx = df['Suitability Score'].idxmax()
            best = df.loc[best_idx]
            lat, lng = float(best['Latitude']), float(best['Longitude'])

            # Create Interactive Map
            m = folium.Map(location=[lat, lng], zoom_start=14, tiles='cartodbpositron')
            HeatMap(
                df[['Latitude', 'Longitude', 'Suitability Score']].values.tolist(),
                radius=20,
                gradient={'0.4': 'blue', '0.6': 'yellow', '1.0': 'red'}
            ).add_to(m)
            folium.Marker(
                location=[lat, lng],
                tooltip="Best Location",
                popup=f"Suitability: {best['Suitability Score']}",
                icon=folium.Icon(color='gold', icon='star', prefix='fa')
            ).add_to(m)

            # Display Map
            folium_static(m)

            # Display Results
            st.write(f"Best Location: Latitude = {lat}, Longitude = {lng}")
            st.write(f"Suitability Score: {best['Suitability Score']}") 