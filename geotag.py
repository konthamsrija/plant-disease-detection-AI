import streamlit as st
import pymongo
import pandas as pd
import pydeck as pdk

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://shantanusingh1807:GyDJ8g8833ZZGuHz@cluster0.fa78xzn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["plantai"]
collection = db["predictions"]

# Load all documents
data = list(collection.find({}))

if not data:
    st.warning("No prediction data available.")
    st.stop()

# Create a DataFrame
df = pd.DataFrame(data)

# Convert data types
df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')

# Drop rows with missing coordinates
df.dropna(subset=["latitude", "longitude"], inplace=True)

# Display on map
st.title("📍 Disease Prediction Map")

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/streets-v12",
    initial_view_state=pdk.ViewState(
        latitude=df["latitude"].mean(),
        longitude=df["longitude"].mean(),
        zoom=4,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=50000,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_text='prediction',
            get_size=16,
            get_color=[0, 0, 0],
            get_angle=0,
            get_alignment_baseline="'bottom'",
        )
    ],
    tooltip={"text": "Prediction: {prediction}\nLat: {latitude}\nLon: {longitude}"}
))
