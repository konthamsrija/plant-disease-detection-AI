## **🌿 Plant Detection AI**

AI-Powered Plant Identification and Health Monitoring System

**📸 Overview**

Plant Detection AI is a deep learning-based application designed to identify plant species and analyze plant health conditions (e.g., disease detection) using real-time image input. Built for both desktop and mobile platforms, this system can assist in smart farming, botany research, and environmental monitoring.

Whether you're a farmer, researcher, or developer, this project is tailored for robust, scalable plant classification and health diagnostics.

**🚀 Features**

🌱 Plant Species Identification
Identify crop types such as Tomato, Potato, Corn, and more.

🦠 Disease Detection
Detect leaf conditions like Leaf Spot, Blight, Mildew, and other common infections.

📲 Camera/Image Upload Support
Web and mobile-friendly image input via Streamlit.

📦 Lightweight Deployment with Streamlit
Run locally or deploy to the web without heavy infrastructure.

🎯 High Accuracy with Constant Learning
Continuous model improvement via fine-tuning and transfer learning.

📉 Real-Time Inference
Fast detection using OpenCV and PyTorch inference pipelines.

🔐 History Storage
Image upload and diagnostic history securely stored using Cloudinary.

📊 Crop Viability Guide
Offers insights on whether the detected crop is viable for the scanned region, based on basic geo-location and season.

🌾 Farming Guide
Provides brief guides on crop care, disease prevention, and fertilizer recommendations based on the detected species and condition.

🗺️ Geo-Tracking Integration
Capture location data using MongoDB and streamlit_js_eval.get_geolocation() for geo-tagged results.


**🛠️ Tech Stack**

Python - Core programming language
Streamlit	- Web-based UI for interactive inference
TensorFlow	- Deep learning model development and inference
PyTorch	- Alternative framework for training and inference
OpenCV - Image preprocessing and camera handling
Torchvision	- Image transformations and model utilities
Cloudinary - Image upload, hosting, and storage of history
MongoDB (pymongo) -	Database for storing geo-tagged predictions
streamlit_js_eval	- JavaScript geolocation support in Streamlit apps
NumPy	- Numerical operations on image arrays
dotenv - Secure management of API keys and environment variables
datetime - Timestamp generation for saved entries
io & os	- File I/O and system path handling




**🧠 Model Architecture**

This project uses a hybrid pipeline:

CNN-based Classifier (ResNet/MobileNet): for species classification and disease detection

Custom Dataset Support: Can be trained on any labeled plant dataset (e.g., PlantVillage, custom images)

**📦 Installation**

git clone https://github.com/ShantanuSingh08/Plant-Disease-Detection-AI

cd Plant-Disease-Detection-AI

pip install -r requirements.txt
