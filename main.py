import streamlit as st
import tensorflow as tf
import numpy as np
import io
import os
import torch
import cloudinary
import torch._classes
import cloudinary.uploader
from streamlit_js_eval import get_geolocation
from torchvision import transforms
from datetime import datetime
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation
from pymongo.mongo_client import MongoClient


Mongo_URI = os.getenv("Mongo_URI")

# Create a new client and connect to the server
client = MongoClient(Mongo_URI)
db = client["plantai"]
collection = db["predictions"]
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

    
# Ensure CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "history"
os.makedirs(SAVE_DIR, exist_ok=True)
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)


load_dotenv()
# Configuration       
cloudinary.config( 
    cloud_name = os.getenv("CLOUD_NAME"),
    api_key = os.getenv("CLOUD_API_KEY"),
    api_secret = os.getenv("CLOUD_API_SECRET"),
    secure=True,
)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("AI Plant Detection")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE DETECTION", "CROP VIABILITY GUIDE", "FARMING GUIDE", "ABOUT US"])
#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
st.image(img)

loc = get_geolocation()

if loc and "coords" in loc:
    coords = loc["coords"]
    latitude = coords.get("latitude")
    longitude = coords.get("longitude")

    if latitude and longitude:
        st.success(f"You are good to use the app!! 😊 ")
    else:
        st.warning("⚠️ Location coordinates incomplete.")
else:
    st.warning("📍 Location not available. Please allow location access.")
    
#Main Page
if(app_mode=="HOME"):
        # Homepage UI
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection 🌿</h1>
        <p style='text-align: center; font-size: 18px;'>Harness the power of AI to diagnose plant diseases and ensure healthier crops.</p>
        <hr>
    """, unsafe_allow_html=True)

    # About Section
    st.markdown("""
    ### 🌱 About This App
    This application helps farmers and agricultural experts detect plant diseases with the help of AI-powered image processing. 
    Simply upload a picture of a leaf, and our model will analyze and predict potential diseases.

    ### 🔍 How It Works
    1. **Capture or Upload**: Take a clear picture of the affected plant.
    2. **Analyze**: The AI model processes the image and identifies possible diseases.
    3. **Get Results**: Receive an instant diagnosis with suggestions for treatment.

    ### 🚀 Get Started
    Use the sidebar to navigate and start detecting plant diseases!
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <hr>
        <p style='text-align: center;'>© 2025 Plant Health AI | Powered by Machine Learning & Computer Vision</p>
    """, unsafe_allow_html=True)


# CROP VIABILITY GUIDE Page 
elif app_mode == "CROP VIABILITY GUIDE":
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 CROP VIABILITY GUIDE 🌿</h1>
    """, unsafe_allow_html=True)


    cropData = [
        {"name": "Apple", "nitrogen": 20.80, "phosphorus": 134.22, "potassium": 199.89, "temperature": 22.63, "humidity": 92.33, "pH": 5.93, "rainfall": 112.65},
        {"name": "Banana", "nitrogen": 100.23, "phosphorus": 82.01, "potassium": 50.05, "temperature": 27.38, "humidity": 80.36, "pH": 5.98, "rainfall": 104.63},
        {"name": "Blackgram", "nitrogen": 40.02, "phosphorus": 67.47, "potassium": 19.24, "temperature": 29.97, "humidity": 65.12, "pH": 7.13, "rainfall": 67.88},
        {"name": "Chickpea", "nitrogen": 40.09, "phosphorus": 67.79, "potassium": 79.92, "temperature": 18.87, "humidity": 16.86, "pH": 7.34, "rainfall": 80.06},
        {"name": "Coconut", "nitrogen": 21.98, "phosphorus": 16.93, "potassium": 30.59, "temperature": 27.41, "humidity": 94.84, "pH": 5.98, "rainfall": 175.69},
        {"name": "Coffee", "nitrogen": 101.20, "phosphorus": 28.74, "potassium": 29.94, "temperature": 25.54, "humidity": 58.87, "pH": 6.81, "rainfall": 158.07},
        {"name": "Cotton", "nitrogen": 117.77, "phosphorus": 46.24, "potassium": 19.56, "temperature": 23.99, "humidity": 79.84, "pH": 6.92, "rainfall": 80.09},
        {"name": "Grapes", "nitrogen": 23.18, "phosphorus": 132.53, "potassium": 200.11, "temperature": 23.87, "humidity": 81.87, "pH": 6.25, "rainfall": 69.91},
        {"name": "Jute", "nitrogen": 78.40, "phosphorus": 46.86, "potassium": 39.99, "temperature": 24.96, "humidity": 79.64, "pH": 6.73, "rainfall": 174.79},
        {"name": "Lentil", "nitrogen": 18.77, "phosphorus": 68.36, "potassium": 19.41, "temperature": 24.51, "humidity": 64.80, "pH": 6.99, "rainfall": 45.68},
        {"name": "Maize", "nitrogen": 77.76, "phosphorus": 48.44, "potassium": 19.79, "temperature": 22.61, "humidity": 65.92, "pH": 6.26, "rainfall": 84.76},
        {"name": "Mango", "nitrogen": 20.07, "phosphorus": 27.18, "potassium": 29.92, "temperature": 31.90, "humidity": 50.05, "pH": 5.77, "rainfall": 94.99},
        {"name": "Mothbeans", "nitrogen": 21.44, "phosphorus": 48.01, "potassium": 20.23, "temperature": 28.52, "humidity": 53.16, "pH": 6.85, "rainfall": 51.22},
        {"name": "Mungbean", "nitrogen": 20.99, "phosphorus": 47.28, "potassium": 19.87, "temperature": 28.27, "humidity": 85.95, "pH": 6.74, "rainfall": 48.44},
        {"name": "Muskmelon", "nitrogen": 100.32, "phosphorus": 17.72, "potassium": 50.08, "temperature": 28.66, "humidity": 92.34, "pH": 6.36, "rainfall": 24.69},
        {"name": "Orange", "nitrogen": 19.58, "phosphorus": 16.55, "potassium": 10.01, "temperature": 22.77, "humidity": 92.50, "pH": 7.01, "rainfall": 110.41},
        {"name": "Papaya", "nitrogen": 49.88, "phosphorus": 59.05, "potassium": 50.04, "temperature": 33.72, "humidity": 92.40, "pH": 6.74, "rainfall": 142.63},
        {"name": "Pigeonpeas", "nitrogen": 20.73, "phosphorus": 67.73, "potassium": 20.29, "temperature": 27.74, "humidity": 48.06, "pH": 5.79, "rainfall": 149.46},
        {"name": "Pomegranate", "nitrogen": 18.87, "phosphorus": 18.75, "potassium": 40.21, "temperature": 21.84, "humidity": 90.13, "pH": 6.43, "rainfall": 107.53},
        {"name": "Rice", "nitrogen": 79.89, "phosphorus": 47.58, "potassium": 39.87, "temperature": 23.69, "humidity": 82.27, "pH": 6.43, "rainfall": 236.18},
        {"name": "Watermelon", "nitrogen": 99.42, "phosphorus": 17.00, "potassium": 50.22, "temperature": 25.59, "humidity": 85.16, "pH": 6.50, "rainfall": 50.79},
        {"name": "Kidneybeans", "nitrogen": 20.75, "phosphorus": 67.54, "potassium": 20.05, "temperature": 20.05, "humidity": 21.61, "pH": 5.78, "rainfall": 105.92}
    ];

    # Display Team Cards
    cols = st.columns(3)  
    for index, member in enumerate(cropData):
        with cols[index % 3]:
            st.markdown(f"**    **")
            st.markdown(f"**{member['name']}**")
            st.markdown(f"Nitrogen: {member['nitrogen']}")
            st.markdown(f"Phosphorus: {member['phosphorus']}")
            st.markdown(f"Potassium: {member['potassium']}")
            st.markdown(f"Temperature: {member['temperature']}")
            st.markdown(f"pH: {member['pH']}")
            st.markdown(f"Rainfall: {member['rainfall']}")

#
# About Us Page - Team Members Section
elif app_mode == "ABOUT US":
    st.markdown("<h1 style='text-align: center; color: white;'>Team Members</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Meet The Developers</h5>", unsafe_allow_html=True)

    # Team Member Data
    team_members = [
         {"name": "Srija Kontham", "linkedin": "#", "github": "#", "instagram": "#"},
         {"name": "Sowmya Sri Devagoni", "linkedin": "#", "github": "#", "instagram": "#"},
         {"name": "Sai Vishnu Teja Madhanambeti", "linkedin": "#", "github": "#", "instagram": "#"}, 
    ]

    # Display Team Cards
    cols = st.columns(3)  
    for index, member in enumerate(team_members):
        with cols[index % 3]:
            st.markdown(f"**    **")
            st.markdown(f"**{member['name']}**")
            st.markdown(f"**Masters | CSE**  \nCentral Michigan University")


#Prediction Page
elif(app_mode=="DISEASE DETECTION"):


    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 DISEASE DETECTION 🌿</h1>
    """, unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    # if test_image:
    #     # Create a unique filename with timestamp
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     file_path = os.path.join(SAVE_DIR, f"{timestamp}.jpg")

    #     # Open and save the image
    #     image = Image.open(test_image)
    #     image.save(file_path)


    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        if test_image:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(SAVE_DIR, f"{timestamp}_{latitude},{longitude}_.jpg")

            # Open and save the image
            image = Image.open(test_image).convert("RGB")  
            image.save(file_path)

            # Open and save the image
            image = Image.open(test_image).convert("RGB")  
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            public_id = f"predictions/{timestamp}"

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(img_bytes, public_id=public_id, resource_type="image")

            # Get the URL
            image_url = upload_result["secure_url"]
            st.success("Image uploaded to Cloudinary!")
            st.image(image_url, caption="Uploaded Image", use_column_width=True)

            
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(file_path)

        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        disease_name = class_name[result_index]

        predicted_label = class_name[result_index]

        document = {
                "_id": timestamp,
                "latitude": latitude,
                "longitude": longitude,
                "prediction": predicted_label,  
                "timestamp": timestamp
            }

        # Save to MongoDB
        try:
            collection.insert_one(document)
        except Exception as e:
            st.error(f"Error: {e}")

        # Ensure needed variables are defined above this block
        if "last_selected_label" not in st.session_state:
            st.session_state.last_selected_label = class_name[int(result_index)]

        # Dropdown for feedback (with predicted label preselected)
        selected_label = st.selectbox("Select the correct label (if wrong):", class_name, index=int(result_index))

        # If user selects something different, auto-trigger feedback
        if selected_label != st.session_state.last_selected_label:
            st.session_state.last_selected_label = selected_label

            # Save corrected image locally
            feedback_path = os.path.join(FEEDBACK_DIR, selected_label)
            os.makedirs(feedback_path, exist_ok=True)

            feedback_filename = f"WrongPrediction_{predicted_label.replace(' ', '_')}_as_{selected_label.replace(' ', '_')}_{timestamp}_{latitude},{longitude}_.jpg"
            corrected_file_path = os.path.join(feedback_path, feedback_filename)
            image.save(corrected_file_path)

            # Upload to Cloudinary
            public_id_feedback = f"feedback/WrongPrediction_{predicted_label.replace(' ', '_')}_as_{selected_label.replace(' ', '_')}_{timestamp}"
            upload_result_feedback = cloudinary.uploader.upload(
                corrected_file_path,
                public_id=public_id_feedback,
                resource_type="image"
            )

        # Disease Treatment Mapping
        treatment_dict = { 
            'Apple___Apple_scab': "A fungal disease caused by *Venturia inaequalis*, resulting in olive-green to black velvety spots on leaves, fruit, and stems. Severe infections cause defoliation and reduce fruit quality significantly.",
            
            'Apple___Black_rot': "Caused by *Botryosphaeria obtusa*, this disease produces dark, concentric rings on fruit, cankers on branches, and leaf spots. It leads to significant fruit losses if not managed promptly.",
            
            'Apple___Cedar_apple_rust': "A fungal disease (*Gymnosporangium juniperi-virginianae*) needing both apple and cedar trees to complete its life cycle. It causes bright orange or yellow spots on leaves and severe defoliation in apples.",
            
            'Apple___healthy': "No disease detected. The tree exhibits vigorous growth, lush green leaves, and blemish-free fruit without any fungal, bacterial, or viral symptoms.",
            
            'Blueberry___healthy': "No disease detected. Blueberry plants show healthy foliage, firm fruit development, and no signs of fungal infections such as mummy berry or leaf spot.",
            
            'Cherry_(including_sour)___Powdery_mildew': "A fungal disease caused by *Podosphaera clandestina*, characterized by white powdery fungal growth on young leaves, stems, and fruit. It leads to poor fruit quality and reduced yield.",
            
            'Cherry_(including_sour)___healthy': "No disease detected. Trees exhibit healthy green foliage, firm fruit, and no powdery or spotty symptoms on leaves or fruit.",
            
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Caused by *Cercospora zeae-maydis*, it appears as elongated gray or tan lesions on leaves. Severe infections reduce photosynthesis, impacting grain fill and yield.",
            
            'Corn_(maize)___Common_rust_': "A fungal disease (*Puccinia sorghi*) characterized by reddish-brown pustules on leaves. Severe infection leads to leaf yellowing, early senescence, and reduced crop yield.",
            
            'Corn_(maize)___Northern_Leaf_Blight': "Caused by *Exserohilum turcicum*, this disease creates long, elliptical gray-green lesions on leaves. Heavy infection weakens the plant and significantly decreases grain production.",
            
            'Corn_(maize)___healthy': "No disease detected. Corn plants show robust green foliage, upright growth, and lack of spotting, blighting, or pest damage.",
            
            'Grape___Black_rot': "A fungal disease caused by *Guignardia bidwellii*, producing small brown spots on leaves and circular, black shriveled fruit known as 'mummies.' It significantly reduces fruit quality and vine vigor.",
            
            'Grape___Esca_(Black_Measles)': "A complex fungal disease caused by multiple pathogens (e.g., *Phaeomoniella chlamydospora*), leading to tiger-striped leaves, berry shriveling, and vine dieback. A major threat in mature vineyards.",
            
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Caused by *Isariopsis clavispora*, this disease forms angular, brownish spots on grape leaves that merge, causing premature leaf drop and reduced vine health.",
            
            'Grape___healthy': "No disease detected. Vines show continuous, healthy growth with undamaged leaves and plump, healthy grape clusters.",
            
            'Orange___Haunglongbing_(Citrus_greening)': "A bacterial disease caused by *Candidatus Liberibacter* spp., spread by the Asian citrus psyllid. It causes yellow shoot development, lopsided fruit, and ultimately tree death. No cure exists.",
            
            'Peach___Bacterial_spot': "Caused by *Xanthomonas arboricola pv. pruni*, this bacterial disease forms dark, water-soaked lesions on leaves and fruit, leading to premature defoliation and blemished fruit.",
            
            'Peach___healthy': "No disease detected. Peaches develop without leaf spots, fruit blemishes, or canker signs, indicating optimal growing conditions and strong plant health.",
            
            'Pepper,_bell___Bacterial_spot': "Caused by *Xanthomonas campestris pv. vesicatoria*, this disease creates small, water-soaked spots on leaves and fruit that turn brown and crack, affecting pepper marketability and yield.",
            
            'Pepper,_bell___healthy': "No disease detected. Plants display lush green foliage, healthy fruit, and absence of leaf or stem lesions.",
            
            'Potato___Early_blight': "A fungal disease caused by *Alternaria solani*, showing as dark brown spots with concentric rings on older leaves. It weakens plants, reduces tuber size, and lowers yield.",
            
            'Potato___Late_blight': "A devastating disease caused by *Phytophthora infestans*. It results in rapidly spreading water-soaked lesions on leaves and stems, leading to plant collapse and tuber rot.",
            
            'Potato___healthy': "No disease detected. Potato plants show vigorous green foliage, unblemished leaves, and no signs of fungal blight or rot.",
            
            'Raspberry___healthy': "No disease detected. Raspberry canes are strong, leaves are green and full, and fruits develop without deformation or discoloration.",
            
            'Soybean___healthy': "No disease detected. Plants maintain uniform green leaves, proper pod development, and show no symptoms of rusts, rots, or blights.",
            
            'Squash___Powdery_mildew': "A fungal disease (*Podosphaera xanthii*) producing white, powdery growth on leaf surfaces. Severe infections stunt plant growth and reduce fruit production.",
            
            'Strawberry___Leaf_scorch': "A fungal disease caused by *Diplocarpon earliana*, resulting in small purple spots that enlarge and merge, leading to scorched, brittle leaves and reduced berry yield.",
            
            'Strawberry___healthy': "No disease detected. Strawberry plants exhibit strong vegetative growth, bright green leaves, and healthy, juicy fruit production.",
            
            'Tomato___Bacterial_spot': "Caused by *Xanthomonas* spp., it results in small, dark water-soaked spots on leaves, stems, and fruit. It leads to significant blemishing and reduced marketability of tomatoes.",
            
            'Tomato___Early_blight': "A fungal infection (*Alternaria solani*) causing brown leaf spots with concentric rings. The disease weakens the plant and drastically reduces tomato fruit production if unmanaged.",
            
            'Tomato___Late_blight': "Caused by *Phytophthora infestans*, it produces large, greasy-looking lesions on leaves and fruit, leading to rapid plant death and massive crop loss if untreated.",
            
            'Tomato___Leaf_Mold': "A fungal disease caused by *Passalora fulva*, manifesting as yellow spots on the top of leaves and olive-green mold underneath. It thrives in warm, humid conditions and can devastate yields.",
            
            'Tomato___Septoria_leaf_spot': "Caused by *Septoria lycopersici*, this disease forms small, circular spots with dark borders on leaves, leading to defoliation and reduced fruit production in tomatoes.",
            
            'Tomato___Spider_mites Two-spotted_spider_mite': "An infestation by *Tetranychus urticae*, leading to speckled, yellowing leaves and fine webbing. Severe infestations cause leaf drop and stunted plant growth.",
            
            'Tomato___Target_Spot': "Caused by *Corynespora cassiicola*, target spot disease produces round, tan lesions with concentric rings. It can cause premature defoliation and fruit blemishes in tomatoes.",
            
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "A viral disease transmitted by whiteflies, causing severe stunting, yellowing of young leaves, and curled leaf margins. It severely limits tomato plant growth and fruiting.",
            
            'Tomato___Tomato_mosaic_virus': "A highly contagious viral infection leading to mottled, light and dark green leaf patterns, leaf curling, and overall stunted plant growth, reducing tomato quality and yield.",
            
            'Tomato___healthy': "No disease detected. Tomato plants grow vigorously with lush, unblemished leaves and abundant, healthy fruit development."
        }

        marketplace_dict = {
            'Apple___Apple_scab': [
                "[Captan Fungicide Spray](https://a.co/d/0tsB75g)",
                "[Mancozeb Fungicide](Mancozeb Fungicide)"
            ],
            'Apple___Black_rot': [
                "[Copper Fungicide Concentrate](https://a.co/d/e1owWjC)",
                "[Botrytis Fungus Control Spray](https://a.co/d/6vCuDeL)"
            ],
            'Apple___Cedar_apple_rust': [
                "[Neem Oil Spray](https://a.co/d/6vCuDeL)",
                "[Bonide Fung-onil Fungicide](https://a.co/d/hWEmXvf)"
            ],
            'Apple___healthy': [
                "[All-purpose Organic Fertilizer](https://a.co/d/fqabwec)"
            ],
            'Blueberry___healthy': [
                "[Blueberry Plant Fertilizer](https://a.co/d/gywsB52)"
            ],
            'Cherry_(including_sour)___Powdery_mildew': [
                "[Sulfur Plant Fungicide](https://a.co/d/3rRfD9t)",
                "[Neem Oil Organic Pesticide](https://a.co/d/6vCuDeL)"
            ],
            'Cherry_(including_sour)___healthy': [
                "[Cherry Tree Fertilizer Spikes](https://a.co/d/cthjgBh)"
            ],
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
                "[Azoxystrobin Fungicide](https://a.co/d/eW5Ho4l)"
            ],
            'Corn_(maize)___Common_rust_': [
                "[Broad Spectrum Fungicide Spray](https://a.co/d/5EjTRSm)"
            ],
            'Corn_(maize)___Northern_Leaf_Blight': [
                "[Systemic Fungicide for Crops](https://a.co/d/9hmsfUZ)"
            ],
            'Corn_(maize)___healthy': [
                "[Organic Corn Fertilizer](https://a.co/d/2dbXUJy)"
            ],
            'Grape___Black_rot': [
                "[Myclobutanil Systemic Fungicide](https://a.co/d/1A4bhYD)"
            ],
            'Grape___Esca_(Black_Measles)': [
                "[Protective Vine Fungicide](https://a.co/d/fZ16PBY)"
            ],
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
                "[Copper Fungicide Spray for Grapes](https://a.co/d/5ARrqy3)"
            ],
            'Grape___healthy': [
                "[Grape Fertilizer & Growth Booster](https://a.co/d/8j6Bu0L)"
            ],
            'Orange___Haunglongbing_(Citrus_greening)': [
                "[Citrus Tree Soil & Fertilizer Kit](https://a.co/d/eJoAWY5)"
            ],
            'Peach___Bacterial_spot': [
                "[Organic Copper Fungicide](https://a.co/d/8KFGaf8)"
            ],
            'Peach___healthy': [
                "[Peach Tree Fertilizer Spikes](https://a.co/d/9HwwX3e)"
            ],
            'Pepper,_bell___Bacterial_spot': [
                "[Copper-Based Plant Spray](https://a.co/d/5ARrqy3)"
            ],
            'Pepper,_bell___healthy': [
                "[Pepper Plant Growth Booster](https://a.co/d/eaZFSDS)"
            ],
            'Potato___Early_blight': [
                "[Chlorothalonil Fungicide](https://a.co/d/379QdIT)"
            ],
            'Potato___Late_blight': [
                "[Mancozeb Systemic Fungicide](https://a.co/d/h6amrIJ)"
            ],
            'Potato___healthy': [
                "[Organic Potato Fertilizer](https://a.co/d/fQOsWOx)"
            ],
            'Raspberry___healthy': [
                "[Raspberry Plant Food](https://a.co/d/epmKWLW)"
            ],
            'Soybean___healthy': [
                "[Soybean Crop Enhancer Fertilizer](https://www.flipkart.com/cropboost-soyabean-liq-fertilizer-crop-make-helthy-plant/p/itmbaa5e2145da60?pid=SMNH8MGGYCSRSRG3&lid=LSTSMNH8MGGYCSRSRG3NE1M9T&marketplace=FLIPKART&cmpid=content_soil-manure_8965229628_gmc)"
            ],
            'Squash___Powdery_mildew': [
                "[Potassium Bicarbonate Fungicide](https://a.co/d/9g7kKyn)"
            ],
            'Strawberry___Leaf_scorch': [
                "[Copper Fungicide for Strawberries](https://a.co/d/6wWZ2sO)"
            ],
            'Strawberry___healthy': [
                "[Strawberry Plant Fertilizer](https://a.co/d/87SoXQW)"
            ],
            'Tomato___Bacterial_spot': [
                "[Copper Spray for Tomato Plants](https://a.co/d/c4U1CmO)"
            ],
            'Tomato___Early_blight': [
                "[Mancozeb Fungicide for Tomatoes](https://a.co/d/hLLIeN9)"
            ],
            'Tomato___Late_blight': [
                "[Chlorothalonil Tomato Fungicide](https://a.co/d/87RR7b2)"
            ],
            'Tomato___Leaf_Mold': [
                "[Sulfur Fungicide Dust](https://a.co/d/fQ5C8Eb)"
            ],
            'Tomato___Septoria_leaf_spot': [
                "[Broad Spectrum Fungicide](https://a.co/d/8pVKlLN)"
            ],
            'Tomato___Spider_mites Two-spotted_spider_mite': [
                "[Neem Oil Spray for Mites](https://a.co/d/4aOc6X8)"
            ],
            'Tomato___Target_Spot': [
                "[Crop Rotation & Fungicide Kit](https://a.co/d/6ItiPwv)"
            ],
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
                "[Whitefly Control Neem Spray](https://a.co/d/aYxxZRW)"
            ],
            'Tomato___Tomato_mosaic_virus': [
                "[Plant Disinfectant Solution](https://a.co/d/4aWhFIT)"
            ],
            'Tomato___healthy': [
                "[Balanced Tomato Fertilizer](https://a.co/d/57YPDU2)"
            ],
        }

                
        # Get treatment
        treatment = treatment_dict.get(disease_name, "No specific treatment found. Consult an expert.")
        st.info(f"**Suggested Treatment** {treatment}")

        # Get marketplace products
        marketplace_products = marketplace_dict.get(disease_name, ["No product suggestions available."])
        products_list = "\n".join([f"- {product}" for product in marketplace_products])
        st.markdown(f"**Marketplace Recommendations:**\n{products_list}")

        class_nameSpanish = {
            "Apple___Apple_scab": "Manzana___Costra_de_la_manzana",
            "Apple___Black_rot": "Manzana___Podredumbre_negra",
            "Apple___Cedar_apple_rust": "Manzana___Roya_del_manzano_y_el_cedro",
            "Apple___healthy": "Manzana___saludable",
            "Blueberry___healthy": "Arándano___saludable",
            "Cherry_(including_sour)___Powdery_mildew": "Cereza_(incluyendo_ácida)___Oídio",
            "Cherry_(including_sour)___healthy": "Cereza_(incluyendo_ácida)___saludable",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maíz___Mancha_foliar_por_Cercospora_Mancha_gris",
            "Corn_(maize)___Common_rust_": "Maíz___Roya_común",
            "Corn_(maize)___Northern_Leaf_Blight": "Maíz___Tizón_foliar_del_norte",
            "Corn_(maize)___healthy": "Maíz___saludable",
            "Grape___Black_rot": "Uva___Podredumbre_negra",
            "Grape___Esca_(Black_Measles)": "Uva___Esca_(Sarampión_negro)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Uva___Tizón_foliar_(Isariopsis)",
            "Grape___healthy": "Uva___saludable",
            "Orange___Haunglongbing_(Citrus_greening)": "Naranja___Huanglongbing_(Enverdecimiento_cítrico)",
            "Peach___Bacterial_spot": "Durazno___Mancha_bacteriana",
            "Peach___healthy": "Durazno___saludable",
            "Pepper,_bell___Bacterial_spot": "Pimiento_dulce___Mancha_bacteriana",
            "Pepper,_bell___healthy": "Pimiento_dulce___saludable",
            "Potato___Early_blight": "Papa___Tizón_temprano",
            "Potato___Late_blight": "Papa___Tizón_tardío",
            "Potato___healthy": "Papa___saludable",
            "Raspberry___healthy": "Frambuesa___saludable",
            "Soybean___healthy": "Soya___saludable",
            "Squash___Powdery_mildew": "Calabaza___Oídio",
            "Strawberry___Leaf_scorch": "Fresa___Chamuscado_foliar",
            "Strawberry___healthy": "Fresa___saludable",
            "Tomato___Bacterial_spot": "Tomate___Mancha_bacteriana",
            "Tomato___Early_blight": "Tomate___Tizón_temprano",
            "Tomato___Late_blight": "Tomate___Tizón_tardío",
            "Tomato___Leaf_Mold": "Tomate___Moho_foliar",
            "Tomato___Septoria_leaf_spot": "Tomate___Mancha_foliar_por_Septoria",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Tomate___Ácaros_Telaraña_(Tetranychus_urticae)",
            "Tomato___Target_Spot": "Tomate___Mancha_objetivo",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomate___Virus_del_enrollamiento_amarillo_de_la_hoja",
            "Tomato___Tomato_mosaic_virus": "Tomate___Virus_del_mosaico_del_tomate",
            "Tomato___healthy": "Tomate___saludable"
        }


        treatment_dictSpanish = {
                'Apple___Apple_scab': "Aplica fungicidas como Captan o Mancozeb. Poda y destruye las hojas infectadas.",
                'Apple___Black_rot': "Elimina frutos y ramas infectadas. Aplica fungicidas a base de cobre. Mejora la circulación del aire.",
                'Apple___Cedar_apple_rust': "Usa fungicidas antes del brote. Elimina los cedros cercanos para evitar la propagación.",
                'Apple___healthy': "No se detectó enfermedad. Mantén un riego y poda adecuados.",
                'Blueberry___healthy': "No se detectó enfermedad. Asegura un buen drenaje y fertilización equilibrada.",
                'Cherry_(including_sour)___Powdery_mildew': "Usa sprays de azufre o aceite de neem. Poda para mejorar el flujo de aire.",
                'Cherry_(including_sour)___healthy': "No se detectó enfermedad. Evita el exceso de riego y mejora la salud del suelo.",
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Aplica fungicidas como Azoxystrobin. Rota cultivos y usa variedades resistentes.",
                'Corn_(maize)___Common_rust_': "Usa variedades resistentes a la roya. Aplica fungicidas si la infección es severa.",
                'Corn_(maize)___Northern_Leaf_Blight': "Elimina hojas infectadas, mejora la circulación del aire y aplica fungicidas si es necesario.",
                'Corn_(maize)___healthy': "No se detectó enfermedad. Mantén una buena rotación de cultivos y evita el exceso de nitrógeno.",
                'Grape___Black_rot': "Poda las vides infectadas. Aplica fungicidas como Myclobutanil al inicio de la temporada.",
                'Grape___Esca_(Black_Measles)': "Elimina las vides infectadas. Mejora el drenaje y aplica fungicidas protectores.",
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Rocía con fungicidas a base de cobre. Elimina las hojas infectadas.",
                'Grape___healthy': "No se detectó enfermedad. Realiza podas regulares y monitorea enfermedades.",
                'Orange___Haunglongbing_(Citrus_greening)': "No existe cura. Elimina árboles infectados y controla insectos psílidos.",
                'Peach___Bacterial_spot': "Usa sprays de cobre a inicios de primavera. Elimina y destruye hojas infectadas.",
                'Peach___healthy': "No se detectó enfermedad. Mantén un riego e irrigación equilibrados.",
                'Pepper,_bell___Bacterial_spot': "Aplica fungicidas a base de cobre. Evita el riego por aspersión. Rota cultivos.",
                'Pepper,_bell___healthy': "No se detectó enfermedad. Mantén un riego y nutrición óptimos.",
                'Potato___Early_blight': "Aplica fungicidas como Clorotalonil. Elimina hojas infectadas. Rota cultivos.",
                'Potato___Late_blight': "Usa fungicidas con Mancozeb o Clorotalonil. Destruye inmediatamente las plantas infectadas.",
                'Potato___healthy': "No se detectó enfermedad. Asegura buen drenaje y evita el hacinamiento de plantas.",
                'Raspberry___healthy': "No se detectó enfermedad. Poda regularmente y elimina tallos débiles.",
                'Soybean___healthy': "No se detectó enfermedad. Monitorea plagas y asegúrate de una fertilidad adecuada del suelo.",
                'Squash___Powdery_mildew': "Aplica sprays de azufre o bicarbonato de potasio. Asegura buen espacio entre plantas.",
                'Strawberry___Leaf_scorch': "Usa fungicidas a base de cobre. Elimina hojas infectadas. Evita el riego por aspersión.",
                'Strawberry___healthy': "No se detectó enfermedad. Mantén un suelo sano y evita la humedad excesiva.",
                'Tomato___Bacterial_spot': "Usa sprays de cobre. Evita manipular plantas mojadas. Elimina hojas infectadas.",
                'Tomato___Early_blight': "Aplica fungicidas como Mancozeb. Usa mantillo para evitar salpicaduras del suelo.",
                'Tomato___Late_blight': "Destruye las plantas infectadas. Aplica fungicidas con Clorotalonil.",
                'Tomato___Leaf_Mold': "Mejora la ventilación. Usa fungicidas a base de cobre o azufre.",
                'Tomato___Septoria_leaf_spot': "Aplica fungicidas. Elimina las hojas inferiores infectadas.",
                'Tomato___Spider_mites Two-spotted_spider_mite': "Rocía con aceite de neem o jabón insecticida. Aumenta la humedad.",
                'Tomato___Target_Spot': "Usa fungicidas. Rota cultivos. Mejora el flujo de aire entre plantas.",
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Usa variedades resistentes. Controla moscas blancas con neem o jabón insecticida.",
                'Tomato___Tomato_mosaic_virus': "Elimina plantas infectadas. Desinfecta las herramientas con regularidad.",
                'Tomato___healthy': "No se detectó enfermedad. Asegura fertilización equilibrada y medidas preventivas."
            }


        treatment_dictHindi = {
                    "Apple___Apple_scab": "कैप्टन या मैंकोजेब जैसे फफूंदनाशकों का प्रयोग करें। संक्रमित पत्तियों को काटकर नष्ट करें।",
                    "Apple___Black_rot": "संक्रमित फलों और टहनियों को हटा दें। तांबा-आधारित फफूंदनाशकों का प्रयोग करें। हवा के संचार में सुधार करें।",
                    "Apple___Cedar_apple_rust": "कली निकलने से पहले फफूंदनाशकों का उपयोग करें। प्रसार को रोकने के लिए नज़दीकी देवदार के पेड़ों को हटाएं।",
                    "Apple___healthy": "कोई रोग नहीं पाया गया। उचित सिंचाई और छंटाई प्रथाओं का पालन करें।",
                    "Blueberry___healthy": "कोई रोग नहीं पाया गया। उचित जल निकासी और संतुलित उर्वरक का प्रयोग करें।",
                    "Cherry_(including_sour)___Powdery_mildew": "गंधक या नीम तेल स्प्रे का प्रयोग करें। हवा के संचार में सुधार के लिए छंटाई करें।",
                    "Cherry_(including_sour)___healthy": "कोई रोग नहीं पाया गया। अधिक पानी देने से बचें और अच्छी मिट्टी के स्वास्थ्य को बनाए रखें।",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "एज़ॉक्सिस्ट्रोबिन जैसे फफूंदनाशकों का प्रयोग करें। फसल चक्र अपनाएं और प्रतिरोधी किस्मों का उपयोग करें।",
                    "Corn_(maize)___Common_rust_": "जंग-रोधी किस्मों का उपयोग करें। यदि गंभीर हो, तो फफूंदनाशकों का प्रयोग करें।",
                    "Corn_(maize)___Northern_Leaf_Blight": "संक्रमित पत्तियों को हटा दें, हवा के संचार में सुधार करें, और आवश्यकतानुसार फफूंदनाशकों का उपयोग करें।",
                    "Corn_(maize)___healthy": "कोई रोग नहीं पाया गया। उचित फसल चक्र बनाए रखें और अत्यधिक नाइट्रोजन उर्वरक से बचें।",
                    "Grape___Black_rot": "संक्रमित बेलों की छंटाई करें। मौसम की शुरुआत में माइकलोबुटानिल जैसे फफूंदनाशकों का उपयोग करें।",
                    "Grape___Esca_(Black_Measles)": "संक्रमित बेलों को हटा दें। जल निकासी में सुधार करें और सुरक्षात्मक फफूंदनाशकों का प्रयोग करें।",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "तांबा-आधारित फफूंदनाशकों का छिड़काव करें। संक्रमित पत्तियों को हटा दें।",
                    "Grape___healthy": "कोई रोग नहीं पाया गया। नियमित छंटाई और रोग निगरानी बनाए रखें।",
                    "Orange___Haunglongbing_(Citrus_greening)": "कोई इलाज उपलब्ध नहीं है। संक्रमित पेड़ों को हटा दें और सिल्लिड कीड़ों को नियंत्रित करें।",
                    "Peach___Bacterial_spot": "वसंत ऋतु की शुरुआत में तांबा स्प्रे का उपयोग करें। संक्रमित पत्तियों को हटा दें और नष्ट करें।",
                    "Peach___healthy": "कोई रोग नहीं पाया गया। संतुलित उर्वरक और सिंचाई बनाए रखें।",
                    "Pepper,_bell___Bacterial_spot": "तांबा-आधारित फफूंदनाशकों का प्रयोग करें। ओवरहेड सिंचाई से बचें। फसल चक्र अपनाएं।",
                    "Pepper,_bell___healthy": "कोई रोग नहीं पाया गया। इष्टतम सिंचाई और पोषक तत्व संतुलन बनाए रखें।",
                    "Potato___Early_blight": "क्लोरोथालोनिल जैसे फफूंदनाशकों का प्रयोग करें। संक्रमित पत्तियों को हटा दें। फसल चक्र अपनाएं।",
                    "Potato___Late_blight": "मैंकोजेब या क्लोरोथालोनिल युक्त फफूंदनाशकों का प्रयोग करें। संक्रमित पौधों को तुरंत नष्ट करें।",
                    "Potato___healthy": "कोई रोग नहीं पाया गया। उचित मिट्टी जल निकासी सुनिश्चित करें और पौधों की भीड़ से बचें।",
                    "Raspberry___healthy": "कोई रोग नहीं पाया गया। नियमित रूप से छंटाई करें और कमजोर शाखाओं को हटा दें।",
                    "Soybean___healthy": "कोई रोग नहीं पाया गया। कीटों की निगरानी करें और मिट्टी की उर्वरता बनाए रखें।",
                    "Squash___Powdery_mildew": "गंधक या पोटेशियम बाइकार्बोनेट स्प्रे का प्रयोग करें। उचित वायु संचार के लिए पौधों के बीच पर्याप्त दूरी रखें।",
                    "Strawberry___Leaf_scorch": "तांबा-आधारित फफूंदनाशकों का प्रयोग करें। संक्रमित पत्तियों को हटा दें। ओवरहेड सिंचाई से बचें।",
                    "Strawberry___healthy": "कोई रोग नहीं पाया गया। स्वस्थ मिट्टी बनाए रखें और अत्यधिक नमी से बचें।",
                    "Tomato___Bacterial_spot": "तांबा-आधारित स्प्रे का प्रयोग करें। गीले पौधों को न छूएं। संक्रमित पत्तियों को हटा दें।",
                    "Tomato___Early_blight": "मैंकोजेब जैसे फफूंदनाशकों का प्रयोग करें। मिट्टी की छींटों से बचाव के लिए पौधों के चारों ओर मल्च बिछाएं।",
                    "Tomato___Late_blight": "संक्रमित पौधों को नष्ट करें। क्लोरोथालोनिल युक्त फफूंदनाशकों का प्रयोग करें।",
                    "Tomato___Leaf_Mold": "हवा के संचार में सुधार करें। तांबा या गंधक-आधारित फफूंदनाशकों का उपयोग करें।",
                    "Tomato___Septoria_leaf_spot": "फफूंदनाशकों का प्रयोग करें। संक्रमित निचली पत्तियों को हटा दें।",
                    "Tomato___Spider_mites Two-spotted_spider_mite": "नीम तेल या कीटनाशक साबुन का छिड़काव करें। आर्द्रता बढ़ाएं।",
                    "Tomato___Target_Spot": "फफूंदनाशकों का प्रयोग करें। फसल चक्र अपनाएं। पौधों के आसपास वायु संचार में सुधार करें।",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "प्रतिरोधी किस्मों का उपयोग करें। नीम तेल या कीटनाशक साबुन से सफेद मक्खियों को नियंत्रित करें।",
                    "Tomato___Tomato_mosaic_virus": "संक्रमित पौधों को हटा दें। उपकरणों को नियमित रूप से कीटाणुरहित करें।",
                    "Tomato___healthy": "कोई रोग नहीं पाया गया। संतुलित उर्वरक और रोग निवारण उपाय सुनिश्चित करें।",
            }
        
        class_nameHindi = {
                    'Apple___Apple_scab': "सेब का कवक",
                    'Apple___Black_rot': "सेब का काला सड़न.",
                    'Apple___Cedar_apple_rust': "सीडर सेब का रस्ट",
                    'Apple___healthy': "सेब स्वस्थ है",
                    'Blueberry___healthy': "ब्लूबेरी___स्वस्थ",
                    'Cherry_(including_sour)___Powdery_mildew': "चेरी (खट्टे सहित)___पाउडरी फफूंदी",
                    'Cherry_(including_sour)___healthy': "चेरी (खट्टे सहित)___स्वस्थ",
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "मक्का___पत्ते दाग ग्रे पत्ते दाग",
                    'Corn_(maize)___Common_rust_': "मकई सामान्य कवक",
                    'Corn_(maize)___Northern_Leaf_Blight': "मक्का (मकई)___उत्तरी पत्तों का जलना",
                    'Corn_(maize)___healthy': "मक्का (मकई)___स्वस्थ",
                    'Grape___Black_rot': "अंगूर___काली सड़न",
                    'Grape___Esca_(Black_Measles)': "अंगूर___एस्का_(काली_चकत्ते)",
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "अंगूर की पत्तियों का मुंहास",
                    'Grape___healthy': "अंगूर स्वस्थ",
                    'Orange___Haunglongbing_(Citrus_greening)': "संतरा___हुआंगलोंगबिंग_",
                    'Peach___Bacterial_spot': "पीच___बैक्टीरियल स्पॉट",
                    'Peach___healthy': "पीच___स्वस्थ",
                    'Pepper,_bell___Bacterial_spot': "शिमला मिर्च___बैक्टीरियल स्पॉट",
                    'Pepper,_bell___healthy': "शिमला मिर्च___स्वस्थ",
                    'Potato___Early_blight': "आलू___प्रारंभिक रोग",
                    'Potato___Late_blight': "आलू___लेट ब्लाइट",
                    'Potato___healthy': "आलू___स्वस्थ",
                    'Raspberry___healthy': "रास्पबेरी___स्वास्थ्यवर्धक",
                    'Soybean___healthy': "सोयाबीन___स्वस्थ",
                    'Squash___Powdery_mildew': "स्क्वैश___पाउडरी फफूंदी",
                    'Strawberry___Leaf_scorch': "स्ट्रॉबेरी___पत्याँ का जलना",
                    'Strawberry___healthy': "स्ट्रॉबेरी____स्वस्थ",
                    'Tomato___Bacterial_spot': "टमाटर___बैक्टीरियल स्पॉट",
                    'Tomato___Early_blight': "टमाटर___प्रारंभिक बीमारी",
                    'Tomato___Late_blight': "टमाटर___लेट ब्लाइट",
                    'Tomato___Leaf_Mold': "टमाटर___पत्ती___साँचा",
                    'Tomato___Septoria_leaf_spot': "टमाटर___सेप्टोरिया_पत्ते_पर_धब्बा",
                    'Tomato___Spider_mites Two-spotted_spider_mite': "टमाटर___मुंहजुखा",
                    'Tomato___Target_Spot': "टमाटर___टारगेट_स्पॉट",
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "टमाटर___टमाटर_पीला_पत्ता_कर्ल_वायरस",
                    'Tomato___Tomato_mosaic_virus': "टमाटर___टमाटर_मोज़ेक_वायरस",
                    'Tomato___healthy': "टमाटर स्वास्थ्य"
             }

                
        class_nameChinese = {
                    'Apple___Apple_scab': "苹果___苹果黑星病",
                    'Apple___Black_rot': "苹果___苹果黑腐病",
                    'Apple___Cedar_apple_rust': "苹果___苹果雪松锈病",
                    'Apple___healthy': "苹果___健康",

                    'Blueberry___healthy': "蓝莓___健康",

                    'Cherry_(including_sour)___Powdery_mildew': "樱桃（包括酸樱桃）___白粉病",
                    'Cherry_(including_sour)___healthy': "樱桃（包括酸樱桃）___健康",

                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "玉米___尾孢叶斑病 灰斑病",
                    'Corn_(maize)___Common_rust_': "玉米___普通锈病",
                    'Corn_(maize)___Northern_Leaf_Blight': "玉米___北方叶斑病",
                    'Corn_(maize)___healthy': "玉米___健康",

                    'Grape___Black_rot': "葡萄___黑腐病",
                    'Grape___Esca_(Black_Measles)': "葡萄___白腐病（黑痘病）",
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "葡萄___叶枯病（伊萨里奥普西斯叶斑）",
                    'Grape___healthy': "葡萄___健康",

                    'Orange___Haunglongbing_(Citrus_greening)': "橙子___黄龙病（柑橘绿化病）",

                   ' Peach___Bacterial_spot': "桃子___细菌性斑点病",
                    'Peach___healthy': "桃子___健康",

                    'Pepper,_bell___Bacterial_spot': "甜椒___细菌性斑点病",
                    'Pepper,_bell___healthy': "甜椒___健康",

                    'Potato___Early_blight': "马铃薯___早疫病",
                    'Potato___Late_blight': "马铃薯___晚疫病",
                    'Potato___healthy': "马铃薯___健康",

                    'Raspberry___healthy': "覆盆子___健康",

                    'Soybean___healthy': "大豆___健康",

                    'Squash___Powdery_mildew': "南瓜___白粉病",

                    'Strawberry___Leaf_scorch' : "草莓___叶灼病",
                    'Strawberry___healthy' : "草莓___健康",

                    'Tomato___Bacterial_spot': "番茄___细菌性斑点病",
                    'Tomato___Early_blight': "番茄___早疫病",
                    'Tomato___Late_blight': "番茄___晚疫病",
                    'Tomato___Leaf_Mold': "番茄___叶霉病",
                    'Tomato___Septoria_leaf_spot': "番茄___尾孢叶斑病",
                    'Tomato___Spider_mites Two-spotted_spider_mite': "番茄___螨虫（双斑蜘蛛螨）",
                    'Tomato___Target_Spot': "番茄___靶斑病",
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "番茄___黄化卷叶病毒",
                    'Tomato___Tomato_mosaic_virus': "番茄___番茄花叶病毒",
                    'Tomato___healthy': "番茄___健康"
                }

        treatment_dictChinese = { 
                    'Apple___Apple_scab': "使用Captan或Mancozeb等杀菌剂。修剪并销毁受感染的叶子。",
                    'Apple___Black_rot': "移除感染的果实和枝条。使用铜基杀菌剂。改善通风。",
                    'Apple___Cedar_apple_rust': "在芽萌动前使用杀菌剂。移除附近的雪松树以防传播。",
                    'Apple___healthy': "未检测到病害。保持适当浇水和修剪习惯。",
                    
                    'Blueberry___healthy': "未检测到病害。确保良好的排水和均衡施肥。",

                    'Cherry_(including_sour)___Powdery_mildew': "使用硫磺或印楝油喷雾。修剪以改善通风。",
                    'Cherry_(including_sour)___healthy': "未检测到病害。避免过度浇水并保持良好土壤健康。",

                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "使用Azoxystrobin等杀菌剂。轮作并使用抗病品种。",
                    'Corn_(maize)___Common_rust_': "使用抗锈病品种。如严重时使用杀菌剂。",
                    'Corn_(maize)___Northern_Leaf_Blight': "移除感染叶子，改善通风，如有需要使用杀菌剂。",
                    'Corn_(maize)___healthy': "未检测到病害。保持轮作并避免过量施氮。",

                    'Grape___Black_rot': "修剪感染藤蔓。季初使用Myclobutanil等杀菌剂。",
                    'Grape___Esca_(Black_Measles)': "移除感染藤蔓。改善排水并使用保护性杀菌剂。",
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "喷洒铜基杀菌剂。移除感染叶子。",
                    'Grape___healthy': "未检测到病害。保持定期修剪和病害监测。",

                    'Orange___Haunglongbing_(Citrus_greening)': "无治疗方法。移除感染树木并控制木虱传播。",

                    'Peach___Bacterial_spot': "春季早期使用铜制喷雾。移除并销毁感染叶子。",
                    'Peach___healthy': "未检测到病害。保持均衡施肥和灌溉。",

                    'Pepper,_bell___Bacterial_spot': "使用铜基杀菌剂。避免喷洒式浇水。轮作作物。",
                    'Pepper,_bell___healthy': "未检测到病害。保持适当浇水和养分平衡。",

                    'Potato___Early_blight': "使用Chlorothalonil等杀菌剂。移除感染叶子。轮作作物。",
                    'Potato___Late_blight': "使用Mancozeb或Chlorothalonil杀菌剂。立即销毁感染植株。",
                    'Potato___healthy': "未检测到病害。确保良好排水并避免植物过密。",

                    'Raspberry___healthy': "未检测到病害。定期修剪并移除弱枝。",

                    'Soybean___healthy': "未检测到病害。监控虫害并确保土壤肥力。",

                    'Squash___Powdery_mildew': "使用硫磺或碳酸氢钾喷雾。保持适当间距以增强通风。",

                    'Strawberry___Leaf_scorch': "使用铜基杀菌剂。移除感染叶子。避免喷洒式浇水。",
                    'Strawberry___healthy': "未检测到病害。保持土壤健康并避免过多水分。",

                    'Tomato___Bacterial_spot': "使用铜基喷雾。避免在植物潮湿时接触。移除感染叶子。",
                    'Tomato___Early_blight': "使用Mancozeb等杀菌剂。在植物周围铺设覆盖物防止土壤飞溅。",
                    'Tomato___Late_blight': "销毁感染植物。使用含Chlorothalonil的杀菌剂。",
                    'Tomato___Leaf_Mold': "改善通风。使用铜或硫磺类杀菌剂。",
                    'Tomato___Septoria_leaf_spot': "使用杀菌剂。移除感染的下部叶子。",
                    'Tomato___Spider_mites Two-spotted_spider_mite': "使用印楝油或杀虫皂喷洒。增加湿度。",
                    'Tomato___Target_Spot': "使用杀菌剂。轮作作物。改善植物周围通风。",
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "使用抗病品种。用印楝油或杀虫皂控制白粉虱。",
                    'Tomato___Tomato_mosaic_virus': "移除感染植株。定期消毒工具。",
                    'Tomato___healthy': "未检测到病害。保持养分平衡和预防措施。"
                }

        # language = st.selectbox("🌐 Select Language:", ["Hindi", "Spanish", "Chinese"])

        # language = st.selectbox("भाषा चुनें | Select Language:", ["English", "हिन्दी"])
        # st.markdown(f"**HINDI/हिंदी:**")

        treatmentHindi = treatment_dictHindi.get(disease_name, "कोई विशेष उपचार पहचाना नहीं गया है। कृपया किसी विशेषज्ञ से परामर्श करें।")
        disease_nameHindi = class_nameHindi.get(disease_name, "रोग XX")
        # st.success(f"**रोग:** {disease_nameHindi}")

        # st.info(f"**सुझाए गए उपचार:** {treatmentHindi}")
#
        # st.markdown(f"**SPANISH/Español:**")

        treatmentSpanish = treatment_dictSpanish.get(disease_name, "No se ha identificado un tratamiento concreto. Se recomienda consultar a un especialista.")
        disease_nameSpanish = class_nameSpanish.get(disease_name, "enfermedad XX")
        # st.success(f"**enfermedad:** {disease_nameSpanish}")

        # st.info(f"**Tratamiento sugerido:--** {treatmentSpanish}")

        # st.markdown(f"**Chinese/中文:**")

        treatmentChinese = treatment_dictChinese.get(disease_name, " 尚未确定具体治疗方法，建议咨询专业人员。")
        disease_nameChinese = class_nameChinese.get(disease_name, "疾病 XX")
        # st.success(f"**疾病:** {disease_nameChinese}")

        # st.info(f"**建议治疗:--** {treatmentChinese}")

        
        with st.expander("🌐 Hindi / हिंदी"):
            st.success(f"**रोग:** {disease_nameHindi}")
            st.info(f"**सुझाए गए उपचार:** {treatmentHindi}")

        with st.expander("🌐 Spanish / Español"):
            st.success(f"**Enfermedad:** {disease_nameSpanish}")
            st.info(f"**Tratamiento sugerido:** {treatmentSpanish}")

        with st.expander("🌐 Chinese / 中文"):
            st.success(f"**疾病:** {disease_nameChinese}")
            st.info(f"**建议治疗:** {treatmentChinese}")



# FARMING GUIDE Page 
elif(app_mode == "FARMING GUIDE"):
    st.markdown("""
         <h1 style='text-align: center; color: green;'>🌿 CROP FARMING GUIDE 🌿</h1>
    """, unsafe_allow_html=True)

    cropGuideSpanish = [
            {"name": "Guía de Cultivo de Maíz", 
                "Introduction": "El maíz (Zea mays), también conocido como elote, es un cultivo de cereal clave ampliamente cultivado por sus granos. Esta guía cubre el proceso completo para cultivar maíz desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de maíz de alta calidad (híbridas o variedades mejoradas)\n- Fertilizantes (Nitrógeno, Fósforo, Potasio)\n- Maquinaria (tractores, herramientas manuales, sembradoras)\n- Control de plagas (herbicidas, insecticidas)\n- Equipo de riego (riego por goteo o por surcos)",
                "Soil Preparation": "El maíz prospera en suelos francos bien drenados con un pH de 5.8 a 7.0. Are el suelo para mejorar la aireación y romper los terrones.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y resistentes a la sequía. Trate las semillas con fungicidas o insecticidas para protección.",
                "Field Preparation": "Nivele el campo para una distribución uniforme del agua. Optimice el espaciado entre hileras para máxima exposición a la luz solar.",
                "Planting Time": "Típicamente se planta al comienzo de la temporada de lluvias, entre abril y junio, dependiendo de la región.",
                "Spacing & Depth": "Siembre las semillas a 20-25 cm dentro de las filas y 60-75 cm entre filas, a una profundidad de 2-5 cm.",
                "Seeding Methods": "- **Siembra Directa:** Siembre las semillas manualmente o con sembradoras.",
                "Watering Requirements": "Requiere riego regular, especialmente durante la formación de estigmas y espiga. Use irrigación si la lluvia es insuficiente.",
                "Nutrient Management": "Aplique fertilizantes en dosis divididas: al sembrar, durante el crecimiento temprano y en las etapas de espigado.",
                "Weed Control": "Deshierbe manual, azadoneo o herbicidas. Primer deshierbe a los 15-20 días después de la siembra, seguido por otro a los 30-40 días.",
                "Pest & Disease Management": "Monitoree barrenadores del maíz, gusanos cogolleros y áfidos. Use pesticidas y manejo integrado de plagas (MIP).",
                "Harvesting": "Coseche cuando las mazorcas maduren y las hojas se sequen. El contenido de humedad debe ser del 20-25%. Use recolección manual o cosechadoras mecánicas.",
                "Post-Harvest Management": "Seque los granos a 13-14% de humedad. Desgrane, limpie y almacene adecuadamente.",
                "Storage Conditions": "Almacene en un lugar fresco y seco con ventilación para prevenir moho y plagas.",
                "Processing": "Si es necesario, seque y muela el maíz para uso posterior.",
                "Challenges & Solutions": "Problemas comunes: variabilidad climática, plagas y escasez de agua. Soluciones: MIP, monitoreo de humedad del suelo y variedades resilientes."
            },

            {"name": "Guía de Cultivo de Arroz", 
                "Introduction": "El arroz Oryza sativa es un cultivo alimenticio básico en muchas partes del mundo. Esta guía cubre el proceso completo de cultivo de arroz desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de alta calidad\n- Fertilizantes (Nitrógeno, Fósforo, Potasio)\n- Sistema de riego\n- Maquinaria (tractores, máquinas trasplantadoras, hoces)\n- Control de plagas (herbicidas, pesticidas)", 
                "Soil Preparation": "El arroz crece mejor en suelos arcillosos o franco-arcillosos con pH de 5.5 a 6.5. Are el suelo y nivele el campo para una distribución uniforme del agua.", 
                "Seed Selection & Treatment": "Use semillas de alto rendimiento y resistentes a plagas. Trátelas con fungicidas o insecticidas para prevenir infestaciones.", 
                "Field Preparation": "Nivele el campo y cree bordos (bordes elevados) para retener el agua.", 
                "Planting Time": "Siembre al inicio de la temporada de lluvias, generalmente de mayo a junio dependiendo de la región.", 
                "Spacing & Depth": "Para trasplante, use espaciado de 20x15 cm. Para siembra directa, siembre a 2-3 cm de profundidad.",
                "Seeding Methods": "- **Siembra Directa:** Dispersión de semillas o siembra en filas.\n- **Trasplante:** Cultive en un semillero y transfiera las plántulas después de 20-30 días.",
                "Watering Requirements": "Mantenga 5-10 cm de agua durante el crecimiento. Reduzca el agua en la etapa de maduración del grano.",
                "Nutrient Management": "Aplique fertilizantes en dosis divididas: al sembrar, durante el macollamiento y en la iniciación de la panícula.",
                "Weed Control": "Use deshierbe manual o herbicidas. Deshierbe 15-20 días después del trasplante, luego nuevamente a los 40 días.",
                "Pest & Disease Management": "Esté atento a plagas como barrenadores del tallo y saltahojas. Use pesticidas y prácticas de manejo integrado de plagas (MIP).",
                "Harvesting": "Coseche cuando los granos se vuelvan amarillo dorado y el 80-90% de los granos estén maduros. Use hoces para pequeñas granjas o cosechadoras mecánicas para mayor eficiencia.",
                "Post-Harvest Management": "Seque los granos a 14% de humedad, trille, aventado, y almacene en un lugar fresco y seco para prevenir el deterioro.",
                "Challenges & Solutions": "Los problemas comunes incluyen clima adverso, plagas y escasez de agua. Use MIP, monitoree los niveles de agua y diversifique las variedades de cultivos para mitigar riesgos."
            },
            {"name": "Guía de Cultivo de Yute",
                "Introduction": "El yute es un cultivo fibroso cultivado principalmente por sus fibras fuertes y naturales, ampliamente utilizadas en textiles y embalajes. Esta guía cubre el proceso completo para cultivar yute desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de yute certificadas de alta calidad (Corchorus olitorius o Corchorus capsularis)\n- Compost orgánico, fertilizantes de nitrógeno, fósforo y potasio\n- Herramientas manuales o tractores para la preparación del suelo\n- Herbicidas y pesticidas para el control de plagas\n- Sistema de riego para un riego controlado",
                "Soil Preparation": "El yute crece mejor en suelos francos y franco-arenosos con buen drenaje y un rango de pH de 6.0 a 7.5. Prepare el suelo arándolo y nivelándolo para romper los terrones y asegurar una buena preparación del semillero.",
                "Seed Selection & Treatment": "Elija variedades de semillas de alto rendimiento y resistentes a enfermedades. Remoje las semillas en agua durante 24 horas antes de plantar para fomentar la germinación.",
                "Field Preparation": "Limpie y nivele el campo para una distribución uniforme del agua. Cree pequeños bordes alrededor del campo si se espera inundación.",
                "Planting Time": "El yute se planta generalmente con la llegada del monzón, típicamente entre marzo y mayo.",
                "Spacing & Depth": "Siembre las semillas en filas con un espaciado de 25-30 cm entre filas. Plante las semillas a 1-2 cm de profundidad para una germinación óptima.",
                "Seeding Methods": "- **Voleo:** Disperse las semillas uniformemente sobre el campo.\n- **Siembra en Filas:** Siembre las semillas en filas, lo que facilita el deshierbe y otras actividades de manejo.",
                "Watering Requirements": "El yute requiere humedad regular; mantenga humedad adecuada, especialmente durante la fase de crecimiento temprano. Evite el encharcamiento asegurando un drenaje adecuado, particularmente después de lluvias intensas.",
                "Nutrient Management": "Aplique una dosis basal de fertilizantes de nitrógeno, fósforo y potasio al sembrar. Se puede aplicar nitrógeno adicional después del raleo, aproximadamente 20-25 días después de la siembra.",
                "Weed Control": "Realice deshierbe manual o aplique herbicidas selectivos según sea necesario, especialmente en las etapas tempranas. Lleve a cabo el primer deshierbe 15-20 días después de la siembra, seguido por otro después de 30-40 días.",
                "Pest & Disease Management": "Monitoree plagas comunes como orugas peludas del yute y áfidos. Use pesticidas o prácticas de manejo integrado de plagas (MIP) para controlar plagas y enfermedades como la pudrición del tallo y la antracnosis.",
                "Harvesting": "Coseche el yute cuando las plantas tengan 10-12 pies de altura y las hojas inferiores comiencen a amarillear, típicamente 100-120 días después de la siembra. Corte las plantas cerca de la base usando una hoz o cuchillo. Para mejor calidad de fibra, coseche antes de que las plantas comiencen a florecer.",
                "Post-Harvest Management": "Agrupe las plantas de yute cosechadas y sumérjalas en agua limpia de movimiento lento para el enriado (proceso de fermentación para aflojar las fibras). El enriado generalmente toma 10-15 días; verifique regularmente la separación de fibras.",
                "Challenges & Solutions": "Los problemas comunes incluyen disponibilidad de agua, infestaciones de plagas y enriado inadecuado. Use métodos eficientes de riego y control de plagas, y monitoree cuidadosamente los niveles de agua durante el enriado para asegurar la calidad de la fibra."
            },

            {"name": "Guía de Cultivo de Algodón",
                "Introduction": "El algodón es un importante cultivo de fibra valorado por sus fibras suaves y esponjosas utilizadas en textiles. Esta guía cubre el proceso completo para cultivar algodón desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de algodón certificadas de alta calidad (por ejemplo, algodón Bt u otras variedades resistentes a plagas)\n- Fertilizantes de nitrógeno, fósforo, potasio y micronutrientes\n- Sistema de riego por goteo o por surcos\n- Herbicidas y pesticidas para el control de plagas\n- Arados, tractores y pulverizadores para la preparación y mantenimiento del campo",
                "Soil Preparation": "El algodón crece mejor en suelos franco-arenosos bien drenados con un pH de 6.0 a 7.5. Prepare el campo con arado profundo, seguido de rastreo para romper terrones y suavizar la superficie.",
                "Seed Selection & Treatment": "Elija variedades de semillas de alto rendimiento y resistentes a plagas. Trate las semillas con fungicidas o insecticidas para protegerlas contra enfermedades transmitidas por el suelo e infestaciones tempranas de plagas.",
                "Field Preparation": "Cree surcos o camas para plantar, dependiendo del método de riego. Asegure un buen drenaje para prevenir el encharcamiento, al cual el algodón es sensible.",
                "Planting Time": "El algodón se planta típicamente en primavera, de marzo a mayo, dependiendo de la región y la temperatura.",
                "Spacing & Depth": "Siembre las semillas a 3-5 cm de profundidad, con un espaciado de 75-100 cm entre filas y 25-30 cm entre plantas.",
                "Seeding Methods": "- **Siembra Directa:** Siembre las semillas directamente en surcos o camas preparadas usando sembradoras o a mano.",
                "Watering Requirements": "El algodón requiere humedad constante, especialmente durante las etapas de floración y formación de cápsulas. Use riego por goteo o por surcos para mantener adecuada humedad del suelo, particularmente durante períodos secos.",
                "Nutrient Management": "Aplique fertilizante basal con fósforo y potasio al sembrar. Aplique nitrógeno en dosis divididas: un tercio al sembrar, un tercio durante el crecimiento vegetativo y un tercio en la floración.",
                "Weed Control": "Use deshierbe manual, azadoneo o herbicidas para controlar malezas, particularmente durante las etapas tempranas de crecimiento. Realice el deshierbe aproximadamente 20-30 días después de la siembra y nuevamente si es necesario a los 45 días.",
                "Pest & Disease Management": "Monitoree plagas comunes como gusanos de la cápsula, áfidos y moscas blancas. Use prácticas de manejo integrado de plagas (MIP), incluyendo controles biológicos, para minimizar el uso de pesticidas.",
                "Harvesting": "Coseche el algodón cuando las cápsulas estén completamente abiertas y las fibras esponjosas, típicamente 150-180 días después de la siembra. La cosecha manual implica recoger cápsulas maduras a mano, mientras que las granjas grandes utilizan máquinas recolectoras de algodón.",
                "Post-Harvest Management": "Permita que el algodón cosechado se seque en un área sombreada y ventilada. Limpie y desgrane el algodón para separar las semillas de la fibra. Almacene las fibras de algodón en un lugar seco y bien ventilado para evitar daños relacionados con la humedad.",
                "Challenges & Solutions": "Los problemas comunes incluyen infestaciones de plagas, disponibilidad de agua y agotamiento de nutrientes del suelo. Use variedades resistentes a la sequía, implemente riego eficiente y siga prácticas de MIP para manejar plagas."
            },
            {"name": "Guía de Cultivo de Coco",
                "Introduction": "La palma de coco (Cocos nucifera) se cultiva por su fruto, que proporciona aceite, leche y fibra. Esta guía cubre los pasos clave desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Plántulas de coco de alta calidad (variedades enanas o altas)\n- Estiércol orgánico, fertilizantes NPK\n- Riego por goteo o por cuenca\n- Pesticidas o agentes de biocontrol\n- Herramientas manuales o equipo mecánico",
                "Soil Preparation": "Los cocos prosperan en suelos franco-arenosos bien drenados con pH 5.5-7.5. Cave hoyos de 1 x 1 x 1 m, rellene con tierra, compost y abono orgánico para un fuerte crecimiento de raíces.",
                "Seed Selection & Treatment": "Use plántulas resistentes a enfermedades y de alto rendimiento. Las variedades enanas facilitan la cosecha, mientras que las variedades altas son resistentes a la sequía.",
                "Field Preparation": "Limpie malezas y escombros, asegure un drenaje adecuado y espacie los hoyos según las necesidades de la variedad.",
                "Planting Time": "Mejor plantado al inicio de la temporada de lluvias para reducir las necesidades de riego; se puede plantar durante todo el año con riego.",
                "Spacing & Depth": "Variedades altas: 7.5-9m de separación; Enanas: 6.5-7m. Asegúrese de que las raíces estén bien cubiertas.",
                "Seeding Methods": "Coloque las plántulas en hoyos con el cuello justo por encima del nivel del suelo.",
                "Watering Requirements": "Riegue regularmente durante los primeros tres años. Los árboles maduros son resistentes a la sequía pero se benefician del riego constante.",
                "Nutrient Management": "Aplique fertilizantes balanceados tres veces al año con micronutrientes como magnesio y boro. Agregue abono orgánico anualmente.",
                "Weed Control": "Deshierbe regularmente, especialmente en crecimiento temprano. El acolchado ayuda a retener la humedad y suprimir las malezas.",
                "Pest & Disease Management": "Controle plagas como escarabajos rinocerontes y picudos rojos de la palma usando pesticidas o biocontroles. Maneje la marchitez de la raíz y la pudrición del cogollo con fungicidas y poda.",
                "Harvesting": "Los cocos maduros (12 meses después de la floración) se vuelven marrones. Coseche cada 45-60 días usando herramientas para trepar o elevadores mecánicos.",
                "Post-Harvest Management": "Almacene en un área seca y ventilada. Procese la copra mediante secado al sol o secado mecánico. Empaque los cocos secos de manera segura para el transporte.",
                "Challenges & Solutions": "La sequía, las plagas y el agotamiento del suelo pueden manejarse con riego por goteo, manejo de plagas y enmiendas orgánicas del suelo."
            },

            {"name": "Guía de Cultivo de Garbanzos",
                "Introduction": "El garbanzo (Cicer arietinum) es una legumbre popular cultivada por sus semillas ricas en proteínas, ampliamente utilizadas en la producción de alimentos. Esta guía cubre el proceso completo para cultivar garbanzos desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de garbanzo de alta calidad y resistentes a enfermedades (tipos desi o kabuli)\n- Fertilizantes a base de fósforo; nitrógeno mínimo\n- Riego por goteo o aspersión\n- Herbicidas y pesticidas\n- Arados, tractores y pulverizadores",
                "Soil Preparation": "Los garbanzos crecen mejor en suelos francos bien drenados con un pH de 6.0-7.5. Are y rastrille el campo para una buena penetración de raíces.",
                "Seed Selection & Treatment": "Elija semillas de alto rendimiento y resistentes a enfermedades. Trate con bacterias rhizobium para fijación de nitrógeno y fungicidas para prevenir enfermedades.",
                "Field Preparation": "Limpie malezas y nivele el campo. Espacie las filas para permitir circulación de aire y reducir el riesgo de enfermedades.",
                "Planting Time": "Mejor plantado en estaciones secas y frescas, típicamente octubre-noviembre.",
                "Spacing & Depth": "Espacie las plantas 30-40 cm entre sí en filas separadas 45-60 cm. Siembre las semillas a 5-8 cm de profundidad según la humedad del suelo.",
                "Seeding Methods": "Siembra directa usando sembradoras o plantación manual.",
                "Watering Requirements": "Los garbanzos requieren riego mínimo pero se benefician del riego durante la floración y el llenado de vainas. Evite el encharcamiento.",
                "Nutrient Management": "Aplique fósforo al plantar. Use potasio y micronutrientes según sea necesario basado en pruebas de suelo.",
                "Weed Control": "Deshierbe temprano y regularmente, ya sea manualmente o con herbicidas. Primer deshierbe a los 20-30 días, segundo a los 45-50 días si es necesario.",
                "Pest & Disease Management": "Monitoree plagas como barrenadores de vainas y áfidos. Use manejo integrado de plagas (MIP) y biopesticidas según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Proteja de plagas, mantenga humedad moderada.\n- Etapa vegetativa: Mantenga los niveles de fósforo.\n- Floración y llenado de vainas: Asegure humedad adecuada para un rendimiento óptimo.",
                "Harvesting": "Los garbanzos maduran en 3-4 meses. Coseche cuando las plantas se amarillen y las vainas se sequen. Corte a mano para pequeñas granjas; use cosechadoras combinadas para cultivo a gran escala.",
                "Post-Harvest Management": "Seque las semillas al sol para reducir la humedad, trille y limpie antes del almacenamiento o venta.",
                "Storage Conditions": "Almacene en lugares secos y frescos con ventilación para prevenir infestaciones de insectos y deterioro.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen plagas, enfermedades, estrés hídrico y deficiencias de nutrientes. Use MIP, variedades resistentes y pruebas de suelo para mitigar riesgos."
            },

            {"name": "Guía de Cultivo de Guandú",
                "Introduction": "El guandú (Cajanus cajan) es una legumbre resistente a la sequía valorada por su alto contenido de proteínas y uso en varios platos. Esta guía cubre el proceso completo para cultivar guandú desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de guandú de alta calidad y resistentes a enfermedades (variedades de maduración temprana, media o tardía)\n- Fertilizantes de nitrógeno, fósforo y potasio; se necesita mínimo nitrógeno\n- Equipo de riego por goteo o surcos\n- Herbicidas y pesticidas específicos para plagas del guandú\n- Herramientas manuales o tractores para preparación del suelo, siembra y deshierbe",
                "Soil Preparation": "El guandú crece mejor en suelos franco-arenosos a franco-arcillosos bien drenados con un pH de 6.0-7.5. Are y rastrille el campo para crear un lecho de siembra fino.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y resistentes a enfermedades adecuadas para su región. Trate las semillas con fungicidas para prevenir enfermedades transmitidas por semillas.",
                "Field Preparation": "Limpie el campo de malezas y escombros, asegurando un buen drenaje.",
                "Planting Time": "Típicamente plantado al comienzo de la temporada de lluvias o durante la estación seca en regiones subtropicales.",
                "Spacing & Depth": "Espacie las plantas 30-40 cm entre sí en filas separadas 60-75 cm. Siembre las semillas a 3-5 cm de profundidad, dependiendo de la humedad y textura del suelo.",
                "Seeding Methods": "Siembra directa usando sembradoras o plantación manual.",
                "Watering Requirements": "El guandú es resistente a la sequía pero requiere humedad adecuada durante la floración y desarrollo de vainas. El riego puede ser necesario, especialmente en los primeros 60 días.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar y aplique nitrógeno en cobertura si es necesario. Las enmiendas orgánicas pueden mejorar la fertilidad del suelo.",
                "Weed Control": "Controle malezas durante las primeras etapas de crecimiento usando deshierbe manual o herbicidas. El acolchado puede ayudar a suprimir malezas y retener la humedad del suelo.",
                "Pest & Disease Management": "Monitoree plagas como barrenadores de vainas, áfidos y moscas blancas. Implemente estrategias de manejo integrado de plagas (MIP), incluyendo controles biológicos y pesticidas químicos según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plántulas jóvenes de plagas y mantenga la humedad del suelo.\n- Etapa vegetativa: Asegure nutrientes adecuados para un crecimiento fuerte.\n- Floración y llenado de vainas: Mantenga humedad constante para maximizar el rendimiento y la calidad de la semilla.",
                "Harvesting": "El guandú madura en 4-6 meses. Coseche cuando las vainas estén maduras y secas. Corte a mano para pequeñas granjas o use cosechadoras combinadas para cultivo a gran escala.",
                "Post-Harvest Management": "Permita que las plantas cosechadas se sequen al sol antes de trillar para reducir el contenido de humedad de la semilla.",
                "Storage Conditions": "Almacene el guandú en un área seca, fresca y bien ventilada para prevenir deterioro e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas o contenedores transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen infestaciones de plagas, enfermedades, estrés hídrico y deficiencias de nutrientes. Use variedades resistentes a enfermedades, practique rotación de cultivos y aplique estrategias de MIP para manejar riesgos."
            }, 
            {"name": "Guía de Cultivo de Frijol Moth",
                "Introduction": "Los frijoles moth (Vigna aconitifolia) son una legumbre resistente a la sequía comúnmente cultivada en regiones áridas. Son valorados por su alto contenido proteico y aplicaciones culinarias. Esta guía cubre el proceso completo para cultivar frijoles moth desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de frijol moth de alta calidad y resistentes a enfermedades\n- Fertilizantes de fósforo y potasio; nitrógeno mínimo\n- Riego por goteo o por surcos\n- Herbicidas y pesticidas\n- Herramientas manuales o tractores",
                "Soil Preparation": "Los frijoles moth prosperan en suelos arenosos francos o arcillosos con buen drenaje y un pH de 6.0-8.0. Prepare el campo arando y rastrillando para obtener un lecho de siembra fino.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y tolerantes a la sequía. Trate las semillas con fungicidas o insecticidas para prevenir enfermedades transmitidas por semillas.",
                "Field Preparation": "Limpie el campo de malezas y escombros para asegurar un buen contacto entre la semilla y el suelo.",
                "Planting Time": "Típicamente sembrado al inicio de la temporada de monzones, entre junio y julio.",
                "Spacing & Depth": "Espacie las plantas de 30-45 cm en filas separadas por 60-75 cm. Siembre las semillas a 3-5 cm de profundidad según la humedad del suelo.",
                "Seeding Methods": "Siembra directa utilizando sembradoras o plantación manual.",
                "Watering Requirements": "Los frijoles moth son resistentes a la sequía pero se benefician de una humedad constante durante la floración y el desarrollo de vainas. Riegue si la lluvia es insuficiente.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar. Use nitrógeno solo si las pruebas de suelo indican una deficiencia. Las enmiendas orgánicas mejoran la fertilidad del suelo.",
                "Weed Control": "Controle las malezas temprano con deshierbe manual o herbicidas. El acolchado ayuda a suprimir las malezas y retener la humedad del suelo.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, barrenadores de vainas y saltahojas. Utilice estrategias de manejo integrado de plagas (MIP) según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Mantenga humedad moderada y proteja de plagas.\n- Etapa vegetativa: Asegure nutrientes adecuados.\n- Floración y llenado de vainas: Mantenga la humedad para un rendimiento óptimo.",
                "Harvesting": "Coseche cuando las vainas maduren y se sequen, típicamente 90-120 días después de la siembra. Cosecha manual para pequeñas granjas; use cosechadoras combinadas para operaciones a gran escala.",
                "Post-Harvest Management": "Seque las plantas al sol antes de la trilla para reducir el contenido de humedad.",
                "Storage Conditions": "Almacene en lugares secos y frescos con ventilación para prevenir el deterioro e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen plagas, enfermedades y condiciones climáticas adversas. Utilice variedades resistentes a la sequía, prácticas de MIP y manejo adecuado del suelo para mitigar riesgos."
                },

            {"name": "Guía de Cultivo de Frijol Mungo",
                "Introduction": "Los frijoles mungo (Vigna radiata) son pequeñas legumbres verdes altamente valoradas por su contenido nutricional y versatilidad culinaria. Esta guía cubre el proceso completo para cultivar frijoles mungo desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de frijol mungo de alta calidad y resistentes a enfermedades\n- Fertilizantes de nitrógeno, fósforo y potasio (se necesita mínimo nitrógeno)\n- Riego por goteo o por surcos\n- Herbicidas y pesticidas\n- Herramientas manuales o tractores",
                "Soil Preparation": "Los frijoles mungo prefieren suelos franco-arenosos a francos con buen drenaje y un pH de 6.0-7.5. Prepare el campo arando y rastrillando para lograr un lecho de siembra fino.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y resistentes a enfermedades adecuadas para su clima. Trate las semillas con fungicidas para proteger contra enfermedades transmitidas por el suelo.",
                "Field Preparation": "Limpie el campo de malezas y escombros para asegurar un buen contacto entre la semilla y el suelo.",
                "Planting Time": "Típicamente sembrado al comienzo de la temporada de lluvias o en condiciones cálidas y secas entre abril y junio.",
                "Spacing & Depth": "Espacie las plantas de 30-40 cm en filas separadas por 45-60 cm. Siembre las semillas a 2-4 cm de profundidad según la humedad del suelo.",
                "Seeding Methods": "Siembra directa utilizando sembradoras o plantación manual.",
                "Watering Requirements": "Los frijoles mungo requieren humedad adecuada, particularmente durante la germinación y floración. Riegue si la lluvia es insuficiente, asegurándose de no regar en exceso para prevenir la pudrición de raíces.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar. Se puede aplicar nitrógeno adicional si es necesario, pero generalmente, la fijación natural es suficiente. Incorpore materia orgánica para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas temprano mediante deshierbe manual o herbicidas. El acolchado ayuda a suprimir las malezas y conservar la humedad del suelo.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, escarabajos y trips. Utilice estrategias de manejo integrado de plagas (MIP) según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plántulas jóvenes de plagas y mantenga la humedad adecuada.\n- Etapa vegetativa: Asegure nutrientes suficientes para un crecimiento fuerte.\n- Floración y llenado de vainas: Mantenga la humedad para un rendimiento y calidad óptimos.",
                "Harvesting": "Coseche cuando las vainas maduren y se sequen, típicamente 60-90 días después de la siembra. Cosecha manual para pequeñas granjas; use cosechadoras combinadas para operaciones a gran escala.",
                "Post-Harvest Management": "Seque las plantas al sol antes de la trilla para reducir el contenido de humedad.",
                "Storage Conditions": "Almacene en lugares secos y frescos con ventilación para prevenir el deterioro e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen plagas, enfermedades y condiciones climáticas adversas. Utilice variedades resistentes a enfermedades, prácticas de MIP y manejo adecuado del suelo y agua para mitigar riesgos."
                },

            {"name": "Guía de Cultivo de Frijol Negro",
                "Introduction": "El frijol negro (Vigna mungo) es una legumbre altamente nutritiva valorada por su alto contenido de proteínas y es ampliamente utilizado en varios platos culinarios. Esta guía cubre el proceso completo para cultivar frijol negro desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de frijol negro de alta calidad y resistentes a enfermedades\n- Fertilizantes de fósforo y potasio (se necesita mínimo nitrógeno)\n- Riego por goteo o por surcos\n- Herbicidas y pesticidas\n- Herramientas manuales o tractores",
                "Soil Preparation": "El frijol negro prefiere suelos franco-arenosos a franco-arcillosos con buen drenaje y un pH de 6.0-7.5. Prepare el campo arando y rastrillando para crear un lecho de siembra fino.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y resistentes a enfermedades adecuadas para su clima. Trate las semillas con fungicidas o insecticidas para proteger contra enfermedades transmitidas por el suelo.",
                "Field Preparation": "Limpie el campo de malezas y escombros para asegurar un buen contacto entre la semilla y el suelo.",
                "Planting Time": "Típicamente sembrado al comienzo de la temporada de monzones o durante condiciones cálidas y secas entre junio y julio.",
                "Spacing & Depth": "Espacie las plantas de 30-45 cm en filas separadas por 60-75 cm. Siembre las semillas a 3-5 cm de profundidad según la humedad del suelo.",
                "Seeding Methods": "Siembra directa utilizando sembradoras o plantación manual.",
                "Watering Requirements": "El frijol negro requiere humedad adecuada, particularmente durante la germinación y floración. Riegue si la lluvia es insuficiente, asegurándose de no regar en exceso para prevenir la pudrición de raíces.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar. Generalmente no es necesario nitrógeno adicional debido a la fijación de nitrógeno. Incorpore materia orgánica para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas temprano mediante deshierbe manual o herbicidas. El acolchado ayuda a suprimir las malezas y conservar la humedad del suelo.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, barrenadores de vainas y trips. Utilice estrategias de manejo integrado de plagas (MIP) según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plántulas jóvenes de plagas y mantenga la humedad adecuada.\n- Etapa vegetativa: Asegure nutrientes suficientes para un crecimiento fuerte.\n- Floración y llenado de vainas: Mantenga la humedad para un rendimiento y calidad óptimos.",
                "Harvesting": "Coseche cuando las vainas maduren y se sequen, típicamente 60-90 días después de la siembra. Cosecha manual para pequeñas granjas; use cosechadoras combinadas para operaciones a gran escala.",
                "Post-Harvest Management": "Seque las plantas al sol antes de la trilla para reducir el contenido de humedad.",
                "Storage Conditions": "Almacene en lugares secos y frescos con ventilación para prevenir el deterioro e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen plagas, enfermedades y condiciones climáticas adversas. Utilice variedades resistentes a enfermedades, prácticas de MIP y manejo adecuado del suelo y agua para mitigar riesgos."
                },   
                   
            {"name": "Guía de Cultivo de Lentejas",
                "Introduction": "Las lentejas (Lens culinaris) son legumbres nutritivas conocidas por su alto contenido de proteínas y fibra. Son ampliamente cultivadas para la alimentación y son un elemento básico en muchas cocinas. Esta guía cubre el proceso completo para cultivar lentejas desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de lentejas de alta calidad y resistentes a enfermedades\n- Fertilizantes de fósforo y potasio (se necesita mínimo nitrógeno)\n- Riego por goteo o por surcos\n- Herbicidas y pesticidas\n- Herramientas manuales o tractores",
                "Soil Preparation": "Las lentejas prefieren suelos francos o arenosos con buen drenaje y un pH de 6.0-7.5. Prepare el campo arando y rastrillando para crear un lecho de siembra fino.",
                "Seed Selection & Treatment": "Elija variedades de alto rendimiento y resistentes a enfermedades adecuadas para su región. Trate las semillas con fungicidas o insecticidas para proteger contra enfermedades transmitidas por semillas.",
                "Field Preparation": "Limpie el campo de malezas y escombros para asegurar un buen contacto entre la semilla y el suelo.",
                "Planting Time": "Las lentejas se siembran típicamente a principios de primavera o finales de invierno, dependiendo del clima, cuando las temperaturas del suelo alcanzan alrededor de 10-15°C (50-59°F).",
                "Spacing & Depth": "Espacie las plantas 25-30 cm en filas separadas por 45-60 cm. Siembre las semillas a 2-3 cm de profundidad según la humedad del suelo.",
                "Seeding Methods": "Siembra directa utilizando sembradoras o plantación manual.",
                "Watering Requirements": "Las lentejas son tolerantes a la sequía pero necesitan humedad adecuada durante la germinación y el desarrollo de vainas. Riegue si la lluvia es insuficiente, particularmente durante la floración y el llenado de semillas.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar. Generalmente no se necesita nitrógeno adicional debido a la fijación de nitrógeno. Incorpore materia orgánica para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas durante el crecimiento temprano usando deshierbe manual o herbicidas. El acolchado también puede ayudar a suprimir las malezas y retener la humedad del suelo.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, chinches lygus y pudriciones de raíz. Implemente estrategias de manejo integrado de plagas (MIP) según sea necesario.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plántulas jóvenes de plagas y mantenga la humedad adecuada.\n- Etapa vegetativa: Asegure nutrientes suficientes para un crecimiento fuerte.\n- Floración y llenado de vainas: Mantenga la humedad para un rendimiento y calidad óptimos.",
                "Harvesting": "Coseche cuando las vainas se vuelvan marrones y se sequen, típicamente 80-100 días después de la siembra. Cosecha manual para pequeñas granjas; use cosechadoras combinadas para operaciones a gran escala.",
                "Post-Harvest Management": "Seque las plantas al sol antes de la trilla para reducir el contenido de humedad.",
                "Storage Conditions": "Almacene en lugares secos y frescos con ventilación para prevenir el deterioro e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique las semillas antes de empaquetar en bolsas transpirables.",
                "Challenges & Solutions": "Los problemas comunes incluyen plagas, enfermedades y clima variable. Utilice variedades resistentes a enfermedades, prácticas de MIP y manejo adecuado del suelo y agua para mitigar riesgos."
                },

            {"name": "Guía de Cultivo de Granada",
                "Introduction": "Las granadas (Punica granatum) son frutas nutritivas conocidas por sus beneficios para la salud y sabor vibrante. Se cultivan en muchas partes del mundo y prosperan en climas cálidos. Esta guía cubre el proceso completo para cultivar granadas desde la plantación hasta la cosecha.",
                "Materials Required": "- Semillas de granada de alta calidad o plántulas saludables de viveros acreditados\n- Fertilizantes balanceados con nitrógeno, fósforo y potasio\n- Sistemas de riego por goteo o riego por surcos\n- Insecticidas y fungicidas para el manejo de plagas y enfermedades\n- Herramientas manuales o tractores para plantación, poda y mantenimiento",
                "Soil Preparation": "Las granadas prefieren suelos franco-arenosos a francos con buen drenaje y un pH de 5.5 a 7.0. Prepare el sitio de plantación arando e incorporando materia orgánica.",
                "Seed Selection & Treatment": "Elija variedades resistentes a enfermedades adecuadas para el clima de su región. Si usa semillas, remójelas durante la noche en agua antes de plantar para mejorar las tasas de germinación.",
                "Field Preparation": "Limpie el sitio de malezas, rocas y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "Las granadas se plantan típicamente en primavera después de la última helada.",
                "Spacing & Depth": "Espacie las plantas 1.5-2.4 metros para permitir un crecimiento adecuado y circulación de aire. Plante semillas o plántulas a una profundidad de 2.5-5 cm, asegurando un buen contacto con el suelo.",
                "Seeding Methods": "Siembra Directa: Siembre las semillas directamente en el sitio preparado. Trasplante: Para plántulas, cave un hoyo ligeramente más grande que el cepellón y rellene con tierra.",
                "Watering Requirements": "Las granadas requieren riego regular, especialmente durante la fase de establecimiento; una vez establecidas, son tolerantes a la sequía. Riegue profundamente pero con poca frecuencia para fomentar el crecimiento profundo de raíces.",
                "Nutrient Management": "Aplique un fertilizante balanceado durante la temporada de crecimiento, típicamente a principios de primavera y nuevamente a finales de verano. Incorpore compost orgánico para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas usando acolchado y deshierbe manual para reducir la competencia por nutrientes.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, moscas blancas y mariposas de la granada. Implemente estrategias de manejo integrado de plagas (MIP), incluyendo el uso de depredadores naturales y pesticidas orgánicos.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plantas jóvenes del clima extremo y plagas. Use acolchado para retener la humedad.\n- Etapa vegetativa: Revise regularmente las deficiencias de nutrientes e infestaciones de plagas; aplique fertilizantes según sea necesario.\n- Floración y desarrollo de frutos: Asegure agua adecuada durante la floración y el cuajado de frutos para promover un desarrollo saludable.",
                "Harvesting": "Las granadas están típicamente listas para cosechar 5-7 meses después de la floración, cuando la fruta tiene un color profundo y hace un sonido metálico cuando se golpea. Use tijeras de podar afiladas para cortar la fruta del árbol, evitando dañar las ramas y otras frutas.",
                "Post-Harvest Management": "Maneje las frutas con cuidado para prevenir magulladuras; almacene en un lugar fresco y seco.",
                "Storage Conditions": "Almacene las granadas en un ambiente fresco y seco; pueden durar varias semanas a meses en condiciones adecuadas.",
                "Processing & Packaging": "Limpie y clasifique las frutas cosechadas, descartando cualquiera dañada o podrida. Empaque las frutas en contenedores transpirables para mantener la calidad durante el almacenamiento.",
                "Challenges & Solutions": "Los problemas comunes incluyen susceptibilidad a plagas, enfermedades y estrés ambiental como sequía o humedad excesiva. Use variedades resistentes a enfermedades, implemente prácticas adecuadas de riego y monitoree las poblaciones de plagas para mitigar desafíos."
                },

            {"name": "Guía de Cultivo de Frijol Rojo",
                "Introduction": "Los frijoles rojos (Phaseolus vulgaris) son una legumbre alta en proteínas comúnmente usada en varias cocinas. Esta guía cubre el proceso completo para cultivar frijoles rojos desde la selección de semillas hasta la cosecha.",
                "Materials Required": "- Semillas de frijol rojo de alta calidad y resistentes a enfermedades\n- Fertilizantes de fósforo y potasio; mínimo nitrógeno ya que los frijoles fijan su propio nitrógeno\n- Riego por goteo o aspersión\n- Herbicidas y pesticidas para plagas comunes del frijol rojo\n- Herramientas manuales o tractores para preparación del suelo, siembra y deshierbe",
                "Soil Preparation": "Los frijoles rojos prosperan en suelos francos con buen drenaje y un pH entre 6.0 y 7.0. Prepare el campo arando y rastrillando para crear una textura fina para fácil penetración de raíces.",
                "Seed Selection & Treatment": "Elija variedades de semillas de alto rendimiento y resistentes a enfermedades. Trate las semillas con fungicidas o insecticidas para proteger contra enfermedades y plagas tempranas transmitidas por el suelo.",
                "Field Preparation": "Limpie el campo de malezas y escombros, luego nivélelo. Marque filas con espaciado adecuado para circulación de aire y penetración de luz solar.",
                "Planting Time": "Los frijoles rojos se plantan típicamente en primavera cuando las temperaturas del suelo alcanzan 15°C (59°F) y no hay riesgo de heladas.",
                "Spacing & Depth": "Plante las semillas a 3-5 cm de profundidad, con 8-10 cm entre plantas y 45-60 cm entre filas.",
                "Seeding Methods": "Siembra Directa: Siembre las semillas directamente en el campo a mano o usando una sembradora.",
                "Watering Requirements": "Los frijoles rojos necesitan riego regular, particularmente durante la floración y desarrollo de vainas. Evite el exceso de agua, ya que los frijoles son sensibles al encharcamiento.",
                "Nutrient Management": "Aplique fósforo y potasio al plantar. Limite el nitrógeno ya que los frijoles rojos fijan nitrógeno atmosférico. Complemente con micronutrientes si las pruebas de suelo indican deficiencias.",
                "Weed Control": "El control de malezas es esencial, particularmente en las etapas tempranas. Use deshierbe manual o herbicidas según sea necesario. El acolchado alrededor de las plantas puede ayudar a retener la humedad y suprimir malezas.",
                "Pest & Disease Management": "Monitoree plagas como pulgones, saltahojas y escarabajos del frijol. Use prácticas de manejo integrado de plagas (MIP) y aplique pesticidas si es necesario. Prevenga enfermedades como pudrición de raíz y tizón practicando rotación de cultivos y evitando suelos encharcados.",
                "Special Care During Growth": "- Etapa de plántula: Asegure humedad moderada del suelo y proteja las plántulas de plagas.\n- Etapa vegetativa: Mantenga niveles de nutrientes para apoyar un crecimiento robusto de hojas y tallos.\n- Etapa de floración y llenado de vainas: Proporcione humedad consistente durante el desarrollo de vainas para mejorar el rendimiento y calidad de semillas.",
                "Harvesting": "Coseche los frijoles rojos cuando las vainas estén completamente maduras y secas, generalmente 90-120 días después de la siembra. Para pequeñas granjas, coseche a mano arrancando toda la planta. Para granjas más grandes, use una cosechadora combinada para recoger los frijoles eficientemente.",
                "Post-Harvest Management": "Permita que las plantas cosechadas se sequen al sol para reducir la humedad en las semillas. Trille los frijoles para separarlos de las vainas, luego limpie las semillas.",
                "Storage Conditions": "Almacene los frijoles rojos en un lugar seco y bien ventilado para prevenir moho e infestaciones de insectos.",
                "Processing & Packaging": "Limpie y clasifique los frijoles para garantizar la calidad antes de empaquetar. Empaque los frijoles en bolsas o contenedores transpirables para mantener la calidad durante el almacenamiento.",
                "Challenges & Solutions": "Los problemas comunes incluyen susceptibilidad a plagas, enfermedades y desequilibrios de nutrientes. Use semillas resistentes a enfermedades, monitoree la salud del suelo y aplique prácticas de MIP para controlar plagas y enfermedades efectivamente."
                },

            {"name": "Guía de Cultivo de Plátano",
                "Introduction": "Los plátanos (Musa spp.) son frutas tropicales reconocidas por su sabor dulce y beneficios nutricionales. Prosperan en climas cálidos y húmedos y se cultivan en todo el mundo tanto para producción comercial como doméstica. Esta guía describe el proceso completo para cultivar plátanos, desde la plantación hasta la cosecha.",
                "Materials Required": "- Hijuelos de plátano saludables o plántulas de cultivo de tejidos\n- Fertilizantes balanceados con nitrógeno, fósforo y potasio; materia orgánica como compost\n- Sistemas de riego por goteo o aspersión para un manejo adecuado de la humedad\n- Insecticidas y fungicidas para manejar plagas y enfermedades\n- Herramientas manuales (palas, podadoras) o tractores para plantación, mantenimiento y cosecha",
                "Soil Preparation": "Los plátanos prefieren suelos francos ricos con buen drenaje y un pH de 5.5 a 7.0. Prepare el suelo arando e incorporando materia orgánica para mejorar la fertilidad y el drenaje.",
                "Plant Selection & Treatment": "Seleccione hijuelos libres de enfermedades de plantas madre saludables u obtenga plántulas de cultivo de tejidos de una fuente confiable. Si usa hijuelos, córtelos de la planta madre con un cuchillo limpio para evitar contaminación.",
                "Field Preparation": "Limpie el sitio de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para plantar.",
                "Planting Time": "El momento ideal para plantar plátanos es al comienzo de la temporada de lluvias o durante los meses más cálidos.",
                "Spacing & Depth": "Espacie las plantas 2.4-3 metros en filas que estén separadas por 3-3.6 metros para permitir un crecimiento adecuado y circulación de aire. Plante los hijuelos o plántulas a la misma profundidad a la que estaban creciendo en el vivero.",
                "Transplanting Methods": "Trasplante: Cave un hoyo lo suficientemente grande para acomodar las raíces y rellene suavemente para evitar bolsas de aire.",
                "Watering Requirements": "Los plátanos requieren humedad constante; riegue regularmente, especialmente durante períodos secos. Apunte a 2.5-5 cm de agua por semana.",
                "Nutrient Management": "Aplique un fertilizante balanceado a principios de primavera y nuevamente a mitad de temporada. Añada compost o acolchado orgánico para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas usando acolchado, que también ayuda a retener la humedad del suelo, y deshierbe manual para reducir la competencia por nutrientes.",
                "Pest & Disease Management": "Monitoree plagas como el picudo del plátano y pulgones. Maneje enfermedades como el mal de Panamá y la sigatoka con sanidad adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluyendo controles culturales y el uso de métodos de control biológico de plagas.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plantas jóvenes del clima extremo y plagas; considere usar tela de sombra si es necesario.\n- Etapa vegetativa: Revise regularmente deficiencias de nutrientes, especialmente potasio y magnesio, y abórdelas con prontitud.\n- Etapa de floración y desarrollo de frutos: Asegure un suministro adecuado de agua durante la floración y desarrollo de frutos para apoyar la formación saludable de frutos.",
                "Harvesting": "Los plátanos están típicamente listos para cosechar 9-12 meses después de la plantación, dependiendo de la variedad y condiciones de crecimiento. Coseche cuando la fruta esté regordeta, verde y el ángulo entre la fruta y el tallo se vuelva más pronunciado. Use un cuchillo afilado o machete para cortar el racimo entero de la planta. Maneje la fruta con cuidado para evitar magulladuras.",
                "Post-Harvest Management": "Elimine cualquier exceso de hojas y maneje los plátanos cosechados con cuidado para prevenir daños. Almacénelos en un área fresca y sombreada.",
                "Storage Conditions": "Almacene los plátanos a temperatura ambiente hasta que maduren. Evite la exposición a la luz solar directa o calor excesivo.",
                "Processing & Packaging": "Si es necesario, los plátanos pueden procesarse en productos como chips de plátano o puré. Empaque los plátanos en cajas transpirables para permitir el flujo de aire y reducir el deterioro durante el transporte.",
                "Challenges & Solutions": "Los problemas comunes incluyen susceptibilidad a plagas y enfermedades, estrés ambiental y riego inadecuado. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
                },

            {"name": "Guía de Cultivo de Uvas",
                "Introduction": "Las uvas (Vitis vinifera y otras especies) son frutas versátiles utilizadas para consumo fresco, secado (pasas) y producción de vino. Prosperan en climas templados y requieren condiciones de cultivo específicas para producir fruta de alta calidad. Esta guía describe el proceso completo para el cultivo de uvas, desde la plantación hasta la cosecha.",
                "Materials Required": "- Vides de uva de calidad, ya sean de raíz desnuda o en maceta, de viveros de confianza\n- Fertilizantes equilibrados que contengan nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo para una gestión eficiente de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (tijeras de podar, palas) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "Las uvas prefieren suelos bien drenados, franco arenosos o franco arcillosos con un pH de 6.0 a 6.8. Prepare el suelo arando e incorporando materia orgánica para mejorar la fertilidad y el drenaje.",
                "Plant Selection & Treatment": "Seleccione variedades de uva resistentes a enfermedades adecuadas para su clima y propósito (uvas de mesa, uvas para vino, etc.). Inspeccione las vides en busca de signos de enfermedad o daño antes de plantar.",
                "Field Preparation": "Limpie el sitio de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El momento ideal para plantar uvas es a principios de la primavera después de la última helada o en el otoño antes de que el suelo se congele.",
                "Spacing & Depth": "Separe las vides de 1.8 a 3 metros en filas que estén separadas de 2.4 a 3 metros para permitir una circulación de aire y un crecimiento adecuados. Plante las vides a la misma profundidad a la que crecían en el vivero.",
                "Transplanting Methods": "Trasplante: Cave un hoyo lo suficientemente grande como para acomodar las raíces, rellene suavemente y riegue abundantemente después de plantar.",
                "Watering Requirements": "Las uvas requieren riego regular durante el primer año para establecer raíces. Una vez establecidas, son tolerantes a la sequía, pero aún se benefician del riego suplementario durante los períodos secos, especialmente durante el desarrollo de la fruta.",
                "Nutrient Management": "Aplique un fertilizante equilibrado a principios de la primavera y nuevamente a mitad de temporada. Use compost orgánico para mejorar la salud del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, el deshierbe manual o el uso de herbicidas para reducir la competencia por los nutrientes y la humedad.",
                "Pest & Disease Management": "Monitoree las plagas como las polillas de la vid, los pulgones y los ácaros. Controle enfermedades como el mildiu polvoriento y el mildiu velloso con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y los depredadores naturales.",
                "Special Care During Growth": "- Etapa de vid joven: Proteja las vides jóvenes del clima extremo y las plagas; use estacas de soporte o espalderas para ayudar a las plantas jóvenes a crecer hacia arriba.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Pode para fomentar una estructura fuerte y la circulación del aire.\n- Etapa de floración y desarrollo de la fruta: Asegure una humedad constante durante la floración y el cuajado de la fruta para maximizar el rendimiento y la calidad de la fruta. Aclare los racimos si es necesario para promover un mayor tamaño de la fruta.",
                "Harvesting": "Las uvas suelen estar listas para la cosecha de 4 a 6 meses después de la floración, según la variedad. Deben cosecharse cuando estén completamente maduras, mostrando un color profundo y un sabor dulce. Use tijeras de podar afiladas para cortar los racimos de la vid. Manipule la fruta con cuidado para evitar magulladuras.",
                "Post-Harvest Management": "Retire las uvas dañadas o podridas y guárdelas en un área fresca y sombreada.",
                "Storage Conditions": "Guarde las uvas en un lugar fresco y seco. La refrigeración puede prolongar su vida útil, pero deben guardarse en recipientes transpirables.",
                "Processing & Packaging": "Si es necesario, las uvas se pueden procesar en productos como jugo de uva, gelatina o vino. Empaque las uvas en recipientes transpirables para permitir el flujo de aire y reducir el deterioro durante el transporte.",
                "Challenges & Solutions": "Los problemas comunes incluyen la susceptibilidad a plagas y enfermedades, problemas relacionados con el clima y riego inadecuado. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            },

            {"name": "Guía de Cultivo de Melón Cantalupo",
                "Introduction": "Los melones cantalupo (Cucumis melo var. cantaloupe) son frutas dulces y aromáticas conocidas por su pulpa jugosa y su distintiva piel reticulada. Prosperan en climas cálidos y son populares por su sabor refrescante. Esta guía describe el proceso completo para el cultivo de melones cantalupo, desde la plantación hasta la cosecha.",
                "Materials Required": "- Semillas o plántulas de melón cantalupo de calidad de fuentes confiables\n- Fertilizantes equilibrados con nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo o por aspersión para una gestión eficiente de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (palas, azadas, tijeras de podar) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "Los melones cantalupo prefieren suelos bien drenados, franco arenosos o francos con un pH de 6.0 a 6.8. Prepare el suelo arando y mezclando materia orgánica para mejorar el drenaje y la fertilidad.",
                "Plant Selection & Treatment": "Elija variedades resistentes a enfermedades adecuadas para su clima y mercado. Si usa semillas, remójelas en agua durante unas horas antes de plantar para mejorar las tasas de germinación.",
                "Field Preparation": "Limpie el sitio de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El momento ideal para plantar melones cantalupo es después de la última fecha de helada, cuando las temperaturas del suelo están consistentemente por encima de 21°C (70°F).",
                "Spacing & Depth": "Separe las plantas de melón cantalupo de 0.9 a 1.2 metros en filas que estén separadas de 1.8 a 2.4 metros para permitir que las vides se extiendan. Plante semillas o plántulas a una profundidad de aproximadamente 2.5 cm (1 pulgada).",
                "Seeding/Transplanting Methods": "Siembra directa: Plante las semillas directamente en el suelo después de que el suelo se caliente. Trasplante: Comience las plántulas en interiores y trasplántelas una vez que sean lo suficientemente fuertes.",
                "Watering Requirements": "Los melones cantalupo necesitan humedad constante, especialmente durante la germinación y el desarrollo de la fruta. Apunte a aproximadamente 2.5-5 cm (1-2 pulgadas) de agua por semana, ajustando la lluvia.",
                "Nutrient Management": "Aplique un fertilizante equilibrado al plantar y nuevamente cuando las vides comiencen a correr. Use compost orgánico o mantillo para mejorar la salud del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, que ayuda a retener la humedad y suprimir el crecimiento de malezas, y el deshierbe manual para reducir la competencia.",
                "Pest & Disease Management": "Monitoree las plagas como los pulgones, los escarabajos del pepino y los ácaros. Controle enfermedades como el mildiu polvoriento y el mildiu velloso con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y el uso de controles biológicos.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plantas jóvenes de las plagas y el clima extremo. Use cubiertas de hileras si es necesario para proteger contra las plagas y las heladas.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Apoye las vides si es necesario, especialmente cuando la fruta comienza a desarrollarse.\n- Etapa de desarrollo de la fruta: Asegure un suministro de agua adecuado durante el desarrollo de la fruta para promover un crecimiento saludable y dulzura. Evite regar directamente sobre la fruta para prevenir la pudrición.",
                "Harvesting": "Los melones cantalupo suelen estar listos para la cosecha de 70 a 90 días después de la plantación. Los indicadores incluyen un cambio de color de verde a amarillo en el extremo de la flor y un aroma dulce. Use un cuchillo afilado o tijeras de podar para cortar la fruta de la vid, dejando un tallo corto adherido al melón.",
                "Post-Harvest Management": "Manipule los melones cantalupo cosechados con cuidado para evitar magulladuras. Guárdelos en un área fresca y sombreada.",
                "Storage Conditions": "Guarde los melones cantalupo a temperatura ambiente hasta que estén completamente maduros. Una vez maduros, se pueden refrigerar durante un corto período para prolongar la frescura.",
                "Processing & Packaging": "Si es necesario, los melones cantalupo se pueden procesar en batidos, sorbetes o ensaladas de frutas. Empaque los melones cantalupo en recipientes transpirables para ayudar a mantener la calidad durante el almacenamiento y el transporte.",
                "Challenges & Solutions": "Los desafíos comunes incluyen la susceptibilidad a plagas y enfermedades, el estrés ambiental como la sequía o el exceso de humedad, y las prácticas de riego inadecuadas. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            },

            {"name": "Guía de Cultivo de Manzanas",
                "Introduction": "Las manzanas (Malus domestica) son una de las frutas más populares en todo el mundo, apreciadas por su sabor, versatilidad y valor nutricional. Crecen mejor en climas templados y se pueden cultivar en varios tipos de suelo. Esta guía describe el proceso completo para el cultivo de manzanas, desde la plantación hasta la cosecha.",
                "Materials Required": "- Plántulas de manzano de calidad o variedades injertadas de viveros de confianza\n- Fertilizantes equilibrados que contengan nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo o mangueras para una gestión eficaz de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (palas, tijeras de podar, azadas) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "Las manzanas prefieren suelos bien drenados, francos con un pH de 6.0 a 7.0. Prepare el suelo arando e incorporando materia orgánica para mejorar la fertilidad y el drenaje.",
                "Plant Selection & Treatment": "Elija variedades de manzana resistentes a enfermedades adecuadas para su clima, considerando factores como el sabor de la fruta y el tiempo de cosecha. Inspeccione las plántulas en busca de signos de enfermedad o daño antes de plantar.",
                "Field Preparation": "Limpie el área de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El mejor momento para plantar manzanos es en otoño o principios de primavera cuando los árboles están inactivos.",
                "Spacing & Depth": "Separe las variedades enanas de 1.2 a 1.8 metros y las variedades estándar de 3 a 4.5 metros para permitir un crecimiento y una circulación de aire adecuados. Plante los árboles a una profundidad que coincida con su altura de vivero, asegurando que la unión del injerto esté por encima del nivel del suelo.",
                "Seeding/Transplanting Methods": "Trasplante: Cave un hoyo lo suficientemente grande como para acomodar las raíces, coloque el árbol en el hoyo, rellene suavemente y riegue abundantemente después de plantar.",
                "Watering Requirements": "Riegue los manzanos jóvenes regularmente para establecer raíces, especialmente durante los períodos secos. Una vez establecidos, son tolerantes a la sequía, pero se benefician de un riego profundo durante el desarrollo de la fruta.",
                "Nutrient Management": "Aplique un fertilizante equilibrado a principios de la primavera y nuevamente a mitad de temporada. Use compost orgánico para mejorar la salud del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, que ayuda a retener la humedad y suprimir el crecimiento de malezas, y el deshierbe manual para reducir la competencia.",
                "Pest & Disease Management": "Monitoree las plagas como las polillas de la manzana, los pulgones y los ácaros. Controle enfermedades como la sarna de la manzana y el mildiu polvoriento con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y el uso de insectos beneficiosos.",
                "Special Care During Growth": "- Etapa de árbol joven: Proteja los árboles jóvenes del clima extremo y las plagas; considere usar protectores de árboles para prevenir el daño animal.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Pode para dar forma a los árboles y fomentar una estructura fuerte.\n- Etapa de floración y desarrollo de la fruta: Asegure una humedad constante durante la floración y el cuajado de la fruta para maximizar el rendimiento y la calidad de la fruta. Aclare la fruta si es necesario para promover manzanas más grandes.",
                "Harvesting": "Las manzanas suelen estar listas para la cosecha de 4 a 6 meses después de la floración, según la variedad. Los indicadores incluyen un cambio de color, textura firme y facilidad de desprendimiento del árbol. Use tijeras de podar afiladas para cortar las manzanas del árbol, dejando un tallo corto adherido a la fruta.",
                "Post-Harvest Management": "Manipule las manzanas cosechadas con cuidado para evitar magulladuras. Guárdelas en un área fresca y sombreada.",
                "Storage Conditions": "Guarde las manzanas en un lugar fresco y oscuro. Se pueden refrigerar para prolongar su vida útil.",
                "Processing & Packaging": "Si es necesario, las manzanas se pueden procesar en salsa de manzana, sidra o rodajas secas. Empaque las manzanas en recipientes transpirables para ayudar a mantener la calidad durante el almacenamiento y el transporte.",
                "Challenges & Solutions": "Los desafíos comunes incluyen la susceptibilidad a plagas y enfermedades, el estrés ambiental (como la sequía o las heladas) y las técnicas de poda inadecuadas. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            },

            {"name": "Guía de Cultivo de Naranjas",
                "Introduction": "Las naranjas (Citrus sinensis) son una de las frutas cítricas más populares, valoradas por su pulpa dulce y jugosa y su alto contenido de vitamina C. Prosperan en climas cálidos, subtropicales a tropicales. Esta guía describe el proceso completo para el cultivo de naranjas, desde la plantación hasta la cosecha.",
                "Materials Required": "- Plántulas de naranjo de calidad o variedades injertadas de viveros de confianza\n- Fertilizantes específicos para cítricos que contengan nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo o mangueras para una gestión eficiente de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (palas, tijeras de podar, azadas) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "Las naranjas prefieren suelos bien drenados, franco arenosos o franco arcillosos con un pH de 6.0 a 7.5. Prepare el suelo arando e incorporando materia orgánica para mejorar la fertilidad y el drenaje.",
                "Plant Selection & Treatment": "Elija variedades de naranja resistentes a enfermedades adecuadas para su clima, considerando factores como el sabor de la fruta y el tiempo de cosecha. Inspeccione las plántulas en busca de signos de enfermedad o daño antes de plantar.",
                "Field Preparation": "Limpie el área de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El mejor momento para plantar naranjos es en primavera después de que haya pasado el peligro de heladas.",
                "Spacing & Depth": "Separe los árboles de 3.6 a 7.6 metros, según el portainjerto y la variedad del árbol, para permitir un crecimiento y una circulación de aire adecuados. Plante los árboles a una profundidad que coincida con su altura de vivero, asegurando que la unión del injerto esté por encima del nivel del suelo.",
                "Seeding/Transplanting Methods": "Trasplante: Cave un hoyo lo suficientemente grande como para acomodar las raíces, coloque el árbol en el hoyo, rellene suavemente y riegue abundantemente después de plantar.",
                "Watering Requirements": "Riegue los naranjos jóvenes regularmente para establecer raíces, especialmente durante los períodos secos. Los árboles maduros requieren un riego profundo durante los períodos secos.",
                "Nutrient Management": "Aplique un fertilizante específico para cítricos a principios de la primavera y nuevamente a mitad de temporada. Use compost orgánico para mejorar la salud del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, que ayuda a retener la humedad y suprimir el crecimiento de malezas, y el deshierbe manual para reducir la competencia.",
                "Pest & Disease Management": "Monitoree las plagas como los pulgones, los ácaros y los minadores de hojas de cítricos. Controle enfermedades como el cancro de los cítricos y la pudrición de la raíz con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y el uso de insectos beneficiosos.",
                "Special Care During Growth": "- Etapa de árbol joven: Proteja los árboles jóvenes del clima extremo y las plagas; considere usar protectores de árboles para prevenir el daño animal.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Pode para dar forma a los árboles y fomentar una estructura fuerte.\n- Etapa de floración y desarrollo de la fruta: Asegure una humedad constante durante la floración y el cuajado de la fruta para maximizar el rendimiento y la calidad de la fruta. Aclare la fruta si es necesario para promover naranjas más grandes.",
                "Harvesting": "Las naranjas suelen estar listas para la cosecha de 7 a 12 meses después de la floración, según la variedad. Los indicadores incluyen un cambio de color, firmeza y dulzura. Use tijeras de podar afiladas para cortar las naranjas del árbol, dejando un tallo corto adherido a la fruta.",
                "Post-Harvest Management": "Manipule las naranjas cosechadas con cuidado para evitar magulladuras. Guárdelas en un área fresca y sombreada.",
                "Storage Conditions": "Guarde las naranjas en un lugar fresco y oscuro. Se pueden refrigerar para prolongar su vida útil.",
                "Processing & Packaging": "Si es necesario, las naranjas se pueden procesar en jugo, mermelada o rodajas secas. Empaque las naranjas en recipientes transpirables para ayudar a mantener la calidad durante el almacenamiento y el transporte.",
                "Challenges & Solutions": "Los desafíos comunes incluyen la susceptibilidad a plagas y enfermedades, el estrés ambiental (como la sequía o las heladas) y las técnicas de poda inadecuadas. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            },  

            {"name": "Guía de Cultivo de Papaya",
                "Introduction": "Las papayas (Carica papaya) son árboles frutales tropicales conocidos por su pulpa dulce y jugosa y su color naranja vibrante. Prosperan en climas cálidos y pueden producir fruta durante todo el año en condiciones óptimas. Esta guía describe el proceso completo para el cultivo de papayas, desde la plantación hasta la cosecha.",
                "Materials Required": "- Semillas o plántulas de papaya de calidad de viveros de confianza\n- Fertilizantes equilibrados con nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo o mangueras para una gestión eficaz de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (palas, tijeras de podar, azadas) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "Las papayas prefieren suelos bien drenados, franco arenosos o francos con un pH de 6.0 a 6.5. Prepare el suelo arando e incorporando materia orgánica para mejorar el drenaje y la fertilidad.",
                "Plant Selection & Treatment": "Elija variedades de papaya resistentes a enfermedades adecuadas para su clima. Si usa semillas, remójelas durante unas horas antes de plantar para mejorar las tasas de germinación.",
                "Field Preparation": "Limpie el área de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El mejor momento para plantar papayas es en primavera cuando las temperaturas son consistentemente cálidas.",
                "Spacing & Depth": "Separe las plantas de papaya de 1.8 a 3 metros para permitir su gran copa y sistema de raíces. Plante semillas o plántulas a una profundidad de aproximadamente 1.2 a 2.5 cm (0.5 a 1 pulgada).",
                "Seeding/Transplanting Methods": "Siembra directa: Plante las semillas directamente en el suelo después de la última helada.\nTrasplante: Comience las plántulas en interiores y trasplántelas cuando tengan aproximadamente 30 cm (12 pulgadas) de altura.",
                "Watering Requirements": "Riegue las plantas de papaya jóvenes regularmente, especialmente durante los períodos secos. Las papayas requieren humedad constante, pero no toleran el encharcamiento.",
                "Nutrient Management": "Aplique un fertilizante equilibrado cada 4-6 semanas durante la temporada de crecimiento. Use compost orgánico para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, que ayuda a retener la humedad y suprimir el crecimiento de malezas, y el deshierbe manual para reducir la competencia.",
                "Pest & Disease Management": "Monitoree las plagas como los pulgones, las moscas blancas y las moscas de la fruta. Controle enfermedades como el mildiu polvoriento y la pudrición de la raíz con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y el uso de insectos beneficiosos.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plantas jóvenes del clima extremo y las plagas. Use cubiertas de hileras si es necesario para proteger de las heladas y los insectos.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Pode las hojas muertas o dañadas para promover un crecimiento saludable.\n- Etapa de desarrollo de la fruta: Asegure un suministro de agua adecuado durante el desarrollo de la fruta. Aclare el exceso de frutas si es necesario para permitir un mayor tamaño de la fruta.",
                "Harvesting": "Las papayas suelen estar listas para la cosecha de 6 a 12 meses después de la plantación, según la variedad. Los indicadores incluyen un cambio en el color de la piel de verde a amarillo y un aroma dulce. Use un cuchillo afilado para cortar la fruta del árbol, dejando una pequeña porción del tallo adherida.",
                "Post-Harvest Management": "Manipule las papayas cosechadas con cuidado para evitar magulladuras. Guárdelas en un área fresca y sombreada.",
                "Storage Conditions": "Guarde las papayas a temperatura ambiente para que maduren aún más. Una vez maduras, se pueden refrigerar durante un corto período para prolongar la frescura.",
                "Processing & Packaging": "Si es necesario, las papayas se pueden procesar en batidos, ensaladas o fruta seca. Empaque las papayas en recipientes transpirables para mantener la calidad durante el almacenamiento y el transporte.",
                "Challenges & Solutions": "Los desafíos comunes incluyen la susceptibilidad a plagas y enfermedades, el estrés ambiental (como la sequía o las inundaciones) y las prácticas de riego inadecuadas. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            },

            {"name": "Guía de Cultivo de Café",
                "Introduction": "El café (Coffea spp.) es una de las bebidas más consumidas a nivel mundial, conocido por sus propiedades estimulantes y su rico sabor. Prospera en climas tropicales, generalmente en altitudes más altas, donde las condiciones son ideales para su crecimiento. Esta guía describe el proceso completo para el cultivo de café, desde la plantación hasta la cosecha.",
                "Materials Required": "- Plántulas o semillas de café de calidad de viveros de confianza\n- Fertilizantes equilibrados ricos en nitrógeno, fósforo y potasio; compost orgánico\n- Sistemas de riego por goteo o mangueras para una gestión eficaz de la humedad\n- Insecticidas, fungicidas y soluciones orgánicas para el control de plagas\n- Herramientas manuales (palas, tijeras de podar, azadas) o tractores para la plantación, el mantenimiento y la cosecha",
                "Soil Preparation": "El café prefiere suelos bien drenados, francos con un pH de 6.0 a 6.5. Prepare el suelo arando e incorporando materia orgánica para mejorar la fertilidad y el drenaje.",
                "Plant Selection & Treatment": "Elija variedades de café resistentes a enfermedades adecuadas para su clima. Si usa semillas, remójelas durante 24 horas para mejorar las tasas de germinación.",
                "Field Preparation": "Limpie el área de plantación de malezas, piedras y escombros para asegurar un ambiente limpio para la plantación.",
                "Planting Time": "El mejor momento para plantar café es al comienzo de la temporada de lluvias.",
                "Spacing & Depth": "Separe las plantas de café de 1.5 a 2.4 metros para permitir un crecimiento y una circulación de aire adecuados. Plante las plántulas a una profundidad que coincida con su altura de vivero, asegurando que el cuello de la raíz esté nivelado con la superficie del suelo.",
                "Seeding/Transplanting Methods": "Trasplante: Cave un hoyo lo suficientemente grande como para acomodar las raíces, coloque la plántula en el hoyo, rellene suavemente y riegue abundantemente después de plantar.",
                "Watering Requirements": "Riegue las plantas de café jóvenes regularmente para establecer raíces, especialmente durante los períodos secos. Las plantas maduras prefieren humedad constante, pero no deben encharcarse.",
                "Nutrient Management": "Aplique un fertilizante equilibrado cada 3-4 meses durante la temporada de crecimiento. Use compost orgánico para mejorar la fertilidad del suelo.",
                "Weed Control": "Controle las malezas mediante el acolchado, que ayuda a retener la humedad y suprimir el crecimiento de malezas, y el deshierbe manual para reducir la competencia.",
                "Pest & Disease Management": "Monitoree las plagas como los escarabajos barrenadores del café y la roya de la hoja. Controle enfermedades como la pudrición de la raíz y la mancha de la hoja con una higiene adecuada y variedades resistentes. Implemente estrategias de manejo integrado de plagas (MIP), incluidos los controles culturales y el uso de insectos beneficiosos.",
                "Special Care During Growth": "- Etapa de plántula: Proteja las plantas jóvenes del clima extremo y las plagas. Use tela de sombra si es necesario para proteger de la luz solar intensa.\n- Etapa vegetativa: Verifique regularmente las deficiencias de nutrientes y abórdelas rápidamente. Pode para dar forma a las plantas y eliminar las ramas muertas o enfermas.\n- Etapa de floración y desarrollo de la fruta: Asegure un suministro de agua adecuado durante la floración y el cuajado de la fruta para maximizar el rendimiento y la calidad de la fruta. Monitoree las infestaciones de moscas de la fruta y controle según sea necesario.",
                "Harvesting": "Las cerezas de café suelen estar listas para la cosecha de 7 a 9 meses después de la floración, según la variedad. Los indicadores incluyen un cambio de color de verde a rojo brillante o amarillo. Coseche las cerezas de café a mano, recogiendo solo las maduras. Use un método de recolección selectiva para la calidad.",
                "Post-Harvest Management": "Manipule las cerezas cosechadas con cuidado para evitar magulladuras. Procéselas lo antes posible para evitar el deterioro.",
                "Processing Methods": "Use el método seco (cerezas secadas al sol) o el método húmedo (cerezas fermentadas y lavadas) para extraer los granos de café.",
                "Storage Conditions": "Guarde los granos de café procesados en un lugar fresco y seco para evitar el deterioro y mantener el sabor.",
                "Processing & Packaging": "Empaque los granos de café en recipientes herméticos para ayudar a preservar la frescura durante el almacenamiento y el transporte.",
                "Challenges & Solutions": "Los desafíos comunes incluyen la susceptibilidad a plagas y enfermedades, el estrés ambiental (como la sequía o las heladas) y la fluctuación de los precios de mercado. Elija variedades resistentes a enfermedades, implemente buenas prácticas culturales y monitoree las condiciones ambientales para mitigar estos desafíos."
            }                
            
        ]

    cropGuide = [
            {"name": "Maize Cultivation Guide", 
                "Introduction": "Maize (Zea mays), also known as corn, is a key cereal crop widely cultivated for its grains. This guide covers the complete process for cultivating maize from seed selection to harvesting.",
                "Materials Required": "- High-quality maize seeds (hybrid or improved varieties)\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Machinery (tractors, hand tools, seed planters)\n- Pest control (herbicides, insecticides)\n- Irrigation equipment (drip or furrow irrigation)",
                "Soil Preparation": "Maize thrives in well-drained loam soils with a pH of 5.8 to 7.0. Till the soil to improve aeration and break up clods.",
                "Seed Selection & Treatment": "Choose high-yielding, drought-resistant varieties. Treat seeds with fungicides or insecticides for protection.",
                "Field Preparation": "Level the field for even water distribution. Optimize row spacing for maximum sunlight exposure.",
                "Planting Time": "Typically planted at the beginning of the rainy season, between April and June, depending on the region.",
                "Spacing & Depth": "Plant seeds at 20-25 cm within rows and 60-75 cm between rows, at a depth of 2-5 cm.",
                "Seeding Methods": "- **Direct Seeding:** Plant seeds manually or with seed planters.",
                "Watering Requirements": "Requires regular watering, especially during silking and tasseling. Use irrigation if rain is insufficient.",
                "Nutrient Management": "Apply fertilizers in split doses: at planting, early growth, and tasseling stages.",
                "Weed Control": "Manual weeding, hoeing, or herbicides. First weeding at 15-20 days after planting, followed by another at 30-40 days.",
                "Pest & Disease Management": "Monitor for maize borers, armyworms, and aphids. Use pesticides and integrated pest management (IPM).",
                "Harvesting": "Harvest when maize ears mature and husks dry. Moisture content should be 20-25%. Use handpicking or mechanical harvesters.",
                "Post-Harvest Management": "Dry grains to 13-14% moisture. Shell, clean, and store properly.",
                "Storage Conditions": "Store in a cool, dry place with ventilation to prevent mold and pests.",
                "Processing": "If needed, dry and mill the maize for further use.",
                "Challenges & Solutions": "Common issues: weather variability, pests, and water scarcity. Solutions: IPM, soil moisture monitoring, and resilient varieties."
            },
            
            {"name": "Rice Cultivation Guide", 
                "Introduction": "Rice Oryza sativa is a staple food crop in many parts of the world. This guide covers the complete process of cultivating rice from seed selection to harvesting.",
                "Materials Required": "- High-quality seeds\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Irrigation system\n- Machinery (tractors, transplanting machines, sickles)\n- Pest control (herbicides, pesticides)", 
                "Soil Preparation": "Rice grows best in clay or clay-loam soils with pH 5.5 to 6.5. Till the soil and level the field for even water distribution.", 
                "Seed Selection & Treatment": "Use high-yielding, pest-resistant seeds. Treat them with fungicides or insecticides to prevent infestations.", 
                "Field Preparation": "Level the field and create bunds (raised edges) to retain water.", 
                "Planting Time": "Plant at the onset of the rainy season, usually from May to June depending on the region.", 
                "Spacing & Depth": "For transplanting, use 20x15 cm spacing. For direct seeding, plant 2-3 cm deep.",
                "Seeding Methods": "- **Direct Seeding:** Broadcasting seeds or planting in rows.\n- **Transplanting:** Grow in a nursery and transfer seedlings after 20-30 days.",
                "Watering Requirements": "Maintain 5-10 cm of water during growth. Reduce water at the grain ripening stage.",
                "Nutrient Management": "Apply fertilizers in split doses: at planting, during tillering, and at panicle initiation.",
                "Weed Control": "Use manual weeding or herbicides. Weed 15-20 days after transplanting, then again at 40 days.",
                "Pest & Disease Management": "Watch for pests like stem borers and leafhoppers. Use pesticides and integrated pest management (IPM) practices.",
                "Harvesting": "Harvest when grains turn golden-yellow and 80-90% of grains are mature. Use sickles for small farms or mechanical harvesters for efficiency.",
                "Post-Harvest Management": "Dry grains to 14% moisture, thresh, winnow, and store in a cool, dry place to prevent spoilage.",
                "Challenges & Solutions": "Common issues include adverse weather, pests, and water scarcity. Use IPM, monitor water levels, and diversify crop varieties to mitigate risks."
            },
            
            {"name": "Jute Cultivation Guide",
                "Introduction": "Jute is a fibrous crop mainly grown for its strong, natural fibers, widely used in textiles and packaging. This guide covers the complete process for cultivating jute from seed selection to harvesting.",
                "Materials Required": "- High-quality, certified jute seeds (Corchorus olitorius or Corchorus capsularis)\n- Organic compost, nitrogen, phosphorus, and potassium fertilizers\n- Hand tools or tractors for soil preparation\n- Herbicides and pesticides for pest control\n- Irrigation system for controlled watering",
                "Soil Preparation": "Jute grows best in loamy, sandy-loam soils with good drainage and a pH range of 6.0 to 7.5. Prepare the soil by plowing and leveling it to break up clods and ensure good seedbed preparation.",
                "Seed Selection & Treatment": "Choose high-yielding and disease-resistant seed varieties. Soak seeds in water for 24 hours before planting to encourage germination.",
                "Field Preparation": "Clear and level the field for uniform water distribution. Create small bunds around the field if flooding is expected.",
                "Planting Time": "Jute is usually planted with the arrival of the monsoon, typically between March and May.",
                "Spacing & Depth": "Sow seeds in rows with a spacing of 25-30 cm between rows. Plant seeds 1-2 cm deep for optimal germination.",
                "Seeding Methods": "- **Broadcasting:** Scatter seeds evenly over the field.\n- **Row Planting:** Sow seeds in rows, which facilitates weeding and other management activities.",
                "Watering Requirements": "Jute requires regular moisture; maintain adequate moisture, especially during the early growth phase. Avoid waterlogging by ensuring proper drainage, particularly after heavy rains.",
                "Nutrient Management": "Apply a basal dose of nitrogen, phosphorus, and potassium fertilizers at planting. Additional nitrogen can be applied after thinning, about 20-25 days after sowing.",
                "Weed Control": "Perform manual weeding or apply selective herbicides as needed, especially in the early stages. Conduct the first weeding 15-20 days after sowing, followed by another after 30-40 days.",
                "Pest & Disease Management": "Monitor for common pests like jute hairy caterpillars and aphids. Use pesticides or integrated pest management (IPM) practices to control pests and diseases like stem rot and anthracnose.",
                "Harvesting": "Harvest jute when the plants are 10-12 feet tall and the lower leaves start to yellow, typically 100-120 days after planting. Cut the plants close to the base using a sickle or knife. For best fiber quality, harvest before the plants begin to flower.",
                "Post-Harvest Management": "Bundle the harvested jute plants and submerge them in clean, slow-moving water for retting (fermentation process to loosen the fibers). Retting usually takes 10-15 days; check fiber separation regularly.",
                "Challenges & Solutions": "Common issues include water availability, pest infestations, and improper retting. Use efficient irrigation and pest control methods, and monitor water levels carefully during retting to ensure fiber quality."
            },

            {"name": "Cotton Cultivation Guide",
                "Introduction": "Cotton is a major fiber crop valued for its soft, fluffy fibers used in textiles. This guide covers the complete process for cultivating cotton from seed selection to harvesting.",
                "Materials Required": "- High-quality, certified cotton seeds (e.g., Bt cotton or other pest-resistant varieties)\n- Nitrogen, phosphorus, potassium, and micronutrient fertilizers\n- Drip or furrow irrigation system\n- Herbicides and pesticides for pest control\n- Plows, tractors, and sprayers for field preparation and maintenance",
                "Soil Preparation": "Cotton grows best in well-drained sandy-loam soils with a pH of 6.0 to 7.5. Prepare the field by deep plowing, followed by harrowing to break clods and smooth the surface.",
                "Seed Selection & Treatment": "Choose high-yielding, pest-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against soil-borne diseases and early pest infestations.",
                "Field Preparation": "Create furrows or beds for planting, depending on irrigation method. Ensure good drainage to prevent waterlogging, which cotton is sensitive to.",
                "Planting Time": "Cotton is typically planted in spring, from March to May, depending on the region and temperature.",
                "Spacing & Depth": "Plant seeds 3-5 cm deep, with a spacing of 75-100 cm between rows and 25-30 cm between plants.",
                "Seeding Methods": "- **Direct Seeding:** Plant seeds directly into prepared furrows or beds using seed drills or by hand.",
                "Watering Requirements": "Cotton requires consistent moisture, especially during the flowering and boll formation stages. Use drip or furrow irrigation to maintain adequate soil moisture, particularly during dry spells.",
                "Nutrient Management": "Apply basal fertilizer with phosphorus and potassium at planting. Apply nitrogen in split doses: one-third at planting, one-third during vegetative growth, and one-third at flowering.",
                "Weed Control": "Use manual weeding, hoeing, or herbicides to control weeds, particularly during early growth stages. Perform weeding about 20-30 days after planting and again if necessary at 45 days.",
                "Pest & Disease Management": "Monitor for common pests like bollworms, aphids, and whiteflies. Use integrated pest management (IPM) practices, including biological controls, to minimize pesticide use.",
                "Harvesting": "Harvest cotton when the bolls are fully open and the fibers are fluffy, typically 150-180 days after planting. Manual harvesting involves picking mature bolls by hand, while large farms use cotton-picking machines.",
                "Post-Harvest Management": "Allow harvested cotton to dry in a shaded, ventilated area. Clean and gin the cotton to separate seeds from fiber. Store cotton fibers in a dry, well-ventilated place to avoid moisture-related damage.",
                "Challenges & Solutions": "Common issues include pest infestations, water availability, and soil nutrient depletion. Use drought-resistant varieties, implement efficient irrigation, and follow IPM practices to manage pests."
            },

            {"name": "Coconut Cultivation Guide",
                "Introduction": "The coconut palm (Cocos nucifera) is cultivated for its fruit, providing oil, milk, and fiber. This guide covers key steps from seed selection to harvesting.",
                "Materials Required": "- High-quality coconut seedlings (dwarf or tall varieties)\n- Organic manure, NPK fertilizers\n- Drip or basin irrigation\n- Pesticides or biocontrol agents\n- Hand tools or mechanical equipment",
                "Soil Preparation": "Coconuts thrive in well-drained sandy loam with pH 5.5-7.5. Dig 1 x 1 x 1 m pits, fill with soil, compost, and organic manure for strong root growth.",
                "Seed Selection & Treatment": "Use disease-resistant, high-yielding seedlings. Dwarf varieties allow easy harvesting, while tall varieties are drought-resistant.",
                "Field Preparation": "Clear weeds and debris, ensure proper drainage, and space pits as per variety needs.",
                "Planting Time": "Best planted at the rainy season’s onset to reduce irrigation needs; can be planted year-round with irrigation.",
                "Spacing & Depth": "Tall varieties: 7.5-9m apart; Dwarf: 6.5-7m. Ensure roots are well covered.",
                "Seeding Methods": "Place seedlings in pits with the collar just above ground level.",
                "Watering Requirements": "Water regularly for the first three years. Mature trees are drought-resistant but benefit from consistent irrigation.",
                "Nutrient Management": "Apply balanced fertilizers three times a year with micronutrients like magnesium and boron. Add organic manure annually.",
                "Weed Control": "Weed regularly, especially in early growth. Mulching helps retain moisture and suppress weeds.",
                "Pest & Disease Management": "Control pests like rhinoceros beetles and red palm weevils using pesticides or biocontrols. Manage root wilt and bud rot with fungicides and pruning.",
                "Harvesting": "Mature coconuts (12 months after flowering) turn brown. Harvest every 45-60 days using climbing tools or mechanical lifters.",
                "Post-Harvest Management": "Store in a dry, ventilated area. Process copra by sun-drying or mechanical drying. Pack dried coconuts securely for transport.",
                "Challenges & Solutions": "Drought, pests, and soil depletion can be managed with drip irrigation, pest management, and organic soil amendments."
            },

            {"name": "Chickpea Cultivation Guide",
                "Introduction": "Chickpea (Cicer arietinum) is a popular legume grown for its protein-rich seeds, widely used in food production. This guide covers the complete process for cultivating chickpeas from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant chickpea seeds (desi or kabuli types)\n- Phosphorus-based fertilizers; minimal nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides\n- Plows, tractors, and sprayers",
                "Soil Preparation": "Chickpeas grow best in well-drained, loamy soils with a pH of 6.0-7.5. Plow and harrow the field for good root penetration.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seeds. Treat with rhizobium bacteria for nitrogen fixation and fungicides to prevent diseases.",
                "Field Preparation": "Clear weeds and level the field. Space rows to allow air circulation and reduce disease risk.",
                "Planting Time": "Best planted in cool, dry seasons, typically October-November.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 5-8 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Chickpeas require minimal watering but benefit from irrigation during flowering and pod filling. Avoid waterlogging.",
                "Nutrient Management": "Apply phosphorus at planting. Use potassium and micronutrients as needed based on soil tests.",
                "Weed Control": "Weed early and regularly, either manually or with herbicides. First weeding at 20-30 days, second at 45-50 days if needed.",
                "Pest & Disease Management": "Monitor for pests like pod borers and aphids. Use integrated pest management (IPM) and biopesticides as needed.",
                "Special Care During Growth": "- Seedling stage: Protect from pests, maintain moderate moisture.\n- Vegetative stage: Maintain phosphorus levels.\n- Flowering & pod-filling: Ensure adequate moisture for optimal yield.",
                "Harvesting": "Chickpeas mature in 3-4 months. Harvest when plants yellow and pods dry. Cut by hand for small farms; use combine harvesters for large-scale farming.",
                "Post-Harvest Management": "Sun-dry seeds to reduce moisture, thresh, and clean before storage or sale.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent insect infestations and spoilage.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, water stress, and nutrient deficiencies. Use IPM, resistant varieties, and soil testing to mitigate risks."
            },

            {"name": "Pigeon Pea Cultivation Guide",
                "Introduction": "Pigeon peas (Cajanus cajan) are a drought-resistant legume valued for their high protein content and use in various dishes. This guide covers the complete process for cultivating pigeon peas from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant pigeon pea seeds (early, medium, or late-maturing varieties)\n- Nitrogen, phosphorus, and potassium fertilizers; minimal nitrogen needed\n- Drip or furrow irrigation equipment\n- Herbicides and pesticides specific to pigeon pea pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
                "Soil Preparation": "Pigeon peas grow best in well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Plow and harrow the field to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your region. Treat seeds with fungicides to prevent seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris, ensuring good drainage.",
                "Planting Time": "Typically planted at the beginning of the rainy season or during the dry season in subtropical regions.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep, depending on soil moisture and texture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Pigeon peas are drought-resistant but require adequate moisture during flowering and pod development. Irrigation may be necessary, especially in the first 60 days.",
                "Nutrient Management": "Apply phosphorus and potassium at planting and top-dress with nitrogen if necessary. Organic amendments can improve soil fertility.",
                "Weed Control": "Control weeds during early growth stages using manual weeding or herbicides. Mulching can help suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests such as pod borers, aphids, and whiteflies. Implement integrated pest management (IPM) strategies, including biological controls and chemical pesticides as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain soil moisture.\n- Vegetative stage: Ensure adequate nutrients for strong growth.\n- Flowering & pod-filling: Maintain consistent moisture to maximize yield and seed quality.",
                "Harvesting": "Pigeon peas mature in 4-6 months. Harvest when pods are mature and dry. Cut by hand for small farms or use combine harvesters for large-scale farming.",
                "Post-Harvest Management": "Allow harvested plants to sun-dry before threshing to reduce seed moisture content.",
                "Storage Conditions": "Store pigeon peas in a dry, cool, and well-ventilated area to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags or containers.",
                "Challenges & Solutions": "Common issues include pest infestations, diseases, water stress, and nutrient deficiencies. Use disease-resistant varieties, practice crop rotation, and apply IPM strategies to manage risks."
            },

            {"name": "Moth Bean Cultivation Guide",
                "Introduction": "Moth beans (Vigna aconitifolia) are a drought-resistant legume commonly grown in arid regions. They are valued for their high protein content and culinary applications. This guide covers the complete process for cultivating moth beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant moth bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Moth beans thrive in well-drained sandy loam or clay soils with a pH of 6.0-8.0. Prepare the field by plowing and harrowing for a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, drought-tolerant varieties. Treat seeds with fungicides or insecticides to prevent seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the onset of the monsoon season, between June and July.",
                "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Moth beans are drought-resistant but benefit from consistent moisture during flowering and pod development. Water if rainfall is insufficient.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Use nitrogen only if soil tests indicate a deficiency. Organic amendments improve soil fertility.",
                "Weed Control": "Control weeds early with manual weeding or herbicides. Mulching helps suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and leafhoppers. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Maintain moderate moisture and protect from pests.\n- Vegetative stage: Ensure adequate nutrients.\n- Flowering & pod-filling: Maintain moisture for optimal yield.",
                "Harvesting": "Harvest when pods mature and dry, typically 90-120 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use drought-resistant varieties, IPM practices, and proper soil management to mitigate risks."
            },

            {"name": "Mung Bean Cultivation Guide",
                "Introduction": "Mung beans (Vigna radiata) are small, green legumes highly valued for their nutritional content and culinary versatility. This guide covers the complete process for cultivating mung beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant mung bean seeds\n- Nitrogen, phosphorus, and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Mung beans prefer well-drained sandy loam to loamy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to achieve a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides to protect against soil-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the beginning of the rainy season or in warm, dry conditions between April and June.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 2-4 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Mung beans require adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen may be applied if needed, but usually, the natural fixation suffices. Incorporate organic matter to improve soil fertility.",
                "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, beetles, and thrips. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Black Gram Cultivation Guide",
                "Introduction": "Black gram (Vigna mungo) is a highly nutritious legume valued for its high protein content and is widely used in various culinary dishes. This guide covers the complete process for cultivating black gram from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant black gram seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Black gram prefers well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides or insecticides to protect against soil-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the beginning of the monsoon season or during warm, dry conditions between June and July.",
                "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Black gram requires adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is generally not necessary due to nitrogen fixation. Incorporate organic matter to improve soil fertility.",
                "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and thrips. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Lentil Cultivation Guide",
                "Introduction": "Lentils (Lens culinaris) are nutritious legumes known for their high protein and fiber content. They are widely cultivated for food and are a staple in many cuisines. This guide covers the complete process for cultivating lentils from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant lentil seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Lentils prefer well-drained loamy or sandy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suited to your region. Treat seeds with fungicides or insecticides to protect against seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Lentils are typically planted in early spring or late winter, depending on the climate, when soil temperatures reach around 10-15°C (50-59°F).",
                "Spacing & Depth": "Space plants 25-30 cm apart in rows 45-60 cm apart. Sow seeds 2-3 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Lentils are drought-tolerant but need adequate moisture during germination and pod development. Water if rainfall is insufficient, particularly during flowering and seed filling.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is typically not needed due to nitrogen fixation. Incorporate organic matter to enhance soil fertility.",
                "Weed Control": "Control weeds during early growth using manual weeding or herbicides. Mulching can also help suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests such as aphids, lygus bugs, and root rots. Implement integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods turn brown and dry, typically 80-100 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and variable weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Pomegranate Cultivation Guide",
                "Introduction": "Pomegranates (Punica granatum) are nutritious fruits known for their health benefits and vibrant flavor. They are cultivated in many parts of the world and thrive in warm climates. This guide covers the complete process for cultivating pomegranates from planting to harvesting.",
                "Materials Required": "- High-quality pomegranate seeds or healthy seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium\n- Drip irrigation systems or furrow irrigation\n- Insecticides and fungicides for pest and disease management\n- Hand tools or tractors for planting, pruning, and maintenance",
                "Soil Preparation": "Pomegranates prefer well-drained, sandy loam to loamy soils with a pH of 5.5 to 7.0. Prepare the planting site by plowing and incorporating organic matter.",
                "Seed Selection & Treatment": "Choose disease-resistant varieties suitable for your region's climate. If using seeds, soak them overnight in water before planting to improve germination rates.",
                "Field Preparation": "Clear the site of weeds, rocks, and debris to ensure a clean planting environment.",
                "Planting Time": "Pomegranates are typically planted in spring after the last frost.",
                "Spacing & Depth": "Space plants 5-8 feet apart to allow for proper growth and air circulation. Plant seeds or seedlings at a depth of 1-2 inches, ensuring good soil contact.",
                "Seeding Methods": "Direct Seeding: Sow seeds directly into the prepared site. Transplanting: For seedlings, dig a hole slightly larger than the root ball and backfill with soil.",
                "Watering Requirements": "Pomegranates require regular watering, especially during the establishment phase; once established, they are drought-tolerant. Water deeply but infrequently to encourage deep root growth.",
                "Nutrient Management": "Apply a balanced fertilizer during the growing season, typically in early spring and again in late summer. Incorporate organic compost to improve soil fertility.",
                "Weed Control": "Control weeds using mulching and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and pomegranate butterflies. Implement integrated pest management (IPM) strategies, including the use of natural predators and organic pesticides.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests. Use mulch to retain moisture.\n- Vegetative stage: Regularly check for nutrient deficiencies and pest infestations; apply fertilizers as needed.\n- Flowering & fruit development: Ensure adequate water during flowering and fruit set to promote healthy development.",
                "Harvesting": "Pomegranates are typically ready for harvest 5-7 months after flowering, when the fruit has a deep color and makes a metallic sound when tapped. Use sharp pruning shears to cut the fruit from the tree, avoiding damage to the branches and other fruit.",
                "Post-Harvest Management": "Handle fruits gently to prevent bruising; store in a cool, dry place.",
                "Storage Conditions": "Store pomegranates in a cool, dry environment; they can last several weeks to months in proper conditions.",
                "Processing & Packaging": "Clean and sort harvested fruits, discarding any damaged or rotten ones. Pack fruits in breathable containers to maintain quality during storage.",
                "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and environmental stresses such as drought or excessive moisture. Use disease-resistant varieties, implement proper irrigation practices, and monitor pest populations to mitigate challenges."
            },

            {"name": "Kidney Bean Cultivation Guide",
                "Introduction": "Kidney beans (Phaseolus vulgaris) are a high-protein legume commonly used in various cuisines. This guide covers the complete process for cultivating kidney beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant kidney bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen as beans fix their own nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides for common kidney bean pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
                "Soil Preparation": "Kidney beans thrive in well-drained, loamy soils with a pH between 6.0 and 7.0. Prepare the field by plowing and harrowing to create a fine tilth for easy root penetration.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against early soil-borne diseases and pests.",
                "Field Preparation": "Clear the field of weeds and debris, then level it. Mark rows with adequate spacing for air circulation and sunlight penetration.",
                "Planting Time": "Kidney beans are typically planted in spring when soil temperatures reach 15°C (59°F) and there is no risk of frost.",
                "Spacing & Depth": "Plant seeds 3-5 cm deep, with 8-10 cm between plants and 45-60 cm between rows.",
                "Seeding Methods": "Direct Seeding: Sow seeds directly into the field by hand or using a seed drill.",
                "Watering Requirements": "Kidney beans need regular watering, particularly during flowering and pod development. Avoid overwatering, as beans are sensitive to waterlogging.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Limit nitrogen since kidney beans fix atmospheric nitrogen. Supplement micronutrients if soil tests indicate deficiencies.",
                "Weed Control": "Weed control is essential, particularly in the early stages. Use manual weeding or herbicides as needed. Mulching around plants can help retain moisture and suppress weeds.",
                "Pest & Disease Management": "Monitor for pests like aphids, leafhoppers, and bean beetles. Use integrated pest management (IPM) practices and apply pesticides if necessary. Prevent diseases like root rot and blight by practicing crop rotation and avoiding waterlogged soil.",
                "Special Care During Growth": "- Seedling stage: Ensure moderate soil moisture and protect seedlings from pests.\n- Vegetative stage: Maintain nutrient levels to support robust leaf and stem growth.\n- Flowering & pod-filling stage: Provide consistent moisture during pod development to enhance yield and seed quality.",
                "Harvesting": "Harvest kidney beans when the pods are fully mature and dry, usually 90-120 days after planting. For small farms, harvest by hand by pulling up the entire plant. For larger farms, use a combine harvester to gather beans efficiently.",
                "Post-Harvest Management": "Allow the harvested plants to dry in the sun to reduce moisture in the seeds. Thresh the beans to separate them from the pods, then clean the seeds.",
                "Storage Conditions": "Store kidney beans in a dry, well-ventilated place to prevent mold and insect infestations.",
                "Processing & Packaging": "Clean and grade the beans for quality assurance before packaging. Pack beans in breathable bags or containers to maintain quality during storage.",
                "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and nutrient imbalances. Use disease-resistant seeds, monitor soil health, and apply IPM practices to control pests and diseases effectively."
            },

            {"name": "Banana Cultivation Guide",
                "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
                "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
                "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
                "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
                "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
                "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
                "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
                "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Banana Cultivation Guide",
                "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
                "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
                "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
                "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
                "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
                "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
                "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
                "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },


            {"name": "Grape Cultivation Guide",
                "Introduction": "Grapes (Vitis vinifera and other species) are versatile fruits used for fresh eating, drying (raisins), and wine production. They thrive in temperate climates and require specific growing conditions to produce high-quality fruit. This guide outlines the complete process for cultivating grapes, from planting to harvesting.",
                "Materials Required": "- Quality grapevines, either bare-root or potted, from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (pruners, shovels) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Grapes prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-resistant grape varieties suitable for your climate and purpose (table grapes, wine grapes, etc.). Inspect vines for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant grapes is in early spring after the last frost or in the fall before the ground freezes.",
                "Spacing & Depth": "Space vines 6-10 feet apart in rows that are 8-10 feet apart to allow for proper air circulation and growth. Plant vines at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Grapes require regular watering during the first year to establish roots. Once established, they are drought-tolerant but still benefit from supplemental irrigation during dry spells, especially during fruit development.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Use organic compost to improve soil health.",
                "Weed Control": "Control weeds through mulching, hand weeding, or the use of herbicides to reduce competition for nutrients and moisture.",
                "Pest & Disease Management": "Monitor for pests such as grapevine moths, aphids, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and natural predators.",
                "Special Care During Growth": "- Young Vine Stage: Protect young vines from extreme weather and pests; use support stakes or trellises to help young plants grow upward.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to encourage a strong structure and air circulation.\n- Flowering & Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin clusters if necessary to promote larger fruit size.",
                "Harvesting": "Grapes are typically ready for harvest 4-6 months after flowering, depending on the variety. They should be harvested when fully ripe, showing deep color and sweet flavor. Use sharp pruning shears to cut clusters from the vine. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any damaged or rotten grapes and store them in a cool, shaded area.",
                "Storage Conditions": "Store grapes in a cool, dry place. Refrigeration can extend their shelf life, but they should be kept in breathable containers.",
                "Processing & Packaging": "If needed, grapes can be processed into products like grape juice, jelly, or wine. Pack grapes in breathable containers to allow airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, climate-related issues, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Muskmelon Cultivation Guide",
                "Introduction": "Muskmelons (Cucumis melo var. cantaloupe) are sweet, aromatic fruits known for their juicy flesh and distinctive netted skin. They thrive in warm climates and are popular for their refreshing taste. This guide outlines the complete process for cultivating muskmelons, from planting to harvesting.",
                "Materials Required": "- Quality muskmelon seeds or seedlings from reputable sources\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip or overhead irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, hoes, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Muskmelons prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and mixing in organic matter to enhance drainage and fertility.",
                "Plant Selection & Treatment": "Choose disease-resistant varieties suited for your climate and market. If using seeds, soak them in water for a few hours before planting to improve germination rates.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant muskmelons is after the last frost date when soil temperatures are consistently above 70°F (21°C).",
                "Spacing & Depth": "Space muskmelon plants 3-4 feet apart in rows that are 6-8 feet apart to allow for sprawling vines. Plant seeds or seedlings at a depth of about 1 inch.",
                "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly into the ground after the soil warms up. Transplanting: Start seedlings indoors and transplant them once they are strong enough.",
                "Watering Requirements": "Muskmelons need consistent moisture, especially during germination and fruit development. Aim for about 1-2 inches of water per week, adjusting for rainfall.",
                "Nutrient Management": "Apply a balanced fertilizer at planting and again when vines begin to run. Use organic compost or mulch to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, cucumber beetles, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological controls.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from pests and extreme weather. Use row covers if necessary to protect against pests and frost.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Support vines if necessary, especially when fruit begins to develop.\n- Fruit Development Stage: Ensure adequate water supply during fruit development to promote healthy growth and sweetness. Avoid watering directly on the fruit to prevent rot.",
                "Harvesting": "Muskmelons are typically ready for harvest 70-90 days after planting. Indicators include a change in color from green to yellow at the blossom end and a sweet aroma. Use a sharp knife or pruning shears to cut the fruit from the vine, leaving a short stem attached to the melon.",
                "Post-Harvest Management": "Handle harvested muskmelons gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store muskmelons at room temperature until they are fully ripe. Once ripe, they can be refrigerated for a short period to extend freshness.",
                "Processing & Packaging": "If needed, muskmelons can be processed into smoothies, sorbets, or fruit salads. Pack muskmelons in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses such as drought or excessive moisture, and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Apple Cultivation Guide",
                "Introduction": "Apples (Malus domestica) are one of the most popular fruits worldwide, appreciated for their taste, versatility, and nutritional value. They grow best in temperate climates and can be cultivated in various soil types. This guide outlines the complete process for cultivating apples, from planting to harvesting.",
                "Materials Required": "- Quality apple tree seedlings or grafted varieties from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Apples prefer well-drained, loamy soils with a pH of 6.0 to 7.0. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant apple varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant apple trees is in the fall or early spring when the trees are dormant.",
                "Spacing & Depth": "Space dwarf varieties 4-6 feet apart and standard varieties 10-15 feet apart to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young apple trees regularly to establish roots, especially during dry spells. Once established, they are drought-tolerant but benefit from deep watering during fruit development.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as codling moths, aphids, and spider mites. Manage diseases like apple scab and powdery mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger apples.",
                "Harvesting": "Apples are typically ready for harvest 4-6 months after flowering, depending on the variety. Indicators include a change in color, firm texture, and ease of detachment from the tree. Use sharp pruning shears to cut apples from the tree, leaving a short stem attached to the fruit.",
                "Post-Harvest Management": "Handle harvested apples gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store apples in a cool, dark place. They can be refrigerated to extend their shelf life.",
                "Processing & Packaging": "If needed, apples can be processed into applesauce, cider, or dried slices. Pack apples in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Orange Cultivation Guide",
                "Introduction": "Oranges (Citrus sinensis) are one of the most popular citrus fruits, valued for their sweet, juicy flesh and high vitamin C content. They thrive in warm, subtropical to tropical climates. This guide outlines the complete process for cultivating oranges, from planting to harvesting.",
                "Materials Required": "- Quality orange tree seedlings or grafted varieties from reputable nurseries\n- Citrus-specific fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Oranges prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 7.5. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant orange varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant orange trees is in the spring after the danger of frost has passed.",
                "Spacing & Depth": "Space trees 12-25 feet apart, depending on the rootstock and tree variety, to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young orange trees regularly to establish roots, especially during dry spells. Mature trees require deep watering during dry periods.",
                "Nutrient Management": "Apply a citrus-specific fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, spider mites, and citrus leaf miners. Manage diseases like citrus canker and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger oranges.",
                "Harvesting": "Oranges are typically ready for harvest 7-12 months after flowering, depending on the variety. Indicators include a change in color, firmness, and sweetness. Use sharp pruning shears to cut oranges from the tree, leaving a short stem attached to the fruit.",
                "Post-Harvest Management": "Handle harvested oranges gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store oranges in a cool, dark place. They can be refrigerated to extend their shelf life.",
                "Processing & Packaging": "If needed, oranges can be processed into juice, marmalade, or dried slices. Pack oranges in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Papaya Cultivation Guide",
                "Introduction": "Papayas (Carica papaya) are tropical fruit trees known for their sweet, juicy flesh and vibrant orange color. They thrive in warm climates and can produce fruit year-round under optimal conditions. This guide outlines the complete process for cultivating papayas, from planting to harvesting.",
                "Materials Required": "- Quality papaya seeds or seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Papayas prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance drainage and fertility.",
                "Plant Selection & Treatment": "Choose disease-resistant papaya varieties suited to your climate. If using seeds, soak them for a few hours before planting to improve germination rates.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant papayas is in the spring when temperatures are consistently warm.",
                "Spacing & Depth": "Space papaya plants 6-10 feet apart to allow for their large canopy and root system. Plant seeds or seedlings at a depth of about 0.5 to 1 inch.",
                "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly in the ground after the last frost.\nTransplanting: Start seedlings indoors and transplant them when they are about 12 inches tall.",
                "Watering Requirements": "Water young papaya plants regularly, especially during dry spells. Papayas require consistent moisture but do not tolerate waterlogging.",
                "Nutrient Management": "Apply a balanced fertilizer every 4-6 weeks during the growing season. Use organic compost to enhance soil fertility.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and fruit flies. Manage diseases like powdery mildew and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use row covers if necessary to shield from frost and insects.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune any dead or damaged leaves to promote healthy growth.\n- Fruit Development Stage: Ensure adequate water supply during fruit development. Thin excess fruits if necessary to allow for larger fruit size.",
                "Harvesting": "Papayas are typically ready for harvest 6-12 months after planting, depending on the variety. Indicators include a change in skin color from green to yellow and a sweet aroma. Use a sharp knife to cut the fruit from the tree, leaving a small portion of the stem attached.",
                "Post-Harvest Management": "Handle harvested papayas gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store papayas at room temperature to ripen further. Once ripe, they can be refrigerated for a short period to extend freshness.",
                "Processing & Packaging": "If needed, papayas can be processed into smoothies, salads, or dried fruit. Pack papayas in breathable containers to maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or flooding), and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Coffee Cultivation Guide",
                "Introduction": "Coffee (Coffea spp.) is one of the most widely consumed beverages globally, known for its stimulating properties and rich flavor. It thrives in tropical climates, typically at higher altitudes, where conditions are ideal for its growth. This guide outlines the complete process for cultivating coffee, from planting to harvesting.",
                "Materials Required": "- Quality coffee seedlings or seeds from reputable nurseries\n- Balanced fertilizers rich in nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Coffee prefers well-drained, loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant coffee varieties suitable for your climate. If using seeds, soak them for 24 hours to improve germination rates.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant coffee is at the beginning of the rainy season.",
                "Spacing & Depth": "Space coffee plants 5-8 feet apart to allow for proper growth and air circulation. Plant seedlings at a depth that matches their nursery height, ensuring the root collar is level with the soil surface.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the seedling in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young coffee plants regularly to establish roots, especially during dry spells. Mature plants prefer consistent moisture but should not be waterlogged.",
                "Nutrient Management": "Apply a balanced fertilizer every 3-4 months during the growing season. Use organic compost to enhance soil fertility.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as coffee borer beetles and leaf rust. Manage diseases like root rot and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use shade cloth if necessary to shield from intense sunlight.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape plants and remove any dead or diseased branches.\n- Flowering and Fruit Development Stage: Ensure adequate water supply during flowering and fruit set to maximize yield and fruit quality. Monitor for fruit fly infestations and control as necessary.",
                "Harvesting": "Coffee cherries are typically ready for harvest 7-9 months after flowering, depending on the variety. Indicators include a change in color from green to bright red or yellow. Harvest coffee cherries by hand, picking only the ripe ones. Use a selective picking method for quality.",
                "Post-Harvest Management": "Handle harvested cherries gently to avoid bruising. Process them as soon as possible to prevent spoilage.",
                "Processing Methods": "Use either the dry method (sun-drying cherries) or the wet method (fermenting and washing cherries) to extract the coffee beans.",
                "Storage Conditions": "Store processed coffee beans in a cool, dry place to prevent spoilage and maintain flavor.",
                "Processing & Packaging": "Pack coffee beans in airtight containers to help preserve freshness during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and fluctuating market prices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            }
        ]
    
    cropGuideHindi = [
        
            {
                "name": "मक्का की खेती गाइड",
                "Introduction": "मक्का (Zea mays), जिसे मकई के name से भी जाना जाता है, एक प्रमुख अनाज फसल है जिसे इसके दानों के लिए व्यापक रूप से उगाया जाता है। यह गाइड बीज चयन से लेकर कटाई तक मक्का की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले मक्का के बीज (संकर या सुधारित किस्में)\n- उर्वरक (नाइट्रोजन, फॉस्फोरस, पोटैशियम)\n- मशीनरी (ट्रैक्टर, हाथ उपकरण, बीज बोने की मशीन)\n- कीट नियंत्रण (हर्बिसाइड्स, कीटनाशक)\n- सिंचाई उपकरण (ड्रिप या फरो सिंचाई)",
                "Soil Preparation": "मक्का अच्छी जल निकासी वाली दोमट मिट्टी में अच्छी तरह से उगता है, जिसका pH 5.8 से 7.0 हो। मिट्टी को हवादार बनाने और ढेले तोड़ने के लिए जुताई करें।",
                "Seed Selection & Treatment": "उच्च उपज वाली, सूखा प्रतिरोधी किस्मों का चयन करें। बीजों को फफूंदनाशक या कीटनाशक से उपचारित करें।",
                "Field Preparation": "समान जल वितरण के लिए खेत को समतल करें। अधिकतम सूर्य के प्रकाश के लिए पंक्ति की दूरी को अनुकूलित करें।",
                "Planting Time": "आमतौर पर बारिश के मौसम की शुरुआत में, अप्रैल से जून के बीच बोया जाता है।",
                "Spacing & Depth": "पंक्तियों में 20-25 सेमी और पंक्तियों के बीच 60-75 सेमी की दूरी पर बीज बोएं, 2-5 सेमी की गहराई पर।",
                "Seeding Methods": "- **सीधी बुवाई:** बीजों को हाथ से या बीज बोने की मशीन से बोएं।",
                "Watering Requirements": "मक्का को नियमित सिंचाई की आवश्यकता होती है, विशेष रूप से सिल्किंग और टैसलिंग के दौरान। यदि बारिश कम हो तो सिंचाई का उपयोग करें।",
                "Nutrient Management": "उर्वरकों को विभाजित मात्रा में लगाएं: बुवाई के समय, प्रारंभिक विकास के दौरान और टैसलिंग के दौरान।",
                "Weed Control": "हाथ से निराई, होइंग या हर्बिसाइड्स का उपयोग करें। पहली निराई 15-20 दिनों के बाद और दूसरी 30-40 दिनों के बाद करें।",
                "Pest & Disease Management": "मक्का बोरर, आर्मीवर्म और एफिड्स के लिए निगरानी करें। कीटनाशक और एकीकृत कीट प्रबंधन (IPM) का उपयोग करें।",
                "Harvesting": "जब मक्का के भुट्टे पक जाएं और भूसी सूख जाए तो कटाई करें। नमी की मात्रा 20-25% होनी चाहिए। हाथ से या मशीन से कटाई करें।",
                "Post-Harvest Management": "दानों को 13-14% नमी तक सुखाएं। छिलके निकालें, साफ करें और ठीक से भंडारण करें।",
                "Storage Conditions": "दानों को ठंडी, सूखी और हवादार जगह पर रखें ताकि फफूंद और कीटों से बचाव हो सके।",
                "Processing": "यदि आवश्यक हो, तो मक्का को सुखाकर पीस लें।",
                "Challenges & Solutions": "सामान्य समस्याएं: मौसम में परिवर्तन, कीट और पानी की कमी। समाधान: IPM, मिट्टी की नमी की निगरानी और प्रतिरोधी किस्में।"
            },

            {
                "name": "चावल की खेती गाइड",
                "Introduction": "चावल (Oryza sativa) दुनिया के कई हिस्सों में एक मुख्य खाद्य फसल है। यह गाइड बीज चयन से लेकर कटाई तक चावल की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले बीज\n- उर्वरक (नाइट्रोजन, फॉस्फोरस, पोटैशियम)\n- सिंचाई प्रणाली\n- मशीनरी (ट्रैक्टर, रोपाई मशीन, सिकल)\n- कीट नियंत्रण (हर्बिसाइड्स, कीटनाशक)",
                "Soil Preparation": "चावल मिट्टी या मिट्टी-दोमट मिट्टी में सबसे अच्छा उगता है, जिसका pH 5.5 से 6.5 हो। मिट्टी को जोतकर और समतल करें।",
                "Seed Selection & Treatment": "उच्च उपज वाले, कीट प्रतिरोधी बीजों का उपयोग करें। बीजों को फफूंदनाशक या कीटनाशक से उपचारित करें।",
                "Field Preparation": "खेत को समतल करें और पानी को रोकने के लिए मेड़ बनाएं।",
                "Planting Time": "बारिश के मौसम की शुरुआत में, आमतौर पर मई से जून के बीच बोया जाता है।",
                "Spacing & Depth": "रोपाई के लिए 20x15 सेमी की दूरी का उपयोग करें। सीधी बुवाई के लिए 2-3 सेमी की गहराई पर बोएं।",
                "Seeding Methods": "- **सीधी बुवाई:** बीजों को छिड़काव या पंक्तियों में बोएं।\n- **रोपाई:** नर्सरी में उगाएं और 20-30 दिनों के बाद पौधों को स्थानांतरित करें।",
                "Watering Requirements": "विकास के दौरान 5-10 सेमी पानी बनाए रखें। दाने पकने के दौरान पानी कम करें।",
                "Nutrient Management": "उर्वरकों को विभाजित मात्रा में लगाएं: बुवाई के समय, टिलरिंग के दौरान और पैनिकल इनिशिएशन के दौरान।",
                "Weed Control": "हाथ से निराई या हर्बिसाइड्स का उपयोग करें। रोपाई के 15-20 दिनों के बाद और फिर 40 दिनों के बाद निराई करें।",
                "Pest & Disease Management": "स्टेम बोरर और लीफहॉपर जैसे कीटों के लिए निगरानी करें। कीटनाशक और एकीकृत कीट प्रबंधन (IPM) का उपयोग करें।",
                "Harvesting": "जब दाने सुनहरे पीले हो जाएं और 80-90% दाने पक जाएं तो कटाई करें। छोटे खेतों के लिए सिकल का उपयोग करें, बड़े खेतों के लिए मशीन का उपयोग करें।",
                "Post-Harvest Management": "दानों को 14% नमी तक सुखाएं, फिर भंडारण करें।",
                "Challenges & Solutions": "सामान्य समस्याएं: प्रतिकूल मौसम, कीट और पानी की कमी। समाधान: IPM, पानी के स्तर की निगरानी और फसल विविधीकरण।"
            },

            {
                "name": "जूट की खेती गाइड",
                "Introduction": "जूट एक रेशेदार फसल है जिसे मुख्य रूप से इसके मजबूत, प्राकृतिक रेशों के लिए उगाया जाता है, जो कपड़े और पैकेजिंग में व्यापक रूप से उपयोग किए जाते हैं। यह गाइड बीज चयन से लेकर कटाई तक जूट की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, प्रमाणित जूट के बीज (Corchorus olitorius या Corchorus capsularis)\n- जैविक खाद, नाइट्रोजन, फॉस्फोरस और पोटैशियम उर्वरक\n- Soil Preparation के लिए हाथ उपकरण या ट्रैक्टर\n- कीट नियंत्रण के लिए हर्बिसाइड्स और कीटनाशक\n- नियंत्रित सिंचाई के लिए सिंचाई प्रणाली",
                "Soil Preparation": "जूट दोमट, बलुई दोमट मिट्टी में सबसे अच्छा उगता है, जिसका pH 6.0 से 7.5 हो। मिट्टी को जोतकर और समतल करें।",
                "Seed Selection & Treatment": "उच्च उपज वाले और रोग प्रतिरोधी बीजों का चयन करें। बुवाई से पहले बीजों को 24 घंटे के लिए पानी में भिगोएं।",
                "Field Preparation": "खेत को साफ करें और समतल करें। यदि बाढ़ की संभावना हो तो छोटे मेड़ बनाएं।",
                "Planting Time": "जूट आमतौर पर मानसून की शुरुआत में, मार्च से मई के बीच बोया जाता है।",
                "Spacing & Depth": "पंक्तियों में 25-30 सेमी की दूरी पर बीज बोएं। बीजों को 1-2 सेमी की गहराई पर बोएं।",
                "Seeding Methods": "- **छिड़काव:** बीजों को खेत में समान रूप से छिड़कें।\n- **पंक्ति बुवाई:** बीजों को पंक्तियों में बोएं।",
                "Watering Requirements": "जूट को नियमित नमी की आवश्यकता होती है। भारी बारिश के बाद जल निकासी सुनिश्चित करें।",
                "Nutrient Management": "बुवाई के समय नाइट्रोजन, फॉस्फोरस और पोटैशियम उर्वरक लगाएं। 20-25 दिनों के बाद अतिरिक्त नाइट्रोजन लगाएं।",
                "Weed Control": "हाथ से निराई या हर्बिसाइड्स का उपयोग करें। बुवाई के 15-20 दिनों के बाद और फिर 30-40 दिनों के बाद निराई करें।",
                "Pest & Disease Management": "जूट के कीटों जैसे जूट हेयरी कैटरपिलर और एफिड्स के लिए निगरानी करें। कीटनाशक या एकीकृत कीट प्रबंधन (IPM) का उपयोग करें।",
                "Harvesting": "जब पौधे 10-12 फीट लंबे हो जाएं और निचली पत्तियां पीली होने लगें तो कटाई करें। सिकल या चाकू का उपयोग करें।",
                "Post-Harvest Management": "कटाई के बाद पौधों को बांधकर साफ, धीमी गति वाले पानी में डुबोएं। रेटिंग प्रक्रिया 10-15 दिनों तक चलती है।",
                "Challenges & Solutions": "सामान्य समस्याएं: पानी की उपलब्धता, कीट और अनुचित रेटिंग। समाधान: कुशल सिंचाई और कीट नियंधन का उपयोग करें।"
            },

            {
                "name": "कपास की खेती गाइड",
                "Introduction": "कपास एक प्रमुख रेशेदार फसल है जिसे इसके नरम, रूईदार रेशों के लिए उगाया जाता है, जो कपड़े बनाने में उपयोग किए जाते हैं। यह गाइड बीज चयन से लेकर कटाई तक कपास की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, प्रमाणित कपास के बीज (जैसे Bt कपास या अन्य कीट प्रतिरोधी किस्में)\n- नाइट्रोजन, फॉस्फोरस, पोटैशियम और सूक्ष्म पोषक तत्व उर्वरक\n- ड्रिप या फरो सिंचाई प्रणाली\n- कीट और बीमारी नियंत्रण के लिए हर्बिसाइड्स और कीटनाशक\n- खेत की तैयारी और रखरखाव के लिए हल, ट्रैक्टर और स्प्रेयर",
                "Soil Preparation": "कपास अच्छी जल निकासी वाली बलुई दोमट मिट्टी में सबसे अच्छा उगता है, जिसका pH 6.0 से 7.5 हो। खेत को गहरी जुताई करके और ढेले तोड़कर तैयार करें।",
                "Seed Selection & Treatment": "उच्च उपज वाले, कीट प्रतिरोधी बीजों का चयन करें। बीजों को फफूंदनाशक या कीटनाशक से उपचारित करें।",
                "Field Preparation": "बुवाई के लिए फरो या बेड बनाएं। जल निकासी सुनिश्चित करें।",
                "Planting Time": "कपास आमतौर पर वसंत ऋतु में, मार्च से मई के बीच बोया जाता है।",
                "Spacing & Depth": "बीजों को 3-5 सेमी की गहराई पर बोएं, पंक्तियों के बीच 75-100 सेमी और पौधों के बीच 25-30 सेमी की दूरी रखें।",
                "Seeding Methods": "- **सीधी बुवाई:** बीजों को तैयार फरो या बेड में सीधे बोएं।",
                "Watering Requirements": "कपास को नियमित नमी की आवश्यकता होती है, विशेष रूप से फूल आने और बोल बनने के दौरान। ड्रिप या फरो सिंचाई का उपयोग करें।",
                "Nutrient Management": "बुवाई के समय फॉस्फोरस और पोटैशियम उर्वरक लगाएं। नाइट्रोजन को विभाजित मात्रा में लगाएं: एक तिहाई बुवाई के समय, एक तिहाई वानस्पतिक विकास के दौरान और एक तिहाई फूल आने के दौरान।",
                "Weed Control": "हाथ से निराई, होइंग या हर्बिसाइड्स का उपयोग करें। बुवाई के 20-30 दिनों के बाद और फिर 45 दिनों के बाद निराई करें।",
                "Pest & Disease Management": "बोलवर्म, एफिड्स और व्हाइटफ्लाइ जैसे कीटों के लिए निगरानी करें। एकीकृत कीट प्रबंधन (IPM) का उपयोग करें।",
                "Harvesting": "जब बोल पूरी तरह से खुल जाएं और रूई फूल जाए तो कटाई करें। छोटे खेतों के लिए हाथ से कटाई करें, बड़े खेतों के लिए मशीन का उपयोग करें।",
                "Post-Harvest Management": "कटाई के बाद रूई को छायादार, हवादार जगह पर सुखाएं। बीजों को अलग करें और रूई को साफ करके भंडारण करें।",
                "Challenges & Solutions": "सामान्य समस्याएं: कीट, पानी की उपलब्धता और मिट्टी की पोषक तत्वों की कमी। समाधान: सूखा प्रतिरोधी किस्में, कुशल सिंचाई और IPM का उपयोग करें।"
            },

            {
                "name": "नारियल की खेती गाइड",
                "Introduction": "नारियल (Cocos nucifera) एक प्रमुख फल है जिसे इसके तेल, दूध और रेशों के लिए उगाया जाता है। यह गाइड बीज चयन से लेकर कटाई तक नारियल की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले नारियल के पौधे (बौनी या लंबी किस्में)\n- जैविक खाद, NPK उर्वरक\n- ड्रिप या बेसिन सिंचाई\n- कीटनाशक या जैविक नियंत्रण एजेंट\n- हाथ उपकरण या मशीनरी",
                "Soil Preparation": "नारियल अच्छी जल निकासी वाली बलुई दोमट मिट्टी में सबसे अच्छा उगता है, जिसका pH 5.5-7.5 हो। 1 x 1 x 1 मीटर के गड्ढे खोदें और उन्हें मिट्टी, खाद और जैविक खाद से भरें।",
                "Seed Selection & Treatment": "रोग प्रतिरोधी, उच्च उपज वाले पौधों का चयन करें। बौनी किस्में आसान कटाई के लिए उपयुक्त हैं, जबकि लंबी किस्में सूखा प्रतिरोधी हैं।",
                "Field Preparation": "खेत को साफ करें और जल निकासी सुनिश्चित करें। पौधों के बीच उचित दूरी रखें।",
                "Planting Time": "बारिश के मौसम की शुरुआत में लगाएं ताकि सिंचाई की आवश्यकता कम हो।",
                "Spacing & Depth": "लंबी किस्मों के लिए 7.5-9 मीटर की दूरी रखें; बौनी किस्मों के लिए 6.5-7 मीटर। जड़ों को अच्छी तरह से ढकें।",
                "Seeding Methods": "पौधों को गड्ढे में लगाएं, जड़ गर्दन जमीन से ऊपर रखें।",
                "Watering Requirements": "पहले तीन वर्षों तक नियमित सिंचाई करें। परिपक्व पेड़ सूखा प्रतिरोधी होते हैं लेकिन नियमित सिंचाई से लाभ होता है।",
                "Nutrient Management": "संतुलित उर्वरक साल में तीन बार लगाएं। सालाना जैविक खाद डालें।",
                "Weed Control": "नियमित निराई करें, विशेष रूप से प्रारंभिक विकास के दौरान। मल्चिंग से नमी बनाए रखें।",
                "Pest & Disease Management": "राइनोसेरोस बीटल और रेड पाम वीविल जैसे कीटों को नियंत्रित करें। रूट विल्ट और बड रोट को प्रबंधित करें।",
                "Harvesting": "नारियल 12 महीने के बाद पक जाते हैं। हर 45-60 दिनों में कटाई करें।",
                "Post-Harvest Management": "नारियल को सुखाकर भंडारण करें।",
                "Challenges & Solutions": "सूखा, कीट और मिट्टी की कमी को ड्रिप सिंचाई, कीट प्रबंधन और जैविक खाद से नियंत्रित करें।"
            },

            {
                "name": "चने की खेती गाइड",
                "Introduction": "चना (Cicer arietinum) एक प्रमुख दलहनी फसल है जिसे इसके प्रोटीन युक्त दानों के लिए उगाया जाता है। यह गाइड बीज चयन से लेकर कटाई तक चने की खेती की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग प्रतिरोधी चने के बीज (देसी या काबुली प्रकार)\n- फॉस्फोरस आधारित उर्वरक; न्यूनतम नाइट्रोजन\n- ड्रिप या स्प्रिंकलर सिंचाई\n- हर्बिसाइड्स और कीटनाशक\n- हल, ट्रैक्टर और स्प्रेयर",
                "Soil Preparation": "चना अच्छी जल निकासी वाली दोमट मिट्टी में सबसे अच्छा उगता है, जिसका pH 6.0-7.5 हो। मिट्टी को जोतकर और हैरो करके तैयार करें।",
                "Seed Selection & Treatment": "उच्च उपज वाले, रोग प्रतिरोधी बीजों का चयन करें। बीजों को राइजोबियम बैक्टीरिया से उपचारित करें।",
                "Field Preparation": "खेत को साफ करें और समतल करें। पंक्तियों के बीच उचित दूरी रखें।",
                "Planting Time": "ठंडे, शुष्क मौसम में, आमतौर पर अक्टूबर-नवंबर में बोया जाता है।",
                "Spacing & Depth": "पौधों के बीच 30-40 सेमी और पंक्तियों के बीच 45-60 सेमी की दूरी रखें। बीजों को 5-8 सेमी की गहराई पर बोएं।",
                "Seeding Methods": "सीधी बुवाई का उपयोग करें।",
                "Watering Requirements": "चने को कम पानी की आवश्यकता होती है, लेकिन फूल आने और फली भरने के दौरान सिंचाई करें।",
                "Nutrient Management": "बुवाई के समय फॉस्फोरस उर्वरक लगाएं। मिट्टी परीक्षण के आधार पर पोटैशियम और सूक्ष्म पोषक तत्व लगाएं।",
                "Weed Control": "हाथ से निराई या हर्बिसाइड्स का उपयोग करें। बुवाई के 20-30 दिनों के बाद और फिर 45-50 दिनों के बाद निराई करें।",
                "Pest & Disease Management": "पॉड बोरर और एफिड्स जैसे कीटों के लिए निगरानी करें। एकीकृत कीट प्रबंधन (IPM) का उपयोग करें।",
                "Harvesting": "चने 3-4 महीने में पक जाते हैं। जब पौधे पीले पड़ जाएं और फलियां सूख जाएं तो कटाई करें।",
                "Post-Harvest Management": "दानों को सुखाकर भंडारण करें।",
                "Challenges & Solutions": "सामान्य समस्याएं: कीट, बीमारियां और पोषक तत्वों की कमी। समाधान: IPM, प्रतिरोधी किस्में और मिट्टी परीक्षण का उपयोग करें।"
            },

            {
                "name": "चने की खेती का मार्गदर्शिका",
                "Introduction": "चना (Cicer arietinum) एक लोकप्रिय फलीदार फसल है जिसे इसके प्रोटीन से भरपूर बीजों के लिए उगाया जाता है, जो खाद्य उत्पादन में व्यापक रूप से उपयोग किए जाते हैं। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक चने की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी चने के बीज (देसी या काबुली प्रकार)\n- फास्फोरस आधारित उर्वरक; न्यूनतम नाइट्रोजन\n- ड्रिप या स्प्रिंकलर सिंचाई\n- खरपतवारनाशक और कीटनाशक\n- हल, ट्रैक्टर और स्प्रेयर",
                "Soil Preparation": "चने की खेती के लिए अच्छी जल निकासी वाली दोमट मिट्टी जिसका पीएच 6.0-7.5 हो, सबसे उपयुक्त है। अच्छे जड़ प्रवेश के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "उच्च उपज देने वाले, रोग-प्रतिरोधी बीज चुनें। नाइट्रोजन स्थिरीकरण के लिए राइजोबियम बैक्टीरिया से और रोगों को रोकने के लिए फफूंदनाशकों से उपचारित करें।",
                "Field Preparation": "खरपतवार साफ करें और खेत को समतल करें। हवा के संचार की अनुमति देने और रोग के जोखिम को कम करने के लिए पंक्तियों को उचित दूरी पर रखें।",
                "Planting Time": "ठंडे, शुष्क मौसम में सबसे अच्छा लगाया जाता है, आमतौर पर अक्टूबर-नवंबर में।",
                "Spacing & Depth": "पौधों को पंक्तियों में 30-40 सेमी की दूरी पर और पंक्तियों को 45-60 सेमी की दूरी पर रखें। मिट्टी की नमी के आधार पर बीज 5-8 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "चने को न्यूनतम पानी की आवश्यकता होती है लेकिन फूल आने और फली भरने के दौरान सिंचाई से लाभ होता है। जलभराव से बचें।",
                "Nutrient Management": "बुवाई के समय फास्फोरस डालें। मिट्टी परीक्षण के आधार पर आवश्यकतानुसार पोटैशियम और सूक्ष्म पोषक तत्वों का उपयोग करें।",
                "Weed Control": "जल्दी और नियमित रूप से निराई करें, या तो मैनुअल रूप से या खरपतवारनाशकों के साथ। पहली निराई 20-30 दिनों पर, यदि आवश्यक हो तो दूसरी 45-50 दिनों पर करें।",
                "Pest & Disease Management": "फली छेदक और एफिड जैसे कीटों की निगरानी करें। आवश्यकतानुसार एकीकृत कीट प्रबंधन (IPM) और जैव-कीटनाशकों का उपयोग करें।",
                "Special Care During Growth": "- अंकुरण अवस्था: कीटों से बचाव करें, मध्यम नमी बनाए रखें।\n- वानस्पतिक अवस्था: फास्फोरस स्तर बनाए रखें।\n- फूल और फली भरने की अवस्था: इष्टतम उपज के लिए पर्याप्त नमी सुनिश्चित करें।",
                "Harvesting": "चने 3-4 महीने में पकते हैं। जब पौधे पीले हो जाएं और फलियां सूख जाएं तब कटाई करें। छोटे खेतों के लिए हाथ से काटें; बड़े पैमाने पर खेती के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "भंडारण या बिक्री से पहले नमी कम करने के लिए बीजों को धूप में सुखाएं, थ्रेश करें और साफ करें।",
                "Storage Conditions": "कीट संक्रमण और खराब होने से बचाने के लिए सूखे, ठंडे स्थानों पर वेंटिलेशन के साथ स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग, पानी का तनाव और पोषक तत्वों की कमी शामिल है। जोखिमों को कम करने के लिए IPM, प्रतिरोधी किस्मों और मिट्टी परीक्षण का उपयोग करें।"
            },

            {
                "name": "अरहर की खेती का मार्गदर्शिका",
                "Introduction": "अरहर (Cajanus cajan) एक सूखा-प्रतिरोधी फलीदार फसल है जिसे इसकी उच्च प्रोटीन सामग्री और विभिन्न व्यंजनों में उपयोग के लिए महत्व दिया जाता है। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक अरहर की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी अरहर के बीज (जल्दी, मध्यम, या देर से पकने वाली किस्में)\n- नाइट्रोजन, फास्फोरस और पोटैशियम उर्वरक; न्यूनतम नाइट्रोजन की आवश्यकता\n- ड्रिप या फरो सिंचाई उपकरण\n- अरहर के कीटों के लिए विशिष्ट खरपतवारनाशक और कीटनाशक\n- Soil Preparation, रोपण और निराई के लिए हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "अरहर अच्छी जल निकासी वाली बलुई दोमट से लेकर चिकनी दोमट मिट्टी में सबसे अच्छी तरह से उगती है, जिसका पीएच 6.0-7.5 हो। एक अच्छे बीज बिस्तर बनाने के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "अपने क्षेत्र के लिए उपयुक्त उच्च उपज देने वाली, रोग-प्रतिरोधी किस्मों का चयन करें। बीज जनित रोगों को रोकने के लिए बीजों को फफूंदनाशकों से उपचारित करें।",
                "Field Preparation": "खरपतवार और मलबे से खेत को साफ करें, अच्छी जल निकासी सुनिश्चित करें।",
                "Planting Time": "आमतौर पर बारिश के मौसम की शुरुआत में या उपोष्णकटिबंधीय क्षेत्रों में शुष्क मौसम के दौरान लगाया जाता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 30-40 सेमी की दूरी पर और पंक्तियों को 60-75 सेमी की दूरी पर रखें। मिट्टी की नमी और बनावट के आधार पर बीज 3-5 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "अरहर सूखा-प्रतिरोधी है लेकिन फूल और फली विकास के दौरान पर्याप्त नमी की आवश्यकता होती है। सिंचाई की आवश्यकता हो सकती है, विशेष रूप से पहले 60 दिनों में।",
                "Nutrient Management": "बुवाई के समय फास्फोरस और पोटैशियम डालें और यदि आवश्यक हो तो नाइट्रोजन का टॉप-ड्रेसिंग करें। जैविक संशोधन मिट्टी की उर्वरता में सुधार कर सकते हैं।",
                "Weed Control": "प्रारंभिक विकास चरणों के दौरान मैनुअल निराई या खरपतवारनाशकों का उपयोग करके खरपतवार नियंत्रित करें। मल्चिंग खरपतवार को दबाने और मिट्टी की नमी बनाए रखने में मदद कर सकती है।",
                "Pest & Disease Management": "फली छेदक, एफिड और सफेदमक्खी जैसे कीटों की निगरानी करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिसमें जैविक नियंत्रण और आवश्यकतानुसार रासायनिक कीटनाशक शामिल हैं।",
                "Special Care During Growth": "- अंकुरण अवस्था: युवा अंकुरों को कीटों से बचाएं और मिट्टी की नमी बनाए रखें।\n- वानस्पतिक अवस्था: मजबूत विकास के लिए पर्याप्त पोषक तत्व सुनिश्चित करें।\n- फूल और फली भरने की अवस्था: उपज और बीज गुणवत्ता को अधिकतम करने के लिए लगातार नमी बनाए रखें।",
                "Harvesting": "अरहर 4-6 महीने में पकती है। जब फलियां पक जाएं और सूख जाएं तब कटाई करें। छोटे खेतों के लिए हाथ से काटें या बड़े पैमाने पर खेती के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "बीज की नमी सामग्री को कम करने के लिए थ्रेशिंग से पहले कटी हुई फसल को धूप में सुखाएं।",
                "Storage Conditions": "खराब होने और कीट संक्रमण को रोकने के लिए अरहर को सूखे, ठंडे और अच्छे वेंटिलेशन वाले क्षेत्र में स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग या कंटेनरों में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट संक्रमण, रोग, पानी का तनाव और पोषक तत्वों की कमी शामिल हैं। जोखिमों को प्रबंधित करने के लिए रोग-प्रतिरोधी किस्मों का उपयोग करें, फसल चक्र का अभ्यास करें और IPM रणनीतियों को लागू करें।"
            },

            {
                "name": "मोठ की खेती का मार्गदर्शिका",
                "Introduction": "मोठ (Vigna aconitifolia) एक सूखा-प्रतिरोधी फलीदार फसल है जो आमतौर पर शुष्क क्षेत्रों में उगाई जाती है। इन्हें उच्च प्रोटीन सामग्री और पाक अनुप्रयोगों के लिए महत्व दिया जाता है। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक मोठ की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी मोठ के बीज\n- फास्फोरस और पोटैशियम उर्वरक; न्यूनतम नाइट्रोजन\n- ड्रिप या फरो सिंचाई\n- खरपतवारनाशक और कीटनाशक\n- हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "मोठ अच्छी जल निकासी वाली बलुई दोमट या चिकनी मिट्टी में फलती-फूलती है, जिसका पीएच 6.0-8.0 हो। एक अच्छे बीज बिस्तर के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "उच्च उपज देने वाली, सूखा-सहिष्णु किस्मों का चयन करें। बीज जनित रोगों को रोकने के लिए बीजों को फफूंदनाशक या कीटनाशकों से उपचारित करें।",
                "Field Preparation": "अच्छे बीज-से-मिट्टी संपर्क सुनिश्चित करने के लिए खेत को खरपतवार और मलबे से साफ करें।",
                "Planting Time": "आमतौर पर मानसून के मौसम की शुरुआत में, जून और जुलाई के बीच बोया जाता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 30-45 सेमी की दूरी पर और पंक्तियों को 60-75 सेमी की दूरी पर रखें। मिट्टी की नमी के आधार पर बीज 3-5 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "मोठ सूखा-प्रतिरोधी है लेकिन फूल और फली विकास के दौरान लगातार नमी से लाभ होता है। यदि वर्षा अपर्याप्त है तो पानी दें।",
                "Nutrient Management": "बुवाई के समय फास्फोरस और पोटैशियम डालें। नाइट्रोजन का उपयोग केवल तभी करें जब मिट्टी परीक्षण कमी का संकेत दें। जैविक संशोधन मिट्टी की उर्वरता में सुधार करते हैं।",
                "Weed Control": "मैनुअल निराई या खरपतवारनाशकों के साथ जल्दी खरपतवार नियंत्रित करें। मल्चिंग खरपतवार को दबाने और मिट्टी की नमी बनाए रखने में मदद करती है।",
                "Pest & Disease Management": "एफिड, फली छेदक और लीफहॉपर जैसे कीटों की निगरानी करें। आवश्यकतानुसार एकीकृत कीट प्रबंधन (IPM) रणनीतियों का उपयोग करें।",
                "Special Care During Growth": "- अंकुरण अवस्था: मध्यम नमी बनाए रखें और कीटों से बचाव करें।\n- वानस्पतिक अवस्था: पर्याप्त पोषक तत्व सुनिश्चित करें।\n- फूल और फली भरने की अवस्था: इष्टतम उपज के लिए नमी बनाए रखें।",
                "Harvesting": "जब फलियां पक जाएं और सूख जाएं, आमतौर पर बुवाई के 90-120 दिनों बाद कटाई करें। छोटे खेतों के लिए हाथ से कटाई करें; बड़े पैमाने पर संचालन के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "नमी सामग्री को कम करने के लिए थ्रेशिंग से पहले पौधों को धूप में सुखाएं।",
                "Storage Conditions": "खराब होने और कीट संक्रमण को रोकने के लिए सूखे, ठंडे स्थानों पर वेंटिलेशन के साथ स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग और प्रतिकूल मौसम शामिल हैं। जोखिमों को कम करने के लिए सूखा-प्रतिरोधी किस्मों, IPM प्रथाओं और उचित मिट्टी प्रबंधन का उपयोग करें।"
            },

            {
                "name": "मूंग की खेती का मार्गदर्शिका",
                "Introduction": "मूंग (Vigna radiata) छोटी, हरी फलीदार फसलें हैं जिन्हें उनकी पोषण सामग्री और पाक बहुमुखी प्रतिभा के लिए अत्यधिक महत्व दिया जाता है। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक मूंग की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी मूंग के बीज\n- नाइट्रोजन, फास्फोरस और पोटैशियम उर्वरक (न्यूनतम नाइट्रोजन की आवश्यकता)\n- ड्रिप या फरो सिंचाई\n- खरपतवारनाशक और कीटनाशक\n- हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "मूंग अच्छी जल निकासी वाली बलुई दोमट से लेकर दोमट मिट्टी पसंद करती है जिसका पीएच 6.0-7.5 हो। एक अच्छे बीज बिस्तर प्राप्त करने के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "अपनी जलवायु के लिए उपयुक्त उच्च उपज देने वाली, रोग-प्रतिरोधी किस्मों का चयन करें। मिट्टी जनित रोगों से बचाव के लिए बीजों को फफूंदनाशकों से उपचारित करें।",
                "Field Preparation": "अच्छे बीज-से-मिट्टी संपर्क सुनिश्चित करने के लिए खेत को खरपतवार और मलबे से साफ करें।",
                "Planting Time": "आमतौर पर बारिश के मौसम की शुरुआत में या अप्रैल और जून के बीच गर्म, शुष्क परिस्थितियों में बोया जाता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 30-40 सेमी की दूरी पर और पंक्तियों को 45-60 सेमी की दूरी पर रखें। मिट्टी की नमी के आधार पर बीज 2-4 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "मूंग को पर्याप्त नमी की आवश्यकता होती है, विशेष रूप से अंकुरण और फूल आने के दौरान। यदि वर्षा अपर्याप्त है तो पानी दें, जड़ सड़न को रोकने के लिए अधिक पानी न दें।",
                "Nutrient Management": "बुवाई के समय फास्फोरस और पोटैशियम डालें। अतिरिक्त नाइट्रोजन यदि आवश्यक हो तो लगाया जा सकता है, लेकिन आमतौर पर, प्राकृतिक स्थिरीकरण पर्याप्त होता है। मिट्टी की उर्वरता में सुधार के लिए जैविक पदार्थ शामिल करें।",
                "Weed Control": "मैनुअल निराई या खरपतवारनाशकों के माध्यम से जल्दी खरपतवार नियंत्रित करें। मल्चिंग खरपतवार को दबाने और मिट्टी की नमी संरक्षित करने में मदद करती है।",
                "Pest & Disease Management": "एफिड, बीटल और थ्रिप्स जैसे कीटों की निगरानी करें। आवश्यकतानुसार एकीकृत कीट प्रबंधन (IPM) रणनीतियों का उपयोग करें।",
                "Special Care During Growth": "- अंकुरण अवस्था: युवा अंकुरों को कीटों से बचाएं और पर्याप्त नमी बनाए रखें।\n- वानस्पतिक अवस्था: मजबूत विकास के लिए पर्याप्त पोषक तत्व सुनिश्चित करें।\n- फूल और फली भरने की अवस्था: इष्टतम उपज और गुणवत्ता के लिए नमी बनाए रखें।",
                "Harvesting": "जब फलियां पक जाएं और सूख जाएं, आमतौर पर बुवाई के 60-90 दिनों बाद कटाई करें। छोटे खेतों के लिए हाथ से कटाई करें; बड़े पैमाने पर संचालन के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "नमी सामग्री को कम करने के लिए थ्रेशिंग से पहले पौधों को धूप में सुखाएं।",
                "Storage Conditions": "खराब होने और कीट संक्रमण को रोकने के लिए सूखे, ठंडे स्थानों पर वेंटिलेशन के साथ स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग और प्रतिकूल मौसम शामिल हैं। जोखिमों को कम करने के लिए रोग-प्रतिरोधी किस्मों, IPM प्रथाओं और उचित मिट्टी और जल प्रबंधन का उपयोग करें।"
            },

            {
                "name": "उड़द की खेती का मार्गदर्शिका",
                "Introduction": "उड़द (Vigna mungo) एक अत्यधिक पौष्टिक फलीदार फसल है जिसे इसकी उच्च प्रोटीन सामग्री के लिए महत्व दिया जाता है और इसका उपयोग विभिन्न व्यंजनों में व्यापक रूप से किया जाता है। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक उड़द की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी उड़द के बीज\n- फास्फोरस और पोटैशियम उर्वरक (न्यूनतम नाइट्रोजन की आवश्यकता)\n- ड्रिप या फरो सिंचाई\n- खरपतवारनाशक और कीटनाशक\n- हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "उड़द अच्छी जल निकासी वाली बलुई दोमट से लेकर चिकनी दोमट मिट्टी पसंद करता है जिसका पीएच 6.0-7.5 हो। अच्छा बीज बिस्तर बनाने के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "अपनी जलवायु के लिए उपयुक्त उच्च उपज देने वाली, रोग-प्रतिरोधी किस्मों का चयन करें। मिट्टी जनित रोगों से बचाव के लिए बीजों को फफूंदनाशकों या कीटनाशकों से उपचारित करें।",
                "Field Preparation": "अच्छे बीज-से-मिट्टी संपर्क सुनिश्चित करने के लिए खेत को खरपतवार और मलबे से साफ करें।",
                "Planting Time": "आमतौर पर मानसून के मौसम की शुरुआत में या जून और जुलाई के बीच गर्म, शुष्क परिस्थितियों में बोया जाता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 30-45 सेमी की दूरी पर और पंक्तियों को 60-75 सेमी की दूरी पर रखें। मिट्टी की नमी के आधार पर बीज 3-5 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "उड़द को पर्याप्त नमी की आवश्यकता होती है, विशेष रूप से अंकुरण और फूल आने के दौरान। यदि वर्षा अपर्याप्त है तो पानी दें, जड़ सड़न को रोकने के लिए अधिक पानी न दें।",
                "Nutrient Management": "बुवाई के समय फास्फोरस और पोटैशियम डालें। नाइट्रोजन स्थिरीकरण के कारण अतिरिक्त नाइट्रोजन आमतौर पर आवश्यक नहीं होता। मिट्टी की उर्वरता में सुधार के लिए जैविक पदार्थ शामिल करें।",
                "Weed Control": "मैनुअल निराई या खरपतवारनाशकों के माध्यम से जल्दी खरपतवार नियंत्रित करें। मल्चिंग खरपतवार को दबाने और मिट्टी की नमी संरक्षित करने में मदद करती है।",
                "Pest & Disease Management": "एफिड, फली छेदक और थ्रिप्स जैसे कीटों की निगरानी करें। आवश्यकतानुसार एकीकृत कीट प्रबंधन (IPM) रणनीतियों का उपयोग करें।",
                "Special Care During Growth": "- अंकुरण अवस्था: युवा अंकुरों को कीटों से बचाएं और पर्याप्त नमी बनाए रखें।\n- वानस्पतिक अवस्था: मजबूत विकास के लिए पर्याप्त पोषक तत्व सुनिश्चित करें।\n- फूल और फली भरने की अवस्था: इष्टतम उपज और गुणवत्ता के लिए नमी बनाए रखें।",
                "Harvesting": "जब फलियां पक जाएं और सूख जाएं, आमतौर पर बुवाई के 60-90 दिनों बाद कटाई करें। छोटे खेतों के लिए हाथ से कटाई करें; बड़े पैमाने पर संचालन के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "नमी सामग्री को कम करने के लिए थ्रेशिंग से पहले पौधों को धूप में सुखाएं।",
                "Storage Conditions": "खराब होने और कीट संक्रमण को रोकने के लिए सूखे, ठंडे स्थानों पर वेंटिलेशन के साथ स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग और प्रतिकूल मौसम शामिल हैं। जोखिमों को कम करने के लिए रोग-प्रतिरोधी किस्मों, IPM प्रथाओं और उचित मिट्टी और जल प्रबंधन का उपयोग करें।"
            },

            {
                "name": "मसूर की खेती का मार्गदर्शिका",
                "Introduction": "मसूर (Lens culinaris) पौष्टिक फलीदार फसलें हैं जो अपनी उच्च प्रोटीन और फाइबर सामग्री के लिए जानी जाती हैं। इनकी खेती व्यापक रूप से खाद्य पदार्थों के लिए की जाती है और ये कई व्यंजनों में मुख्य भोजन हैं। यह मार्गदर्शिका बीज चयन से लेकर फसल कटाई तक मसूर की खेती की पूरी प्रक्रिया को कवर करती है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग-प्रतिरोधी मसूर के बीज\n- फास्फोरस और पोटैशियम उर्वरक (न्यूनतम नाइट्रोजन की आवश्यकता)\n- ड्रिप या फरो सिंचाई\n- खरपतवारनाशक और कीटनाशक\n- हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "मसूर अच्छी जल निकासी वाली दोमट या बलुई मिट्टी पसंद करती है जिसका पीएच 6.0-7.5 हो। अच्छा बीज बिस्तर बनाने के लिए खेत को जोतें और हैरो करें।",
                "Seed Selection & Treatment": "अपने क्षेत्र के अनुकूल उच्च उपज देने वाली, रोग-प्रतिरोधी किस्मों का चयन करें। बीज जनित रोगों से बचाव के लिए बीजों को फफूंदनाशकों या कीटनाशकों से उपचारित करें।",
                "Field Preparation": "अच्छे बीज-से-मिट्टी संपर्क सुनिश्चित करने के लिए खेत को खरपतवार और मलबे से साफ करें।",
                "Planting Time": "मसूर आमतौर पर वसंत की शुरुआत या सर्दियों के अंत में बोई जाती है, जलवायु के आधार पर, जब मिट्टी का तापमान लगभग 10-15°C (50-59°F) तक पहुंच जाता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 25-30 सेमी की दूरी पर और पंक्तियों को 45-60 सेमी की दूरी पर रखें। मिट्टी की नमी के आधार पर बीज 2-3 सेमी गहराई पर बोएं।",
                "Seeding Methods": "सीड ड्रिल का उपयोग करके या मैनुअल रूप से सीधे बीज बोना।",
                "Watering Requirements": "मसूर सूखा-सहिष्णु होती है लेकिन अंकुरण और फली विकास के दौरान पर्याप्त नमी की आवश्यकता होती है। यदि वर्षा अपर्याप्त है तो पानी दें, विशेष रूप से फूल आने और बीज भरने के दौरान।",
                "Nutrient Management": "बुवाई के समय फास्फोरस और पोटैशियम डालें। नाइट्रोजन स्थिरीकरण के कारण अतिरिक्त नाइट्रोजन आमतौर पर आवश्यक नहीं होता। मिट्टी की उर्वरता बढ़ाने के लिए जैविक पदार्थ शामिल करें।",
                "Weed Control": "प्रारंभिक विकास के दौरान मैनुअल निराई या खरपतवारनाशकों का उपयोग करके खरपतवार नियंत्रित करें। मल्चिंग भी खरपतवार को दबाने और मिट्टी की नमी बनाए रखने में मदद कर सकती है।",
                "Pest & Disease Management": "एफिड, लाइगस बग और रूट रॉट जैसे कीटों की निगरानी करें। आवश्यकतानुसार एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें।",
                "Special Care During Growth": "- अंकुरण अवस्था: युवा अंकुरों को कीटों से बचाएं और पर्याप्त नमी बनाए रखें।\n- वानस्पतिक अवस्था: मजबूत विकास के लिए पर्याप्त पोषक तत्व सुनिश्चित करें।\n- फूल और फली भरने की अवस्था: इष्टतम उपज और गुणवत्ता के लिए नमी बनाए रखें।",
                "Harvesting": "जब फलियां भूरी हो जाएं और सूख जाएं, आमतौर पर बुवाई के 80-100 दिनों बाद कटाई करें। छोटे खेतों के लिए हाथ से कटाई करें; बड़े पैमाने पर संचालन के लिए कंबाइन हार्वेस्टर का उपयोग करें।",
                "Post-Harvest Management": "नमी सामग्री को कम करने के लिए थ्रेशिंग से पहले पौधों को धूप में सुखाएं।",
                "Storage Conditions": "खराब होने और कीट संक्रमण को रोकने के लिए सूखे, ठंडे स्थानों पर वेंटिलेशन के साथ स्टोर करें।",
                "Processing & Packaging": "सांस लेने वाले बैग में पैकेजिंग से पहले बीजों को साफ और ग्रेड करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग और परिवर्तनशील मौसम शामिल हैं। जोखिमों को कम करने के लिए रोग-प्रतिरोधी किस्मों, IPM प्रथाओं और उचित मिट्टी और जल प्रबंधन का उपयोग करें।"
            },

            {
                "name": "अनार की खेती गाइड",
                "Introduction": "अनार (Punica granatum) एक पौष्टिक फल है जो अपने स्वास्थ्य लाभों और समृद्ध स्वाद के लिए जाना जाता है। यह कई हिस्सों में उगाया जाता है और गर्म जलवायु में अच्छा पनपता है। यह गाइड रोपण से लेकर कटाई तक की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले अनार के बीज या विश्वसनीय नर्सरी से स्वस्थ पौधे\n- नाइट्रोजन, फास्फोरस और पोटैशियम युक्त संतुलित उर्वरक\n- ड्रिप सिंचाई प्रणाली या फरो सिंचाई\n- कीटनाशक और कवकनाशक कीट और रोग प्रबंधन के लिए\n- रोपण, छंटाई और रखरखाव के लिए हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "अनार को अच्छे जल निकास वाली, रेतीली दोमट से दोमट मिट्टी पसंद है, जिसका pH 5.5 से 7.0 के बीच हो। जैविक पदार्थ मिलाकर भूमि की जुताई करें।",
                "Seed Selection & Treatment": "अपने क्षेत्र की जलवायु के लिए उपयुक्त रोग प्रतिरोधी किस्में चुनें। यदि बीजों का उपयोग कर रहे हैं, तो अंकुरण दर में सुधार के लिए उन्हें रात भर पानी में भिगोएं।",
                "Field Preparation": "जमीन से खरपतवार, पत्थर और मलबे को हटा दें ताकि एक स्वच्छ रोपण वातावरण सुनिश्चित हो।",
                "Planting Time": "अनार को आमतौर पर वसंत में अंतिम ठंढ के बाद लगाया जाता है।",
                "Spacing & Depth": "पौधों को 5-8 फीट की दूरी पर लगाएं ताकि उचित विकास और वायु संचार सुनिश्चित हो सके। बीजों या पौधों को 1-2 इंच गहराई में लगाएं और मिट्टी को अच्छे से दबाएं।",
                "Seeding Methods": "सीधा बुआई: बीजों को सीधे तैयार किए गए स्थान पर बोएं।\nप्रतिरोपण: यदि पौधे लगा रहे हैं, तो जड़ के आकार से थोड़ा बड़ा गड्ढा खोदें और मिट्टी से भरें।",
                "Watering Requirements": "अनार को विशेष रूप से प्रारंभिक अवस्था में नियमित पानी देने की आवश्यकता होती है। एक बार स्थापित होने के बाद, यह सूखा सहिष्णु होता है। गहरे जड़ विकास को बढ़ावा देने के लिए गहराई से लेकिन कम बार पानी दें।",
                "Nutrient Management": "विकास के मौसम के दौरान संतुलित उर्वरक डालें, आमतौर पर शुरुआती वसंत और देर से गर्मियों में। मिट्टी की उर्वरता बढ़ाने के लिए जैविक खाद मिलाएं।",
                "Weed Control": "पोषक तत्वों के लिए प्रतिस्पर्धा कम करने के लिए मल्चिंग और हाथ से निराई करके खरपतवारों को नियंत्रित करें।",
                "Pest & Disease Management": "कीटों जैसे एफिड्स, सफेद मक्खी और अनार तितलियों पर नजर रखें। प्राकृतिक शत्रुओं और जैविक कीटनाशकों का उपयोग करके एकीकृत कीट प्रबंधन (IPM) रणनीतियाँ लागू करें।",
                "Special Care During Growth": "- अंकुर अवस्था: युवा पौधों को अत्यधिक मौसम और कीटों से बचाएं। नमी बनाए रखने के लिए मल्च का उपयोग करें।\n- वनस्पति अवस्था: पोषक तत्वों की कमी और कीट संक्रमण के लिए नियमित रूप से जाँच करें और आवश्यकतानुसार उर्वरक डालें।\n- फूल और फल बनने की अवस्था: स्वस्थ विकास को बढ़ावा देने के लिए फूल लगने और फल बनने के दौरान पर्याप्त पानी सुनिश्चित करें।",
                "Harvesting": "अनार आमतौर पर फूल आने के 5-7 महीने बाद कटाई के लिए तैयार होता है, जब फल गहरे रंग का हो जाता है और थपथपाने पर धातु जैसी आवाज करता है। फलों को तेज कैंची से काटें ताकि शाखाओं और अन्य फलों को नुकसान न पहुंचे।",
                "Post-Harvest Management": "फलों को धीरे से संभालें ताकि चोट न लगे; उन्हें ठंडी और सूखी जगह पर रखें।",
                "Storage Conditions": "अनार को ठंडी और सूखी जगह पर स्टोर करें; उचित परिस्थितियों में यह कई हफ्तों से महीनों तक टिक सकता है।",
                "Processing & Packaging": "फलों को साफ और छांटकर किसी भी खराब या सड़े हुए फलों को अलग करें। भंडारण के दौरान गुणवत्ता बनाए रखने के लिए फलों को सांस लेने योग्य कंटेनरों में पैक करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट, रोग और सूखा या अत्यधिक नमी जैसी पर्यावरणीय चुनौतियाँ शामिल हैं। रोग प्रतिरोधी किस्मों का उपयोग करें, उचित सिंचाई तकनीकों को लागू करें, और कीट नियंत्रण के लिए नियमित निगरानी करें।"
            },

            {
                "name": "राजमा की खेती गाइड",
                "Introduction": "राजमा (Phaseolus vulgaris) एक उच्च प्रोटीन युक्त दलहन है जो विभिन्न व्यंजनों में उपयोग किया जाता है। यह गाइड बीज चयन से लेकर कटाई तक की पूरी प्रक्रिया को कवर करता है।",
                "Materials Required": "- उच्च गुणवत्ता वाले, रोग प्रतिरोधी राजमा के बीज\n- फास्फोरस और पोटैशियम उर्वरक; सीमित नाइट्रोजन क्योंकि राजमा स्वयं नाइट्रोजन फिक्स करता है\n- ड्रिप या स्प्रिंकलर सिंचाई प्रणाली\n- खरपतवारनाशी और कीटनाशक सामान्य राजमा कीटों के लिए\n- Soil Preparation, रोपण और निराई के लिए हाथ के उपकरण या ट्रैक्टर",
                "Soil Preparation": "राजमा अच्छे जल निकास वाली, दोमट मिट्टी में सबसे अच्छा बढ़ता है, जिसका pH 6.0 से 7.0 के बीच होता है। जुताई और जड़ें आसानी से फैलाने के लिए हल्की मिट्टी तैयार करें।",
                "Seed Selection & Treatment": "उच्च उपज देने वाली, रोग प्रतिरोधी किस्में चुनें। बीजों को शुरुआती मिट्टी जनित रोगों और कीटों से बचाने के लिए कवकनाशी या कीटनाशक से उपचारित करें।",
                "Field Preparation": "खेत से खरपतवार और मलबे को साफ करें, फिर समतल करें। पंक्तियों को इस तरह चिह्नित करें कि वायु संचार और सूर्य का प्रकाश अच्छी तरह मिल सके।",
                "Planting Time": "राजमा को आमतौर पर वसंत में तब बोया जाता है जब मिट्टी का तापमान 15°C (59°F) तक पहुँच जाता है और ठंढ का कोई खतरा नहीं होता।",
                "Spacing & Depth": "बीजों को 3-5 सेमी गहराई में लगाएं, पौधों के बीच 8-10 सेमी और पंक्तियों के बीच 45-60 सेमी दूरी रखें।",
                "Seeding Methods": "सीधी बुआई: बीजों को सीधे खेत में हाथ से या बीज ड्रिल से बोएं।",
                "Watering Requirements": "राजमा को नियमित रूप से पानी देने की आवश्यकता होती है, विशेष रूप से फूल और फली बनने के दौरान। अधिक पानी देने से बचें क्योंकि यह जलभराव के प्रति संवेदनशील होता है।",
                "Nutrient Management": "रोपण के समय फास्फोरस और पोटैशियम लागू करें। नाइट्रोजन की मात्रा सीमित रखें क्योंकि राजमा स्वयं नाइट्रोजन का स्थिरीकरण करता है।",
                "Weed Control": "खरपतवारों को नियंत्रित करने के लिए शुरुआती चरणों में निराई करें। जरूरत पड़ने पर खरपतवारनाशी का उपयोग करें।",
                "Harvesting": "राजमा की कटाई तब करें जब फली पूरी तरह परिपक्व और सूखी हो, आमतौर पर 90-120 दिनों में।",
                "Storage Conditions": "राजमा को सूखी, हवादार जगह पर स्टोर करें ताकि फफूंदी और कीटों से बचा जा सके।"
            },

            {
                "name": "केला खेती गाइड",
                "Introduction": "केले (Musa spp.) एक उष्णकटिबंधीय फल हैं जो अपने मीठे स्वाद और पोषण गुणों के लिए प्रसिद्ध हैं। ये गर्म, आर्द्र जलवायु में अच्छी तरह से विकसित होते हैं और व्यावसायिक तथा घरेलू उत्पादन दोनों के लिए उगाए जाते हैं। यह गाइड केले की खेती की पूरी प्रक्रिया को कवर करता है, जिसमें रोपण से लेकर कटाई तक की जानकारी दी गई है।",
                "Materials Required": "- स्वस्थ केला चूसक या ऊतक-संस्कृत पौधे\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम युक्त संतुलित उर्वरक; जैविक खाद जैसे कंपोस्ट\n- सिंचाई प्रबंधन के लिए ड्रिप या स्प्रिंकलर प्रणाली\n- कीटनाशक और कवकनाशक ताकि कीट और रोगों को प्रबंधित किया जा सके\n- रोपण, रखरखाव और कटाई के लिए हाथ के उपकरण (फावड़ा, छंटाई कैंची) या ट्रैक्टर",
                "Soil Preparation": "केले अच्छी जल निकासी वाली, समृद्ध दोमट मिट्टी को पसंद करते हैं जिसका पीएच 5.5 से 7.0 के बीच हो। मिट्टी को जोतकर उसमें जैविक खाद मिलाएं ताकि उर्वरता और जल निकासी में सुधार हो।",
                "Plant Selection & Treatment": "स्वस्थ माता-पिता पौधों से रोग-मुक्त चूसक चुनें या प्रमाणित स्रोत से ऊतक-संस्कृत पौधे प्राप्त करें। चूसक को माता-पिता पौधे से काटने के लिए स्वच्छ चाकू का उपयोग करें ताकि संक्रमण न फैले।",
                "Field Preparation": "रोपण स्थल को खरपतवार, पत्थरों और मलबे से साफ करें ताकि स्वस्थ वातावरण बनाया जा सके।",
                "Planting Time": "केले के लिए सबसे अच्छा रोपण समय वर्षा ऋतु की शुरुआत या गर्म महीनों के दौरान होता है।",
                "Spacing & Depth": "पौधों को पंक्तियों में 8-10 फीट की दूरी पर और पंक्तियों के बीच 10-12 फीट की दूरी पर लगाएं ताकि उचित वृद्धि और वायु संचार हो सके। चूसकों या पौधों को उसी गहराई पर लगाएं जिस गहराई पर वे नर्सरी में उग रहे थे।",
                "Seeding Methods": "केले को लगातार नमी की आवश्यकता होती है; विशेष रूप से सूखे समय में नियमित रूप से सिंचाई करें। प्रति सप्ताह 1-2 इंच पानी देने का प्रयास करें।",
                "Nutrient Management": "वसंत ऋतु की शुरुआत में और फिर मध्य ऋतु में संतुलित उर्वरक लगाएं। मिट्टी की उर्वरता बढ़ाने के लिए जैविक खाद या गीली घास का उपयोग करें।",
                "Weed Control": "गीली घास का उपयोग करके खरपतवारों को नियंत्रित करें, जिससे नमी भी बनी रहती है, और हाथ से निराई करके पोषक तत्वों की प्रतिस्पर्धा को कम करें।",
                "Pest & Disease Management": "केले के भूरे धब्बे की बीमारी और बनाना वीविल जैसे कीटों की निगरानी करें। उचित स्वच्छता और प्रतिरोधी किस्मों के उपयोग से रोगों को रोकें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिनमें जैविक नियंत्रण विधियाँ भी शामिल हैं।",
                "Harvesting": "केले आमतौर पर रोपण के 9-12 महीने बाद कटाई के लिए तैयार होते हैं। जब फल मोटे हो जाते हैं और डंठल और फल के बीच का कोण अधिक स्पष्ट हो जाता है, तो उन्हें काट लें। तेज चाकू या खुरपी का उपयोग करके पूरे गुच्छे को काटें। फलों को सावधानीपूर्वक संभालें ताकि वे क्षतिग्रस्त न हों।",
                "Storage Conditions": "केले को कमरे के तापमान पर रखें जब तक वे पूरी तरह से पक न जाएँ। सीधे धूप या अत्यधिक गर्मी से बचाएं।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट और रोग संवेदनशीलता, पर्यावरणीय तनाव और अनुचित सिंचाई शामिल हैं। रोग प्रतिरोधी किस्मों का चयन करें, अच्छे कृषि अभ्यासों को लागू करें और पर्यावरणीय स्थितियों की निगरानी करें।"
            },


            {"name": "अंगूर की खेती मार्गदर्शिका",
                "Introduction": "अंगूर (Vitis vinifera और अन्य प्रजातियाँ) बहुउद्देश्यीय फल हैं, जिनका उपयोग ताजे फल के रूप में खाने, सूखाकर किशमिश बनाने और वाइन उत्पादन के लिए किया जाता है। ये समशीतोष्ण जलवायु में अच्छे से विकसित होते हैं और उच्च गुणवत्ता वाले फल उत्पादन के लिए विशिष्ट बढ़ती परिस्थितियों की आवश्यकता होती है। यह मार्गदर्शिका अंगूर की खेती की पूरी प्रक्रिया को कवर करती है, जिसमें रोपण से लेकर कटाई तक की जानकारी दी गई है।",
                "Materials Required": "- उच्च गुणवत्ता वाली अंगूर की बेलें, नग्न जड़ या गमले में उगाई गई, विश्वसनीय नर्सरी से\n- संतुलित उर्वरक जिसमें नाइट्रोजन, फास्फोरस और पोटैशियम हों; जैविक खाद\n- प्रभावी नमी प्रबंधन के लिए ड्रिप सिंचाई प्रणाली\n- कीटनाशक, फफूंदनाशक और जैविक कीट प्रबंधन समाधान\n- रोपण, रखरखाव और कटाई के लिए हाथ के औजार (प्रूनर, फावड़ा) या ट्रैक्टर",
                "Soil Preparation": "अंगूर को अच्छी जल निकासी वाली, रेतीली दोमट या चिकनी दोमट मिट्टी पसंद होती है, जिसकी पीएच 6.0 से 6.8 के बीच हो। मिट्टी को जोतकर और जैविक पदार्थ मिलाकर उर्वरता और जल निकासी में सुधार करें।",
                "Plant Selection & Treatment": "अपने जलवायु और उद्देश्य (टेबल अंगूर, वाइन अंगूर आदि) के लिए रोग-प्रतिरोधी अंगूर की किस्में चुनें। रोपण से पहले बेलों की बीमारी या क्षति के लिए जाँच करें।",
                "Field Preparation": "रोपण स्थल को खरपतवार, पत्थरों और मलबे से साफ करें ताकि स्वच्छ वातावरण सुनिश्चित हो।",
                "Planting Time": "अंगूर को शुरुआती वसंत में अंतिम ठंढ के बाद या सर्दियों से पहले पतझड़ में लगाना सबसे अच्छा होता है।",
                "Spacing & Depth": "बेलों को 6-10 फीट की दूरी पर और पंक्तियों को 8-10 फीट की दूरी पर लगाएँ ताकि उचित वायु संचार और विकास सुनिश्चित हो सके। बेलों को उसी गहराई पर लगाएँ जिस पर वे नर्सरी में उग रही थीं।",
                "Seed Selection & Treatment": "पुनः प्रत्यारोपण: जड़ों को समायोजित करने के लिए पर्याप्त बड़ा गड्ढा खोदें, धीरे-धीरे मिट्टी भरें और रोपण के बाद अच्छी तरह पानी दें।",
                "Watering Requirements": "अंगूर को पहले वर्ष में नियमित रूप से पानी देने की आवश्यकता होती है ताकि जड़ें स्थापित हो सकें। एक बार स्थापित हो जाने के बाद, वे सूखा-सहिष्णु होते हैं, लेकिन सूखे की स्थिति में, विशेष रूप से फल विकास के दौरान, अतिरिक्त सिंचाई लाभकारी होती है।",
                "Nutrient Management": "शुरुआती वसंत में और मध्य सीजन में संतुलित उर्वरक डालें। जैविक खाद का उपयोग करें ताकि मिट्टी की सेहत में सुधार हो।",
                "Weed Control": "खरपतवारों को रोकने के लिए गीली घास (मल्चिंग), हाथ से निराई या शाकनाशी का उपयोग करें ताकि पोषक तत्वों और नमी के लिए प्रतिस्पर्धा कम हो।",
                "Pest & Disease Management": "अंगूर कीट जैसे अंगूर की बेल कीट, एफिड्स और मकड़ी के कणों के लिए निगरानी रखें। पाउडरी मिल्ड्यू और डाउनरी मिल्ड्यू जैसी बीमारियों को स्वच्छता और रोग-प्रतिरोधी किस्मों के माध्यम से नियंत्रित करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को अपनाएँ, जिसमें सांस्कृतिक नियंत्रण और प्राकृतिक शिकारी शामिल हों।",
                "Special Care During Growth": "- युवा बेल चरण: युवा बेलों को चरम मौसम और कीटों से बचाएँ; उन्हें ऊपर बढ़ने में मदद के लिए सहारा स्टेक्स या ट्रेलिस का उपयोग करें।\n- वनस्पति चरण: पोषक तत्वों की कमी के लिए नियमित रूप से जाँच करें और उन्हें तुरंत पूरा करें। मजबूत संरचना और वायु संचार को प्रोत्साहित करने के लिए छँटाई करें।\n- फूल और फल विकास चरण: फूल आने और फल बनने के दौरान निरंतर नमी सुनिश्चित करें ताकि उपज और गुणवत्ता बढ़ सके। बड़े फलों को बढ़ावा देने के लिए यदि आवश्यक हो तो गुच्छों को पतला करें।",
                "Harvesting": "अंगूर फूल आने के 4-6 महीने बाद कटाई के लिए तैयार होते हैं, जो किस्म के आधार पर भिन्न हो सकता है। उन्हें पूरी तरह से पका होने पर काटना चाहिए, जब वे गहरे रंग के हो जाएँ और मीठे स्वाद वाले हों। बेल से गुच्छों को काटने के लिए तेज कैंची या प्रूनर का उपयोग करें। फलों को नुकसान से बचाने के लिए सावधानीपूर्वक संभालें।",
                "Post-Harvest Management": "किसी भी क्षतिग्रस्त या सड़े हुए अंगूर को हटा दें और उन्हें ठंडी, छायादार जगह पर रखें।",
                "Storage Conditions": "अंगूर को ठंडी, सूखी जगह पर स्टोर करें। प्रशीतन से उनका शेल्फ लाइफ बढ़ाया जा सकता है, लेकिन उन्हें हवादार कंटेनरों में रखना चाहिए।",
                "Processing & Packaging": "यदि आवश्यक हो, तो अंगूर को अंगूर का रस, जैली या वाइन में संसाधित किया जा सकता है। परिवहन के दौरान खराब होने से बचाने के लिए अंगूर को हवादार कंटेनरों में पैक करें।",
                "Challenges & Solutions": "सामान्य समस्याओं में कीट और बीमारियों की संवेदनशीलता, जलवायु से संबंधित समस्याएँ और अनुचित सिंचाई शामिल हैं। रोग-प्रतिरोधी किस्में चुनें, अच्छे कृषि पद्धतियों को अपनाएँ और पर्यावरणीय परिस्थितियों की निगरानी करें ताकि इन चुनौतियों को कम किया जा सके।"
            },

            {
                "name": "मस्कमेलन की खेती गाइड",
                "Introduction": "मस्कमेलन (Cucumis melo var. cantaloupe) मीठे, सुगंधित फल होते हैं, जो अपने रसीले गूदे और विशिष्ट जालदार छिलके के लिए जाने जाते हैं। ये गर्म जलवायु में अच्छी तरह से पनपते हैं और अपने ताजगी भरे स्वाद के लिए लोकप्रिय हैं। यह गाइड मस्कमेलन की खेती की पूरी प्रक्रिया को कवर करता है, रोपण से लेकर कटाई तक।",
                "Materials Required": "- विश्वसनीय स्रोतों से उच्च गुणवत्ता वाले मस्कमेलन के बीज या पौधे\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम युक्त संतुलित उर्वरक; जैविक खाद\n- नमी प्रबंधन के लिए ड्रिप या ओवरहेड सिंचाई प्रणाली\n- कीटनाशक, फफूंदनाशी और जैविक कीट प्रबंधन समाधान\n- फावड़े, कुदाल, छंटाई कैंची जैसे हाथ के औजार या खेती के लिए ट्रैक्टर",
                "Soil Preparation": "मस्कमेलन को अच्छी जल निकासी वाली, बलुई दोमट या दोमट मिट्टी पसंद होती है, जिसकी pH 6.0 से 6.8 के बीच हो। मिट्टी को जोतकर और जैविक पदार्थ मिलाकर जल निकासी और उर्वरता बढ़ाएं।",
                "Plant Selection & Treatment": "अपने जलवायु और बाजार के अनुसार रोग प्रतिरोधी किस्मों का चयन करें। यदि बीज उपयोग कर रहे हैं, तो उन्हें बोने से पहले कुछ घंटों के लिए पानी में भिगोएँ ताकि अंकुरण दर में सुधार हो सके।",
                "Field Preparation": "रोपण स्थल को खरपतवार, पत्थरों और मलबे से साफ करें ताकि एक स्वच्छ वातावरण सुनिश्चित हो सके।",
                "Planting Time": "मस्कमेलन लगाने का आदर्श समय अंतिम पाले के बाद होता है जब मिट्टी का तापमान लगातार 70°F (21°C) से अधिक हो।",
                "Spacing & Depth": "मस्कमेलन के पौधों को 3-4 फीट की दूरी पर और पंक्तियों को 6-8 फीट की दूरी पर लगाएं ताकि बेलें फैल सकें। बीजों या पौधों को लगभग 1 इंच की गहराई में लगाएं।",
                "Seed Selection & Treatment": "- प्रत्यक्ष बीजारोपण: जब मिट्टी गर्म हो जाए तो बीजों को सीधे जमीन में बोएं।\n- पुनःरोपण: पौधों को पहले अंदर उगाएं और जब वे मजबूत हो जाएं तो उन्हें खेत में प्रत्यारोपित करें।",
                "Watering Requirements": "मस्कमेलन को विशेष रूप से अंकुरण और फल विकास के दौरान लगातार नमी की आवश्यकता होती है। प्रति सप्ताह लगभग 1-2 इंच पानी देने का लक्ष्य रखें, वर्षा के अनुसार समायोजन करें।",
                "Nutrient Management": "रोपण के समय और जब बेलें बढ़ने लगें तो संतुलित उर्वरक लगाएं। जैविक खाद या गीली घास का उपयोग मिट्टी के स्वास्थ्य को बढ़ाने के लिए करें।",
                "Weed Control": "गीली घास के उपयोग से नमी बनाए रखने और खरपतवार के विकास को दबाने में मदद मिलती है। नियमित रूप से हाथ से खरपतवार निकालें ताकि वे पौधों से पोषक तत्व न छीनें।",
                "Pest & Disease Management": "कीटों जैसे कि एफिड्स, ककड़ी बीटल और मकड़ी के कणों की निगरानी करें। पाउडरी मिल्ड्यू और डाउनी मिल्ड्यू जैसे रोगों का प्रबंधन उचित स्वच्छता और प्रतिरोधी किस्मों के माध्यम से करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को अपनाएं।",
                "Special Care During Growth": "- अंकुर अवस्था: युवा पौधों को कीटों और अत्यधिक मौसम से बचाएं।\n- वनस्पति अवस्था: पोषक तत्वों की कमी की नियमित जांच करें और तुरंत समाधान करें।\n- फल विकास अवस्था: फल के विकास के दौरान पर्याप्त पानी की आपूर्ति करें ताकि फल स्वस्थ और मीठे बनें।",
                "Harvesting": "मस्कमेलन आमतौर पर रोपण के 70-90 दिन बाद कटाई के लिए तैयार हो जाते हैं। संकेतों में रंग का हरे से पीले की ओर परिवर्तन और मीठी सुगंध शामिल हैं। फल को बेल से काटने के लिए तेज चाकू या छंटाई कैंची का उपयोग करें।",
                "Post-Harvest Management": "कटे हुए फलों को सावधानीपूर्वक संभालें ताकि चोट या क्षति से बचा जा सके। उन्हें एक ठंडी, छायादार जगह में रखें।",
                "Storage Conditions": "मस्कमेलन को पूरी तरह पकने तक कमरे के तापमान पर रखें। एक बार पक जाने के बाद, उन्हें थोड़े समय के लिए फ्रिज में रखा जा सकता है ताकि ताजगी बनी रहे।",
                "Processing & Packaging": "यदि आवश्यक हो, तो मस्कमेलन को स्मूदी, शर्बत या फलों के सलाद में संसाधित किया जा सकता है। भंडारण और परिवहन के दौरान गुणवत्ता बनाए रखने के लिए मस्कमेलन को सांस लेने योग्य कंटेनरों में पैक करें।",
                "Challenges & Solutions": "सामान्य चुनौतियों में कीट और रोग संवेदनशीलता, पर्यावरणीय तनाव जैसे सूखा या अत्यधिक नमी, और अनुचित सिंचाई प्रथाएँ शामिल हैं। रोग-प्रतिरोधी किस्मों का चयन करें, अच्छी खेती की प्रथाएँ अपनाएँ और पर्यावरणीय परिस्थितियों की निगरानी करें।"
            },

            {
                "name": "सेब की खेती गाइड",
                "Introduction": "सेब (Malus domestica) दुनिया में सबसे लोकप्रिय फलों में से एक हैं, जो अपने स्वाद, बहुमुखी उपयोग और पोषण मूल्य के लिए सराहे जाते हैं। ये समशीतोष्ण जलवायु में सबसे अच्छा विकसित होते हैं और विभिन्न प्रकार की मिट्टी में उगाए जा सकते हैं। यह गाइड सेब की खेती की पूरी प्रक्रिया को रेखांकित करता है, जिसमें रोपण से लेकर कटाई तक की जानकारी शामिल है।",
                "Materials Required": "- प्रतिष्ठित नर्सरी से उच्च गुणवत्ता वाले सेब के पौधे या ग्राफ्टेड किस्में\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम युक्त संतुलित उर्वरक; जैविक खाद\n- प्रभावी नमी प्रबंधन के लिए ड्रिप सिंचाई प्रणाली या नली\n- कीटनाशक, फफूंदनाशी और जैविक कीट प्रबंधन समाधान\n- रोपण, रखरखाव और कटाई के लिए हाथ के उपकरण (फावड़ा, छंटाई कैंची, कुदाल) या ट्रैक्टर",
                "Soil Preparation": "सेब को अच्छी जल निकासी वाली, दोमट मिट्टी पसंद होती है, जिसका pH 6.0 से 7.0 के बीच हो। मिट्टी को जोतकर उसमें जैविक पदार्थ मिलाएं ताकि उपजाऊपन और जल निकासी में सुधार हो।",
                "Plant Selection & Treatment": "अपने जलवायु के अनुसार रोग-प्रतिरोधी सेब की किस्में चुनें, जिसमें फल के स्वाद और कटाई के समय को ध्यान में रखें। पौधों को लगाने से पहले किसी भी बीमारी या क्षति के लक्षणों की जांच करें।",
                "Field Preparation": "रोपण क्षेत्र को खरपतवार, पत्थर और मलबे से साफ करें ताकि एक स्वच्छ वातावरण सुनिश्चित हो।",
                "Planting Time": "सेब के पौधों को लगाने का सबसे अच्छा समय पतझड़ या शुरुआती वसंत ऋतु होता है, जब पेड़ सुप्त अवस्था में होते हैं।",
                "Spacing & Depth": "बौनी किस्मों को 4-6 फीट की दूरी पर और मानक किस्मों को 10-15 फीट की दूरी पर लगाएं ताकि उचित वृद्धि और वायु संचलन हो सके। पेड़ों को उसी गहराई पर लगाएं जिस गहराई पर वे नर्सरी में थे, और यह सुनिश्चित करें कि ग्राफ्ट यूनियन मिट्टी के स्तर से ऊपर रहे।",
                "Seeding/Transplanting Methods": "रोपण: जड़ों के आकार के अनुसार एक गड्ढा खोदें, पौधे को उसमें रखें, धीरे-धीरे मिट्टी भरें और रोपण के बाद अच्छी तरह पानी दें।",
                "Watering Requirements": "छोटे सेब के पौधों को जड़ जमाने के लिए नियमित रूप से पानी दें, विशेष रूप से शुष्क मौसम में। स्थापित पेड़ सूखा-सहिष्णु होते हैं, लेकिन फल के विकास के दौरान गहरे पानी की आवश्यकता होती है।",
                "Nutrient Management": "वसंत ऋतु की शुरुआत में और मध्य मौसम में संतुलित उर्वरक लगाएं। जैविक खाद का उपयोग करके मिट्टी के स्वास्थ्य में सुधार करें।",
                "Weed Control": "मल्चिंग से खरपतवारों को नियंत्रित करें, जिससे नमी बनाए रखने और खरपतवार वृद्धि को दबाने में मदद मिलती है। साथ ही, प्रतिस्पर्धा को कम करने के लिए समय-समय पर खरपतवार निकालें।",
                "Pest & Disease Management": "कोडिंग मॉथ, एफिड्स और स्पाइडर माइट्स जैसे कीटों की निगरानी करें। सेब स्कैब और पाउडरी मिल्ड्यू जैसी बीमारियों का उचित स्वच्छता और रोग प्रतिरोधी किस्मों के माध्यम से प्रबंधन करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिसमें सांस्कृतिक नियंत्रण और लाभकारी कीटों का उपयोग शामिल हो।",
                "Special Care During Growth": "- युवा पौधा चरण: युवा पेड़ों को चरम मौसम और कीटों से बचाएं; पशु क्षति से बचाने के लिए ट्री गार्ड का उपयोग करें।\n- वनस्पति वृद्धि चरण: नियमित रूप से पोषक तत्वों की कमी की जांच करें और उन्हें तुरंत ठीक करें। पेड़ों की सही आकार में छंटाई करें और मजबूत संरचना विकसित करने में मदद करें।\n- फूल और फल विकास चरण: अधिकतम उपज और फल की गुणवत्ता सुनिश्चित करने के लिए फूल आने और फल लगने के दौरान निरंतर नमी बनाए रखें। यदि आवश्यक हो, तो बड़े सेब पैदा करने के लिए कुछ फलों को पतला करें।",
                "Harvesting": "सेब आमतौर पर फूल आने के 4-6 महीने बाद कटाई के लिए तैयार होते हैं, जो किस्म पर निर्भर करता है। कटाई के संकेतों में रंग परिवर्तन, मजबूत बनावट और पेड़ से आसानी से अलग होना शामिल हैं। तेज छंटाई कैंची से सेब काटें, जिससे फल से एक छोटा तना जुड़ा रहे।",
                "Post-Harvest Management": "कटे हुए सेबों को धीरे से संभालें ताकि चोट लगने से बचा जा सके। उन्हें ठंडी और छायादार जगह पर संग्रहित करें।",
                "Storage Conditions": "सेब को ठंडी, अंधेरी जगह में रखें। उनकी शेल्फ लाइफ बढ़ाने के लिए इन्हें रेफ्रिजरेटर में संग्रहीत किया जा सकता है।",
                "Processing & Packaging": "यदि आवश्यक हो, तो सेब को सेब सॉस, साइडर या सूखे टुकड़ों में संसाधित किया जा सकता है। सेबों को सांस लेने योग्य कंटेनरों में पैक करें ताकि भंडारण और परिवहन के दौरान उनकी गुणवत्ता बनी रहे।",
                "Challenges & Solutions": "आम चुनौतियों में कीट और रोगों की संवेदनशीलता, पर्यावरणीय तनाव (जैसे सूखा या पाला) और अनुचित छंटाई तकनीक शामिल हैं। रोग-प्रतिरोधी किस्मों का चयन करें, अच्छे कृषि अभ्यासों को लागू करें, और इन चुनौतियों को कम करने के लिए पर्यावरणीय परिस्थितियों की निगरानी करें।"
            },

           {
                "name": "संतरा खेती गाइड",
                "Introduction": "संतरा (Citrus sinensis) सबसे लोकप्रिय खट्टे फलों में से एक है, जो अपने मीठे, रसदार गूदे और उच्च विटामिन C सामग्री के लिए मूल्यवान है। ये गर्म, उपोष्णकटिबंधीय से लेकर उष्णकटिबंधीय जलवायु में पनपते हैं। यह गाइड संतरे की खेती की पूरी प्रक्रिया को रेखांकित करता है, जिसमें रोपण से लेकर कटाई तक की जानकारी शामिल है।",
                "Materials Required": "- प्रतिष्ठित नर्सरी से उच्च गुणवत्ता वाले संतरे के पौधे या ग्राफ्टेड किस्में\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम युक्त खट्टे फलों के लिए विशेष उर्वरक; जैविक खाद\n- प्रभावी नमी प्रबंधन के लिए ड्रिप सिंचाई प्रणाली या नली\n- कीटनाशक, फफूंदनाशी और जैविक कीट प्रबंधन समाधान\n- रोपण, रखरखाव और कटाई के लिए हाथ के उपकरण (फावड़ा, छंटाई कैंची, कुदाल) या ट्रैक्टर",
                "Soil Preparation": "संतरा अच्छी जल निकासी वाली, बलुई दोमट या चिकनी दोमट मिट्टी को पसंद करता है, जिसका pH 6.0 से 7.5 के बीच हो। मिट्टी को जोतकर उसमें जैविक पदार्थ मिलाएं ताकि उपजाऊपन और जल निकासी में सुधार हो।",
                "Plant Selection & Treatment": "अपने जलवायु के अनुसार रोग-प्रतिरोधी संतरे की किस्में चुनें, जिसमें फल के स्वाद और कटाई के समय को ध्यान में रखें। पौधों को लगाने से पहले किसी भी बीमारी या क्षति के लक्षणों की जांच करें।",
                "Field Preparation": "रोपण क्षेत्र को खरपतवार, पत्थर और मलबे से साफ करें ताकि एक स्वच्छ वातावरण सुनिश्चित हो।",
                "Planting Time": "संतरे के पौधों को लगाने का सबसे अच्छा समय वसंत ऋतु होता है, जब ठंढ का खतरा समाप्त हो जाता है।",
                "Spacing & Depth": "पेड़ों को 12-25 फीट की दूरी पर लगाएं, जो कि जड़स्टॉक और पेड़ की किस्म पर निर्भर करता है, ताकि उचित वृद्धि और वायु संचलन हो सके। पेड़ों को उसी गहराई पर लगाएं जिस गहराई पर वे नर्सरी में थे, और यह सुनिश्चित करें कि ग्राफ्ट यूनियन मिट्टी के स्तर से ऊपर रहे।",
                "Seeding/Transplanting Methods": "रोपण: जड़ों के आकार के अनुसार एक गड्ढा खोदें, पौधे को उसमें रखें, धीरे-धीरे मिट्टी भरें और रोपण के बाद अच्छी तरह पानी दें।",
                "Watering Requirements": "छोटे संतरे के पौधों को जड़ जमाने के लिए नियमित रूप से पानी दें, विशेष रूप से शुष्क मौसम में। स्थापित पेड़ शुष्क अवधि के दौरान गहरे पानी की आवश्यकता रखते हैं।",
                "Nutrient Management": "वसंत ऋतु की शुरुआत में और मध्य मौसम में खट्टे फलों के लिए विशेष उर्वरक लगाएं। जैविक खाद का उपयोग करके मिट्टी के स्वास्थ्य में सुधार करें।",
                "Weed Control": "मल्चिंग से खरपतवारों को नियंत्रित करें, जिससे नमी बनाए रखने और खरपतवार वृद्धि को दबाने में मदद मिलती है। साथ ही, प्रतिस्पर्धा को कम करने के लिए समय-समय पर खरपतवार निकालें।",
                "Pest & Disease Management": "एफिड्स, स्पाइडर माइट्स और साइट्रस लीफ माइनर जैसे कीटों की निगरानी करें। साइट्रस कैंकर और रूट रॉट जैसी बीमारियों का उचित स्वच्छता और रोग प्रतिरोधी किस्मों के माध्यम से प्रबंधन करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिसमें सांस्कृतिक नियंत्रण और लाभकारी कीटों का उपयोग शामिल हो।",
                "Special Care During Growth": "- युवा पौधा चरण: युवा पेड़ों को चरम मौसम और कीटों से बचाएं; पशु क्षति से बचाने के लिए ट्री गार्ड का उपयोग करें।\n- वनस्पति वृद्धि चरण: नियमित रूप से पोषक तत्वों की कमी की जांच करें और उन्हें तुरंत ठीक करें। पेड़ों की सही आकार में छंटाई करें और मजबूत संरचना विकसित करने में मदद करें।\n- फूल और फल विकास चरण: अधिकतम उपज और फल की गुणवत्ता सुनिश्चित करने के लिए फूल आने और फल लगने के दौरान निरंतर नमी बनाए रखें। यदि आवश्यक हो, तो बड़े संतरे पैदा करने के लिए कुछ फलों को पतला करें।",
                "Harvesting": "संतरे आमतौर पर फूल आने के 7-12 महीने बाद कटाई के लिए तैयार होते हैं, जो किस्म पर निर्भर करता है। कटाई के संकेतों में रंग परिवर्तन, मजबूत बनावट और मिठास शामिल हैं। तेज छंटाई कैंची से संतरे काटें, जिससे फल से एक छोटा तना जुड़ा रहे।",
                "Post-Harvest Management": "कटे हुए संतरों को धीरे से संभालें ताकि चोट लगने से बचा जा सके। उन्हें ठंडी और छायादार जगह पर संग्रहित करें।",
                "Storage Conditions": "संतरे को ठंडी, अंधेरी जगह में रखें। उनकी शेल्फ लाइफ बढ़ाने के लिए इन्हें रेफ्रिजरेटर में संग्रहीत किया जा सकता है।",
                "Processing & Packaging": "यदि आवश्यक हो, तो संतरे को जूस, मुरब्बा या सूखे टुकड़ों में संसाधित किया जा सकता है। संतरों को सांस लेने योग्य कंटेनरों में पैक करें ताकि भंडारण और परिवहन के दौरान उनकी गुणवत्ता बनी रहे।",
                "Challenges & Solutions": "आम चुनौतियों में कीट और रोगों की संवेदनशीलता, पर्यावरणीय तनाव (जैसे सूखा या पाला) और अनुचित छंटाई तकनीक शामिल हैं। रोग-प्रतिरोधी किस्मों का चयन करें, अच्छे कृषि अभ्यासों को लागू करें, और इन चुनौतियों को कम करने के लिए पर्यावरणीय परिस्थितियों की निगरानी करें।"
            },


           {
                "name": "पपीता खेती गाइड",
                "Introduction": "पपीता (Carica papaya) एक उष्णकटिबंधीय फलदार वृक्ष है, जो अपने मीठे, रसदार गूदे और चमकीले नारंगी रंग के लिए प्रसिद्ध है। यह गर्म जलवायु में पनपता है और अनुकूल परिस्थितियों में वर्षभर फल प्रदान कर सकता है। यह गाइड पपीते की खेती की पूरी प्रक्रिया को रेखांकित करता है, जिसमें रोपण से लेकर कटाई तक की जानकारी शामिल है।",
                "Materials Required": "- प्रतिष्ठित नर्सरी से उच्च गुणवत्ता वाले पपीते के बीज या पौधे\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम युक्त संतुलित उर्वरक; जैविक खाद\n- प्रभावी नमी प्रबंधन के लिए ड्रिप सिंचाई प्रणाली या नली\n- कीटनाशक, फफूंदनाशी और जैविक कीट प्रबंधन समाधान\n- रोपण, रखरखाव और कटाई के लिए हाथ के उपकरण (फावड़ा, छंटाई कैंची, कुदाल) या ट्रैक्टर",
                "Soil Preparation": "पपीता अच्छी जल निकासी वाली, बलुई दोमट या दोमट मिट्टी को पसंद करता है, जिसका pH 6.0 से 6.5 के बीच हो। मिट्टी को जोतकर उसमें जैविक पदार्थ मिलाएं ताकि जल निकासी और उपजाऊपन में सुधार हो।",
                "Plant Selection & Treatment": "अपने जलवायु के अनुसार रोग-प्रतिरोधी पपीते की किस्में चुनें। यदि बीजों का उपयोग कर रहे हैं, तो रोपण से पहले उन्हें कुछ घंटों के लिए भिगोएं ताकि अंकुरण दर में सुधार हो।",
                "Field Preparation": "रोपण क्षेत्र को खरपतवार, पत्थर और मलबे से साफ करें ताकि एक स्वच्छ वातावरण सुनिश्चित हो।",
                "Planting Time": "पपीते के पौधों को लगाने का सबसे अच्छा समय वसंत ऋतु होता है, जब तापमान लगातार गर्म रहता है।",
                "Spacing & Depth": "पपीते के पौधों को 6-10 फीट की दूरी पर लगाएं ताकि उनकी बड़ी छतरी और जड़ प्रणाली के लिए पर्याप्त जगह हो। बीजों या पौधों को 0.5 से 1 इंच की गहराई पर लगाएं।",
                "Seeding/Transplanting Methods": "प्रत्यक्ष बीजाई: अंतिम ठंढ के बाद बीजों को सीधे जमीन में बोएं।\nरोपाई: बीजों को घर के अंदर अंकुरित करें और जब वे लगभग 12 इंच लंबे हो जाएं, तब उन्हें खेत में प्रत्यारोपित करें।",
                "Watering Requirements": "छोटे पपीते के पौधों को नियमित रूप से पानी दें, विशेष रूप से शुष्क मौसम में। पपीते को लगातार नमी की आवश्यकता होती है लेकिन जलभराव सहन नहीं होता।",
                "Nutrient Management": "वृद्धि के मौसम में हर 4-6 सप्ताह में संतुलित उर्वरक लगाएं। जैविक खाद का उपयोग करके मिट्टी की उपजाऊपन में सुधार करें।",
                "Weed Control": "मल्चिंग से खरपतवारों को नियंत्रित करें, जिससे नमी बनाए रखने और खरपतवार वृद्धि को दबाने में मदद मिलती है। साथ ही, प्रतिस्पर्धा को कम करने के लिए समय-समय पर खरपतवार निकालें।",
                "Pest & Disease Management": "एफिड्स, सफेद मक्खियाँ और फल मक्खियों जैसे कीटों की निगरानी करें। पाउडरी मिल्ड्यू और जड़ सड़न जैसी बीमारियों का उचित स्वच्छता और रोग-प्रतिरोधी किस्मों के माध्यम से प्रबंधन करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिसमें सांस्कृतिक नियंत्रण और लाभकारी कीटों का उपयोग शामिल हो।",
                "Special Care During Growth": "- अंकुर अवस्था: युवा पौधों को चरम मौसम और कीटों से बचाएं। यदि आवश्यक हो तो पाले और कीड़ों से बचाने के लिए रो कवर का उपयोग करें।\n- वनस्पति वृद्धि अवस्था: नियमित रूप से पोषक तत्वों की कमी की जांच करें और उन्हें तुरंत ठीक करें। स्वस्थ वृद्धि को प्रोत्साहित करने के लिए मरे हुए या क्षतिग्रस्त पत्तों की छंटाई करें।\n- फल विकास अवस्था: फल बनने के दौरान पर्याप्त पानी की आपूर्ति सुनिश्चित करें। यदि आवश्यक हो, तो बड़े फल प्राप्त करने के लिए अतिरिक्त फलों को पतला करें।",
                "Harvesting": "पपीते आमतौर पर रोपण के 6-12 महीने बाद कटाई के लिए तैयार होते हैं, जो किस्म पर निर्भर करता है। कटाई के संकेतों में त्वचा का हरा से पीला रंग में परिवर्तन और मीठी सुगंध शामिल हैं। तेज चाकू से फल को पेड़ से काटें, जिससे फल के साथ थोड़ा सा तना जुड़ा रहे।",
                "Post-Harvest Management": "कटे हुए पपीते को धीरे से संभालें ताकि चोट लगने से बचा जा सके। उन्हें ठंडी और छायादार जगह पर संग्रहित करें।",
                "Storage Conditions": "पपीते को कमरे के तापमान पर रखा जा सकता है ताकि वे और अधिक पक सकें। एक बार पकने के बाद, उनकी ताजगी बढ़ाने के लिए उन्हें थोड़े समय के लिए रेफ्रिजरेटर में संग्रहीत किया जा सकता है।",
                "Processing & Packaging": "यदि आवश्यक हो, तो पपीते को स्मूदी, सलाद या सूखे फलों में संसाधित किया जा सकता है। पपीतों को सांस लेने योग्य कंटेनरों में पैक करें ताकि भंडारण और परिवहन के दौरान उनकी गुणवत्ता बनी रहे।",
                "Challenges & Solutions": "आम चुनौतियों में कीट और रोगों की संवेदनशीलता, पर्यावरणीय तनाव (जैसे सूखा या बाढ़) और अनुचित सिंचाई पद्धतियाँ शामिल हैं। रोग-प्रतिरोधी किस्मों का चयन करें, अच्छे कृषि अभ्यासों को लागू करें, और इन चुनौतियों को कम करने के लिए पर्यावरणीय परिस्थितियों की निगरानी करें।"
            },


            {
                "name": "कॉफी की खेती गाइड",
                "Introduction": "कॉफी (Coffea spp.) दुनिया में सबसे अधिक उपभोग किए जाने वाले पेयों में से एक है, जो अपनी उत्तेजक विशेषताओं और समृद्ध स्वाद के लिए प्रसिद्ध है। यह उष्णकटिबंधीय जलवायु में पनपती है, विशेष रूप से ऊँचाई वाले क्षेत्रों में, जहाँ इसकी वृद्धि के लिए अनुकूल परिस्थितियाँ होती हैं। यह गाइड रोपण से लेकर कटाई तक कॉफी की खेती की पूरी प्रक्रिया को रेखांकित करता है।",
                "Materials Required": "- प्रतिष्ठित नर्सरी से उच्च गुणवत्ता वाले कॉफी के पौधे या बीज\n- नाइट्रोजन, फॉस्फोरस और पोटैशियम से भरपूर संतुलित उर्वरक; जैविक खाद\n- प्रभावी नमी प्रबंधन के लिए ड्रिप सिंचाई प्रणाली या नली\n- कीटनाशक, फफूंदनाशी और जैविक कीट प्रबंधन समाधान\n- रोपण, रखरखाव और कटाई के लिए हाथ के उपकरण (फावड़ा, छंटाई कैंची, कुदाल) या ट्रैक्टर",
                "Soil Preparation": "कॉफी अच्छी जल निकासी वाली, दोमट मिट्टी को पसंद करती है, जिसका pH 6.0 से 6.5 के बीच होना चाहिए। मिट्टी को जोतकर उसमें जैविक पदार्थ मिलाएँ ताकि उपजाऊपन और जल निकासी में सुधार हो।",
                "Plant Selection & Treatment": "अपने जलवायु के अनुसार रोग-प्रतिरोधी कॉफी की किस्में चुनें। यदि बीजों का उपयोग कर रहे हैं, तो रोपण से पहले उन्हें 24 घंटे के लिए भिगोएँ ताकि अंकुरण दर में सुधार हो।",
                "Field Preparation": "रोपण क्षेत्र को खरपतवार, पत्थर और मलबे से साफ करें ताकि एक स्वच्छ वातावरण सुनिश्चित हो।",
                "Planting Time": "कॉफी लगाने का सबसे अच्छा समय बारिश के मौसम की शुरुआत में होता है।",
                "Spacing & Depth": "कॉफी के पौधों को 5-8 फीट की दूरी पर लगाएँ ताकि उचित वृद्धि और वायु संचार सुनिश्चित हो सके। पौधों को इतनी गहराई पर रोपें कि उनकी जड़ गर्दन मिट्टी की सतह के समान रहे।",
                "Seeding/Transplanting Methods": "रोपाई: गड्ढा इतना बड़ा खोदें कि जड़ों के लिए पर्याप्त जगह हो, फिर पौधे को उसमें रखें, हल्के से मिट्टी भरें और रोपण के बाद अच्छी तरह से पानी दें।",
                "Watering Requirements": "छोटे कॉफी के पौधों को नियमित रूप से पानी दें ताकि जड़ें स्थापित हो सकें, विशेष रूप से शुष्क मौसम में। परिपक्व पौधों को लगातार नमी की आवश्यकता होती है लेकिन जलभराव नहीं होना चाहिए।",
                "Nutrient Management": "वृद्धि के मौसम में हर 3-4 महीने में संतुलित उर्वरक लगाएँ। जैविक खाद का उपयोग करके मिट्टी की उपजाऊपन में सुधार करें।",
                "Weed Control": "मल्चिंग से खरपतवारों को नियंत्रित करें, जिससे नमी बनाए रखने और खरपतवार वृद्धि को दबाने में मदद मिलती है। साथ ही, प्रतिस्पर्धा को कम करने के लिए समय-समय पर खरपतवार निकालें।",
                "Pest & Disease Management": "कॉफी बोरर बीटल और लीफ रस्ट जैसे कीटों की निगरानी करें। जड़ सड़न और पत्ती के धब्बे जैसी बीमारियों का उचित स्वच्छता और रोग-प्रतिरोधी किस्मों के माध्यम से प्रबंधन करें। एकीकृत कीट प्रबंधन (IPM) रणनीतियों को लागू करें, जिसमें सांस्कृतिक नियंत्रण और लाभकारी कीटों का उपयोग शामिल हो।",
                "Special Care During Growth": "- अंकुर अवस्था: युवा पौधों को चरम मौसम और कीटों से बचाएँ। यदि आवश्यक हो तो तेज धूप से बचाने के लिए छायादार कपड़े का उपयोग करें।\n- वनस्पति वृद्धि अवस्था: नियमित रूप से पोषक तत्वों की कमी की जांच करें और उन्हें तुरंत ठीक करें। पौधों को आकार देने और मृत या रोगग्रस्त शाखाओं को हटाने के लिए छँटाई करें।\n- फूल और फल विकास अवस्था: फूल और फल बनने के दौरान पर्याप्त पानी की आपूर्ति सुनिश्चित करें ताकि उपज और गुणवत्ता में सुधार हो सके। फलों पर मक्खियों के संक्रमण की निगरानी करें और आवश्यकतानुसार नियंत्रण करें।",
                "Harvesting": "कॉफी चेरी फूल आने के 7-9 महीने बाद कटाई के लिए तैयार होती हैं, जो किस्म पर निर्भर करती है। कटाई के संकेतों में चेरी का रंग हरे से चमकदार लाल या पीले में बदलना शामिल है। कॉफी चेरी को हाथ से चुनें, केवल पकी हुई चेरी ही तोड़ें। उच्च गुणवत्ता के लिए चयनात्मक कटाई विधि अपनाएँ।",
                "Post-Harvest Management": "कटे हुए चेरी को धीरे से संभालें ताकि चोट लगने से बचा जा सके। खराब होने से रोकने के लिए उन्हें यथाशीघ्र प्रोसेस करें।",
                "Processing Methods": "कॉफी बीज निकालने के लिए या तो सूखी विधि (सूरज में चेरी सुखाना) या गीली विधि (किण्वन और धोने की प्रक्रिया) का उपयोग करें।",
                "Storage Conditions": "प्रसंस्करण किए गए कॉफी बीजों को ठंडी, शुष्क जगह पर संग्रहीत करें ताकि खराबी से बचा जा सके और स्वाद बरकरार रहे।",
                "Processing & Packaging": "कॉफी बीजों को एयरटाइट कंटेनरों में पैक करें ताकि भंडारण और परिवहन के दौरान उनकी ताजगी बनी रहे।",
                "Challenges & Solutions": "आम चुनौतियों में कीट और रोगों की संवेदनशीलता, पर्यावरणीय तनाव (जैसे सूखा या पाला), और बाजार मूल्य में उतार-चढ़ाव शामिल हैं। रोग-प्रतिरोधी किस्मों का चयन करें, अच्छे कृषि अभ्यासों को लागू करें, और इन चुनौतियों को कम करने के लिए पर्यावरणीय परिस्थितियों की निगरानी करें।"
            }

        ]

    cropGuideChinese = [
        {"name": "玉米种植指南", 
            "简介": "玉米（Zea mays），又称玉蜀黍，是一种重要的谷类作物，广泛种植以获取其籽粒。本指南涵盖从选种到收获的完整玉米种植流程。",
            "所需材料": "- 优质玉米种子（杂交或改良品种）\n- 肥料（氮、磷、钾）\n- 机械设备（拖拉机、手工工具、播种机）\n- 病虫害防治用品（除草剂、杀虫剂）\n- 灌溉设备（滴灌或沟灌）",
            "土壤准备": "玉米适宜在排水良好的壤土中生长，土壤pH值应在5.8至7.0之间。翻耕土壤以改善通气性并打碎土块。",
            "选种与种子处理": "选择高产、抗旱的品种。用杀菌剂或杀虫剂处理种子以提供保护。",
            "田间准备": "平整田地以确保水分均匀分布。优化行距以最大化阳光照射。",
            "播种时间": "通常在雨季开始时播种，具体时间因地区而异，一般在4月至6月之间。",
            "间距与深度": "行内间距20-25厘米，行间间距60-75厘米，播种深度2-5厘米。",
            "播种方法": "- **直播：** 手动或使用播种机直接播种。",
            "浇水需求": "需要定期浇水，尤其是在抽穗和吐丝期。如果雨水不足，需进行灌溉。",
            "养分管理": "分次施肥：播种时、生长初期和抽穗期各施一次。",
            "杂草控制": "人工除草、锄地或使用除草剂。第一次除草在播种后15-20天，第二次在30-40天后。",
            "病虫害管理": "注意玉米螟、草地贪夜蛾和蚜虫等害虫。使用杀虫剂和综合病虫害管理（IPM）措施。",
            "收获": "当玉米穗成熟且苞叶干燥时收获。籽粒含水量应在20-25%之间。可手工采摘或使用机械收割机。",
            "收获后管理": "将籽粒干燥至含水量13-14%。脱粒、清理并妥善储存。",
            "储存条件": "储存在通风、阴凉干燥处，以防霉变和虫害。",
            "加工": "如需进一步加工，可将玉米干燥并磨粉。",
            "挑战与解决方案": "常见问题包括天气多变、病虫害和缺水。解决方案包括IPM、土壤水分监测和抗逆品种。"
        },
        
        {"name": "水稻种植指南", 
            "简介": "水稻（Oryza sativa）是全球许多地区的主粮作物。本指南涵盖从选种到收获的完整水稻种植流程。",
            "所需材料": "- 优质稻种\n- 肥料（氮、磷、钾）\n- 灌溉系统\n- 机械设备（拖拉机、插秧机、镰刀）\n- 病虫害防治用品（除草剂、杀虫剂）", 
            "土壤准备": "水稻最适宜在黏土或黏壤土中生长，pH值5.5至6.5。翻耕并平整田地以确保水分均匀分布。", 
            "选种与种子处理": "选择高产、抗病虫害的种子。用杀菌剂或杀虫剂处理以防感染。", 
            "田间准备": "平整田地并修筑田埂以蓄水。", 
            "播种时间": "通常在雨季开始时播种，具体时间因地区而异，一般为5月至6月。", 
            "间距与深度": "移栽时采用20x15厘米间距。直播时播种深度2-3厘米。",
            "播种方法": "- **直播：** 撒播或条播。\n- **移栽：** 在苗床育苗20-30天后移栽。",
            "浇水需求": "生长期保持5-10厘米水深。籽粒成熟期减少水量。",
            "养分管理": "分次施肥：播种时、分蘖期和孕穗期各施一次。",
            "杂草控制": "人工除草或使用除草剂。移栽后15-20天第一次除草，40天后第二次。",
            "病虫害管理": "注意螟虫、叶蝉等害虫。使用杀虫剂和综合病虫害管理（IPM）措施。",
            "收获": "当稻谷变金黄且80-90%籽粒成熟时收获。小规模种植可用镰刀，大规模种植建议使用收割机。",
            "收获后管理": "将稻谷干燥至含水量14%，脱粒、扬净并存放在阴凉干燥处以防变质。",
            "挑战与解决方案": "常见问题包括恶劣天气、病虫害和缺水。可通过IPM、水位监测和品种多样化来应对。"
        },
        
        {"name": "黄麻种植指南",
            "简介": "黄麻是一种纤维作物，主要用于生产坚固的天然纤维，广泛应用于纺织和包装行业。本指南涵盖从选种到收获的完整黄麻种植流程。",
            "所需材料": "- 优质黄麻种子（长果黄麻或圆果黄麻）\n- 有机堆肥、氮磷钾肥料\n- 手工工具或拖拉机用于整地\n- 除草剂和杀虫剂\n- 灌溉系统用于控水",
            "土壤准备": "黄麻适宜在排水良好的壤土或砂壤土中生长，pH值6.0至7.5。通过犁地和耙地整地，确保苗床良好。",
            "选种与种子处理": "选择高产抗病的品种。播种前将种子浸泡24小时以促进发芽。",
            "田间准备": "清理并平整田地以确保水分均匀分布。如可能发生涝灾，可在田边修筑小田埂。",
            "播种时间": "黄麻通常在雨季开始时播种，一般为3月至5月。",
            "间距与深度": "行距25-30厘米，播种深度1-2厘米以确保最佳发芽率。",
            "播种方法": "- **撒播：** 将种子均匀撒在田里。\n- **条播：** 按行播种，便于除草和其他管理。",
            "浇水需求": "黄麻需要定期保湿，尤其是生长初期。避免积水，大雨后确保排水通畅。",
            "养分管理": "播种时施基肥（氮磷钾）。间苗后20-25天追加氮肥。",
            "杂草控制": "早期人工除草或使用选择性除草剂。第一次除草在播种后15-20天，第二次在30-40天后。",
            "病虫害管理": "注意黄麻毛虫、蚜虫等害虫。使用杀虫剂或综合病虫害管理（IPM）防治病虫害。",
            "收获": "当植株长到10-12英尺高且下部叶片开始发黄时收获（通常播种后100-120天）。用镰刀或刀具贴近基部割取。为获得最佳纤维质量，应在开花前收割。",
            "收获后管理": "将收割的黄麻捆扎后浸入清洁缓流水中进行沤麻（发酵以分离纤维）。沤麻通常需10-15天，需定期检查纤维分离情况。",
            "挑战与解决方案": "常见问题包括水源、虫害和沤麻不当。可通过高效灌溉、害虫防治和沤麻期水位监控来应对。"
        },

        {"name": "棉花种植指南",
            "简介": "棉花是一种重要的纤维作物，其柔软蓬松的纤维广泛用于纺织业。本指南涵盖从选种到收获的完整棉花种植流程。",
            "所需材料": "- 优质棉花种子（如Bt棉等抗虫品种）\n- 氮磷钾及微量元素肥料\n- 滴灌或沟灌系统\n- 除草剂和杀虫剂\n- 犁、拖拉机和喷雾器等整地及维护设备",
            "土壤准备": "棉花适宜在排水良好的砂壤土中生长，pH值6.0至7.5。深耕后耙地以破碎土块并平整表面。",
            "选种与种子处理": "选择高产抗虫品种。用杀菌剂或杀虫剂处理种子以防土传病害和早期虫害。",
            "田间准备": "根据灌溉方式开沟或做畦。确保排水良好以防涝害。",
            "播种时间": "棉花通常在春季播种，具体时间为3月至5月，依地区温度而定。",
            "间距与深度": "播种深度3-5厘米，行距75-100厘米，株距25-30厘米。",
            "播种方法": "- **直播：** 用播种机或手动在准备好的沟或畦中直接播种。",
            "浇水需求": "棉花需要持续保湿，尤其是开花和结铃期。干旱时使用滴灌或沟灌保持土壤湿度。",
            "养分管理": "播种时施磷钾基肥。氮肥分三次施：播种时、营养生长期和开花期各施三分之一。",
            "杂草控制": "早期人工除草、锄地或使用除草剂。播种后20-30天第一次除草，必要时45天后第二次。",
            "病虫害管理": "注意棉铃虫、蚜虫和粉虱等害虫。采用综合病虫害管理（IPM），包括生物防治以减少农药使用。",
            "收获": "棉铃完全开裂且纤维蓬松时收获（通常播种后150-180天）。人工采摘需用手摘取成熟棉铃，大型农场可用采棉机。",
            "收获后管理": "将收获的棉花阴干。清理并轧棉以分离棉籽和纤维。将纤维储存在干燥通风处以防受潮。",
            "挑战与解决方案": "常见问题包括虫害、缺水和土壤养分流失。可通过抗旱品种、高效灌溉和IPM措施应对。"
        },

        {"name": "椰子种植指南",
            "简介": "椰子树（Cocos nucifera）因其果实可提供椰油、椰奶和椰纤维而被广泛种植。本指南涵盖从选种到收获的关键步骤。",
            "所需材料": "- 优质椰苗（矮种或高种）\n- 有机肥、NPK肥料\n- 滴灌或树盘灌溉\n- 杀虫剂或生物防治剂\n- 手工工具或机械设备",
            "土壤准备": "椰子适宜在排水良好的砂壤土中生长，pH值5.5-7.5。挖1x1x1米的定植穴，填入土壤、堆肥和有机肥以促进根系生长。",
            "选种与种子处理": "选用抗病高产的椰苗。矮种便于收获，高种更抗旱。",
            "田间准备": "清除杂草和杂物，确保排水良好，按品种需求间距挖定植穴。",
            "播种时间": "最佳种植时间为雨季初期以减少灌溉需求；若有灌溉条件可全年种植。",
            "间距与深度": "高种株距7.5-9米；矮种6.5-7米。确保根系充分覆土。",
            "播种方法": "将椰苗放入定植穴，茎基部略高于地面。",
            "浇水需求": "前三年定期浇水。成树虽抗旱，但持续灌溉有益。",
            "养分管理": "每年分三次施平衡肥并补充镁、硼等微量元素。每年添加有机肥。",
            "杂草控制": "定期除草，尤其是生长初期。覆盖可保湿抑草。",
            "病虫害管理": "用杀虫剂或生物防治对付犀角金龟、红棕象甲等害虫。通过杀菌剂和修剪防治根萎病和芽腐病。",
            "收获": "成熟椰子（开花后12个月）会变褐色。每45-60天采收一次，可用攀爬工具或机械升降机。",
            "收获后管理": "储存在干燥通风处。通过晒干或机械干燥制作椰干。干燥椰子需密封包装运输。",
            "挑战与解决方案": "干旱、虫害和土壤退化可通过滴灌、害虫管理和有机土壤改良应对。"
        },

        {"name": "鹰嘴豆种植指南",
            "简介": "鹰嘴豆（Cicer arietinum）是一种富含蛋白质的豆类，广泛用于食品生产。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病鹰嘴豆种子（Desi或Kabuli类型）\n- 磷肥为主，少量氮肥\n- 滴灌或喷灌\n- 除草剂和杀虫剂\n- 犁、拖拉机和喷雾器",
            "土壤准备": "鹰嘴豆适宜在排水良好的壤土中生长，pH值6.0-7.5。犁地耙地以利根系穿透。",
            "选种与种子处理": "选择高产抗病种子。用根瘤菌处理以固氮，并用杀菌剂防病。",
            "田间准备": "清除杂草并平整田地。合理行距以促进通风减少病害。",
            "播种时间": "最佳播种期为凉爽干燥季节，通常为10-11月。",
            "间距与深度": "株距30-40厘米，行距45-60厘米。根据土壤湿度播种深度5-8厘米。",
            "播种方法": "使用播种机或人工直接播种。",
            "浇水需求": "鹰嘴豆需水少，但开花结荚期灌溉有益。避免积水。",
            "养分管理": "播种时施磷肥。根据土壤检测补充钾和微量元素。",
            "杂草控制": "早期定期除草，人工或化学除草。第一次除草在播后20-30天，第二次在45-50天（如需）。",
            "病虫害管理": "注意豆荚螟、蚜虫等害虫。采用综合病虫害管理（IPM）和生物农药。",
            "生长阶段特殊护理": "- 苗期：防虫保湿\n- 营养生长期：保持磷水平\n- 开花结荚期：保证水分以提高产量",
            "收获": "鹰嘴豆3-4个月成熟。植株变黄、豆荚干燥时收获。小农场手工收割，大规模用联合收割机。",
            "收获后管理": "晒干种子降低水分，脱粒清理后储存或销售。",
            "储存条件": "储存在干燥阴凉通风处以防虫害和霉变。",
            "加工与包装": "清理分级后装入透气袋。",
            "挑战与解决方案": "常见问题包括病虫害、水分胁迫和营养缺乏。可通过IPM、抗病品种和土壤测试应对。"
        },

        {"name": "木豆种植指南",
            "简介": "木豆（Cajanus cajan）是一种抗旱豆类，因其高蛋白含量和多样烹饪用途而受重视。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病木豆种子（早、中、晚熟品种）\n- 氮磷钾肥料（需氮少）\n- 滴灌或沟灌设备\n- 木豆专用除草剂和杀虫剂\n- 手工工具或拖拉机用于整地、播种和除草",
            "土壤准备": "木豆适宜在排水良好的砂壤土至黏壤土中生长，pH值6.0-7.5。犁地耙地以创造细碎苗床。",
            "选种与种子处理": "选择适合当地的高产抗病品种。用杀菌剂处理种子以防种传病害。",
            "田间准备": "清理田间的杂草和杂物，确保排水良好。",
            "播种时间": "通常在雨季初期或亚热带地区的旱季播种。",
            "间距与深度": "株距30-40厘米，行距60-75厘米。播种深度3-5厘米（依土壤湿度和质地而定）。",
            "播种方法": "使用播种机或人工直接播种。",
            "浇水需求": "木豆抗旱，但开花和豆荚发育期需保湿。前60天可能需要灌溉。",
            "养分管理": "播种时施磷钾肥，必要时追施氮肥。有机改良剂可提高土壤肥力。",
            "杂草控制": "生长初期通过人工除草或除草剂控草。覆盖可抑草保墒。",
            "病虫害管理": "注意豆荚螟、蚜虫和粉虱等害虫。实施综合病虫害管理（IPM），包括生物防治和化学农药。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防虫保湿\n- 营养生长期：确保养分促进健壮生长\n- 开花结荚期：保持水分以提高产量和品质",
            "收获": "木豆4-6个月成熟。豆荚成熟干燥时收获。小农场手工收割，大规模用联合收割机。",
            "收获后管理": "收割后晒干植株以降低种子含水量。",
            "储存条件": "储存在干燥阴凉通风处以防变质和虫害。",
            "加工与包装": "清理分级后装入透气袋或容器。",
            "挑战与解决方案": "常见问题包括虫害、病害、水分胁迫和营养缺乏。可通过抗病品种、轮作和IPM策略应对。"
        },

        {"name": "绿豆种植指南",
            "简介": "绿豆（Vigna radiata）是一种小而绿的豆类，因其营养价值和烹饪多样性备受推崇。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病绿豆种子\n- 氮磷钾肥料（需氮少）\n- 滴灌或沟灌\n- 除草剂和杀虫剂\n- 手工工具或拖拉机",
            "土壤准备": "绿豆适宜在排水良好的砂壤土至壤土中生长，pH值6.0-7.5。犁地耙地以创造细碎苗床。",
            "选种与种子处理": "选择适合气候的高产抗病品种。用杀菌剂处理种子以防土传病害。",
            "田间准备": "清理田间杂草和杂物以确保良好种土接触。",
            "播种时间": "通常在雨季初期或温暖干燥的4-6月播种。",
            "间距与深度": "株距30-40厘米，行距45-60厘米。播种深度2-4厘米（依土壤湿度而定）。",
            "播种方法": "使用播种机或人工直接播种。",
            "浇水需求": "绿豆需充足水分，尤其是发芽和开花期。雨水不足时灌溉，避免过湿以防根腐。",
            "养分管理": "播种时施磷钾肥。如需可追加氮肥，但通常自然固氮足够。添加有机质提高土壤肥力。",
            "杂草控制": "早期通过人工或化学除草控草。覆盖可抑草保墒。",
            "病虫害管理": "注意蚜虫、甲虫和蓟马等害虫。采用综合病虫害管理（IPM）策略。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防虫保湿\n- 营养生长期：确保养分促进健壮生长\n- 开花结荚期：保持水分以提高产量和品质",
            "收获": "绿豆60-90天成熟。豆荚干燥时收获。小农场手工收割，大规模用联合收割机。",
            "收获后管理": "收割后晒干植株以降低种子含水量。",
            "储存条件": "储存在干燥阴凉通风处以防变质和虫害。",
            "加工与包装": "清理分级后装入透气袋。",
            "挑战与解决方案": "常见问题包括病虫害和恶劣天气。可通过抗病品种、IPM和合理水土管理应对。"
        },

        {"name": "黑豆种植指南",
            "简介": "黑豆（Vigna mungo）是一种高营养豆类，因其高蛋白含量和多样烹饪用途而受重视。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病黑豆种子\n- 磷钾肥料（需氮少）\n- 滴灌或沟灌\n- 除草剂和杀虫剂\n- 手工工具或拖拉机",
            "土壤准备": "黑豆适宜在排水良好的砂壤土至黏壤土中生长，pH值6.0-7.5。犁地耙地以创造细碎苗床。",
            "选种与种子处理": "选择适合气候的高产抗病品种。用杀菌剂或杀虫剂处理种子以防土传病害。",
            "田间准备": "清理田间杂草和杂物以确保良好种土接触。",
            "播种时间": "通常在雨季初期或温暖干燥的6-7月播种。",
            "间距与深度": "株距30-45厘米，行距60-75厘米。播种深度3-5厘米（依土壤湿度而定）。",
            "播种方法": "使用播种机或人工直接播种。",
            "浇水需求": "黑豆需充足水分，尤其是发芽和开花期。雨水不足时灌溉，避免过湿以防根腐。",
            "养分管理": "播种时施磷钾肥。通常无需额外氮肥（因固氮作用）。添加有机质提高土壤肥力。",
            "杂草控制": "早期通过人工或化学除草控草。覆盖可抑草保墒。",
            "病虫害管理": "注意蚜虫、豆荚螟和蓟马等害虫。采用综合病虫害管理（IPM）策略。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防虫保湿\n- 营养生长期：确保养分促进健壮生长\n- 开花结荚期：保持水分以提高产量和品质",
            "收获": "黑豆60-90天成熟。豆荚干燥时收获。小农场手工收割，大规模用联合收割机。",
            "收获后管理": "收割后晒干植株以降低种子含水量。",
            "储存条件": "储存在干燥阴凉通风处以防变质和虫害。",
            "加工与包装": "清理分级后装入透气袋。",
            "挑战与解决方案": "常见问题包括病虫害和恶劣天气。可通过抗病品种、IPM和合理水土管理应对。"
        },

        {"name": "扁豆种植指南",
            "简介": "扁豆（Lens culinaris）是一种营养丰富的豆类，以高蛋白和高纤维含量闻名，是许多菜肴的主料。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病扁豆种子\n- 磷钾肥料（需氮少）\n- 滴灌或沟灌\n- 除草剂和杀虫剂\n- 手工工具或拖拉机",
            "土壤准备": "扁豆适宜在排水良好的壤土或砂土中生长，pH值6.0-7.5。犁地耙地以创造细碎苗床。",
            "选种与种子处理": "选择适合地区的高产抗病品种。用杀菌剂或杀虫剂处理种子以防种传病害。",
            "田间准备": "清理田间杂草和杂物以确保良好种土接触。",
            "播种时间": "扁豆通常在早春或冬末播种，具体时间依气候而定，当土壤温度达10-15°C（50-59°F）时。",
            "间距与深度": "株距25-30厘米，行距45-60厘米。播种深度2-3厘米（依土壤湿度而定）。",
            "播种方法": "使用播种机或人工直接播种。",
            "浇水需求": "扁豆耐旱，但发芽和豆荚发育期需保湿。开花和籽粒充实期如雨水不足需灌溉。",
            "养分管理": "播种时施磷钾肥。通常无需额外氮肥（因固氮作用）。添加有机质提高土壤肥力。",
            "杂草控制": "生长初期通过人工或化学除草控草。覆盖可抑草保墒。",
            "病虫害管理": "注意蚜虫、盲蝽和根腐病等病虫害。实施综合病虫害管理（IPM）策略。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防虫保湿\n- 营养生长期：确保养分促进健壮生长\n- 开花结荚期：保持水分以提高产量和品质",
            "收获": "扁豆80-100天成熟。豆荚变褐干燥时收获。小农场手工收割，大规模用联合收割机。",
            "收获后管理": "收割后晒干植株以降低种子含水量。",
            "储存条件": "储存在干燥阴凉通风处以防变质和虫害。",
            "加工与包装": "清理分级后装入透气袋。",
            "挑战与解决方案": "常见问题包括病虫害和天气多变。可通过抗病品种、IPM和合理水土管理应对。"
        },

        {"name": "石榴种植指南",
            "简介": "石榴（Punica granatum）是一种营养丰富的水果，以其健康益处和鲜美的风味闻名。它们在温暖气候下生长良好，全球许多地区都有种植。本指南涵盖从种植到收获的完整流程。",
            "所需材料": "- 优质石榴种子或健康苗木（来自可靠苗圃）\n- 氮磷钾平衡肥料\n- 滴灌或沟灌系统\n- 杀虫剂和杀菌剂\n- 手工工具或拖拉机用于种植、修剪和维护",
            "土壤准备": "石榴适宜在排水良好的砂壤土至壤土中生长，pH值5.5至7.0。种植前犁地并掺入有机质。",
            "选种与种子处理": "选择适合当地气候的抗病品种。如用种子，播种前浸泡一夜以提高发芽率。",
            "田间准备": "清理种植地的杂草、石块和杂物。",
            "播种时间": "石榴通常在春季最后一次霜冻后种植。",
            "间距与深度": "株距5-8英尺以利生长和通风。种子或苗木种植深度1-2英寸，确保与土壤良好接触。",
            "播种方法": "直播：将种子直接播入准备好的地块。移栽：对苗木，挖略大于根球的穴，回填土壤。",
            "浇水需求": "石榴需定期浇水，尤其是定植期；成株后耐旱。深浇少灌以促根系下扎。",
            "养分管理": "生长季初期和夏末各施一次平衡肥。添加有机堆肥提高土壤肥力。",
            "杂草控制": "通过覆盖和人工除草减少杂草竞争。",
            "病虫害管理": "注意蚜虫、粉虱和石榴蝶等害虫。采用综合病虫害管理（IPM）策略，包括天敌和有机农药。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防极端天气和害虫；必要时用遮阳布。\n- 营养生长期：定期检查营养缺乏和虫害；按需施肥。\n- 开花结果期：花果期保证水分以促进健康发育。",
            "收获": "石榴通常在开花后5-7个月成熟，果色深且敲击有金属声时采收。用锋利修枝剪剪下果实，避免伤及枝条和其他果实。",
            "收获后管理": "轻拿轻放以防碰伤；存放于阴凉干燥处。",
            "储存条件": "石榴在适当条件下可储存数周至数月。",
            "加工与包装": "清洁并分拣果实，剔除损伤或腐烂者。用透气容器包装以保持品质。",
            "挑战与解决方案": "常见问题包括易感病虫害、干旱或过湿等环境胁迫。选择抗病品种、合理灌溉并监控虫口以应对。"
        },

        {"name": "菜豆种植指南",
            "简介": "菜豆（Phaseolus vulgaris）是一种高蛋白豆类，广泛用于各类菜肴。本指南涵盖从选种到收获的完整种植流程。",
            "所需材料": "- 优质抗病菜豆种子\n- 磷钾肥料（菜豆可固氮，需氮少）\n- 滴灌或喷灌\n- 菜豆专用除草剂和杀虫剂\n- 手工工具或拖拉机用于整地、播种和除草",
            "土壤准备": "菜豆适宜在排水良好的壤土中生长，pH值6.0至7.0。犁地耙地以利根系穿透。",
            "选种与种子处理": "选择高产抗病品种。用杀菌剂或杀虫剂处理种子以防早期土传病害和虫害。",
            "田间准备": "清理田间杂草和杂物后整平。划行确保通风和光照。",
            "播种时间": "菜豆通常在春季播种，当土壤温度达15°C（59°F）且无霜冻风险时。",
            "间距与深度": "播种深度3-5厘米，株距8-10厘米，行距45-60厘米。",
            "播种方法": "直播：手动或使用播种机直接在田间播种。",
            "浇水需求": "菜豆需定期浇水，尤其是开花和结荚期。避免过湿，因菜豆不耐涝。",
            "养分管理": "播种时施磷钾肥。限制氮肥，因菜豆可固氮。如土壤检测显示缺乏，补充微量元素。",
            "杂草控制": "早期除草至关重要。人工除草或按需使用除草剂。植株周围覆盖可保湿抑草。",
            "病虫害管理": "注意蚜虫、叶蝉和豆甲等害虫。采用综合病虫害管理（IPM）措施，必要时施药。通过轮作和避免积水预防根腐病和疫病。",
            "生长阶段特殊护理": "- 苗期：保持适度土壤湿度并防虫。\n- 营养生长期：维持养分以支持茎叶生长。\n- 开花结荚期：结荚期保持水分以提高产量和品质。",
            "收获": "菜豆90-120天成熟。豆荚完全成熟干燥时收获。小农场可整株拔起，大型农场用联合收割机。",
            "收获后管理": "收割后晒干植株以降低种子含水量。脱粒后清理种子。",
            "储存条件": "将菜豆储存在干燥通风处以防霉变和虫害。",
            "加工与包装": "清理并分级后包装。使用透气袋或容器以保持储存品质。",
            "挑战与解决方案": "常见问题包括易感病虫害和营养失衡。使用抗病种子、监控土壤健康并采用IPM措施有效防控。"
        },

        {"name": "香蕉种植指南",
            "简介": "香蕉（Musa spp.）是一种热带水果，以其甜美的味道和营养价值闻名。它们在温暖湿润的气候下生长良好，全球广泛种植用于商业和家庭消费。本指南概述从种植到收获的完整流程。",
            "所需材料": "- 健康的香蕉吸芽或组培苗\n- 氮磷钾平衡肥料；堆肥等有机质\n- 滴灌或喷灌系统\n- 杀虫剂和杀菌剂\n- 手工工具（铲、修枝剪）或拖拉机用于种植、维护和采收",
            "土壤准备": "香蕉适宜在排水良好、肥沃的壤土中生长，pH值5.5至7.0。犁地并加入有机质以提高肥力和排水性。",
            "选种与处理": "从健康母株选择无病吸芽或从可靠来源获取组培苗。如用吸芽，用干净刀具从母株分离以避免污染。",
            "田间准备": "清理种植地的杂草、石块和杂物。",
            "播种时间": "最佳种植时间为雨季初期或温暖月份。",
            "间距与深度": "行距10-12英尺，株距8-10英尺以利生长和通风。吸芽或组培苗按原深度种植。",
            "移栽方法": "移栽：挖足够大的穴容纳根系，回填时避免气穴。",
            "浇水需求": "香蕉需持续保湿；旱季定期灌溉。每周需水1-2英寸。",
            "养分管理": "早春和生长季中期各施一次平衡肥。添加堆肥或有机覆盖物提高肥力。",
            "杂草控制": "通过覆盖保湿并人工除草减少竞争。",
            "病虫害管理": "注意香蕉象甲和蚜虫等害虫。通过清洁和抗病品种防控巴拿马病和叶斑病。采用IPM策略，包括生物防治。",
            "生长阶段特殊护理": "- 苗期：保护幼苗防极端天气和害虫；必要时用遮阳布。\n- 营养生长期：定期检查钾镁等缺素症并及时处理。\n- 花果期：花果发育期保证水分以支持果实形成。",
            "收获": "香蕉通常种植后9-12个月可收，具体依品种和条件而定。当果实饱满、绿色且果柄角度明显时采收。用利刀或砍刀从植株割下整串。轻拿轻放以防碰伤。",
            "收获后管理": "去除多余叶片，轻拿轻放以防损伤。存放于阴凉处。",
            "储存条件": "香蕉室温存放至成熟。避免阳光直射或过热。",
            "加工与包装": "如需可加工成香蕉片或果泥。用透气盒包装以利运输中通风减损。",
            "挑战与解决方案": "常见问题包括易感病虫害、环境胁迫和不当灌溉。选择抗病品种、良好栽培实践并监控环境以应对。"
        },

        {"name": "葡萄种植指南",
            "简介": "葡萄（Vitis vinifera等）是一种多用途水果，可用于鲜食、制干（葡萄干）和酿酒。它们适应温带气候，需要特定条件以生产高品质果实。本指南概述从种植到收获的完整流程。",
            "所需材料": "- 优质葡萄苗（裸根或盆栽，来自可靠苗圃）\n- 氮磷钾平衡肥料；有机堆肥\n- 滴灌系统以高效管理水分\n- 杀虫剂、杀菌剂和有机防治方案\n- 手工工具（修枝剪、铲）或拖拉机用于种植、维护和采收",
            "土壤准备": "葡萄适宜在排水良好的砂壤土或黏壤土中生长，pH值6.0至6.8。整地时掺入有机质以提高肥力和排水性。",
            "选种与处理": "根据气候和用途（鲜食、酿酒等）选择抗病品种。种植前检查苗木是否有病害或损伤。",
            "田间准备": "清理种植地的杂草、石块和杂物。",
            "播种时间": "葡萄最佳种植时间为春季最后一次霜冻后或秋季土地封冻前。",
            "间距与深度": "行距8-10英尺，株距6-10英尺以利通风和生长。按苗圃原深度种植。",
            "移栽方法": "移栽：挖足够大的穴容纳根系，回填后浇透水。",
            "浇水需求": "葡萄定植第一年需定期浇水。成株后耐旱，但果实发育期需深灌。",
            "养分管理": "早春和生长季中期各施一次平衡肥。使用有机堆肥改善土壤健康。",
            "杂草控制": "通过覆盖保湿抑草，或人工除草及除草剂减少竞争。",
            "病虫害管理": "注意葡萄蛾、蚜虫和红蜘蛛等害虫。通过清洁和抗病品种防控白粉病和霜霉病。采用IPM策略，包括生物防治。",
            "生长阶段特殊护理": "- 幼藤期：保护幼苗防极端天气和害虫；用支架或棚架助其向上生长。\n- 营养生长期：定期检查缺素症并及时处理；修剪以促健壮结构和通风。\n- 花果期：花果期保持水分以提高产量和品质；必要时疏果以增大果粒。",
            "收获": "葡萄通常开花后4-6个月成熟，具体依品种而定。当果实完全成熟、色泽深且味甜时采收。用锋利修枝剪剪下果串，轻拿轻放。",
            "收获后管理": "剔除损伤或腐烂果实，存放于阴凉处。",
            "储存条件": "葡萄室温存放。冷藏可延长保质期，但需用透气容器。",
            "加工与包装": "如需可加工成果汁、果冻或葡萄酒。用透气容器包装以利运输中保鲜。",
            "挑战与解决方案": "常见问题包括易感病虫害、气候相关问题和不当灌溉。选择抗病品种、良好栽培实践并监控环境以应对。"
        },
        {"name": "甜瓜栽培指南",
            "Introduction": "甜瓜（学名：Cucumis melo var. cantaloupe）是以其多汁的果肉和独特的网状外皮而闻名的香甜水果。它们在温暖的气候中生长良好，因其清爽的口感而广受欢迎。本指南概述了从种植到收获的甜瓜栽培完整过程。",
            "Materials Required": "- 来自可靠来源的优质甜瓜种子或幼苗\n- 含氮、磷、钾的平衡肥料；有机堆肥\n- 滴灌或喷灌系统以高效管理水分\n- 杀虫剂、杀菌剂和有机害虫管理解决方案\n- 手工工具（铲子、锄头、修枝剪）或拖拉机用于种植、维护和收获",
            "Soil Preparation": "甜瓜喜欢排水良好的沙质壤土或壤土，pH值为6.0至6.8。通过耕作并混入有机物质来准备土壤，以增强排水性和肥力。",
            "Plant Selection & Treatment": "选择适合您气候和市场的抗病品种。如果使用种子，在种植前将其浸泡在水中几个小时以提高发芽率。",
            "Field Preparation": "清除种植地点的杂草、石头和碎屑，确保种植环境干净。",
            "Planting Time": "种植甜瓜的理想时间是在最后一次霜冻日期之后，当土壤温度持续保持在70°F（21°C）以上时。",
            "Spacing & Depth": "甜瓜植株间距为3-4英尺，行距为6-8英尺，以便藤蔓可以舒展。将种子或幼苗种植在约1英寸深的土中。",
            "Seeding/Transplanting Methods": "直接播种：在土壤变暖后直接将种子种入地中。移栽：室内培育幼苗，待其足够强壮后再移栽。",
            "Watering Requirements": "甜瓜需要持续的水分，特别是在发芽和果实发育期间。每周提供约1-2英寸的水，根据降雨情况调整。",
            "Nutrient Management": "在种植时施用平衡肥料，当藤蔓开始蔓延时再次施用。使用有机堆肥或覆盖物来提高土壤健康。",
            "Weed Control": "通过覆盖物控制杂草，这有助于保持水分并抑制杂草生长，以及手工除草以减少竞争。",
            "Pest & Disease Management": "监测蚜虫、黄瓜甲虫和蜘蛛螨等害虫。通过适当的卫生措施和抗性品种来管理白粉病和霜霉病等疾病。实施综合害虫管理（IPM）策略，包括文化控制和使用生物控制。",
            "Special Care During Growth": "- 幼苗阶段：保护幼苗免受害虫和极端天气影响。必要时使用行覆盖物保护幼苗免受害虫和霜冻。\n- 营养生长阶段：定期检查营养缺乏症并及时解决。必要时支撑藤蔓，特别是当果实开始发育时。\n- 果实发育阶段：在果实发育期间确保充足的水分供应，以促进健康生长和甜度。避免直接在果实上浇水以防止腐烂。",
            "Harvesting": "甜瓜通常在种植后70-90天准备收获。指标包括花萼端从绿色变为黄色以及散发出甜香。使用锋利的刀或修枝剪从藤蔓上切下果实，在瓜上留下一小段茎。",
            "Post-Harvest Management": "轻柔处理收获的甜瓜以避免碰伤。将其存放在阴凉处。",
            "Storage Conditions": "将甜瓜在室温下存放直至完全成熟。一旦成熟，可短期冷藏以延长新鲜度。",
            "Processing & Packaging": "如有需要，甜瓜可加工成冰沙、冰糕或水果沙拉。将甜瓜装在透气容器中，以帮助在储存和运输期间维持质量。",
            "Challenges & Solutions": "常见挑战包括易受害虫和疾病影响、环境压力如干旱或过度湿润，以及不当的浇水方式。选择抗病品种，实施良好的栽培实践，监测环境条件以缓解这些挑战。"
            },

            {
            "name": "苹果栽培指南",
            "Introduction": "苹果（学名：Malus domestica）是全球最受欢迎的水果之一，因其口味、多用途性和营养价值而备受赞赏。它们在温带气候中生长最佳，可以在各种土壤类型中种植。本指南概述了从种植到收获的苹果栽培完整过程。",
            "Materials Required": "- 来自可靠苗圃的优质苹果树幼苗或嫁接品种\n- 含氮、磷、钾的平衡肥料；有机堆肥\n- 滴灌系统或水管以有效管理水分\n- 杀虫剂、杀菌剂和有机害虫管理解决方案\n- 手工工具（铲子、修枝剪、锄头）或拖拉机用于种植、维护和收获",
            "Soil Preparation": "苹果喜欢排水良好的壤土，pH值为6.0至7.0。通过耕作并添加有机物质来准备土壤，以提高肥力和排水性。",
            "Plant Selection & Treatment": "选择适合您气候的抗病苹果品种，考虑诸如果实风味和收获时间等因素。在种植前检查幼苗是否有疾病或损伤迹象。",
            "Field Preparation": "清除种植区域的杂草、石头和碎屑，确保种植环境干净。",
            "Planting Time": "种植苹果树的最佳时间是在秋季或早春，当树木处于休眠状态时。",
            "Spacing & Depth": "矮生品种间距为4-6英尺，标准品种间距为10-15英尺，以便适当生长和空气流通。将树木种植在与其苗圃高度相匹配的深度，确保嫁接点高于土壤表面。",
            "Seeding/Transplanting Methods": "移栽：挖一个足够容纳根系的洞，将树放入洞中，轻轻回填，种植后彻底浇水。",
            "Watering Requirements": "定期给年轻的苹果树浇水以建立根系，特别是在干旱期间。一旦确立，它们能够忍受干旱，但在果实发育期间受益于深层浇水。",
            "Nutrient Management": "在早春和季中再次施用平衡肥料。使用有机堆肥提高土壤健康。",
            "Weed Control": "通过覆盖物控制杂草，这有助于保持水分并抑制杂草生长，以及手工除草以减少竞争。",
            "Pest & Disease Management": "监测蛀果蛾、蚜虫和蜘蛛螨等害虫。通过适当的卫生措施和抗性品种来管理苹果黑星病和白粉病等疾病。实施综合害虫管理（IPM）策略，包括文化控制和使用有益昆虫。",
            "Special Care During Growth": "- 幼树阶段：保护幼树免受极端天气和害虫影响；考虑使用树木保护罩以防止动物损伤。\n- 营养生长阶段：定期检查营养缺乏症并及时解决。修剪以塑造树形并鼓励强健的结构。\n- 开花和果实发育阶段：在开花和结果期间确保水分稳定，以最大化产量和果实质量。必要时疏果以促进更大的苹果。",
            "Harvesting": "苹果通常在开花后4-6个月准备收获，具体取决于品种。指标包括颜色变化、坚实的质地和易从树上脱落。使用锋利的修枝剪从树上剪下苹果，在果实上留下一小段茎。",
            "Post-Harvest Management": "轻柔处理收获的苹果以避免碰伤。将其存放在阴凉处。",
            "Storage Conditions": "将苹果存放在阴凉黑暗的地方。可以冷藏以延长保质期。",
            "Processing & Packaging": "如有需要，苹果可加工成苹果酱、苹果汁或干片。将苹果装在透气容器中，以帮助在储存和运输期间维持质量。",
            "Challenges & Solutions": "常见挑战包括易受害虫和疾病影响、环境压力（如干旱或霜冻）和不当的修剪技术。选择抗病品种，实施良好的栽培实践，监测环境条件以缓解这些挑战。"
            },

            {
            "name": "橙子栽培指南",
            "Introduction": "橙子（学名：Citrus sinensis）是最受欢迎的柑橘类水果之一，因其甜美多汁的果肉和高维生素C含量而备受重视。它们在温暖的亚热带至热带气候中茁壮成长。本指南概述了从种植到收获的橙子栽培完整过程。",
            "Materials Required": "- 来自可靠苗圃的优质橙树幼苗或嫁接品种\n- 含氮、磷、钾的柑橘专用肥料；有机堆肥\n- 滴灌系统或水管以高效管理水分\n- 杀虫剂、杀菌剂和有机害虫管理解决方案\n- 手工工具（铲子、修枝剪、锄头）或拖拉机用于种植、维护和收获",
            "Soil Preparation": "橙子喜欢排水良好的沙质壤土或粘质壤土，pH值为6.0至7.5。通过耕作并添加有机物质来准备土壤，以提高肥力和排水性。",
            "Plant Selection & Treatment": "选择适合您气候的抗病橙子品种，考虑诸如果实风味和收获时间等因素。在种植前检查幼苗是否有疾病或损伤迹象。",
            "Field Preparation": "清除种植区域的杂草、石头和碎屑，确保种植环境干净。",
            "Planting Time": "种植橙树的最佳时间是在春季，当霜冻危险已过去之后。",
            "Spacing & Depth": "根据根茎和树种不同，树木间距为12-25英尺，以便适当生长和空气流通。将树木种植在与其苗圃高度相匹配的深度，确保嫁接点高于土壤表面。",
            "Seeding/Transplanting Methods": "移栽：挖一个足够容纳根系的洞，将树放入洞中，轻轻回填，种植后彻底浇水。",
            "Watering Requirements": "定期给年轻的橙树浇水以建立根系，特别是在干旱期间。成熟的树木在干旱期间需要深层浇水。",
            "Nutrient Management": "在早春和季中再次施用柑橘专用肥料。使用有机堆肥提高土壤健康。",
            "Weed Control": "通过覆盖物控制杂草，这有助于保持水分并抑制杂草生长，以及手工除草以减少竞争。",
            "Pest & Disease Management": "监测蚜虫、蜘蛛螨和柑橘叶甲等害虫。通过适当的卫生措施和抗性品种来管理柑橘溃疡病和根腐病等疾病。实施综合害虫管理（IPM）策略，包括文化控制和使用有益昆虫。",
            "Special Care During Growth": "- 幼树阶段：保护幼树免受极端天气和害虫影响；考虑使用树木保护罩以防止动物损伤。\n- 营养生长阶段：定期检查营养缺乏症并及时解决。修剪以塑造树形并鼓励强健的结构。\n- 开花和果实发育阶段：在开花和结果期间确保水分稳定，以最大化产量和果实质量。必要时疏果以促进更大的橙子。",
            "Harvesting": "橙子通常在开花后7-12个月准备收获，具体取决于品种。指标包括颜色变化、坚实度和甜度。使用锋利的修枝剪从树上剪下橙子，在果实上留下一小段茎。",
            "Post-Harvest Management": "轻柔处理收获的橙子以避免碰伤。将其存放在阴凉处。",
            "Storage Conditions": "将橙子存放在阴凉黑暗的地方。可以冷藏以延长保质期。",
            "Processing & Packaging": "如有需要，橙子可加工成果汁、果酱或干片。将橙子装在透气容器中，以帮助在储存和运输期间维持质量。",
            "Challenges & Solutions": "常见挑战包括易受害虫和疾病影响、环境压力（如干旱或霜冻）和不当的修剪技术。选择抗病品种，实施良好的栽培实践，监测环境条件以缓解这些挑战。"
            },

            {
            "name": "木瓜栽培指南",
            "Introduction": "木瓜（学名：Carica papaya）是以其甜美多汁的果肉和鲜艳的橙色而闻名的热带水果树。它们在温暖的气候中茁壮成长，在最佳条件下可全年结果。本指南概述了从种植到收获的木瓜栽培完整过程。",
            "Materials Required": "- 来自可靠苗圃的优质木瓜种子或幼苗\n- 含氮、磷、钾的平衡肥料；有机堆肥\n- 滴灌系统或水管以有效管理水分\n- 杀虫剂、杀菌剂和有机害虫管理解决方案\n- 手工工具（铲子、修枝剪、锄头）或拖拉机用于种植、维护和收获",
            "Soil Preparation": "木瓜喜欢排水良好的沙质壤土或壤土，pH值为6.0至6.5。通过耕作并添加有机物质来准备土壤，以增强排水性和肥力。",
            "Plant Selection & Treatment": "选择适合您气候的抗病木瓜品种。如果使用种子，在种植前将其浸泡几个小时以提高发芽率。",
            "Field Preparation": "清除种植区域的杂草、石头和碎屑，确保种植环境干净。",
            "Planting Time": "种植木瓜的最佳时间是在春季，当温度持续温暖时。",
            "Spacing & Depth": "木瓜植株间距为6-10英尺，以适应其大型树冠和根系。将种子或幼苗种植在约0.5至1英寸深的土中。",
            "Seeding/Transplanting Methods": "直接播种：在最后一次霜冻后直接将种子种入地中。\n移栽：室内培育幼苗，当它们约12英寸高时进行移栽。",
            "Watering Requirements": "定期给年轻的木瓜植株浇水，特别是在干旱期间。木瓜需要持续的水分但不能忍受积水。",
            "Nutrient Management": "在生长季节每4-6周施用一次平衡肥料。使用有机堆肥提高土壤肥力。",
            "Weed Control": "通过覆盖物控制杂草，这有助于保持水分并抑制杂草生长，以及手工除草以减少竞争。",
            "Pest & Disease Management": "监测蚜虫、粉虱和果蝇等害虫。通过适当的卫生措施和抗性品种来管理白粉病和根腐病等疾病。实施综合害虫管理（IPM）策略，包括文化控制和使用有益昆虫。",
            "Special Care During Growth": "- 幼苗阶段：保护幼苗免受极端天气和害虫影响。必要时使用行覆盖物以防霜冻和昆虫。\n- 营养生长阶段：定期检查营养缺乏症并及时解决。修剪任何死亡或受损的叶子以促进健康生长。\n- 果实发育阶段：在果实发育期间确保充足的水分供应。必要时疏除多余的果实以允许更大的果实尺寸。",
            "Harvesting": "木瓜通常在种植后6-12个月准备收获，具体取决于品种。指标包括皮肤颜色从绿色变为黄色以及散发出甜香。使用锋利的刀从树上切下果实，留下一小段茎。",
            "Post-Harvest Management": "轻柔处理收获的木瓜以避免碰伤。将其存放在阴凉处。",
            "Storage Conditions": "将木瓜在室温下存放以进一步成熟。一旦成熟，可短期冷藏以延长新鲜度。",
            "Processing & Packaging": "如有需要，木瓜可加工成冰沙、沙拉或干果。将木瓜装在透气容器中，以维持在储存和运输期间的质量。",
            "Challenges & Solutions": "常见挑战包括易受害虫和疾病影响、环境压力（如干旱或洪水）和不当的浇水方式。选择抗病品种，实施良好的栽培实践，监测环境条件以缓解这些挑战。"
            },

            {
            "name": "咖啡栽培指南",
            "Introduction": "咖啡（学名：Coffea spp.）是全球消费最广泛的饮料之一，以其提神特性和丰富风味而闻名。它在热带气候中茁壮成长，通常在较高海拔地区，那里的条件非常适合其生长。本指南概述了从种植到收获的咖啡栽培完整过程。",
            "Materials Required": "- 来自可靠苗圃的优质咖啡幼苗或种子\n- 富含氮、磷、钾的平衡肥料；有机堆肥\n- 滴灌系统或水管以有效管理水分\n- 杀虫剂、杀菌剂和有机害虫管理解决方案\n- 手工工具（铲子、修枝剪、锄头）或拖拉机用于种植、维护和收获",
            "Soil Preparation": "咖啡喜欢排水良好的壤土，pH值为6.0至6.5。通过耕作并添加有机物质来准备土壤，以提高肥力和排水性。",
            "Plant Selection & Treatment": "选择适合您气候的抗病咖啡品种。如果使用种子，浸泡24小时以提高发芽率。",
            "Field Preparation": "清除种植区域的杂草、石头和碎屑，确保种植环境干净。",
            "Planting Time": "种植咖啡的最佳时间是在雨季开始时。",
            "Spacing & Depth": "咖啡植株间距为5-8英尺，以便适当生长和空气流通。将幼苗种植在与其苗圃高度相匹配的深度，确保根领与土壤表面齐平。",
            "Seeding/Transplanting Methods": "移栽：挖一个足够容纳根系的洞，将幼苗放入洞中，轻轻回填，种植后彻底浇水。",
            "Watering Requirements": "定期给年轻的咖啡植株浇水以建立根系，特别是在干旱期间。成熟的植株喜欢持续的水分但不应积水。",
            "Nutrient Management": "在生长季节每3-4个月施用一次平衡肥料。使用有机堆肥提高土壤肥力。",
            "Weed Control": "通过覆盖物控制杂草，这有助于保持水分并抑制杂草生长，以及手工除草以减少竞争。",
            "Pest & Disease Management": "监测咖啡小蠹甲虫和叶锈病等害虫。通过适当的卫生措施和抗性品种来管理根腐病和叶斑病等疾病。实施综合害虫管理（IPM）策略，包括文化控制和使用有益昆虫。",
            "Special Care During Growth": "- 幼苗阶段：保护幼苗免受极端天气和害虫影响。必要时使用遮阳布以防强烈阳光。\n- 营养生长阶段：定期检查营养缺乏症并及时解决。修剪以塑造植株并移除任何死亡或患病的枝条。\n- 开花和果实发育阶段：在开花和结果期间确保充足的水分供应，以最大化产量和果实质量。监测果蝇侵扰并根据需要控制。",
            "Harvesting": "咖啡樱桃通常在开花后7-9个月准备收获，具体取决于品种。指标包括颜色从绿色变为鲜红色或黄色。手工收获咖啡樱桃，只采摘成熟的果实。使用选择性采摘方法以确保质量。",
            "Post-Harvest Management": "轻柔处理收获的樱桃以避免碰伤。尽快处理它们以防止腐坏。",
            "Processing Methods": "使用干法（阳光下晒干樱桃）或湿法（发酵和洗涤樱桃）来提取咖啡豆。",
            "Storage Conditions": "将处理过的咖啡豆存放在阴凉干燥的地方，以防止腐坏并保持风味。",
            "Processing & Packaging": "将咖啡豆装在密封容器中，以帮助在储存和运输期间保持新鲜度。",
            "Challenges & Solutions": "常见挑战包括易受害虫和疾病影响、环境压力（如干旱或霜冻）和市场价格波动。选择抗病品种，实施良好的栽培实践，监测环境条件以缓解这些挑战。"
            }
        ]
        
    # # Dropdown to select crop
    # selected_crop = st.selectbox("Select a crop to view details:", [crop["name"] for crop in cropGuide])

    # # Display selected crop details
    # crop_details = next((crop for crop in cropGuide if crop["name"] == selected_crop), None)

    # if crop_details:
    #     st.subheader(f"{selected_crop} Cultivation Details")
    #     for index, (key, value) in enumerate(crop_details.items()):
    #         if key != "name":
    #                 st.markdown(f"**{key}:** {value}")

    language = st.selectbox("भाषा चुनें | Select Language:", ["English", "हिन्दी", "Español","中文"])
        
        # Select crop guide based on language choice
    selected_guide = cropGuide if language == "English" else cropGuideHindi if language == "हिन्दी" else cropGuideSpanish if language == "Español" else cropGuideChinese

        # Dropdown to select crop
    selected_crop = st.selectbox("Select a crop to view details:", [crop["name"] for crop in selected_guide])

        # Display selected crop details
    crop_details = next((crop for crop in selected_guide if crop["name"] == selected_crop), None)

    if crop_details:
            st.subheader(f"{selected_crop} Cultivation Details")
            for key, value in crop_details.items():
                if key != "name":
                    st.markdown(f"**{key}:** {value}")


