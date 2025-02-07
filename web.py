import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os 


file_id = "14SmIXmGBDdxVQCbi3H4HqLjDYulsqvGT"
url = 'https://drive.google.com/file/d/14SmIXmGBDdxVQCbi3H4HqLjDYulsqvGT/view?usp=sharing'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloding model from Google Drive...")
    gdown.download(url,model_path, quiet=False)


    
# Function to load model and make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = test_image.resize((128, 128))  # Resize using PIL
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Streamlit Sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

# Display Logo
img = Image.open('plant leaf disease detection logo.png')
#img = Image.open('plant leaf disease detection logo.png')
st.image(img)

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System for Sustainable Agriculture')

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    if st.button('Show Image'):
        st.image(image, use_column_width=True)  # Correct spelling

    if st.button('Predict'):
        st.snow()
        st.write('Our Prediction...')
        result_index = model_prediction(image)
        class_names = ['Potato___Early_blight', 'Potato___Late_blight','Potato___healthy']
        st.success(f'Model is predicting it as: {class_names[result_index]}')
