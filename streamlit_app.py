import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
import zipfile

st.title('🎈 Hello World')

# Function to download, preprocess, and display an image from a URL
def preprocess_and_display_image(image_url):
    """Downloads, preprocesses, and displays an image from a URL in Streamlit."""
    try:
        # Download the image.
        image_path = tf.keras.utils.get_file(origin=image_url)
        image = tf.keras.preprocessing.image.load_img(image_path)

        # Preprocess the image.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, [224, 224])

        # Display the image in Streamlit.
        st.image(image.numpy(), caption='Processed Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Define the class names for predictions
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# Google Drive file ID and download path
file_id = '14M4lfoNrQb3j3z0ggBYPP7q8KjvCghwu'  # File ID of the zip file
zip_path = 'model.zip'
extracted_model_path = 'downloaded_model'  # Folder name to store the extracted model

# Download and extract the model zip file
def download_and_extract_model(file_id, zip_path, extracted_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the zip file
    gdown.download(url, zip_path, quiet=False)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

# Ensure the extracted model directory exists
os.makedirs(extracted_model_path, exist_ok=True)

# Download and extract if not already done
if not os.listdir(extracted_model_path):  # Check if the directory is empty
    download_and_extract_model(file_id, zip_path, extracted_model_path)

# List contents of the extracted model directory for debugging
st.write("Contents of the extracted model directory:")
for root, dirs, files in os.walk(extracted_model_path):
    for file in files:
        st.write(os.path.join(root, file))

# Load the SavedModel using tf.saved_model.load
try:
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.saved_model.load(extracted_model_path, options=load_options)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Streamlit app for predictions
st.title("Prediction App")
user_input = st.text_input("Enter input for prediction:")

if st.button("Predict"):
    try:
        # Ensure user input is formatted correctly for your model
        prediction = model.predict([user_input])  # Adjust this according to your model's input requirements
        st.write(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Additional Input for URL to process images
image_url = st.text_input("Enter image URL to process:")
if image_url:
    preprocess_and_display_image(image_url)

st.write('Hello World!')
