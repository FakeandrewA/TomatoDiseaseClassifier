import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import gdown
import os

st.title('🎈 Hello world')
 
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_and_display_image(image_url):
  """Downloads, preprocesses, and displays an image from a URL in Streamlit.

  Args:
    image_url: The URL of the image.
  """
  # Download the image.
  image_path = tf.keras.utils.get_file(origin=image_url)
  image = tf.keras.preprocessing.image.load_img(image_path)

  # Preprocess the image.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, [224, 224])

  # Display the image in Streamlit.
  st.image(image.numpy(), caption='Processed Image', use_column_width=True)

class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Step 1: Define your Google Drive file ID and download path
file_id = '1LnvMfTLyMJWkDG2QS3P8ejMmMkJM4_8c'  # Extracted from your link
model_path = 'downloaded_model'  # Choose a folder name to store the model

# Step 2: Download the model from Google Drive
def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download_folder(url, quiet=False, output=model_path, use_cookies=False)

# Ensure the model directory exists
os.makedirs(model_path, exist_ok=True)

# Download if not already downloaded
if not os.listdir(model_path):  # Check if the directory is empty
    download_model(file_id, model_path)

# Load the SavedModel using tf.saved_model.load
model = tf.saved_model.load(model_path)  

# Streamlit app for predictions
st.title("Prediction App")
user_input = st.text_input("Enter input for prediction:")

if st.button("Predict"):
    prediction = model.predict([user_input])
    st.write(f"Prediction: {prediction}")

st.write('Hello world!')

# Display a chat input widget at the bottom of the app.
u=st.text_input("Say something");
if(u):
 preprocess_and_display_image(u)



