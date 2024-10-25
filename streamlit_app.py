import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
import zipfile
import requests
from PIL import Image
import io

st.title('🎈 Tomato Leaf Disease Prediction App')

# Function to download, preprocess, and display an image from a URL
def preprocess_img(url):
    try:
        # Fetch the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        
        # Open the image and convert to RGB
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize((244, 244))  # Resize to match model input

        # Convert the image to a numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

file_id = '14M4lfoNrQb3j3z0ggBYPP7q8KjvCghwu'
zip_path = 'model.zip'
extracted_model_path = 'downloaded_model/model/Resnet_model'  # Adjusted path

def download_and_extract_model(file_id, zip_path, extracted_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

os.makedirs(extracted_model_path, exist_ok=True)

if not os.listdir(extracted_model_path):
    download_and_extract_model(file_id, zip_path, extracted_model_path)

# Load the SavedModel
try:
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.saved_model.load(extracted_model_path, options=load_options)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

def predict(model, url):
    img = preprocess_img(url)  # No label needed for prediction
    if img is None:
        return None, None  # Return if image processing fails

    img = tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to tensor
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    try:
        pred_probs = model(img)  # Use the model to predict
        predicted_class_index = tf.argmax(pred_probs[0]).numpy()
        
        if 0 <= predicted_class_index < len(class_names):
            predicted_class = class_names[predicted_class_index]
        else:
            predicted_class = "Unknown"
    except ValueError as e:
        st.error(f"Model prediction error: {e}")
        predicted_class = None

    return predicted_class, img[0]  # Return the class name and the image tensor

# Streamlit app for user input
image_url = st.text_input("Enter image URL to process:")
if st.button("Predict"):
    if image_url:
        predicted_class, processed_image = predict(model, image_url)

        # Display the processed image
        if processed_image is not None:
            # Convert tensor to numpy array and ensure it's in the correct range
            processed_image = (processed_image.numpy() * 255).astype(np.uint8)  # Convert back to uint8
            st.image(processed_image, caption='Processed Image', use_column_width=True)

        # Display the prediction result
        if predicted_class is not None:
            st.write(f"Predicted class: {predicted_class}")
    else:
        st.error("Please enter a valid image URL.")

st.write('Hello World!')
