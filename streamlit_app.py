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
print(class_names[0])
st.write('Hello world!')

# Display a chat input widget at the bottom of the app.
u=st.text_input("Say something");
if(u):
 preprocess_and_display_image(u)



