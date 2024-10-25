import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
st.title('🎈 Hello world')

def preprocess_img(url):
  # Download an image and preprocess it
  image_path = tf.keras.utils.get_file(origin=url)
  image = tf.keras.preprocessing.image.load_img(image_path)
  image = tf.cast(image, dtype=tf.float32) / 255.0
  image = tf.image.resize(image, [224, 224]).numpy()
  return image
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
print(class_names[0])
st.write('Hello world!')

# Display a chat input widget at the bottom of the app.
user_input = st.text_input("Say something")
img = preprocess_img(user_input)
plt.imshow(img)
plt.title("Your Image")
plt.axis("off")


