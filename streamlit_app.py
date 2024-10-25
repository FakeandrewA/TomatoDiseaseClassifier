import streamlit as st

st.title('🎈 App')

def preprocess_img(url):
  # Download an image and preprocess it
  image_path = tf.keras.utils.get_file(origin=url)
  image = tf.keras.preprocessing.image.load_img(image_path)
  image = tf.cast(image, dtype=tf.float32) / 255.0
  image = tf.image.resize(image, [224, 224]).numpy()
  return image


st.write('Hello world!')
