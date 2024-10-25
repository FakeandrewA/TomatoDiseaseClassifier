import streamlit as st
import numpy as np
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
# Insert a chat message container.
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# Display a chat input widget at the bottom of the app.
>>> st.chat_input("Say something")

# Display a chat input widget inline.
with st.container():
    st.chat_input("Say something")
st.write(class_names[0])
