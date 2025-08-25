import streamlit as st
from PIL import Image
import numpy as np


st.title("AI model image prediction app")
st.set_page_config(page_title="Pet Classifier", page_icon="🐾", layout="centered")

class_names=['bear','chinkara','elephant','lion','peacock','pig','sheep','tiger']
upload_file=st.file_uploader("upload image",type=["jpg","png","jpeg"])

if upload_file is not None:
    image= Image.open(upload_file)
    st.image(image,caption="Uploaded_image",use_container_width=True)
    img_array=np.array(image)
    st.write("image ready for model processing")

from tensorflow.keras.models import load_model
def load_my_model():
    return load_model("animal_cnn.h5")

model=load_my_model()

from tensorflow.keras.preprocessing.image import img_to_array
import cv2

if upload_file is not None:
    image= Image.open(upload_file).convert("RGB")
    image=image.resize((128,128))
    img_array=img_to_array(image)
    img_array=np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.success(f"Prediction: **{class_names[predicted_class]}**")
