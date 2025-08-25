import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pet Classifier", page_icon="🐾", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
        font-family: 'Poppins', sans-serif;
    }
    

    .prediction-box {
        background: #00796b;
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)



st.title("🐾 AI Model Image Prediction App")
st.write("Upload an image of an animal, and the AI model will classify it.")

upload_file = st.file_uploader("", type=["jpg", "png", "jpeg"])



class_names=['bear','chinkara','elephant','lion','peacock','pig','sheep','tiger']


from tensorflow.keras.models import load_model

def load_my_model():
    return load_model("animal_cnn.h5")

model=load_my_model()


from tensorflow.keras.preprocessing.image import img_to_array

if upload_file is not None:
    image= Image.open(upload_file).convert("RGB")
    st.image(image,caption="Uploaded_image",use_container_width=True)
    image=image.resize((128,128))
    img_array=img_to_array(image)
    img_array=np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    st.markdown(f'<div class="prediction-box">Prediction: {class_names[predicted_class]}</div>', unsafe_allow_html=True)
