from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_cnn_model(path='model/result/cifar10_cnn.keras'):
    model = load_model(path)
    return model

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = predictions.argmax(axis=1)[0]
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence, predictions[0]
