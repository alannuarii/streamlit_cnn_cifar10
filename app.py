# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Label kelas CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load model
@st.cache_resource  # Cache model agar tidak muat ulang setiap kali
def load_cnn_model():
    model = load_model('result/cifar10_cnn.keras')
    return model

model = load_cnn_model()

# Judul aplikasi
st.title("📷 Deteksi Objek dengan CNN | CIFAR-10")
st.markdown("Unggah gambar atau gunakan contoh dari CIFAR-10 untuk prediksi kelas.")

# Input: Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"])

# Jika gambar diunggah
if uploaded_file is not None:
    # Baca gambar
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Tampilkan gambar dan hasil
    st.subheader("Gambar yang Diupload:")
    st.image(img, caption="Gambar Masukan", use_column_width=True)

    st.subheader("Hasil Prediksi:")
    st.success(f"Kelas: **{class_names[predicted_class].title()}**")
    st.info(f"Confidence: **{confidence:.2%}**")

    # Tampilkan semua prediksi
    st.write("Probabilitas Kelas:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i].title()}: {prob:.2%}")

# Opini: Contoh gambar CIFAR-10 (jika ingin tampilkan contoh)
with st.expander("🎲 Lihat Contoh Gambar dari CIFAR-10"):
    st.write("Berikut adalah beberapa contoh gambar dari CIFAR-10:")
    # Kita bisa tampilkan gambar dari dataset secara acak
    example_img = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
    st.image(example_img, caption="Contoh Gambar (Acak)", use_column_width=True)
    st.markdown(
        "[CIFAR-10 Dataset (via TensorFlow)](https://www.cs.toronto.edu/~kriz/cifar.html)"
    )
