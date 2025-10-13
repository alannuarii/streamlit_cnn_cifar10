import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
from modules import data, model as mdl, visualization, utils

# Label kelas CIFAR-10
class_names = data.get_class_names()

# Load model
model = mdl.load_cnn_model()

# Inisialisasi menu aktif di session_state
if 'menu' not in st.session_state:
    st.session_state.menu = 'Deteksi Objek'

def set_menu(menu_name):
    st.session_state.menu = menu_name

# Sidebar dengan tombol-tombol menu
st.sidebar.image("assets/img/Logo-BO-01-1.png")
st.sidebar.title("Menu Navigasi")
if st.sidebar.button('Deteksi Objek', width="stretch", ):
    set_menu('Deteksi Objek')
if st.sidebar.button('Deteksi Objek Realtime', width="stretch"):
    set_menu('Deteksi Objek Realtime')
if st.sidebar.button('Karakteristik Model', width="stretch"):
    set_menu('Karakteristik Model')
if st.sidebar.button('Karakteristik Dataset', width="stretch"):
    set_menu('Karakteristik Dataset')

menu = st.session_state.menu

if menu == "Deteksi Objek":
    st.header("Deteksi Objek dengan CNN | CIFAR-10")
    
    # Input: Upload gambar
    uploaded_file = st.file_uploader("Unggah sample gambar untuk prediksi kelas", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi

        predicted_class, confidence, predictions = mdl.predict_image(model, img_array)

        st.subheader("Gambar yang Diupload:")
        st.image(img, caption="Gambar Masukan", use_container_width=True)

        st.subheader("Hasil Prediksi:")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Kelas: **{class_names[predicted_class].title()}**")
        with col2:
            st.info(f"Confidence: **{confidence:.2%}**")

        prob_df = pd.DataFrame({
            'Kelas': [k.title() for k in class_names],
            'Probabilitas': predictions
        })
        prob_df = prob_df.set_index('Kelas')

        st.markdown("### Probabilitas Kelas")
        st.bar_chart(prob_df)

elif menu == "Deteksi Objek Realtime":
    st.header("Deteksi Objek Realtime dengan Kamera")

    camera_image = st.camera_input("Ambil gambar atau objek untuk prediksi")

    if camera_image is not None:
        img = image.load_img(camera_image)  # Memuat image dari input kamera
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi

        predicted_class, confidence, predictions = mdl.predict_image(model, img_array)

        st.subheader("Hasil Prediksi dari Kamera:")
        st.image(img, caption="Gambar Kamera", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Kelas: **{class_names[predicted_class].title()}**")
        with col2:
            st.info(f"Confidence: **{confidence:.2%}**")
        
        prob_df = pd.DataFrame({
            'Kelas': [k.title() for k in class_names],
            'Probabilitas': predictions
        })
        prob_df = prob_df.set_index('Kelas')

        st.markdown("### Probabilitas Kelas")
        st.bar_chart(prob_df)

elif menu == "Karakteristik Model":
    st.header("Karakteristik Model CNN - CIFAR-10")

    training_metrics = utils.load_training_metrics()
    if training_metrics:
        epochs = len(training_metrics['accuracy'])
        st.subheader("Statistik Training Terakhir")
        
        st.write(f"- **Jumlah Epoch:** {epochs}")
        st.write(f"- **Akurasi Training Terakhir:** {training_metrics['accuracy'][-1]:.2%}")
        st.write(f"- **Akurasi Validasi Terakhir:** {training_metrics['val_accuracy'][-1]:.2%}")
        st.write(f"- **Loss Training Terakhir:** {training_metrics['loss'][-1]:.4f}")
        st.write(f"- **Loss Validasi Terakhir:** {training_metrics['val_loss'][-1]:.4f}")

        # Menampilkan seluruh metrik tiap epoch dalam tabel
        st.subheader("Metrik Training per Epoch")
        df_metrics = pd.DataFrame(training_metrics)
        df_metrics.index = df_metrics.index + 1
        st.dataframe(df_metrics)

    # Grafik Akurasi dan Loss
    df_history = utils.load_training_history()
    if df_history is not None:
        st.subheader("Grafik Akurasi dan Loss Selama Training")
        visualization.plot_training_metrics(df_history)

    st.markdown("""
    ---  
    *Data yang ditampilkan diambil langsung dari proses training model di `model.ipynb`.*
    """)
elif menu == "Karakteristik Dataset":
    st.header("Karakteristik Dataset CIFAR-10")

    # Load dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    st.write(f"- **Jumlah Data Training:** {x_train.shape[0]}")
    st.write(f"- **Jumlah Data Testing:** {x_test.shape[0]}")
    st.write(f"- **Ukuran Gambar:** {x_train.shape[1]}x{x_train.shape[2]} pixels")
    st.write(f"- **Jumlah Kelas:** {len(np.unique(y_train))}")

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    st.subheader("Kelas CIFAR-10")
    st.write(", ".join([cls.title() for cls in class_names]))

    st.subheader("Contoh Gambar dari Dataset Training")
    utils.show_cifar10_samples(x_train, y_train, class_names, n_samples=10)

    st.subheader("Distribusi Jumlah Data per Kelas (Training Set)")
    # Hitung jumlah per kelas
    unique, counts = np.unique(y_train, return_counts=True)
    distribution = pd.DataFrame({'Kelas': [class_names[i].title() for i in unique], 'Jumlah': counts})

    visualization.show_class_distribution(distribution)

