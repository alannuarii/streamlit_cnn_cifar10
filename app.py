import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
from modules import data, visualization, utils, model

# CIFAR-10 class labels
class_names = data.get_class_names()

# Load the trained CNN model
model_loaded = model.load_cnn_model()

# Initialize active menu in session_state
if 'menu' not in st.session_state:
    st.session_state.menu = 'Image Classification'

def set_menu(menu_name):
    st.session_state.menu = menu_name

# Sidebar with menu buttons
st.sidebar.image("assets/img/Logo-BO-01-1.png")
st.sidebar.title("Navigation Menu")
if st.sidebar.button('Image Classification', width="stretch"):
    set_menu('Image Classification')
if st.sidebar.button('Model Characteristics', width="stretch"):
    set_menu('Model Characteristics')
if st.sidebar.button('Dataset Characteristics', width="stretch"):
    set_menu('Dataset Characteristics')

menu = st.session_state.menu

if menu == "Image Classification":
    st.header("Image Classification with CNN | CIFAR-10")
    uploaded_file = st.file_uploader("Upload image sample for class prediction", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        img_processed = img.resize((32, 32))
        img_array = image.img_to_array(img_processed)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalization
        
        predicted_class, confidence, predictions = model.predict_image(model_loaded, img_array)

        st.image(img, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Class: **{class_names[predicted_class].title()}**")
        with col2:
            st.info(f"Confidence: **{confidence:.2%}**")

        prob_df = pd.DataFrame({
            'Class': [k.title() for k in class_names],
            'Probability': predictions
        }).set_index('Class')

        st.markdown("### Class Probability")
        st.bar_chart(prob_df)
    
elif menu == "Model Characteristics":
    st.header("Characteristics of the CNN Model CIFAR-10")

    training_metrics = utils.load_training_metrics()
    if training_metrics:
        epochs = len(training_metrics['accuracy'])
        st.subheader("Latest Training Statistics")

        st.write(f"- **Number of Epochs:** {epochs}")
        st.write(f"- **Latest Training Accuracy:** {training_metrics['accuracy'][-1]:.2%}")
        st.write(f"- **Latest Validation Accuracy:** {training_metrics['val_accuracy'][-1]:.2%}")
        st.write(f"- **Latest Training Loss:** {training_metrics['loss'][-1]:.4f}")
        st.write(f"- **Latest Validation Loss:** {training_metrics['val_loss'][-1]:.4f}")

        st.subheader("Training Metrics per Epoch")
        df_metrics = pd.DataFrame(training_metrics)
        df_metrics.index = df_metrics.index + 1
        st.dataframe(df_metrics)

    df_history = utils.load_training_history()
    if df_history is not None:
        st.subheader("Accuracy and Loss Charts During Training")
        visualization.plot_training_metrics(df_history)

    st.markdown("""
    ---  
    *Data displayed is taken directly from the model training process in `model.ipynb`.*
    """)

elif menu == "Dataset Characteristics":
    st.header("Characteristics of the CIFAR-10 Dataset")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    st.write(f"- **Number of Training Images:** {x_train.shape[0]}")
    st.write(f"- **Number of Testing Images:** {x_test.shape[0]}")
    st.write(f"- **Image Size:** {x_train.shape[1]}x{x_train.shape[2]} pixels")
    st.write(f"- **Number of Classes:** {len(np.unique(y_train))}")

    class_names = data.get_class_names()

    st.subheader("CIFAR-10 Classes")
    st.write(", ".join([cls.title() for cls in class_names]))

    st.subheader("Sample Images from Training Dataset")
    utils.show_cifar10_samples(x_train, y_train, class_names, n_samples=10)

    st.subheader("Class Distribution (Training Set)")
    unique, counts = np.unique(y_train, return_counts=True)
    distribution = pd.DataFrame({'Class': [class_names[i].title() for i in unique], 'Count': counts})

    visualization.show_class_distribution(distribution)
