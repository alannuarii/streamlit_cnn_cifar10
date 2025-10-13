import matplotlib.pyplot as plt
import streamlit as st

#Grafik Akurasi dan Loss Selama Training
def plot_training_metrics(df_history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(df_history['accuracy'], label='Training Accuracy')
    ax[0].plot(df_history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy per Epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(df_history['loss'], label='Training Loss')
    ax[1].plot(df_history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss per Epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

#Distribusi Jumlah Data per Kelas (Training Set)
def show_class_distribution(distribution_df):
    st.bar_chart(distribution_df.set_index('Kelas'))
