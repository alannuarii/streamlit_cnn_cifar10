import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Fungsi untuk load metrik dari file JSON
def load_training_metrics(json_path='model/metric/training_metrics.json'):
    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        st.warning(f"File '{json_path}' tidak ditemukan. Pastikan metrik training sudah disimpan.")
    except Exception as e:
        st.error(f"Error memuat metrik training: {e}")
    return None

# Fungsi untuk load history lengkap dari CSV
def load_training_history(csv_path='model/metric/training_history.csv'):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.warning(f"File '{csv_path}' tidak ditemukan. Tidak dapat menampilkan grafik history.")
    except Exception as e:
        st.error(f"Error memuat file history: {e}")
    return None

# Fungsi tampilkan contoh gambar dari dataset
def show_cifar10_samples(x, y, class_names, n_samples=10):
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(x[i])
        ax.axis('off')
        ax.set_title(class_names[y[i][0]].title())
    st.pyplot(fig)