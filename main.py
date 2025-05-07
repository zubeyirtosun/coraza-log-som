import streamlit as st
import json
import pandas as pd
import numpy as np
from data_processing import preprocess_data, handle_missing_data, train_som
from visualizations import (show_summary_table, show_visualizations, handle_meta_clustering, 
                           handle_neuron_details, handle_anomaly_detection)
from session_state import initialize_session_state, reset_session_state
from text_content import get_main_description, get_som_description, get_user_gains

initialize_session_state()

st.set_page_config(page_title="Log Analizi ve SOM", layout="wide")

st.title("Log Analizi ve SOM Kümeleme")
st.write(get_main_description())

with st.expander("Self-Organizing Map (SOM) Nedir?"):
    st.markdown(get_som_description())

st.markdown(get_user_gains())

st.info("Başlamak için bir JSON log dosyası yükleyin. Ardından eksik veri işleme seçeneklerini belirtebilirsiniz.")

uploaded_file = st.file_uploader("Log dosyanızı yükleyin (JSON formatında)", type=["json"])
if uploaded_file and st.session_state.df is None:
    try:
        raw_data = json.load(uploaded_file)
        
        # JSON yapısını işle
        transactions = []
        if isinstance(raw_data, dict):
            if 'transaction' in raw_data:
                transactions.append(raw_data['transaction'])
            else:
                transactions.append(raw_data)
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    if 'transaction' in item:
                        transactions.append(item['transaction'])
                    else:
                        transactions.append(item)
        
        # Nokta ayracı ile normalize et
        try:
            st.session_state.df = pd.json_normalize(transactions, sep='.')
        except:
            # Basit düz DataFrame oluştur
            st.session_state.df = pd.DataFrame(transactions)
        
        # Hata ayıklama çıktıları
        st.write("Yüklenen Veri Önizleme:")
        st.write(st.session_state.df.head())
        st.write("Mevcut Sütunlar:", st.session_state.df.columns.tolist())
        
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
        st.stop()

if st.session_state.df is not None:
    handle_missing_data()


if st.session_state.X is not None and st.session_state.som is None:
    n_samples = len(st.session_state.df)
    grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
    st.session_state.grid_size = grid_size  # Grid boyutunu session_state'e kaydet
    sigma = grid_size / 2.0
    learning_rate = 0.5
    num_iterations = 1000
    
    with st.spinner("SOM eğitimi yapılıyor..."):
        st.session_state.som = train_som(st.session_state.X, grid_size, sigma, learning_rate, num_iterations)
    
    # BMU koordinatlarını ekle
    bmu_coords = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
    st.session_state.df['bmu_x'] = bmu_coords[:, 0]
    st.session_state.df['bmu_y'] = bmu_coords[:, 1]
    st.session_state.df['quantization_error'] = [st.session_state.som.quantization_error(x.reshape(1, -1)) for x in st.session_state.X]

if st.session_state.som is not None:
    st.subheader("SOM Parametre Ayarları")
    grid_size = st.session_state.grid_size  # Grid boyutunu session_state'den al
    sigma = st.slider("Sigma", 0.1, 10.0, value=float(grid_size/2), step=0.1)
    learning_rate = st.slider("Öğrenme Oranı", 0.1, 1.0, 0.5, 0.1)
    num_iterations = st.slider("İterasyon Sayısı", 100, 5000, 1000, 100)
    
    if st.button("SOM'u Yeniden Eğit"):
        with st.spinner("SOM yeniden eğitiliyor..."):
            st.session_state.som = train_som(st.session_state.X, grid_size, sigma, learning_rate, num_iterations)
        bmu_coords = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
        st.session_state.df['bmu_x'] = bmu_coords[:, 0]
        st.session_state.df['bmu_y'] = bmu_coords[:, 1]
        st.session_state.df['quantization_error'] = [st.session_state.som.quantization_error(x.reshape(1, -1)) for x in st.session_state.X]
        st.session_state.summary_df = None


if st.session_state.som is not None:
    show_summary_table()
    show_visualizations()
    handle_neuron_details()  # Yeni eklenen nöron detayları bölümü
    handle_anomaly_detection()  # Yeni eklenen anomali tespiti bölümü
    handle_meta_clustering()

if st.button("Analizi Sıfırla"):
    reset_session_state()
    st.rerun()

# Sonuç ve Yorum
st.markdown("""
### Sonuç ve Değerlendirme
Bu analiz, log verilerindeki desenleri ve potansiyel anomalileri tespit etmenize yardımcı olur.
Yüksek quantization error değerlerine sahip loglar, anormal davranışları işaret edebilir ve daha detaylı inceleme gerektirebilir.
Meta kümeleme, benzer davranış gösteren log gruplarını belirlemenize yardımcı olur.
""")
