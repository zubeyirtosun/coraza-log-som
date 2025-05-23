import streamlit as st
import json
import pandas as pd
import numpy as np
from data_processing import preprocess_data, handle_missing_data, train_som, fix_unhashable_columns, preprocess_data_interactive
from visualizations import (show_summary_table, show_visualizations, handle_meta_clustering, 
                           handle_neuron_details, handle_anomaly_detection, show_som_validation, 
                           show_meta_clustering_validation, show_advanced_analysis)
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

# Veri yükleme seçenekleri
upload_method = st.radio(
    "Veri yükleme yöntemi seçin:",
    ["Dosya Yükle", "Örnek Veri Kullan"]
)

if upload_method == "Dosya Yükle":
    # Dosya formatı açıklaması
    with st.expander("Desteklenen JSON formatı hakkında bilgi"):
        st.markdown("""
        ### Dosya Formatı Gereksinimleri:
        
        Yüklediğiniz JSON dosyası aşağıdaki formatlardan birinde olmalıdır:
        
        **1. Tek transaction:**
        ```json
        {
          "transaction": {
            "client_port": 12345,
            "request": {
              "uri": "/login",
              "method": "POST"
            },
            "timestamp": "2023-01-01T10:20:30Z",
            "is_interrupted": false
          }
        }
        ```
        
        **2. Transaction listesi:**
        ```json
        [
          {
            "transaction": {
              "client_port": 12345,
              "request": {
                "uri": "/login",
                "method": "POST"
              },
              "timestamp": "2023-01-01T10:20:30Z",
              "is_interrupted": false
            }
          },
          {
            "transaction": {
              "client_port": 67890,
              "request": {
                "uri": "/admin",
                "method": "GET"
              },
              "timestamp": "2023-01-01T10:25:30Z",
              "is_interrupted": true
            }
          }
        ]
        ```
        
        **3. Düz format:**
        ```json
        [
          {
            "client_port": 12345,
            "request.uri": "/login",
            "request.method": "POST",
            "timestamp": "2023-01-01T10:20:30Z",
            "is_interrupted": false
          }
        ]
        ```
        
        Eksik alanlar otomatik olarak varsayılan değerlerle doldurulacaktır.
        """)
    
    uploaded_file = st.file_uploader("Log dosyanızı yükleyin (JSON formatında)", type=["json"])
elif upload_method == "Örnek Veri Kullan":
    if st.button("Örnek Veri Yükle"):
        # Örnek veri
        import io
        import json
        
        example_data = [
            {
                "transaction": {
                    "client_port": 12345,
                    "request": {
                        "uri": "/login",
                        "method": "POST"
                    },
                    "timestamp": "2023-01-01T10:20:30Z",
                    "is_interrupted": False
                }
            },
            {
                "transaction": {
                    "client_port": 67890,
                    "request": {
                        "uri": "/admin",
                        "method": "GET"
                    },
                    "timestamp": "2023-01-01T10:25:30Z",
                    "is_interrupted": True
                }
            },
            {
                "transaction": {
                    "client_port": 54321,
                    "request": {
                        "uri": "/dashboard",
                        "method": "GET"
                    },
                    "timestamp": "2023-01-01T10:30:30Z",
                    "is_interrupted": False
                }
            },
            # Birkaç çeşit örnek ekliyoruz
            {
                "transaction": {
                    "client_port": 11111,
                    "request": {
                        "uri": "/api/users",
                        "method": "GET"
                    },
                    "timestamp": "2023-01-01T11:20:30Z",
                    "is_interrupted": False
                }
            },
            {
                "transaction": {
                    "client_port": 22222,
                    "request": {
                        "uri": "/login",
                        "method": "POST"
                    },
                    "timestamp": "2023-01-01T12:20:30Z",
                    "is_interrupted": True
                }
            }
        ]
        
        for i in range(20):  # Analiz için daha fazla örnek ekle
            import random
            import datetime
            
            uri_choices = ["/login", "/admin", "/dashboard", "/api/users", "/logout", "/profile", "/settings"]
            method_choices = ["GET", "POST", "PUT", "DELETE"]
            
            example_data.append({
                "transaction": {
                    "client_port": random.randint(10000, 60000),
                    "request": {
                        "uri": random.choice(uri_choices),
                        "method": random.choice(method_choices)
                    },
                    "timestamp": (datetime.datetime(2023, 1, 1, 10, 0, 0) + 
                                  datetime.timedelta(minutes=random.randint(0, 1440))).isoformat(),
                    "is_interrupted": random.random() < 0.3  # %30 olasılıkla engellendi
                }
            })
        
        # Örnek veriyi JSON'a dönüştür
        example_json = json.dumps(example_data)
        uploaded_file = io.BytesIO(example_json.encode())
        uploaded_file.name = "example_data.json"
        
        st.success("Örnek veri yüklendi! İşlemeye devam edebilirsiniz.")
    else:
        uploaded_file = None

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
            df = pd.json_normalize(transactions, sep='.')
            # Hashlenemeyen değerleri düzelt
            df = fix_unhashable_columns(df)
            st.session_state.df = df
        except Exception as e:
            # Hata çıktısı
            st.warning(f"JSON normalize hatası: {str(e)}. Basit formata dönüştürülüyor.")
            # Basit düz DataFrame oluştur
            df = pd.DataFrame(transactions)
            # Hashlenemeyen değerleri düzelt
            df = fix_unhashable_columns(df)
            st.session_state.df = df
        
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
    
    # BMU koordinatlarını tekrar hashlenebilir yap
    st.session_state.df = fix_unhashable_columns(st.session_state.df)

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
        
        # BMU koordinatlarını tekrar hashlenebilir yap
        st.session_state.df = fix_unhashable_columns(st.session_state.df)
        
        st.session_state.summary_df = None


if st.session_state.som is not None:
    # Main tabs
    main_tabs = st.tabs(["Temel Analizler", "Gelişmiş Analizler"])
    
    with main_tabs[0]:
        show_summary_table()
        show_visualizations()
        handle_neuron_details()  # Yeni eklenen nöron detayları bölümü
        handle_anomaly_detection()  # Yeni eklenen anomali tespiti bölümü
        handle_meta_clustering()
        show_som_validation()
        show_meta_clustering_validation()
    
    with main_tabs[1]:
        show_advanced_analysis()

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
