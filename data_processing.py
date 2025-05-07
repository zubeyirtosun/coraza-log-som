import streamlit as st
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

def handle_missing_data():
    st.subheader("Eksik Veri İşleme")
    df = st.session_state.df
    
    if df is None:
        st.error("Önce veri yükleyin!")
        return
    
    missing_values = df.isnull().sum()
    
    if missing_values.any():
        missing_method = st.selectbox(
            "Eksik Veri İşleme Yöntemi",
            options=["Satırları Çıkar", "Ortalama ile Doldur", "Medyan ile Doldur", "Bir Şey Yapma"]
        )
    else:
        missing_method = "Bir Şey Yapma"

    if st.button("Veriyi İşle"):
        processed_df, result = preprocess_data(df.copy(), missing_method)
        
        if processed_df is None:
            st.error(result)
            st.stop()
        
        st.session_state.df = processed_df
        st.session_state.X = result
        st.success("Veri başarıyla işlendi!")

@st.cache_data
def preprocess_data(df, missing_method, uri_threshold=10):
    try:
        # Zorunlu sütun kontrolü (nokta ayraçlı veya düz format)
        required_columns_dot = [
            'transaction.client_port',
            'transaction.request.uri',
            'transaction.timestamp',
            'transaction.is_interrupted',
            'transaction.request.method'
        ]
        
        required_columns_flat = [
            'client_port',
            'request.uri', 'request_uri',
            'timestamp',
            'is_interrupted',
            'request.method', 'request_method'
        ]
        
        # Nokta ayraçlı sütunları kontrol et
        missing_dot_columns = [col for col in required_columns_dot if col not in df.columns]
        
        # Eğer nokta ayraçlı sütunlar eksikse, düz sütunları kontrol et ve dönüştür
        if missing_dot_columns:
            # Sütun isimlerini standartlaştır
            column_mapping = {
                'client_port': 'transaction.client_port',
                'request.uri': 'transaction.request.uri',
                'request_uri': 'transaction.request.uri',
                'timestamp': 'transaction.timestamp',
                'is_interrupted': 'transaction.is_interrupted',
                'request.method': 'transaction.request.method',
                'request_method': 'transaction.request.method'
            }
            
            # Mevcut sütunları yeniden adlandır
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            # Yeniden kontrol et
            missing_columns = [col for col in required_columns_dot if col not in df.columns]
            if missing_columns:
                return None, f"Eksik sütunlar: {missing_columns}"
        
        df = handle_missing_values(df, missing_method)
        df = create_features(df, uri_threshold)
        X = prepare_final_features(df)
        
        return df, X
        
    except Exception as e:
        return None, f"İşlem hatası: {str(e)}"

def handle_missing_values(df, method):
    if method == "Satırları Çıkar":
        return df.dropna()
    elif method == "Ortalama ile Doldur":
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif method == "Medyan ile Doldur":
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    return df

def create_features(df, uri_threshold):
    # Zaman damgası işleme
    df['transaction.timestamp'] = pd.to_datetime(df['transaction.timestamp'])
    df['hour'] = df['transaction.timestamp'].dt.hour
    
    # URI kategorizasyonu
    df['uri_cat'] = df['transaction.request.uri'].apply(
        lambda x: x.split('/')[1] if len(str(x).split('/')) > 1 else 'root'
    )
    
    uri_counts = df['uri_cat'].value_counts()
    rare_uris = uri_counts[uri_counts < uri_threshold].index
    df['uri_cat'] = df['uri_cat'].apply(lambda x: 'other' if x in rare_uris else x)
    
    return df

def prepare_final_features(df):
    # One-hot encoding
    df = pd.get_dummies(df, columns=['uri_cat', 'transaction.request.method'], drop_first=True)
    
    # Özellik seçimi
    numeric_features = [col for col in df.columns if col.startswith('uri_cat_') 
                      or col.startswith('transaction.request.method_') 
                      or col in ['transaction.client_port', 'hour']]
    
    scaler = StandardScaler()
    return scaler.fit_transform(df[numeric_features].values)

def train_som(X, grid_size, sigma, learning_rate, iterations):
    som = MiniSom(grid_size, grid_size, X.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(X)
    som.train_random(X, iterations)
    return som