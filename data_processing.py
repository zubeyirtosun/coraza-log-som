import streamlit as st
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

def handle_missing_data():
    st.subheader("Eksik Veri İşleme")
    
    st.info("""
    **🎯 Coraza WAF Log İşleme**  
    Bu bölüm Coraza WAF loglarının standart formatını bekler:
    - `transaction.client_port` - İstemci portu
    - `transaction.request.uri` - İstek URI'si  
    - `transaction.timestamp` - Zaman damgası
    - `transaction.is_interrupted` - Engellenme durumu
    - `transaction.request.method` - HTTP metodu
    """)
    
    df = st.session_state.df
    
    if df is None:
        st.error("Önce Coraza WAF log dosyası yükleyin!")
        return
    
    missing_values = df.isnull().sum()
    
    if missing_values.any():
        missing_method = st.selectbox(
            "Eksik Veri İşleme Yöntemi",
            options=["Ortalama ile Doldur", "Medyan ile Doldur", "Bir Şey Yapma"]
        )
    else:
        missing_method = "Bir Şey Yapma"
    
    if st.button("Veriyi İşle"):
        # Eski yöntemle devam edelim, yeni yöntem preprocess_data_interactive olarak kullanılabilir
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
        
        # Uyarı kaydı
        st.warning(f"Mevcut sütunlar: {df.columns.tolist()}")
        st.warning(f"Eksik nokta ayraçlı sütunlar: {missing_dot_columns}")
        
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
            new_columns_added = []
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
                    new_columns_added.append(f"{old_col} -> {new_col}")
            
            if new_columns_added:
                st.info(f"Standartlaştırılan sütunlar: {', '.join(new_columns_added)}")
            
            # Bazı gerekli sütunları eksik verilere rağmen bir şekilde sağla
            # Varsayılan değerler atanabilir veya özel bir işleme yapılabilir
            for col in missing_dot_columns:
                if col == 'transaction.client_port' and col not in df.columns:
                    # Client port eksikse 0 olarak ata
                    df[col] = 0
                    st.warning(f"Eksik sütun {col} için varsayılan değer 0 atandı.")
                
                elif col == 'transaction.request.uri' and col not in df.columns:
                    # URI eksikse '/unknown' olarak ata
                    df[col] = '/unknown'
                    st.warning(f"Eksik sütun {col} için varsayılan değer '/unknown' atandı.")
                
                elif col == 'transaction.is_interrupted' and col not in df.columns:
                    # Engellenme durumu eksikse False olarak ata
                    df[col] = False
                    st.warning(f"Eksik sütun {col} için varsayılan değer False atandı.")
                
                elif col == 'transaction.request.method' and col not in df.columns:
                    # Method eksikse 'UNKNOWN' olarak ata
                    df[col] = 'UNKNOWN'
                    st.warning(f"Eksik sütun {col} için varsayılan değer 'UNKNOWN' atandı.")
                
                elif col == 'transaction.timestamp' and col not in df.columns:
                    # Zaman damgası eksikse şimdiki zamanı ata
                    import datetime
                    df[col] = datetime.datetime.now()
                    st.warning(f"Eksik sütun {col} için varsayılan değer (şimdiki zaman) atandı.")
            
            # Yeniden kontrol et
            remaining_missing = [col for col in required_columns_dot if col not in df.columns]
            if remaining_missing:
                st.warning(f"Hala eksik sütunlar var: {remaining_missing}")
                # Eksik sütunlar olsa bile devam et
                st.info("Eksik sütunlara rağmen işleme devam ediliyor...")
        
        df = handle_missing_values(df, missing_method)
        df = create_features(df, uri_threshold)
        X = prepare_final_features(df)
        
        return df, X
        
    except Exception as e:
        st.error(f"İşlem hatası: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, f"İşlem hatası: {str(e)}"

def handle_missing_values(df, method):
    if method == "Ortalama ile Doldur":
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
    df['transaction.timestamp'] = pd.to_datetime(df['transaction.timestamp'], errors='coerce', format='mixed')
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
    
    # Tüm sütunların sayısal olduğundan emin ol
    safe_numeric_features = []
    for col in numeric_features:
        try:
            # Sütunun veri tipini kontrol et
            if df[col].dtype == 'object':
                # Sayısal değere dönüştürmeyi dene
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # NaN değerleri doldur
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
                
            # Sütun hala sayısal mı kontrol et
            if pd.api.types.is_numeric_dtype(df[col]):
                safe_numeric_features.append(col)
            else:
                st.warning(f"'{col}' sütunu sayısal değil, atlanıyor.")
        except Exception as e:
            st.warning(f"'{col}' sütunu işlenirken hata: {str(e)}")
    
    if not safe_numeric_features:
        st.error("Hiçbir sayısal özellik bulunamadı!")
        # Boş bir array döndür
        return np.zeros((len(df), 1))
    
    # Sadece sayısal sütunları kullan
    feature_array = df[safe_numeric_features].values
    
    # NaN veya inf değerleri kontrol et ve temizle
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Güncel scikit-learn sürümleriyle uyumlu StandardScaler kullanımı
    if len(feature_array) > 0 and feature_array.size > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_array)
        return X_scaled
    else:
        st.error("Özellik dizisi boş!")
        return np.zeros((len(df), 1))

def train_som(X, grid_size, sigma, learning_rate, iterations):
    som = MiniSom(grid_size, grid_size, X.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(X)
    som.train_random(X, iterations)
    return som

def fix_unhashable_columns(df):
    """DataFrame'deki liste veya hashlenemeyen değerleri içeren sütunları düzeltir."""
    for col in df.columns:
        # Listeler ve diğer hashlenemeyen değerleri kontrol et
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            # Bu değerleri string temsiline çevir
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
    return df

