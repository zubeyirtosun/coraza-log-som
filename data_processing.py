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
    
    # Alternatif olarak interaktif veri önişleme kullanma seçeneği
    if st.button("İnteraktif Veri Önişleme"):
        preprocess_data_interactive()

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

def preprocess_data_interactive():
    """
    İnteraktif veri önişleme fonksiyonu. Kullanıcının özellik seçmesine ve 
    veriyi önişlemesine olanak sağlar.
    """
    if st.session_state.df is None:
        st.warning("Önce bir log dosyası yüklemelisiniz.")
        return

    st.subheader("Veri Önişleme")
    
    df = st.session_state.df.copy()
    
    # Özellik seçimi ve ayarları için ayrı bir sekmeli bölüm
    preprocessing_tabs = st.tabs(["Özellik Seçimi", "Özel Özellikler", "Önişleme Sonuçları"])
    
    with preprocessing_tabs[0]:
        # Özellik seçimi
        all_columns = df.columns.tolist()
        default_numeric_cols = [col for col in all_columns if df[col].dtype in ['int64', 'float64']]
        
        st.write("### Sayısal Özellikleri Seçin:")
        numeric_features = st.multiselect(
            "Sayısal Özellikler", 
            options=all_columns,
            default=default_numeric_cols if 'selected_numeric_features' not in st.session_state else st.session_state.selected_numeric_features
        )
        
        st.write("### Kategorik Özellikleri Seçin:")
        non_numeric_cols = [col for col in all_columns if col not in numeric_features]
        categorical_features = st.multiselect(
            "Kategorik Özellikler", 
            options=non_numeric_cols,
            default=non_numeric_cols[:5] if 'selected_categorical_features' not in st.session_state else st.session_state.selected_categorical_features  # İlk 5 kategorik sütunu varsayılan olarak seç
        )
        
        # Seçimleri kaydet
        st.session_state.selected_numeric_features = numeric_features
        st.session_state.selected_categorical_features = categorical_features

    with preprocessing_tabs[1]:
        st.write("### Özel Özellik Oluşturma")
        st.info("Bu bölümde, seçili sütunlarla özel özellikler oluşturabilirsiniz.")
        
        # URI threshold ayarı
        uri_threshold = st.slider(
            "URI Kategori Eşiği (az sayıdaki URI'ler 'other' olarak gruplandırılacak)",
            2, 50, 10
        )
        
        # Zaman özelliği seçimi
        time_feature_options = ['Saat', 'Gün', 'Hafta', 'Ay', 'Yıl']
        time_features = st.multiselect(
            "Zaman Özellikleri (transaction.timestamp sütunu için)",
            options=time_feature_options,
            default=['Saat'] if 'selected_time_features' not in st.session_state else st.session_state.selected_time_features
        )
        st.session_state.selected_time_features = time_features
        
        # İlave özellikler
        add_request_size = st.checkbox("Request Size Özelliği Ekle", 
                                      'add_request_size' in st.session_state and st.session_state.add_request_size)
        st.session_state.add_request_size = add_request_size
        
        add_is_blocked = st.checkbox("Engellenme Durumu Özelliği Ekle", 
                                    'add_is_blocked' in st.session_state and st.session_state.add_is_blocked)
        st.session_state.add_is_blocked = add_is_blocked

    with preprocessing_tabs[2]:
        if 'interactive_preprocessing_done' in st.session_state and st.session_state.interactive_preprocessing_done:
            st.success("Veri önişleme işlemi tamamlanmıştır.")
            
            if 'feature_names' in st.session_state:
                st.write("### İşlenmiş Özellikler")
                st.write(f"Toplam {len(st.session_state.feature_names)} özellik")
                
                # Gruplar halinde özellikleri göster
                feature_groups = {}
                for feat in st.session_state.feature_names:
                    prefix = feat.split('_')[0] if '_' in feat else 'Diğer'
                    if prefix not in feature_groups:
                        feature_groups[prefix] = []
                    feature_groups[prefix].append(feat)
                
                for group, features in feature_groups.items():
                    with st.expander(f"{group} ({len(features)}):"):
                        st.write(", ".join(features))
                
            if 'X' in st.session_state and st.session_state.X is not None:
                st.write("### Önişlenmiş Veri Boyutu:")
                st.write(f"Satır sayısı: {st.session_state.X.shape[0]}, Özellik sayısı: {st.session_state.X.shape[1]}")
                
                # İlk 5 özelliğin dağılımını göster
                if 'feature_names' in st.session_state and len(st.session_state.feature_names) > 0:
                    import matplotlib.pyplot as plt
                    
                    st.write("### Özellik Dağılımları (İlk 5):")
                    fig, axs = plt.subplots(min(5, len(st.session_state.feature_names)), 1, figsize=(10, 10))
                    
                    for i in range(min(5, len(st.session_state.feature_names))):
                        if len(st.session_state.feature_names) > 1:
                            ax = axs[i]
                        else:
                            ax = axs
                        
                        ax.hist(st.session_state.X[:, i], bins=20)
                        ax.set_title(st.session_state.feature_names[i])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("Henüz veri önişleme yapılmamış. Lütfen 'Özellikleri İşle ve Devam Et' butonuna basın.")
    
    # İşleme butonunu ana bölüme ekle
    st.write("### Veri İşleme")
    process_button = st.button("Özellikleri İşle ve Devam Et", key="process_button")
    
    if process_button:
        with st.spinner("Veri işleniyor..."):
            try:
                # İşlenecek özellikleri seç
                selected_features = list(set(numeric_features + categorical_features))
                
                if not selected_features:
                    st.warning("En az bir özellik seçmelisiniz.")
                    return
                
                # Seçilen özelliklere göre veriyi filtrele
                X_df = df[selected_features].copy()
                
                # Sütun isimleri doğrulama - "transaction." önekini ekle
                for col in X_df.columns:
                    if not col.startswith('transaction.') and any(req_col.endswith(col) for req_col in ['client_port', 'request.uri', 'timestamp', 'is_interrupted', 'request.method']):
                        # Bu sütunu "transaction." önekiyle yeniden adlandır
                        new_col = f"transaction.{col}"
                        X_df[new_col] = X_df[col]
                        X_df = X_df.drop(col, axis=1)
                
                # Hashlenemeyen değerleri düzelt
                X_df = fix_unhashable_columns(X_df)
                
                # Zaman damgası işleme
                if "transaction.timestamp" in X_df.columns:
                    X_df['transaction.timestamp'] = pd.to_datetime(X_df['transaction.timestamp'], errors='coerce', format='mixed')
                    
                    if 'Saat' in time_features:
                        X_df['hour'] = X_df['transaction.timestamp'].dt.hour
                    
                    if 'Gün' in time_features:
                        X_df['day'] = X_df['transaction.timestamp'].dt.day
                    
                    if 'Hafta' in time_features:
                        X_df['dayofweek'] = X_df['transaction.timestamp'].dt.dayofweek
                    
                    if 'Ay' in time_features:
                        X_df['month'] = X_df['transaction.timestamp'].dt.month
                    
                    if 'Yıl' in time_features:
                        X_df['year'] = X_df['transaction.timestamp'].dt.year
                
                # URI kategorizasyonu
                if "transaction.request.uri" in X_df.columns:
                    X_df['uri_cat'] = X_df['transaction.request.uri'].apply(
                        lambda x: str(x).split('/')[1] if len(str(x).split('/')) > 1 else 'root'
                    )
                    
                    # Nadir URI'leri 'other' olarak grupla
                    uri_counts = X_df['uri_cat'].value_counts()
                    rare_uris = uri_counts[uri_counts < uri_threshold].index
                    X_df['uri_cat'] = X_df['uri_cat'].apply(lambda x: 'other' if x in rare_uris else x)
                
                # Request size özelliği
                if add_request_size and "transaction.request.body" in X_df.columns:
                    X_df['request_size'] = X_df['transaction.request.body'].apply(
                        lambda x: len(str(x)) if isinstance(x, (str, bytes)) else 0
                    )
                
                # Engellenme durumu özelliği
                if add_is_blocked and "transaction.is_interrupted" in X_df.columns:
                    X_df['is_blocked'] = X_df['transaction.is_interrupted'].astype(int)
                
                # Kategorik sütunları belirle (URI ve Method dahil)
                cat_columns = [col for col in X_df.columns if (
                    col in categorical_features or
                    col == 'uri_cat' or
                    (col == 'transaction.request.method' and col in X_df.columns)
                )]
                
                # Kategorik özelliklerde eksik değerleri doldur
                for col in cat_columns:
                    X_df[col] = X_df[col].fillna('unknown')
                
                # One-hot encoding
                X_encoded = pd.get_dummies(X_df, columns=cat_columns, drop_first=False)
                
                # Sayısal özelliklerde eksik değerleri doldur
                num_columns = [col for col in X_encoded.columns if X_encoded[col].dtype in ['int64', 'float64']]
                for col in num_columns:
                    X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
                
                # Sayısal olmayan sütunları kontrol et ve dönüştür
                non_numeric_cols = [col for col in X_encoded.columns if X_encoded[col].dtype == 'object']
                for col in non_numeric_cols:
                    try:
                        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
                        # Dönüştürülen sütun için eksik değerleri doldur
                        X_encoded[col] = X_encoded[col].fillna(0)
                        num_columns.append(col)
                    except:
                        st.warning(f"'{col}' sütunu sayısal değere dönüştürülemedi, bu sütun atlanacak.")
                        X_encoded = X_encoded.drop(columns=[col])
                
                # Sadece sayısal sütunları al
                safe_columns = [col for col in num_columns if pd.api.types.is_numeric_dtype(X_encoded[col])]
                
                if not safe_columns:
                    st.error("Hiçbir sayısal özellik bulunamadı!")
                    return
                
                # İçerik kontrolü
                feature_array = X_encoded[safe_columns].values
                
                # NaN veya inf değerleri doğrudan temizle
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Güncel scikit-learn sürümleriyle uyumlu StandardScaler kullanımı
                scaler = StandardScaler()
                
                X_scaled = scaler.fit_transform(feature_array)
                
                # Sonuçları kaydet
                st.session_state.X = X_scaled
                st.session_state.feature_names = X_encoded.columns.tolist()
                st.session_state.interactive_preprocessing_done = True
                
                st.success(f"Veri önişleme tamamlandı. {X_encoded.shape[1]} özellik oluşturuldu.")
                
                # Sekme 2'ye geç (Önişleme Sonuçları)
                preprocessing_tabs[2].active = True
                
            except Exception as e:
                st.error(f"Veri önişleme sırasında hata oluştu: {str(e)}")
                import traceback
                st.error(traceback.format_exc())