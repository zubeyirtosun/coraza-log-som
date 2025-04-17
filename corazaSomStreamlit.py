import streamlit as st
import json
import pandas as pd
import numpy as np
from minisom import MiniSom
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Başlık ve açıklama
st.title("Log Analizi ve SOM Kümeleme")
st.write("Bu uygulama, JSON formatındaki log dosyalarını analiz eder, SOM ile kümeler ve anomalileri tespit eder.")

# 1. Esnek Log Girişi
st.write("### Log Dosyası Yükleme")
uploaded_file = st.file_uploader("Log dosyanızı yükleyin (JSON formatında)", type=["json"])

if uploaded_file is not None:
    logs = json.load(uploaded_file)
    df = pd.json_normalize(logs)
else:
    st.error("Lütfen bir log dosyası yükleyin.")
    st.stop()

# Gerekli sütunların kontrolü
required_columns = ['transaction.client_port', 'transaction.request.uri', 'transaction.timestamp', 'transaction.is_interrupted']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Gerekli sütun eksik: {col}")
        st.stop()

# 2. Dinamik URI Kategorizasyonu
df['uri_cat'] = df['transaction.request.uri'].apply(
    lambda x: x.split('/')[1] if len(x.split('/')) > 1 else 'root'
)

# 3. Özellik Seçimi ve Ön İşleme
df['transaction.timestamp'] = pd.to_datetime(df['transaction.timestamp'])
df['hour'] = df['transaction.timestamp'].dt.hour
features = ['transaction.client_port', 'hour']
X = pd.get_dummies(df[features], columns=['hour']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Uyarlanabilir SOM Izgara Boyutu
n_samples = len(df)
grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
som = MiniSom(x=grid_size, y=grid_size, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

# BMU ve quantization error hesaplama
df[['bmu_x', 'bmu_y']] = np.array([som.winner(x) for x in X])
df['quantization_error'] = [som.quantization_error(np.array([x])) for x in X]

# 5. Nöron Bazında Özet Tablosu
st.write("### Nöron Bazında Özet Tablosu")
st.write("Bu tablo, her SOM nöronundaki logların özet istatistiklerini gösterir.")
summary_df = df.groupby(['bmu_x', 'bmu_y']).agg({
    'transaction.is_interrupted': 'mean',
    'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
    'quantization_error': 'mean',
    'transaction.client_port': 'count'
}).reset_index()
summary_df.columns = ['BMU_X', 'BMU_Y', 'Engellenmiş Oranı', 'En Sık URI', 'Ort. Quantization Error', 'Log Sayısı']
summary_df['Nöron'] = summary_df.apply(lambda row: f"({row['BMU_X']},{row['BMU_Y']})", axis=1)
st.table(summary_df)

# 6. Etkileşimli Görselleştirmeler
st.write("### SOM Izgarasındaki Log Dağılımı")
fig_scatter = px.scatter(
    df,
    x='bmu_x',
    y='bmu_y',
    color='quantization_error',
    hover_data=['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted'],
    title='SOM Izgarasındaki Log Dağılımı'
)
st.plotly_chart(fig_scatter)

# 7. Nöron Seçimi ve Detaylı İstatistikler
st.write("### Nöron Detayları")
selected_neuron = st.selectbox("Bir nöron seçin", options=summary_df['Nöron'])
bmu_x, bmu_y = map(int, selected_neuron.strip('()').split(','))
neuron_group = df[(df['bmu_x'] == bmu_x) & (df['bmu_y'] == bmu_y)]
st.write(f"Seçilen Nöron: ({bmu_x},{bmu_y})")
st.write(f"Toplam log sayısı: {len(neuron_group)}")
st.write(f"Engellenmiş istek sayısı: {neuron_group['transaction.is_interrupted'].sum()}")
st.write(f"En sık URI: {neuron_group['transaction.request.uri'].mode()[0] if not neuron_group.empty else 'Yok'}")
st.dataframe(neuron_group)

# 8. Ayarlanabilir Anomali Eşiği
st.write("### Anomali Tespiti")
st.info("Anomali eşiği, logların quantization error'ının üst percentile'ına göre belirlenir.")
threshold_percentile = st.slider("Anomali eşiği percentile", min_value=80, max_value=99, value=95)
threshold = np.percentile(df['quantization_error'], threshold_percentile)
anomalies = df[df['quantization_error'] > threshold]
st.write(f"Anomali eşiği: {threshold:.4f} (Percentile: {threshold_percentile})")
st.dataframe(anomalies[['transaction.client_port', 'transaction.request.method', 'transaction.is_interrupted', 'quantization_error']])

# 9. Meta-Kümeleme
st.write("### Meta-Kümeleme Analizi")
weights = som.get_weights().reshape(-1, X.shape[1])
kmeans = KMeans(n_clusters=5)
meta_clusters = kmeans.fit_predict(weights)
meta_cluster_map = meta_clusters.reshape(grid_size, grid_size)
df['meta_cluster'] = df.apply(lambda row: meta_cluster_map[int(row['bmu_x']), int(row['bmu_y'])], axis=1)
meta_summary = df.groupby('meta_cluster').agg({
    'transaction.is_interrupted': 'mean',
    'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
    'bmu_x': 'count'
}).rename(columns={'bmu_x': 'Log Sayısı'})
st.write("#### Meta-Küme Özet İstatistikleri")
st.table(meta_summary)
fig_meta = px.imshow(meta_cluster_map, color_continuous_scale='viridis', title='Meta-Küme Haritası')
st.plotly_chart(fig_meta)

# 10. Geliştirilmiş Kullanıcı Arayüzü (Zaten entegre edildi)
st.write("Analiz tamamlandı! Daha fazla bilgi için yukarıdaki bölümleri inceleyin.")
