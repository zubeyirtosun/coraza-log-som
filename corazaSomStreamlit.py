import streamlit as st
import json
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Session state'i başlat
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'som' not in st.session_state:
    st.session_state.som = None
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = None

# Başlık ve açıklama
st.title("Log Analizi ve SOM Kümeleme")
st.write("Bu uygulama, JSON formatındaki log dosyalarını analiz eder, SOM ile kümeler ve anomalileri tespit eder.")

# SOM Nedir? Açıklama
with st.expander("Self-Organizing Map (SOM) Nedir?"):
    st.write("""
    Self-Organizing Map (SOM), yüksek boyutlu verileri 2 boyutlu bir ızgarada görselleştiren bir yapay sinir ağıdır. Log analizinde, benzer davranışları gruplamak ve anormal istekleri tespit etmek için kullanılır. Her log, ızgarada en uygun nörona (Best Matching Unit - BMU) atanır ve quantization error ile anomaliler belirlenir.
    Daha fazla bilgi için: [Analytics Vidhya SOM Kılavuzu](https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/)
    """)

# Kazanımlarım Bölümü
st.markdown("""
### Kazanımlarım
- **Neden yaptım?**: Log verilerini SOM ile analiz ederek, loglardaki desenleri ve anomalileri keşfetmek istedim. Bu sayede, potansiyel güvenlik tehditlerini tespit edebileceğimi düşündüm.
- **Ne anladım?**: SOM’un, log verilerini benzerliklerine göre kümelere ayırdığını ve quantization error ile anormal davranışları tespit ettiğini öğrendim. Meta-kümeleme ile daha geniş desenleri keşfettim.
- **Neler öğrendim?**: SOM algoritmasını uygulamayı, log verilerini ön işlemden geçirmeyi, eksik verileri işlemeyi ve görselleştirme tekniklerini geliştirmeyi öğrendim.
""")

st.info("Başlamak için bir JSON log dosyası yükleyin. Ardından eksik veri işleme seçeneklerini belirtebilirsiniz.")

# Analizi sıfırlama butonu
if st.button("Analizi Sıfırla"):
    st.session_state.clear()
    st.rerun()

# 1. Esnek Log Girişi
st.subheader("Log Dosyası Yükleme")
st.info("JSON formatında bir log dosyası yükleyin. Dosya, gerekli sütunları (client_port, uri, timestamp, is_interrupted, request.method) içermelidir.")
uploaded_file = st.file_uploader("Log dosyanızı yükleyin (JSON formatında)", type=["json"])

if uploaded_file is not None and st.session_state.df is None:
    try:
        logs = json.load(uploaded_file)
        st.session_state.df = pd.json_normalize(logs)
    except json.JSONDecodeError:
        st.error("Geçersiz JSON dosyası. Lütfen doğru formatta bir dosya yükleyin.")
        st.stop()

# Gerekli sütunların kontrolü ve hata ayıklama
if st.session_state.df is not None:
    required_columns = ['transaction.client_port', 'transaction.request.uri', 'transaction.timestamp', 'transaction.is_interrupted', 'transaction.request.method']
    missing_columns = [col for col in required_columns if col not in st.session_state.df.columns]
    if missing_columns:
        st.error(f"Gerekli sütunlar eksik: {missing_columns}")
        st.write("Veri setinde bulunan sütunlar:")
        st.write(st.session_state.df.columns.tolist())
        st.stop()

# 2. Eksik Değer Kontrolü ve Kullanıcı Seçimi
@st.cache_data
def preprocess_data(df, features, missing_method, uri_threshold=10):
    df = df.copy()
    if 'transaction.timestamp' not in df.columns:
        return None, "transaction.timestamp sütunu eksik."
    
    df['transaction.timestamp'] = pd.to_datetime(df['transaction.timestamp'], errors='coerce')
    df['hour'] = df['transaction.timestamp'].dt.hour

    # Eksik sütun kontrolü
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        return None, f"Şu sütunlar veri setinde bulunamadı: {missing_cols}"

    if missing_method == "Satırları Çıkar":
        df = df.dropna()
        if df.empty:
            return None, "Tüm satırlar eksik veriler içeriyor ve çıkarıldı."
    elif missing_method == "Ortalama ile Doldur":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_method == "Medyan ile Doldur":
        df = df.fillna(df.median(numeric_only=True))
    elif missing_method == "Bir Şey Yapma":
        if df[features].isnull().any().any():
            return None, "Eksik veriler mevcut ve 'Bir Şey Yapma' seçildi. Lütfen başka bir yöntem seçin veya eksik verileri temizleyin."

    # URI kategorisi oluşturma ve nadir URI'leri "other" ile değiştirme
    df['uri_cat'] = df['transaction.request.uri'].apply(
        lambda x: x.split('/')[1] if len(x.split('/')) > 1 else 'root'
    )
    uri_counts = df['uri_cat'].value_counts()
    rare_uris = uri_counts[uri_counts < uri_threshold].index
    df['uri_cat'] = df['uri_cat'].apply(lambda x: 'other' if x in rare_uris else x)

    # Kategorik sütunları one-hot encoding ile sayısal hale getirme
    df = pd.get_dummies(df, columns=['uri_cat', 'transaction.request.method'], drop_first=True)

    # Sayısal sütunları seçme (orijinal sayısal sütunlar + one-hot encoded sütunlar)
    numeric_features = [col for col in df.columns if col.startswith('uri_cat_') or col.startswith('transaction.request.method_') or col in ['transaction.client_port', 'hour']]
    X = df[numeric_features].values

    if X.shape[0] == 0:
        return None, "Veri seti boş. Analiz yapılamaz."

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return df, X

if st.session_state.df is not None:
    st.subheader("Eksik Veri İşleme")
    st.info("Eksik veriler tespit edilirse, aşağıdaki seçeneklerden birini seçerek nasıl işleneceğini belirtebilirsiniz.")
    missing_values = st.session_state.df.isnull().sum()
    if missing_values.any():
        st.write("Eksik veriler tespit edildi.")
        missing_method = st.selectbox(
            "Eksik Veri İşleme Yöntemi",
            options=["Satırları Çıkar", "Ortalama ile Doldur", "Medyan ile Doldur", "Bir Şey Yapma"]
        )
    else:
        st.write("Eksik veri bulunmamaktadır.")
        missing_method = "Bir Şey Yapma"

    if st.button("Veriyi İşle"):
        # Sadece veri setinde var olan sütunları features listesine ekliyoruz
        features = ['transaction.client_port', 'hour', 'transaction.request.method']
        processed_df, result = preprocess_data(st.session_state.df, features, missing_method, uri_threshold=10)
        if processed_df is None:
            st.error(result)
            st.stop()
        st.session_state.df = processed_df
        st.session_state.X = result
        st.write(f"Eksik veriler '{missing_method}' yöntemiyle işlendi.")

        # SOM’u otomatik eğit
        n_samples = len(st.session_state.df)
        st.session_state.grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
        sigma = float(st.session_state.grid_size) / 2.0  # Sigma’yı ızgara boyutunun yarısına ayarla ve float yap
        learning_rate = 0.5
        num_iterations = 1000  # İterasyon sayısını artır
        st.session_state.som = MiniSom(x=st.session_state.grid_size, y=st.session_state.grid_size, input_len=st.session_state.X.shape[1], sigma=sigma, learning_rate=learning_rate)
        st.session_state.som.random_weights_init(st.session_state.X)
        with st.spinner("SOM eğitimi yapılıyor..."):
            st.session_state.som.train_random(st.session_state.X, num_iterations)

        st.session_state.df[['bmu_x', 'bmu_y']] = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
        st.session_state.df['quantization_error'] = [st.session_state.som.quantization_error(np.array([x])) for x in st.session_state.X]

# 3. SOM Parametre Ayarları ve Yeniden Eğitim
if st.session_state.X is not None:
    st.subheader("SOM Parametre Ayarları")
    st.info("SOM parametrelerini özelleştirmek için ayarları yapın ve 'SOM’u Yeniden Eğit' butonuna basın.")
    
    # Grid size tanımlı değilse varsayılan bir değer kullan
    max_grid_size = float(st.session_state.grid_size) if st.session_state.grid_size is not None else 10.0
    default_sigma = max_grid_size / 2.0
    
    sigma = st.slider("Sigma (Komşuluk Yayılımı)", min_value=0.1, max_value=max_grid_size, value=default_sigma, step=0.1)
    learning_rate = st.slider("Öğrenme Oranı", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    num_iterations = st.slider("Eğitim İterasyon Sayısı", min_value=100, max_value=5000, value=1000, step=100)

    if st.button("SOM’u Yeniden Eğit"):
        st.session_state.som = MiniSom(x=st.session_state.grid_size, y=st.session_state.grid_size, input_len=st.session_state.X.shape[1], sigma=sigma, learning_rate=learning_rate)
        st.session_state.som.random_weights_init(st.session_state.X)
        with st.spinner("SOM eğitimi yapılıyor..."):
            st.session_state.som.train_random(st.session_state.X, num_iterations)

        st.session_state.df[['bmu_x', 'bmu_y']] = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
        st.session_state.df['quantization_error'] = [st.session_state.som.quantization_error(np.array([x])) for x in st.session_state.X]
        st.session_state.summary_df = None  # Özet tabloyu sıfırla

# 4. Nöron Bazında Özet Tablosu
if st.session_state.som is not None and st.session_state.summary_df is None:
    st.subheader("Nöron Bazında Özet Tablosu")
    with st.expander("Bu Tablo Ne Anlama Geliyor?"):
        st.write("""
        Bu tablo, SOM ızgarasındaki her nöronun (BMU) özet istatistiklerini gösterir. Her nöron, benzer logları temsil eder. Tablo, nöronun koordinatlarını, engellenmiş istek oranını, en sık URI’yi, ortalama quantization error’u ve log sayısını içerir.
        - **Yüksek engellenmiş oranı**: Potansiyel güvenlik tehditlerini işaret edebilir.
        - **Yüksek quantization error**: Anormal davranışları gösterebilir.
        """)
    st.session_state.summary_df = st.session_state.df.groupby(['bmu_x', 'bmu_y']).agg({
        'transaction.is_interrupted': 'mean',
        'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
        'quantization_error': 'mean',
        'transaction.client_port': 'count'
    }).reset_index()
    st.session_state.summary_df.columns = ['BMU_X', 'BMU_Y', 'Engellenmiş Oranı', 'En Sık URI', 'Ort. Quantization Error', 'Log Sayısı']
    st.session_state.summary_df['Nöron'] = st.session_state.summary_df.apply(lambda row: f"({row['BMU_X']},{row['BMU_Y']})", axis=1)

if st.session_state.summary_df is not None:
    st.table(st.session_state.summary_df)

# 5. Etkileşimli Görselleştirmeler
if st.session_state.summary_df is not None:
    st.subheader("SOM Izgarasındaki Log Dağılımı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu grafik, log verilerinin SOM ızgarasındaki dağılımını gösterir. Her nokta, bir log kaydını temsil eder ve en uygun nörona (BMU) atanmıştır. Renkler, quantization error’u gösterir:
        - **Mavi (düşük error)**: Normal davranışları temsil eder.
        - **Kırmızı (yüksek error)**: Potansiyel anomalileri işaret eder.
        Üzerine gelindiğinde, logun client port, URI ve engellenme durumu gibi detayları görünür.
        Örnek: `/WEB-INF/web.xml` gibi hassas dosyalara erişim girişimleri genellikle yüksek quantization error ile anormal olarak işaretlenir.
        """)
    fig_scatter = px.scatter(
        st.session_state.df,
        x='bmu_x',
        y='bmu_y',
        color='quantization_error',
        hover_data=['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted'],
        title='SOM Izgarasındaki Log Dağılımı',
        color_continuous_scale='Viridis'
    )
    fig_scatter.update_layout(
        legend_title_text='Quantization Error',
        legend=dict(itemsizing='constant')
    )
    st.plotly_chart(fig_scatter)
    st.markdown("**Lejant:** Renkler, quantization error değerlerini gösterir. Daha yüksek değerler (kırmızı), potansiyel anomalileri işaret eder.")

    # Ek Grafik: Quantization Error Dağılımı
    st.subheader("Quantization Error Dağılımı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu histogram, logların quantization error değerlerinin dağılımını gösterir. Yüksek quantization error’lar (sağda) potansiyel anomalileri temsil eder. Bu grafik, anomalilerin ne kadar yaygın olduğunu anlamanıza yardımcı olur.
        """)
    fig_hist = px.histogram(
        st.session_state.df,
        x='quantization_error',
        nbins=50,
        title='Quantization Error Dağılımı',
        labels={'quantization_error': 'Quantization Error'}
    )
    st.plotly_chart(fig_hist)

# 6. Nöron Seçimi ve Detaylı İstatistikler
if st.session_state.summary_df is not None:
    st.subheader("Nöron Detayları")
    with st.expander("Bu Bölüm Ne Anlama Geliyor?"):
        st.write("""
        Bu bölüm, seçilen bir SOM nöronundaki logların detaylarını gösterir. Bir nöron seçerek, o nörona atanan logların özelliklerini (örneğin, engellenmiş istek sayısı, en sık URI) inceleyebilirsiniz. Vurgulanan kırmızı 'X', seçilen nöronun ızgaradaki konumunu gösterir.
        """)
    selected_neuron = st.selectbox("Bir nöron seçin", options=st.session_state.summary_df['Nöron'])
    bmu_x, bmu_y = map(int, selected_neuron.strip('()').split(','))
    neuron_group = st.session_state.df[(st.session_state.df['bmu_x'] == bmu_x) & (st.session_state.df['bmu_y'] == bmu_y)]
    
    fig_scatter = px.scatter(
        st.session_state.df,
        x='bmu_x',
        y='bmu_y',
        color='quantization_error',
        hover_data=['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted'],
        title='SOM Izgarasındaki Log Dağılımı (Seçilen Nöron Vurgulandı)',
        color_continuous_scale='Viridis'
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[bmu_x],
            y=[bmu_y],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Seçilen Nöron'
        )
    )
    fig_scatter.update_layout(
        legend_title_text='Quantization Error',
        legend=dict(itemsizing='constant')
    )
    st.plotly_chart(fig_scatter)

    st.write(f"Seçilen Nöron: ({bmu_x},{bmu_y})")
    st.write(f"Toplam log sayısı: {len(neuron_group)}")
    st.write(f"Engellenmiş istek sayısı: {neuron_group['transaction.is_interrupted'].sum()}")
    st.write(f"En sık URI: {neuron_group['transaction.request.uri'].mode()[0] if not neuron_group.empty else 'Yok'}")
    st.dataframe(neuron_group)

# 7. Ayarlanabilir Anomali Eşiği
if st.session_state.som is not None:
    st.subheader("Anomali Tespiti")
    with st.expander("Bu Bölüm Ne Anlama Geliyor?"):
        st.write("""
        Bu bölüm, yüksek quantization error’a sahip logları anomaliler olarak tespit eder. Yüzdebirlik eşiğini ayarlayarak, hangi logların anormal olduğunu belirleyebilirsiniz. Daha yüksek bir eşik, daha az ama daha belirgin anomaliler gösterir.
        Örnek: `/WEB-INF/web.xml` gibi hassas dosyalara erişim girişimleri genellikle yüksek quantization error ile anormal olarak işaretlenir.
        """)
    threshold_percentile = st.slider("Anomali eşiği yüzdebirlik (percentile)", min_value=50, max_value=99, value=95, step=1)
    if 'quantization_error' not in st.session_state.df.columns:
        st.error("Quantization error verileri eksik. SOM modelinin doğru eğitildiğinden emin olun.")
        st.stop()
    threshold = np.percentile(st.session_state.df['quantization_error'], threshold_percentile)
    anomalies = st.session_state.df[st.session_state.df['quantization_error'] > threshold]
    st.write(f"Anomali eşiği: {threshold:.4f} (Yüzdebirlik: {threshold_percentile})")
    if anomalies.empty:
        st.warning("Seçilen yüzdebirlik değeriyle hiçbir anomali tespit edilmedi. Daha düşük bir yüzdebirlik deneyin.")
    else:
        st.dataframe(anomalies[['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted', 'quantization_error']])

# 8. Meta-Kümeleme Analizi
if st.session_state.som is not None and st.session_state.X is not None:
    st.subheader("Meta-Kümeleme Analizi")
    with st.expander("Meta-Kümeleme Nedir?"):
        st.write("""
        Meta-kümeleme, SOM nöronlarını K-means algoritmasıyla daha büyük kümelere ayırır. Bu, log verilerindeki geniş davranış modellerini tespit etmeye yardımcı olur. Her meta-küme, benzer özelliklere sahip logları temsil eder.
        Örnek: Bir meta-küme, `/login` endpoint’ine yönelik şüpheli istekleri içerebilir ve yüksek engellenmiş istek oranıyla dikkat çekebilir.
        """)
    weights = st.session_state.som.get_weights().reshape(-1, st.session_state.X.shape[1])
    n_meta_clusters = st.slider("Meta-Küme Sayısı", min_value=2, max_value=10, value=5)
    kmeans = KMeans(n_clusters=n_meta_clusters)
    meta_clusters = kmeans.fit_predict(weights)
    meta_cluster_map = meta_clusters.reshape(st.session_state.grid_size, st.session_state.grid_size)
    st.session_state.df['meta_cluster'] = st.session_state.df.apply(lambda row: meta_cluster_map[int(row['bmu_x']), int(row['bmu_y'])], axis=1)
    meta_summary = st.session_state.df.groupby('meta_cluster').agg({
        'transaction.is_interrupted': 'mean',
        'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
        'bmu_x': 'count'
    }).rename(columns={'bmu_x': 'Log Sayısı'})
    st.write("#### Meta-Küme Özet İstatistikleri")
    st.table(meta_summary)

    # Meta-Küme Haritası Görselleştirme
    st.subheader("Meta-Küme Haritası")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu grafik, SOM nöronlarının K-means ile meta-kümelere ayrıldığını gösterir. Her renk, farklı bir meta-kümeyi temsil eder. Üzerine gelindiğinde, logun client port ve URI gibi detayları görünür.
        Örnek: Meta Cluster 0’da 147 log varsa ve çoğu `/WEB-INF/web.xml` gibi hassas URI’lere yönelikse, bu küme bir güvenlik tehdidi modelini temsil edebilir.
        """)
    fig_meta = px.scatter(
        st.session_state.df,
        x='bmu_x',
        y='bmu_y',
        color='meta_cluster',
        hover_data=['transaction.client_port', 'transaction.request.uri'],
        title='Meta-Küme Haritası',
        color_continuous_scale='Viridis'
    )
    fig_meta.update_layout(
        legend_title_text='Meta-Kümeler',
        legend=dict(itemsizing='constant')
    )
    st.plotly_chart(fig_meta)
    st.markdown("**Lejant:** Renkler, farklı meta kümeleri temsil eder.")

    # Ek Grafik: Meta-Küme Bazında Log Sayısı
    st.subheader("Meta-Küme Bazında Log Sayısı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu çubuk grafik, her meta-kümedeki log sayısını gösterir. Daha yüksek çubuklar, daha yaygın davranış modellerini temsil eder. Örneğin, Meta Cluster 0’da 147 log varsa, bu küme sistemdeki baskın bir davranışı gösterebilir.
        """)
    cluster_counts = st.session_state.df['meta_cluster'].value_counts()
    fig_bar = px.bar(
        cluster_counts,
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Meta-Küme', 'y': 'Log Sayısı'},
        title='Meta-Küme Bazında Log Sayısı'
    )
    st.plotly_chart(fig_bar)
    st.markdown("**Açıklama:** Her çubuk, bir meta-kümedeki log sayısını gösterir. Bu, kümelerin büyüklüğünü anlamaya yardımcı olur.")

    # Ek Grafik: En Sık URI’lerin Dağılımı
    st.subheader("En Sık URI’lerin Dağılımı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu pasta grafik, loglardaki en sık URI’lerin dağılımını gösterir. En büyük dilimler, sistemde en çok erişilen endpoint’leri temsil eder. Örneğin, `/login` büyük bir dilimse, bu endpoint sıkça hedefleniyor demektir.
        """)
    uri_counts = st.session_state.df['transaction.request.uri'].value_counts().head(10)
    fig_pie = px.pie(
        values=uri_counts.values,
        names=uri_counts.index,
        title='En Sık URI’lerin Dağılımı'
    )
    st.plotly_chart(fig_pie)

# 9. Ek Görselleştirmeler
if st.session_state.som is not None:
    st.subheader("Ek Görselleştirmeler")
    st.write("#### SOM Izgarası Üzerinde Özellik Isı Haritaları")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu ısı haritası, SOM ızgarasındaki nöronlar arasındaki mesafeleri gösterir. Daha koyu renkler, nöronlar arasında daha büyük farklılıkları (yani farklı davranış modellerini) temsil eder.
        """)
    fig_heatmap = px.imshow(st.session_state.som.distance_map().T, color_continuous_scale='viridis', title='SOM Izgarası Isı Haritası')
    st.plotly_chart(fig_heatmap)

    st.write("#### Zaman Serisi Analizi")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu grafik, saat bazında engellenmiş istek oranını gösterir. Örneğin, belirli saatlerde engellenmiş isteklerin artması, bir saldırı girişimini işaret edebilir.
        """)
    fig_time_series = px.line(st.session_state.df.groupby('hour').agg({'transaction.is_interrupted': 'mean'}),
                              title='Saat Bazında Engellenmiş İstek Oranı')
    st.plotly_chart(fig_time_series)

# Sonuç ve Yorum
st.markdown("""
### Sonuç
Bu analizle, SOM kullanarak log verilerindeki desenleri ve anomalileri başarıyla tespit edebildim. Meta-kümeleme ile daha büyük yapıları ortaya çıkardım ve görselleştirme tekniklerimi geliştirdim. Quantization error, potansiyel güvenlik tehditlerini belirlemede etkili bir araç oldu.
""")
