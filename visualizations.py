import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd

def show_summary_table():
    st.subheader("Nöron Bazında Özet Tablosu")
    with st.expander("Bu Tablo Ne Anlama Geliyor?"):
        st.write("""
        Bu tablo, SOM ızgarasındaki her nöronun (BMU) özet istatistiklerini gösterir. Her nöron, benzer logları temsil eder. 
        Tablo, nöronun koordinatlarını, engellenmiş istek oranını, en sık URI'yi, ortalama quantization error'u ve log sayısını içerir.
        - **Yüksek engellenmiş oranı**: Potansiyel güvenlik tehditlerini işaret edebilir.
        - **Yüksek quantization error**: Anormal davranışları gösterebilir.
        """)
    
    if 'summary_df' not in st.session_state or st.session_state.summary_df is None:
        create_summary_table()
    st.dataframe(st.session_state.summary_df)

def create_summary_table():
    df = st.session_state.df
    if 'bmu_x' not in df.columns or 'bmu_y' not in df.columns:
        st.error("BMU koordinatları bulunamadı!")
        return
    
    summary = df.groupby(['bmu_x', 'bmu_y']).agg({
        'transaction.is_interrupted': 'mean',
        'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
        'quantization_error': 'mean',
        'transaction.client_port': 'count'
    }).reset_index()
    
    summary.columns = ['BMU_X', 'BMU_Y', 'Engellenme Oranı', 'En Sık URI', 'Ort. Hata', 'Log Sayısı']
    summary['Nöron'] = summary.apply(lambda row: f"({row['BMU_X']},{row['BMU_Y']})", axis=1)
    st.session_state.summary_df = summary

def show_visualizations():
    st.subheader("Etkileşimli Görselleştirmeler")
    with st.expander("Bu Grafikler Ne Anlama Geliyor?"):
        st.write("""
        Bu grafikler, log verilerinin SOM ızgarasındaki dağılımını ve quantization error değerlerini gösterir.
        - **SOM Izgara Dağılımı**: Her nokta bir log kaydını temsil eder. Renkler quantization error'u gösterir.
          Mavi (düşük error) normal davranışları, kırmızı (yüksek error) potansiyel anomalileri işaret eder.
        - **Quantization Error Dağılımı**: Logların quantization error değerlerinin dağılımını gösterir.
          Yüksek değerler (sağda) potansiyel anomalileri temsil eder.
        """)
    
    create_scatter_plot()
    create_error_distribution()
    create_time_series_analysis()
    create_uri_distribution()
    create_heatmap()

def create_scatter_plot():
    fig = px.scatter(
        st.session_state.df,
        x='bmu_x',
        y='bmu_y',
        color='quantization_error',
        hover_data=['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted'],
        title='SOM Izgara Dağılımı',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        legend_title_text='Quantization Error',
        legend=dict(itemsizing='constant')
    )
    st.plotly_chart(fig)
    st.markdown("**Lejant:** Renkler, quantization error değerlerini gösterir. Daha yüksek değerler (kırmızı), potansiyel anomalileri işaret eder.")

def create_error_distribution():
    fig = px.histogram(
        st.session_state.df,
        x='quantization_error',
        nbins=50,
        title='Quantization Error Dağılımı'
    )
    st.plotly_chart(fig)

def create_time_series_analysis():
    st.subheader("Zaman Serisi Analizi")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu grafik, saat bazında engellenmiş istek oranını gösterir. Örneğin, belirli saatlerde engellenmiş 
        isteklerin artması, bir saldırı girişimini işaret edebilir.
        """)
    
    if 'hour' in st.session_state.df.columns:
        time_series = st.session_state.df.groupby('hour').agg({'transaction.is_interrupted': 'mean'}).reset_index()
        fig = px.line(
            time_series, 
            x='hour', 
            y='transaction.is_interrupted',
            title='Saat Bazında Engellenmiş İstek Oranı',
            labels={'transaction.is_interrupted': 'Engellenme Oranı', 'hour': 'Saat'}
        )
        st.plotly_chart(fig)
    else:
        st.warning("Zaman serisi analizi için 'hour' sütunu bulunamadı.")

def create_uri_distribution():
    st.subheader("En Sık URI'lerin Dağılımı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu pasta grafik, loglardaki en sık URI'lerin dağılımını gösterir. En büyük dilimler, 
        sistemde en çok erişilen endpoint'leri temsil eder. Örneğin, `/login` büyük bir dilimse, 
        bu endpoint sıkça hedefleniyor demektir.
        """)
    
    uri_counts = st.session_state.df['transaction.request.uri'].value_counts().head(10)
    fig = px.pie(
        values=uri_counts.values,
        names=uri_counts.index,
        title='En Sık URI\'lerin Dağılımı'
    )
    st.plotly_chart(fig)

def create_heatmap():
    st.subheader("SOM Izgarası Isı Haritası")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu ısı haritası, SOM ızgarasındaki nöronlar arasındaki mesafeleri gösterir. 
        Daha koyu renkler, nöronlar arasında daha büyük farklılıkları (yani farklı davranış modellerini) temsil eder.
        """)
    
    if st.session_state.som is not None:
        fig = px.imshow(
            st.session_state.som.distance_map().T, 
            color_continuous_scale='viridis', 
            title='SOM Izgarası Isı Haritası'
        )
        st.plotly_chart(fig)
    else:
        st.warning("SOM modeli bulunamadı.")

def handle_meta_clustering():
    st.subheader("Meta Kümeleme Analizi")
    with st.expander("Meta-Kümeleme Nedir?"):
        st.write("""
        Meta-kümeleme, SOM nöronlarını K-means algoritmasıyla daha büyük kümelere ayırır. 
        Bu, log verilerindeki geniş davranış modellerini tespit etmeye yardımcı olur. 
        Her meta-küme, benzer özelliklere sahip logları temsil eder.
        Örnek: Bir meta-küme, `/login` endpoint'ine yönelik şüpheli istekleri içerebilir 
        ve yüksek engellenmiş istek oranıyla dikkat çekebilir.
        """)
    
    n_clusters = st.slider("Meta Küme Sayısı", 2, 10, 5)
    
    from sklearn.cluster import KMeans
    weights = st.session_state.som.get_weights().reshape(-1, st.session_state.X.shape[1])
    kmeans = KMeans(n_clusters=n_clusters)
    meta_clusters = kmeans.fit_predict(weights)
    
    # SOM ızgarasındaki her hücre için meta küme etiketlerini oluştur
    grid_size = st.session_state.som.get_weights().shape[0]  # SOM ızgara boyutu
    meta_cluster_map = {}
    
    # Her BMU koordinatı için ilgili meta küme etiketini eşleştir
    for i in range(grid_size):
        for j in range(grid_size):
            # (i,j) koordinatındaki hücrenin meta küme etiketi
            meta_cluster_map[(i, j)] = meta_clusters[i * grid_size + j]
    
    # Her veri noktasına BMU koordinatlarına göre meta küme etiketi ata
    df_meta = st.session_state.df.copy()
    df_meta['meta_cluster'] = df_meta.apply(
        lambda row: meta_cluster_map.get((int(row['bmu_x']), int(row['bmu_y'])), -1), 
        axis=1
    )
    
    # Meta küme dağılımı grafiği
    fig = px.scatter(
        df_meta,
        x='bmu_x',
        y='bmu_y',
        color='meta_cluster',
        hover_data=['transaction.client_port', 'transaction.request.uri'],
        title='Meta Küme Dağılımı',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig)
    
    # Meta küme özet istatistikleri
    meta_summary = df_meta.groupby('meta_cluster').agg({
        'transaction.is_interrupted': 'mean',
        'transaction.request.uri': lambda x: x.mode()[0] if not x.empty else 'Yok',
        'transaction.client_port': 'count'
    }).reset_index()
    
    meta_summary.columns = ['Meta Küme', 'Engellenme Oranı', 'En Sık URI', 'Log Sayısı']
    st.write("#### Meta Küme Özet İstatistikleri")
    st.table(meta_summary)
    
    # Meta küme bazında log sayısı grafiği
    st.subheader("Meta Küme Bazında Log Sayısı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu çubuk grafik, her meta-kümedeki log sayısını gösterir. Daha yüksek çubuklar, 
        daha yaygın davranış modellerini temsil eder. Örneğin, Meta Cluster 0'da çok sayıda log varsa, 
        bu küme sistemdeki baskın bir davranışı gösterebilir.
        """)
    
    cluster_counts = df_meta['meta_cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Meta Küme', 'Log Sayısı']
    
    fig = px.bar(
        cluster_counts,
        x='Meta Küme',
        y='Log Sayısı',
        title='Meta Küme Bazında Log Sayısı'
    )
    st.plotly_chart(fig)

def handle_neuron_details():
    st.subheader("Nöron Detayları")
    with st.expander("Bu Bölüm Ne Anlama Geliyor?"):
        st.write("""
        Bu bölüm, seçilen bir SOM nöronundaki logların detaylarını gösterir. 
        Bir nöron seçerek, o nörona atanan logların özelliklerini (örneğin, engellenmiş istek sayısı, en sık URI) 
        inceleyebilirsiniz. Vurgulanan kırmızı 'X', seçilen nöronun ızgaradaki konumunu gösterir.
        """)
    
    if 'summary_df' not in st.session_state or st.session_state.summary_df is None:
        create_summary_table()
    
    selected_neuron = st.selectbox("Bir nöron seçin", options=st.session_state.summary_df['Nöron'])
    bmu_x, bmu_y = map(int, selected_neuron.strip('()').split(','))
    neuron_group = st.session_state.df[(st.session_state.df['bmu_x'] == bmu_x) & (st.session_state.df['bmu_y'] == bmu_y)]
    
    fig = px.scatter(
        st.session_state.df,
        x='bmu_x',
        y='bmu_y',
        color='quantization_error',
        hover_data=['transaction.client_port', 'transaction.request.uri', 'transaction.is_interrupted'],
        title='SOM Izgarasındaki Log Dağılımı (Seçilen Nöron Vurgulandı)',
        color_continuous_scale='Viridis'
    )
    fig.add_trace(
        go.Scatter(
            x=[bmu_x],
            y=[bmu_y],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Seçilen Nöron'
        )
    )
    st.plotly_chart(fig)
    
    st.write(f"Seçilen Nöron: ({bmu_x},{bmu_y})")
    st.write(f"Toplam log sayısı: {len(neuron_group)}")
    st.write(f"Engellenmiş istek sayısı: {neuron_group['transaction.is_interrupted'].sum()}")
    st.write(f"En sık URI: {neuron_group['transaction.request.uri'].mode()[0] if not neuron_group.empty else 'Yok'}")
    st.dataframe(neuron_group)

def handle_anomaly_detection():
    st.subheader("Anomali Tespiti")
    with st.expander("Bu Bölüm Ne Anlama Geliyor?"):
        st.write("""
        Bu bölüm, yüksek quantization error'a sahip logları anomaliler olarak tespit eder. 
        Yüzdebirlik eşiğini ayarlayarak, hangi logların anormal olduğunu belirleyebilirsiniz. 
        Daha yüksek bir eşik, daha az ama daha belirgin anomaliler gösterir.
        Örnek: `/WEB-INF/web.xml` gibi hassas dosyalara erişim girişimleri genellikle 
        yüksek quantization error ile anormal olarak işaretlenir.
        """)
    
    threshold_percentile = st.slider("Anomali eşiği yüzdebirlik (percentile)", 
                                    min_value=50, max_value=99, value=95, step=1)
    
    if 'quantization_error' not in st.session_state.df.columns:
        st.error("Quantization error verileri eksik. SOM modelinin doğru eğitildiğinden emin olun.")
        return
    
    threshold = np.percentile(st.session_state.df['quantization_error'], threshold_percentile)
    anomalies = st.session_state.df[st.session_state.df['quantization_error'] > threshold]
    
    st.write(f"Anomali eşiği: {threshold:.4f} (Yüzdebirlik: {threshold_percentile})")
    
    if anomalies.empty:
        st.warning("Seçilen yüzdebirlik değeriyle hiçbir anomali tespit edilmedi. Daha düşük bir yüzdebirlik deneyin.")
    else:
        st.dataframe(anomalies[['transaction.client_port', 'transaction.request.uri', 
                               'transaction.is_interrupted', 'quantization_error']])
