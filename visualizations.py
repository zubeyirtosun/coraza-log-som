import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

# BytesIO görsellerini doğrudan StreamLit'te göstermek için
def show_matplotlib_figure(fig_buffer):
    if isinstance(fig_buffer, io.BytesIO):
        fig_buffer.seek(0)
        st.image(fig_buffer)
    elif isinstance(fig_buffer, str):
        # Base64 string ise BytesIO'ya çevir
        buffer = io.BytesIO(base64.b64decode(fig_buffer))
        buffer.seek(0)
        st.image(buffer)
    else:
        st.warning("Görselleştirme gösterilemiyor.")

# BytesIO'yu base64 string'e dönüştürmek için yardımcı fonksiyon
def _buffer_to_base64(buffer):
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return img_str

# Base64 string'i BytesIO'ya dönüştürmek için yardımcı fonksiyon
def _base64_to_buffer(base64_str):
    buffer = io.BytesIO(base64.b64decode(base64_str))
    buffer.seek(0)
    return buffer

def show_summary_table():
    st.subheader("Nöron Bazında Özet Tablosu")
    with st.expander("Bu Tablo Ne Anlama Geliyor?"):
        st.write("""
        Bu tablo, SOM ızgarasındaki her nöronun (BMU) özet istatistiklerini gösterir. Her nöron, benzer logları temsil eder. 
        Tablo, nöronun koordinatlarını, engellenmiş istek oranını, en sık URI'yi, ortalama niceleme hatasını ve log sayısını içerir.
        - **Yüksek engellenmiş oranı**: Potansiyel güvenlik tehditlerini işaret edebilir.
        - **Yüksek niceleme hatası**: Anormal davranışları gösterebilir.
        """)
    
    if 'summary_df' not in st.session_state or st.session_state.summary_df is None:
        create_summary_table()
    st.dataframe(st.session_state.summary_df)

def create_summary_table():
    df = st.session_state.df
    if 'bmu_x' not in df.columns or 'bmu_y' not in df.columns:
        st.error("BMU koordinatları bulunamadı!")
        return
    
    # Sütun kontrolleri ve gerekirse alternatif sütunları kullan
    agg_dict = {}
    
    # Engellenme oranı sütunu
    if 'transaction.is_interrupted' in df.columns:
        agg_dict['transaction.is_interrupted'] = 'mean'
    elif 'is_interrupted' in df.columns:
        agg_dict['is_interrupted'] = 'mean'
    else:
        st.warning("Engellenme oranı sütunu bulunamadı!")
    
    # URI sütunu
    if 'transaction.request.uri' in df.columns:
        agg_dict['transaction.request.uri'] = lambda x: x.mode()[0] if not x.empty and len(x.mode()) > 0 else 'Yok'
    elif 'request.uri' in df.columns:
        agg_dict['request.uri'] = lambda x: x.mode()[0] if not x.empty and len(x.mode()) > 0 else 'Yok'
    elif 'request_uri' in df.columns:
        agg_dict['request_uri'] = lambda x: x.mode()[0] if not x.empty and len(x.mode()) > 0 else 'Yok'
    else:
        st.warning("URI sütunu bulunamadı!")
    
    # Niceleme hatası sütunu
    if 'quantization_error' in df.columns:
        agg_dict['quantization_error'] = 'mean'
    
    # Log sayısı için herhangi bir sayısal sütun
    if 'transaction.client_port' in df.columns:
        agg_dict['transaction.client_port'] = 'count'
    elif 'client_port' in df.columns:
        agg_dict['client_port'] = 'count'
    else:
        # İlk sayısal sütunu bul
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            agg_dict[numeric_cols[0]] = 'count'
        else:
            # Herhangi bir sütun kullan
            agg_dict[df.columns[0]] = 'count'
    
    # Agregasyon yapacak sütun yoksa uyarı ver ve çık
    if not agg_dict:
        st.error("Özet tablo için gerekli sütunlar bulunamadı!")
        return
    
    # Özet tabloyu oluştur
    summary = df.groupby(['bmu_x', 'bmu_y']).agg(agg_dict).reset_index()
    
    # Sütun isimlerini düzelt
    rename_dict = {'bmu_x': 'BMU_X', 'bmu_y': 'BMU_Y'}
    
    for old_col, agg_func in agg_dict.items():
        if old_col in ['transaction.is_interrupted', 'is_interrupted']:
            rename_dict[old_col] = 'Engellenme Oranı'
        elif old_col in ['transaction.request.uri', 'request.uri', 'request_uri']:
            rename_dict[old_col] = 'En Sık URI'
        elif old_col == 'quantization_error':
            rename_dict[old_col] = 'Ort. Hata'
        elif agg_func == 'count':
            rename_dict[old_col] = 'Log Sayısı'
    
    summary = summary.rename(columns=rename_dict)
    
    # Nöron sütunu ekle
    summary['Nöron'] = summary.apply(lambda row: f"({row['BMU_X']},{row['BMU_Y']})", axis=1)
    st.session_state.summary_df = summary

def show_visualizations():
    st.subheader("Etkileşimli Görselleştirmeler")
    with st.expander("Bu Grafikler Ne Anlama Geliyor?"):
        st.write("""
        Bu grafikler, log verilerinin SOM ızgarasındaki dağılımını ve niceleme hatası değerlerini gösterir.
        - **SOM Izgara Dağılımı**: Her nokta bir log kaydını temsil eder. Renkler niceleme hatasını gösterir. 
          Mavi (düşük hata) normal davranışları, kırmızı (yüksek hata) potansiyel anomalileri işaret eder.
        - **Niceleme Hatası Dağılımı**: Logların niceleme hatası değerlerinin dağılımını gösterir.
          Yüksek değerler (sağda) potansiyel anomalileri temsil eder.
        """)
    
    create_scatter_plot()
    create_error_distribution()
    create_time_series_analysis()
    create_uri_distribution()
    create_heatmap()

def create_scatter_plot():
    df = st.session_state.df
    
    # Hover data seçeneklerini kontrol et
    hover_data = []
    if 'transaction.client_port' in df.columns:
        hover_data.append('transaction.client_port')
    elif 'client_port' in df.columns:
        hover_data.append('client_port')
        
    if 'transaction.request.uri' in df.columns:
        hover_data.append('transaction.request.uri')
    elif 'request.uri' in df.columns:
        hover_data.append('request.uri')
    elif 'request_uri' in df.columns:
        hover_data.append('request_uri')
        
    if 'transaction.is_interrupted' in df.columns:
        hover_data.append('transaction.is_interrupted')
    elif 'is_interrupted' in df.columns:
        hover_data.append('is_interrupted')
    
    # Scatter plot oluştur
    fig = px.scatter(
        df,
        x='bmu_x',
        y='bmu_y',
        color='quantization_error',
        hover_data=hover_data,
        title='SOM Izgara Dağılımı',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        legend_title_text='Niceleme Hatası',
        legend=dict(itemsizing='constant')
    )
    st.plotly_chart(fig)
    st.markdown("**Lejant:** Renkler, niceleme hatası değerlerini gösterir. Daha yüksek değerler (kırmızı), potansiyel anomalileri işaret eder.")

def create_error_distribution():
    fig = px.histogram(
        st.session_state.df,
        x='quantization_error',
        nbins=50,
        title='Niceleme Hatası Dağılımı'
    )
    st.plotly_chart(fig)

def create_time_series_analysis():
    st.subheader("Zaman Serisi Analizi")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu grafik, saat bazında engellenmiş istek oranını gösterir. Örneğin, belirli saatlerde engellenmiş 
        isteklerin artması, bir saldırı girişimini işaret edebilir.
        """)
    
    df = st.session_state.df
    
    # Saat sütunu kontrolü
    if 'hour' in df.columns:
        # Engellenme sütunu kontrolü
        interrupted_col = None
        if 'transaction.is_interrupted' in df.columns:
            interrupted_col = 'transaction.is_interrupted'
        elif 'is_interrupted' in df.columns:
            interrupted_col = 'is_interrupted'
        
        if interrupted_col:
            try:
                # Zamansal veriye göre gruplama yap
                time_series = df.groupby('hour')[interrupted_col].mean().reset_index()
                
                # Saat değerlerinin tam sayılardan oluştuğunu doğrula
                time_series['hour'] = time_series['hour'].astype(int)
                
                # Saatleri sırala
                time_series = time_series.sort_values('hour')
                
                # İnteraktif çizgi grafiği oluştur
                fig = px.line(
                    time_series, 
                    x='hour', 
                    y=interrupted_col,
                    title='Saat Bazında Engellenmiş İstek Oranı',
                    labels={interrupted_col: 'Engellenme Oranı', 'hour': 'Saat'},
                    markers=True
                )
                
                # Grafiği iyileştir
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 24)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24)]
                    ),
                    yaxis=dict(
                        tickformat='.2%',
                        title='Engellenme Oranı'
                    ),
                    hovermode='x unified'
                )
                
                # Ayrıca bu veriler için bir bar grafiği de ekle
                fig2 = px.bar(
                    time_series,
                    x='hour',
                    y=interrupted_col,
                    title='Saat Bazında Engellenmiş İstek Sayısı',
                    labels={interrupted_col: 'Engellenme Oranı', 'hour': 'Saat'},
                    color=interrupted_col,
                    color_continuous_scale='Viridis'
                )
                
                fig2.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 24)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24)]
                    ),
                    yaxis=dict(
                        tickformat='.2%',
                        title='Engellenme Oranı'
                    )
                )
                
                # Grafikleri göster
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                    
                # İsteğe bağlı, ayrıca saat bazında istek sayısı analizi ekle
                st.subheader("Saat Bazında İstek Sayısı Analizi")
                
                # Her saatteki toplam istek sayısını hesapla
                request_count = df.groupby('hour').size().reset_index(name='istek_sayisi')
                request_count['hour'] = request_count['hour'].astype(int)
                request_count = request_count.sort_values('hour')
                
                # İstek sayısı grafiği
                fig3 = px.bar(
                    request_count,
                    x='hour',
                    y='istek_sayisi',
                    title='Saat Bazında Toplam İstek Sayısı',
                    labels={'istek_sayisi': 'İstek Sayısı', 'hour': 'Saat'},
                    color='istek_sayisi',
                    color_continuous_scale='Viridis'
                )
                
                fig3.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 24)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24)]
                    ),
                    yaxis=dict(
                        title='İstek Sayısı'
                    )
                )
                
                st.plotly_chart(fig3)
                
            except Exception as e:
                st.error(f"Zaman serisi analizi oluşturulurken hata: {str(e)}")
                st.warning("Zaman serisi verilerinde beklenmeyen bir format tespit edildi.")
        else:
            st.warning("Engellenme oranı sütunu bulunamadı.")
    else:
        st.warning("Zaman serisi analizi için 'hour' sütunu bulunamadı.")
        # Zaman sütununu otomatik oluşturmayı dene
        if 'transaction.timestamp' in df.columns:
            st.info("'transaction.timestamp' sütunu bulundu. Saatlik analiz için bu sütunu kullanmak ister misiniz?")
            if st.button("Saat Sütunu Oluştur"):
                try:
                    # Timestamp sütununu datetime formatına çevir
                    df['transaction.timestamp'] = pd.to_datetime(df['transaction.timestamp'], errors='coerce', format='mixed')
                    # Saat bilgisini çıkar
                    df['hour'] = df['transaction.timestamp'].dt.hour
                    st.session_state.df = df
                    st.success("Saat sütunu başarıyla oluşturuldu. Sayfayı yenileyerek analizi görüntüleyebilirsiniz.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Saat sütunu oluşturulurken hata: {str(e)}")
        elif 'timestamp' in df.columns:
            st.info("'timestamp' sütunu bulundu. Saatlik analiz için bu sütunu kullanmak ister misiniz?")
            if st.button("Saat Sütunu Oluştur"):
                try:
                    # Timestamp sütununu datetime formatına çevir
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
                    # Saat bilgisini çıkar
                    df['hour'] = df['timestamp'].dt.hour
                    st.session_state.df = df
                    st.success("Saat sütunu başarıyla oluşturuldu. Sayfayı yenileyerek analizi görüntüleyebilirsiniz.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Saat sütunu oluşturulurken hata: {str(e)}")

def create_uri_distribution():
    st.subheader("En Sık URI'lerin Dağılımı")
    with st.expander("Bu Grafik Ne Anlama Geliyor?"):
        st.write("""
        Bu pasta grafik, loglardaki en sık URI'lerin dağılımını gösterir. En büyük dilimler, 
        sistemde en çok erişilen endpoint'leri temsil eder. Örneğin, `/login` büyük bir dilimse, 
        bu endpoint sıkça hedefleniyor demektir.
        """)
    
    df = st.session_state.df
    
    # URI sütunu kontrolü
    uri_col = None
    if 'transaction.request.uri' in df.columns:
        uri_col = 'transaction.request.uri'
    elif 'request.uri' in df.columns:
        uri_col = 'request.uri'
    elif 'request_uri' in df.columns:
        uri_col = 'request_uri'
        
    if uri_col:
        # URI'leri kısaltma fonksiyonu
        def shorten_uri(uri, max_len=30):
            if isinstance(uri, str) and len(uri) > max_len:
                return uri[:max_len-3] + '...'
            return uri
            
        # URI değerlerini kısalt ve sayımları al
        uri_counts = df[uri_col].value_counts().head(10)
        
        # Okunabilirliği artırmak için URI etiketlerini kısalt
        shortened_uris = {shorten_uri(uri): count for uri, count in uri_counts.items()}
        
        # Kısa açıklamalar için tooltip
        hover_template = '<b>%{label}</b><br>Sayı: %{value}<br>Oran: %{percent}'
        
        fig = px.pie(
            values=list(shortened_uris.values()),
            names=list(shortened_uris.keys()),
            title='En Sık URI\'lerin Dağılımı',
            hover_data=[list(uri_counts.keys())],  # Tam URI'yi hover'da göster
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
    else:
        st.warning("URI dağılımı için gerekli sütun bulunamadı.")

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
    
    n_clusters = st.slider("Meta Küme Sayısı", 2, 25, 5)
    
    # KMeans sınıfını sadece bir kez import et
    from sklearn.cluster import KMeans
    
    try:
        # SOM ağırlıklarını bir kez hesapla ve diğer fonksiyonlarda kullanmak için session_state'e kaydet
        if 'som_weights_reshaped' not in st.session_state or st.session_state.som_weights_reshaped is None:
            # Som ağırlık şeklini ve X verisi boyutunu belirle
            som_weights = st.session_state.som.get_weights()
            weights_shape = som_weights.shape
            
            # Ağırlıkları düzleştir
            grid_size = weights_shape[0]  # SOM ızgara boyutu
            feature_dim = weights_shape[2]  # Özellik boyutu
            
            # Ağırlıkları yeniden şekillendirme
            weights = som_weights.reshape(grid_size * grid_size, feature_dim)
            st.session_state.som_weights_reshaped = weights
        else:
            weights = st.session_state.som_weights_reshaped
        
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
        
        # Meta kümeleri session state'e kaydet
        st.session_state.meta_clusters = meta_clusters
        st.session_state.df_meta = df_meta
        
        # Meta küme dağılımı grafiği
        fig = px.scatter(
            df_meta,
            x='bmu_x',
            y='bmu_y',
            color='meta_cluster',
            hover_data=['meta_cluster'],
            title='Meta Küme Dağılımı',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig)
        
        # Meta küme özet istatistikleri için sütun kontrolü
        agg_dict = {}
        
        # Engellenme oranı sütunu
        if 'transaction.is_interrupted' in df_meta.columns:
            agg_dict['transaction.is_interrupted'] = 'mean'
        elif 'is_interrupted' in df_meta.columns:
            agg_dict['is_interrupted'] = 'mean'
            
        # URI sütunu
        uri_cols = ['transaction.request.uri', 'request.uri', 'request_uri']
        uri_col = next((col for col in uri_cols if col in df_meta.columns), None)
        if uri_col:
            agg_dict[uri_col] = lambda x: x.mode()[0] if not x.empty and len(x.mode()) > 0 else 'Yok'
            
        # Niceleme hatası için özet değerler
        if 'quantization_error' in df_meta.columns:
            agg_dict['quantization_error'] = ['mean', 'std', 'min', 'max']
            
        # Log sayısı için herhangi bir sütun
        count_cols = ['transaction.client_port', 'client_port']
        count_col = next((col for col in count_cols if col in df_meta.columns), df_meta.columns[0])
        agg_dict[count_col] = 'count'
        
        if agg_dict:
            meta_summary = df_meta.groupby('meta_cluster').agg(agg_dict).reset_index()
            
            # Sütun isimleri
            col_names = ['Meta Küme']
            if any(col in agg_dict for col in ['transaction.is_interrupted', 'is_interrupted']):
                col_names.append('Engellenme Oranı')
            if uri_col:
                col_names.append('En Sık URI')
            col_names.append('Log Sayısı')
            
            # Sütun sayısı kontrolü
            if len(meta_summary.columns) == len(col_names):
                meta_summary.columns = col_names
                st.write("#### Meta Küme Özet İstatistikleri")
                st.table(meta_summary)
            else:
                st.warning("Meta küme özeti oluşturulurken sütun sayısı uyumsuzluğu nedeniyle tablo görüntülenemiyor.")
                st.write("Ham Meta Küme Verileri:")
                st.dataframe(meta_summary)
        
        # Meta küme bazında niceleme hatası dağılımı
        st.subheader("Meta Küme Bazında Niceleme Hatası Dağılımı")
        with st.expander("Bu Grafik Ne Anlama Geliyor?"):
            st.write("""
            Bu grafik, her meta-kümedeki logların niceleme hata değerlerinin dağılımını gösterir. 
            Box plot grafiği, her kümenin medyan, çeyreklikler ve aykırı değerlerini gösterir. 
            Yüksek niceleme hatası değerleri, daha anormal logları gösterebilir. Kümeler arasında 
            büyük farklar, farklı davranış modellerini işaret eder.
            """)
            
            if 'quantization_error' in df_meta.columns:
                fig = px.box(
                    df_meta,
                    x='meta_cluster',
                    y='quantization_error',
                    title='Meta Küme Bazında Niceleme Hatası Dağılımı',
                    labels={'meta_cluster': 'Meta Küme', 'quantization_error': 'Niceleme Hatası'},
                    color='meta_cluster'
                )
                st.plotly_chart(fig)
            else:
                st.warning("Niceleme hatası sütunu bulunamadı.")
        
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
        
        # Gelişmiş Görselleştirme Seçenekleri
        st.markdown("---")
        st.markdown("### 🎨 Gelişmiş Görselleştirme Seçenekleri")
        
        visualization_choice = st.selectbox(
            "Görselleştirme Türü Seçin (Temel Analizler)",
            [
                "Standart Meta Kümeleme (Yukarıdaki)",
                "1. Karar Sınırları",
                "2. Büyük Noktalar + Kalın Kenarlık", 
                "3. Küme Başına Ayrı Subplotlar",
                "4. Etiket Farkları Analizi",
                "5. Konveks Hull Sınırları",
                "6. Kapsamlı Karşılaştırma (Hepsi)"
            ],
            help="Küme sınırlarını ve farklarını görmek için farklı görselleştirme yöntemleri",
            key="viz_choice_basic"
        )
        
        if visualization_choice != "Standart Meta Kümeleme (Yukarıdaki)":
            # Advanced visualizations import
            try:
                from advanced_visualizations import (
                    create_decision_boundary_plot,
                    create_large_points_plot, 
                    create_separate_clusters_plot,
                    create_label_differences_plot,
                    create_convex_hull_plot,
                    create_comprehensive_comparison_plot
                )
                
                # Veri hazırlık - Temel analizler için SOM ve Meta kümeleme karşılaştırması
                if 'X' in st.session_state and st.session_state.X is not None:
                    X_viz = st.session_state.X
                    
                    # SOM etiketleri (BMU bazında)
                    if 'bmu_x' in df_meta.columns and 'bmu_y' in df_meta.columns:
                        grid_size = st.session_state.som.get_weights().shape[0]
                        som_labels_viz = df_meta['bmu_x'].values * grid_size + df_meta['bmu_y'].values
                    else:
                        st.error("BMU koordinatları bulunamadı!")
                        return
                    
                    # Meta kümeleme etiketleri
                    meta_labels_viz = df_meta['meta_cluster'].values
                    
                    # NaN kontrol ve temizleme
                    if np.any(np.isnan(X_viz)) or np.any(np.isinf(X_viz)):
                        finite_mask = np.all(np.isfinite(X_viz), axis=1)
                        X_viz = X_viz[finite_mask]
                        som_labels_viz = som_labels_viz[finite_mask]
                        meta_labels_viz = meta_labels_viz[finite_mask]
                    
                    # Görselleştirme seçimi
                    if visualization_choice == "1. Karar Sınırları":
                        st.markdown("#### 🔲 Karar Sınırları Görselleştirmesi")
                        st.info("Meta kümeleme algoritmasının karar bölgelerini arka plan renkleri ile gösterir")
                        
                        fig = create_decision_boundary_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                    
                    elif visualization_choice == "2. Büyük Noktalar + Kalın Kenarlık":
                        st.markdown("#### 🔵 Büyük Noktalar Görselleştirmesi")
                        st.info("Daha net küme aidiyeti gösterimi için büyük noktalar ve kalın kenarlıklar")
                        
                        fig = create_large_points_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                    
                    elif visualization_choice == "3. Küme Başına Ayrı Subplotlar":
                        st.markdown("#### 📊 Küme Başına Ayrı Görselleştirme")
                        st.info("Her kümenin içsel yapısını ayrı grafikte gösterir")
                        
                        fig = create_separate_clusters_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                    
                    elif visualization_choice == "4. Etiket Farkları Analizi":
                        st.markdown("#### 🔍 Etiket Farkları Analizi")
                        st.info("SOM ve Meta kümeleme arasında farklı kümeye atanan logları vurgular")
                        
                        fig = create_label_differences_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                    
                    elif visualization_choice == "5. Konveks Hull Sınırları":
                        st.markdown("#### 🔷 Konveks Hull Sınırları")
                        st.info("Küme sınırlarını çizgilerle gösterir")
                        
                        fig = create_convex_hull_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                    
                    elif visualization_choice == "6. Kapsamlı Karşılaştırma (Hepsi)":
                        st.markdown("#### 🎯 Kapsamlı Karşılaştırma")
                        st.info("Tüm görselleştirme yöntemlerini tek seferde gösterir")
                        
                        fig = create_comprehensive_comparison_plot(X_viz, som_labels_viz, meta_labels_viz)
                        if fig:
                            st.pyplot(fig)
                            
                        # İlave analiz bilgileri
                        with st.expander("Detaylı Analiz Bilgileri"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Toplam Log Sayısı", len(som_labels_viz))
                                st.metric("SOM Küme Sayısı", len(np.unique(som_labels_viz)))
                                
                            with col2:
                                st.metric("Meta Küme Sayısı", len(np.unique(meta_labels_viz)))
                                same_labels = np.sum(som_labels_viz == meta_labels_viz)
                                st.metric("Etiket Benzerliği", f"{same_labels/len(som_labels_viz):.1%}")
                                
                            with col3:
                                som_std = np.std(pd.Series(som_labels_viz).value_counts())
                                meta_std = np.std(pd.Series(meta_labels_viz).value_counts())
                                st.metric("SOM Küme Boyutu Std", f"{som_std:.1f}")
                                st.metric("Meta Küme Boyutu Std", f"{meta_std:.1f}")
                else:
                    st.warning("X veri seti bulunamadı. Önce SOM eğitimi yapılmalı.")
            
            except ImportError as e:
                st.error(f"Gelişmiş görselleştirme modülü yüklenemedi: {str(e)}")
                st.info("advanced_visualizations.py dosyasının mevcut olduğundan emin olun")
            
            except Exception as e:
                st.error(f"Görselleştirme hatası: {str(e)}")
                st.info("Standart meta kümelemeyi kullanın")
    
    except Exception as e:
        st.error(f"Meta kümeleme sırasında bir hata oluştu: {str(e)}")
        import traceback
        st.warning(traceback.format_exc())

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
    
    if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
        selected_neuron = st.selectbox("Bir nöron seçin", options=st.session_state.summary_df['Nöron'])
        bmu_x, bmu_y = map(int, selected_neuron.strip('()').split(','))
        neuron_group = st.session_state.df[(st.session_state.df['bmu_x'] == bmu_x) & (st.session_state.df['bmu_y'] == bmu_y)]
        
        # Hover sütunlarını belirle
        hover_columns = ['bmu_x', 'bmu_y', 'quantization_error']
        
        # URI sütunu
        uri_cols = ['transaction.request.uri', 'request.uri', 'request_uri']
        uri_col = next((col for col in uri_cols if col in st.session_state.df.columns), None)
        if uri_col:
            hover_columns.append(uri_col)
        
        # Engellenme sütunu
        interrupted_cols = ['transaction.is_interrupted', 'is_interrupted']
        interrupted_col = next((col for col in interrupted_cols if col in st.session_state.df.columns), None)
        if interrupted_col:
            hover_columns.append(interrupted_col)
        
        # Client port sütunu
        port_cols = ['transaction.client_port', 'client_port']
        port_col = next((col for col in port_cols if col in st.session_state.df.columns), None)
        if port_col:
            hover_columns.append(port_col)
        
        # Scatter plot
        fig = px.scatter(
            st.session_state.df,
            x='bmu_x',
            y='bmu_y',
            color='quantization_error',
            hover_data=hover_columns,
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
        
        # Seçilen nörondaki logları göster
        st.write(f"**Nöron ({bmu_x},{bmu_y}) Detayları:**")
        st.write(f"Bu nöronda toplam {len(neuron_group)} log bulundu.")
        
        # En sık URI
        if uri_col and len(neuron_group) > 0:
            uri_counts = neuron_group[uri_col].value_counts().head(5)
            st.write("**En Sık URI'ler:**")
            for uri, count in uri_counts.items():
                st.write(f"- {uri}: {count} kez")
        
        # Ortalama niceleme hatası
        avg_qe = neuron_group['quantization_error'].mean() if 'quantization_error' in neuron_group.columns else 0
        st.write(f"**Ortalama Niceleme Hatası:** {avg_qe:.4f}")
        
        # Engellenme oranı
        if interrupted_col and len(neuron_group) > 0:
            block_rate = neuron_group[interrupted_col].mean()
            st.write(f"**Engellenme Oranı:** {block_rate:.2%}")
        
        # Örnek logları göster
        st.write("**Örnek Loglar:**")
        st.dataframe(neuron_group.head(10))
    else:
        st.warning("Henüz nöron analizi yapılamadı. Önce SOM eğitimi tamamlanmalı.")

def handle_anomaly_detection():
    st.subheader("Anomali Tespiti")
    with st.expander("Anomali Tespiti Nedir?"):
        st.write("""
        Bu bölüm, SOM modelini kullanarak potansiyel anomalileri tespit eder.
        Niceleme hatası yüksek olan loglar, veri setindeki genel kalıplara uymayan 'aykırı' 
        kayıtlar olarak kabul edilir. Bunlar, potansiyel güvenlik tehditleri veya sistem anormallikleri olabilir.
        """)
    
    # Anomali eşiği için kullanıcı inputu
    percentile = st.slider("Niceleme Hatası Eşik Yüzdesi", 90, 99, 95, 
                          help="Belirtilen yüzdelik dilimin üzerindeki niceleme hatasına sahip logları anomali olarak işaretle")
    
    if st.session_state.df is not None and 'quantization_error' in st.session_state.df.columns:
        # Belirtilen eşiğe göre anomalileri tespit et
        threshold = np.percentile(st.session_state.df['quantization_error'], percentile)
        anomalies = st.session_state.df[st.session_state.df['quantization_error'] > threshold].copy()
        
        st.write(f"**Tespit Edilen Anomali Sayısı:** {len(anomalies)}")
        st.write(f"**Niceleme Hatası Eşiği:** {threshold:.4f}")
        
        # Anomalilerle ilgili daha detaylı bilgi
        if len(anomalies) > 0:
            # Gösterilecek sütunları belirle
            cols_to_display = ['bmu_x', 'bmu_y', 'quantization_error']
            
            # URI sütunu
            uri_cols = ['transaction.request.uri', 'request.uri', 'request_uri']
            uri_col = next((col for col in uri_cols if col in anomalies.columns), None)
            if uri_col:
                cols_to_display.append(uri_col)
            
            # Client port sütunu
            port_cols = ['transaction.client_port', 'client_port']
            port_col = next((col for col in port_cols if col in anomalies.columns), None)
            if port_col:
                cols_to_display.append(port_col)
                
            # Engellenme sütunu
            interrupted_cols = ['transaction.is_interrupted', 'is_interrupted']
            interrupted_col = next((col for col in interrupted_cols if col in anomalies.columns), None)
            if interrupted_col:
                cols_to_display.append(interrupted_col)
            
            # Anomalileri göster
            st.dataframe(anomalies[cols_to_display].sort_values('quantization_error', ascending=False))
            
            # Anomalilerin dağılımını göster
            fig = px.scatter(
                anomalies,
                x='bmu_x',
                y='bmu_y',
                color='quantization_error',
                hover_data=cols_to_display,
                title='SOM Izgara Anomali Dağılımı',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig)
            
            # Anomalilerin URI dağılımı
            if uri_col:
                uri_counts = anomalies[uri_col].value_counts().head(10)
                fig = px.pie(
                    values=uri_counts.values,
                    names=uri_counts.index,
                    title='Anomali URI\'lerin Dağılımı'
                )
                st.plotly_chart(fig)
                
                st.write("#### En Sık Anomali URI'ler:")
                for uri, count in uri_counts.items():
                    st.write(f"- {uri}: {count} kez")
        else:
            st.info("Seçilen eşiğe göre anomali tespit edilemedi. Eşik değerini düşürebilirsiniz.")
    else:
        st.warning("Veri yüklenmiş ve SOM eğitilmiş olmalıdır.")

def show_som_validation():
    st.subheader("SOM Model Doğrulama")
    
    if st.session_state.som is None or st.session_state.X is None:
        st.warning("SOM modeli henüz eğitilmemiş.")
        return
    
    with st.expander("SOM Doğrulama Nedir?"):
        st.write("""
        1. **Niceleme Hatası**: Her log kaydının SOM'daki en iyi eşleşen birimle (BMU) arasındaki mesafeyi ölçer. 
           Düşük değerler, verinin SOM tarafından iyi temsil edildiğini gösterir.
           
        2. **Topolojik Hata**: Bir veri noktasının BMU'su ve ikinci en iyi eşleşen birimi SOM ızgarasında komşu değilse 
           bir hata oluşur. Bu ölçüm, SOM'un giriş verisinin topolojisini ne kadar iyi koruduğunu gösterir.
           
        3. **Silüet Skoru**: Kümelerin ne kadar iyi ayrıldığını ölçer. Yüksek değerler (1'e yakın), 
           kümelerin iyi tanımlandığını gösterir.
        """)
    
    # SOM model performansını hesapla
    try:
        # Niceleme hatası - doğrudan df'den alıyoruz, çünkü önceden hesaplanmış
        if 'quantization_error' in st.session_state.df.columns:
            total_qe = st.session_state.df['quantization_error'].mean()
        else:
            total_qe = "Hesaplanamadı"
        
        # Topolojik hata
        # Not: SOM ve X özellikleri uyumsuzsa bu kısmı atlıyoruz
        try:
            # Çok fazla veri noktası varsa örnekleme yap
            if len(st.session_state.X) > 1000:
                indices = np.random.choice(len(st.session_state.X), 1000, replace=False)
                sample_X = st.session_state.X[indices]
                topographic_error = calculate_topographic_error(st.session_state.som, sample_X)
            else:
                topographic_error = calculate_topographic_error(st.session_state.som, st.session_state.X)
        except:
            topographic_error = "Hesaplanamadı"
            
        # Silüet skoru hesapla (kümeleme yapılmışsa)
        silhouette = "N/A"
        try:
            if 'df_meta' in st.session_state and st.session_state.df_meta is not None:
                if 'meta_cluster' in st.session_state.df_meta.columns:
                    labels = st.session_state.df_meta['meta_cluster'].values
                    # Silüet skoru en az 2 küme ve her kümede en az 1 eleman gerektirir
                    if len(np.unique(labels)) >= 2:
                        from sklearn.metrics import silhouette_score
                        silhouette = silhouette_score(st.session_state.X, labels)
        except:
            silhouette = "Hesaplanamadı"
        
        # Sonuçları göster
        metrics = {
            "Metrik": ["Niceleme Hatası", "Topolojik Hata", "Silüet Skoru"],
            "Değer": [f"{total_qe:.4f}" if isinstance(total_qe, float) else total_qe, 
                    f"{topographic_error:.4f}" if isinstance(topographic_error, float) else topographic_error, 
                    f"{silhouette:.4f}" if isinstance(silhouette, float) else silhouette],
            "Yorum": [
                "Düşük değer iyi (0'a yakın)",
                "Düşük değer iyi (0'a yakın)",
                "Yüksek değer iyi (1'e yakın)"
            ]
        }
        st.table(pd.DataFrame(metrics))
        
        # Yorumlama
        st.write("#### Sonuçların Yorumlanması")
        
        if isinstance(total_qe, float):
            if total_qe < 0.1:
                st.success("Niceleme Hatası çok düşük. SOM, veriyi çok iyi temsil ediyor.")
            elif total_qe < 0.3:
                st.info("Niceleme Hatası makul düzeyde. SOM, veriyi yeterince iyi temsil ediyor.")
            else:
                st.warning("Niceleme Hatası yüksek. SOM parametrelerini ayarlamak veya eğitim süresini uzatmak faydalı olabilir.")
            
        if isinstance(topographic_error, float):
            if topographic_error < 0.05:
                st.success("Topolojik Hata çok düşük. SOM, veri topolojisini mükemmel koruyor.")
            elif topographic_error < 0.1:
                st.info("Topolojik Hata makul düzeyde. SOM, veri topolojisini iyi koruyor.")
            else:
                st.warning("Topolojik Hata yüksek. SOM ızgara boyutunu arttırmak faydalı olabilir.")
            
        if isinstance(silhouette, float):
            if silhouette > 0.7:
                st.success("Silüet Skoru çok iyi. Kümeler net bir şekilde ayrılmış.")
            elif silhouette > 0.5:
                st.info("Silüet Skoru iyi. Kümeler makul düzeyde ayrılmış.")
            elif silhouette > 0.3:
                st.warning("Silüet Skoru orta düzeyde. Küme sayısını optimize etmek faydalı olabilir.")
            else:
                st.error("Silüet Skoru düşük. Kümeler iyi ayrılmamış. Farklı bir küme sayısı deneyin.")
    
    except Exception as e:
        st.error(f"SOM doğrulama hesaplanırken bir hata oluştu: {str(e)}")
        import traceback
        st.warning(traceback.format_exc())

def calculate_topographic_error(som, data):
    """Topolojik hata hesaplama"""
    error = 0
    for x in data:
        w1 = som.winner(x)
        w2 = find_second_best_matching_unit(som, x)
        if not are_neighbors(som, w1, w2):
            error += 1
    return error / len(data)

def find_second_best_matching_unit(som, x):
    """İkinci en iyi eşleşen birimi bul"""
    # Tüm nöronlar için mesafeleri hesapla
    distances = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
    for i in range(som.get_weights().shape[0]):
        for j in range(som.get_weights().shape[1]):
            distances[i, j] = np.linalg.norm(x - som.get_weights()[i, j])
    
    # En iyi eşleşen birimi bul
    best_matching_unit = som.winner(x)
    
    # En iyi eşleşen birimin mesafesini çok büyük bir sayı yap
    distances[best_matching_unit] = float('inf')
    
    # İkinci en iyi eşleşen birimi bul
    second_best = np.unravel_index(np.argmin(distances), distances.shape)
    
    return second_best

def are_neighbors(som, w1, w2):
    """İki nöronun komşu olup olmadığını kontrol et"""
    return abs(w1[0] - w2[0]) <= 1 and abs(w1[1] - w2[1]) <= 1

def calculate_silhouette_score(X, labels):
    """Silüet skoru hesaplama"""
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels['bmu_x'] * 100 + labels['bmu_y'])

def show_meta_clustering_validation():
    st.subheader("Meta Kümeleme Doğrulama")
    with st.expander("Meta Kümeleme Doğrulama Nedir?"):
        st.write("""
        Bu bölüm, meta kümeleme (K-means ile SOM haritasını kümeleme) sonuçlarının kalitesini değerlendirir:
        
        1. **Silüet Skoru**: Kümelerin ne kadar iyi ayrıldığını ve içlerindeki verilerin ne kadar benzer olduğunu ölçer.
           Değerler -1 ile 1 arasında değişir; yüksek değerler (1'e yakın) daha iyi kümeleme kalitesini gösterir.
           
        2. **Calinski-Harabasz Skoru**: Küme kompaktlığını ve ayrılığını değerlendirir. 
           Yüksek değerler, kümelerin içlerinde kompakt ve birbirlerinden ayrık olduğunu gösterir.
           
        3. **Davies-Bouldin Skoru**: Kümeler arası benzerliği ve küme içi benzerliği ölçer.
           Düşük değerler, daha iyi kümeleme kalitesini gösterir.
        """)
    
    if (st.session_state.som is None or 
        'meta_clusters' not in st.session_state or 
        st.session_state.meta_clusters is None):
        st.warning("Meta kümeleme henüz yapılmamış. Önce Meta Kümeleme Analizi bölümündeki 'Meta Küme Sayısı' belirleyip çalıştırın.")
        return
    
    try:
        # SOM ağırlıkları kontrol et
        if 'som_weights_reshaped' not in st.session_state:
            st.warning("SOM ağırlıkları bulunamadı. Lütfen önce Meta Kümeleme analizini çalıştırın.")
            return
        
        # Meta kümeleme etiketlerini al
        meta_clusters = st.session_state.meta_clusters
        weights = st.session_state.som_weights_reshaped
        
        # Metrikleri hesapla
        if len(np.unique(meta_clusters)) >= 2:  # En az 2 küme olmalı
            try:
                from sklearn.metrics import silhouette_score
                
                # Silhouette skoru hesaplaması için ek kontroller
                unique_clusters = np.unique(meta_clusters)
                min_samples_per_cluster = min([np.sum(meta_clusters == c) for c in unique_clusters])
                
                if min_samples_per_cluster >= 1 and len(unique_clusters) >= 2:
                    silhouette = silhouette_score(weights, meta_clusters)
                else:
                    silhouette = "Hesaplanamadı (Bazı kümeler tek elemanlı veya küme sayısı yetersiz)"
            except Exception as e:
                silhouette = f"Hesaplanamadı: {str(e)}"
                
            try:
                from sklearn.metrics import calinski_harabasz_score
                calinski = calinski_harabasz_score(weights, meta_clusters)
            except:
                calinski = "Hesaplanamadı"
                
            try:
                from sklearn.metrics import davies_bouldin_score
                davies = davies_bouldin_score(weights, meta_clusters)
            except:
                davies = "Hesaplanamadı"
                
            # Sonuçları göster
            metrics = {
                "Metrik": ["Silüet Skoru", "Calinski-Harabasz Skoru", "Davies-Bouldin Skoru"],
                "Değer": [
                    f"{silhouette:.4f}" if isinstance(silhouette, float) else silhouette,
                    f"{calinski:.2f}" if isinstance(calinski, float) else calinski,
                    f"{davies:.4f}" if isinstance(davies, float) else davies
                ],
                "Yorumlama": [
                    "Yüksek değer iyi (-1 ile 1 arası, 1'e yakın = iyi)",
                    "Yüksek değer iyi (bağıl ölçü)",
                    "Düşük değer iyi (0'a yakın = iyi)"
                ]
            }
            st.table(pd.DataFrame(metrics))
            
            # Yorumlama
            if isinstance(silhouette, float):
                st.subheader("Silüet Skoru Yorumu")
                if silhouette > 0.7:
                    st.success(f"Mükemmel kümeleme kalitesi ({silhouette:.4f})")
                elif silhouette > 0.5:
                    st.info(f"İyi kümeleme kalitesi ({silhouette:.4f})")
                elif silhouette > 0.3:
                    st.warning(f"Orta düzeyde kümeleme kalitesi ({silhouette:.4f})")
                else:
                    st.error(f"Zayıf kümeleme kalitesi ({silhouette:.4f}) - Farklı bir küme sayısı deneyin")
                    
            # Görselleştirme
            if st.session_state.df_meta is not None:
                st.subheader("Her Kümeden Log Örnekleri")
                
                unique_clusters = sorted(st.session_state.df_meta['meta_cluster'].unique())
                selected_cluster = st.selectbox("Küme", options=unique_clusters)
                
                cluster_samples = st.session_state.df_meta[st.session_state.df_meta['meta_cluster'] == selected_cluster]
                
                if len(cluster_samples) > 0:
                    st.write(f"Küme {selected_cluster}'de {len(cluster_samples)} log bulundu.")
                    
                    # Görüntülenecek sütunları belirle
                    display_cols = ['bmu_x', 'bmu_y', 'quantization_error']
                    
                    # URI ve diğer önemli sütunları ekle
                    uri_cols = ['transaction.request.uri', 'request.uri', 'request_uri']
                    uri_col = next((col for col in uri_cols if col in cluster_samples.columns), None)
                    if uri_col:
                        display_cols.append(uri_col)
                    
                    # Engellenme durumu
                    interrupted_cols = ['transaction.is_interrupted', 'is_interrupted']
                    interrupted_col = next((col for col in interrupted_cols if col in cluster_samples.columns), None)
                    if interrupted_col:
                        display_cols.append(interrupted_col)
                    
                    st.dataframe(cluster_samples[display_cols].head(10))
                    
                    # Küme içindeki en sık URI'ler
                    if uri_col:
                        uri_counts = cluster_samples[uri_col].value_counts().head(5)
                        st.write("#### Bu Kümedeki En Sık URI'ler:")
                        for uri, count in uri_counts.items():
                            st.write(f"- {uri}: {count} adet")
                    
                    # Kümenin karakteristiği
                    st.write("#### Küme Karakteristiği:")
                    
                    stats = {
                        "Niceleme Hatası (Ortalama)": f"{cluster_samples['quantization_error'].mean():.4f}",
                        "Log Sayısı": f"{len(cluster_samples)}"
                    }
                    
                    if interrupted_col:
                        stats["Engellenme Oranı"] = f"{cluster_samples[interrupted_col].mean():.2%}"
                    
                    stats_df = pd.DataFrame(list(stats.items()), columns=["Metrik", "Değer"])
                    st.table(stats_df)
                else:
                    st.warning(f"Küme {selected_cluster} için log bulunamadı.")
            
        else:
            st.error("En az 2 meta küme olmalıdır. Daha yüksek bir küme sayısı belirleyin.")
    
    except Exception as e:
        st.error(f"Meta kümeleme doğrulama hesaplanırken bir hata oluştu: {str(e)}")
        import traceback
        st.warning(traceback.format_exc())

def show_advanced_analysis():
    if st.session_state.som is None:
        st.info("Önce bir SOM modeli eğitmeniz gerekiyor.")
        return
    
    try:
        # Sekmeleri oluştur
        tabs = st.tabs([
            "Optimal Küme Sayısı",
            "Kümeleme Algoritmaları",
            "Kümeleme Stabilitesi",
            "Boyut İndirgeme",
            "Çapraz Doğrulama",
            "PDF Rapor"
        ])
        
        # Tab 1: Optimal Küme Sayısı
        with tabs[0]:
            st.write("""
            ### Optimal Küme Sayısı Analizi
            Bu analiz, meta kümeleme için en uygun küme sayısını (K) belirler. 
            Dirsek Yöntemi, Silüet Skoru, Calinski-Harabasz Indeksi ve 
            Davies-Bouldin Indeksi gibi çeşitli metrikler kullanılır.
            """)
            
            # Kullanıcıdan max_k değerini al
            max_k = st.slider("Değerlendirilecek maksimum küme sayısı", 5, 30, 15, key="max_k_slider")
            
            # Daha önce hesaplanmamışsa yeni bir analiz yap
            if st.button("Optimal K Analizi Yap"):
                from advanced_clustering import find_optimal_k
                
                with st.spinner("Optimal K değeri hesaplanıyor..."):
                    st.session_state.optimal_k_results = find_optimal_k(max_k=max_k)
                    
                    if st.session_state.optimal_k_results:
                        st.session_state.optimal_k = st.session_state.optimal_k_results.get('optimal_k')
            
            # Sonuçları göster
            if 'optimal_k_results' in st.session_state and st.session_state.optimal_k_results:
                results = st.session_state.optimal_k_results
                
                st.success(f"Önerilen K değeri: **{results.get('optimal_k')}**")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    metrics_table = {
                        "Metrik": ["Dirsek Yöntemi", "Silüet Skoru", "Calinski-Harabasz", "Davies-Bouldin"],
                        "Optimal K": [
                            results.get('elbow_k'),
                            results.get('silhouette_k'),
                            results.get('calinski_k'),
                            results.get('davies_k')
                        ]
                    }
                    st.table(pd.DataFrame(metrics_table))
                
                if 'visualization' in results or 'visualization_base64' in results:
                    vis_key = 'visualization_base64' if 'visualization_base64' in results else 'visualization'
                    show_matplotlib_figure(results[vis_key])
        
        # Tab 2: Kümeleme Algoritmaları
        with tabs[1]:
            st.write("""
            ### Kümeleme Algoritmaları Karşılaştırması
            Farklı kümeleme algoritmalarını (K-means, Hiyerarşik Kümeleme, DBSCAN, HDBSCAN) 
            karşılaştırır. Silüet skoru, Calinski-Harabasz ve Davies-Bouldin metrikleri 
            kullanılarak performansları ölçülür.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_k = st.slider(
                    "Küme Sayısı", 
                    2, 15, 
                    st.session_state.optimal_k if 'optimal_k' in st.session_state and st.session_state.optimal_k else 5
                )
            
            st.subheader("DBSCAN Parametreleri")
            col1, col2 = st.columns(2)
            with col1:
                use_custom_eps = st.checkbox("Özel Epsilon Kullan", False)
                if use_custom_eps:
                    dbscan_eps = st.slider("DBSCAN Epsilon", 0.01, 2.0, 0.5, 0.01)
                else:
                    dbscan_eps = None
            
            with col2:
                dbscan_min_samples = st.slider("DBSCAN Min Samples", 2, 20, 5)
            
            if st.button("Algoritmaları Karşılaştır", key="compare_algo_button"):
                from advanced_clustering import compare_clustering_algorithms
                
                with st.spinner("Kümeleme algoritmaları karşılaştırılıyor..."):
                    st.session_state.alternative_clustering_results = compare_clustering_algorithms(
                        n_clusters=selected_k,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min_samples
                    )
            
            # Önceden hesaplanmış sonuçları göster
            if 'alternative_clustering_results' in st.session_state and st.session_state.alternative_clustering_results is not None:
                comparison_results = st.session_state.alternative_clustering_results
                
                if 'metrics' in comparison_results:
                    st.subheader("Kümeleme Algoritmaları Metrikler Karşılaştırması")
                    st.table(comparison_results['metrics'])
                    
                    if 'visualizations' in comparison_results:
                        if 'Metrikler' in comparison_results['visualizations']:
                            show_matplotlib_figure(comparison_results['visualizations']['Metrikler'])
                    
                    st.subheader("Her Algoritma için Kümeleme Görselleştirmesi")
                    
                    if 'visualizations' in comparison_results:
                        algo_names = [name for name in comparison_results['visualizations'].keys() if name != 'Metrikler']
                        if algo_names:
                            algo_tabs = st.tabs(algo_names)
                            for i, algo_name in enumerate(algo_names):
                                with algo_tabs[i]:
                                    show_matplotlib_figure(comparison_results['visualizations'][algo_name])
        
        # Tab 3: Kümeleme Stabilitesi
        with tabs[2]:
            st.write("""
            ### Kümeleme Stabilitesi Analizi
            Bu analiz, K-means kümelemesinin ne kadar tutarlı sonuçlar verdiğini değerlendirir.
            Farklı başlangıç noktalarıyla K-means çalıştırılır ve sonuçlar karşılaştırılır.
            Yüksek stabilite skoru, daha güvenilir kümeleme demektir.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                stability_k = st.slider(
                    "Stabilite Analizi için K", 
                    2, 15, 
                    st.session_state.optimal_k if 'optimal_k' in st.session_state and st.session_state.optimal_k else 5
                )
            with col2:
                n_runs = st.slider(
                    "Tekrar Sayısı", 
                    3, 10, 5
                )
            
            if st.button("Stabilite Analizi Yap", key="stability_button"):
                st.session_state.stability_results = None  # Önceki sonuçları temizle
                from advanced_clustering import analyze_clustering_stability
                
                with st.spinner("Kümeleme stabilitesi değerlendiriliyor..."):
                    st.session_state.stability_results = analyze_clustering_stability(n_runs=n_runs, n_clusters=stability_k)
            
            if 'stability_results' in st.session_state and st.session_state.stability_results is not None:
                stability_results = st.session_state.stability_results
                
                st.success(f"Ortalama Stabilite Skoru: **{stability_results.get('stability_score', 0):.4f}**")
                st.info("(Skor 1'e yaklaştıkça daha stabil kümeleme anlamına gelir)")
                
                if 'visualization' in stability_results:
                    show_matplotlib_figure(stability_results['visualization'])
        
        # Tab 4: Boyut İndirgeme
        with tabs[3]:
            st.write("""
            ### Boyut İndirgeme Analizi
            PCA, t-SNE ve UMAP gibi farklı boyut indirgeme teknikleriyle SOM nöronlarının
            düşük boyutlu görselleştirmesini yapar. Bu, nöron yapısını daha iyi anlamanıza yardımcı olur.
            """)
            
            dr_k = st.slider(
                "Boyut İndirgeme için K", 
                2, 15, 
                st.session_state.optimal_k if 'optimal_k' in st.session_state and st.session_state.optimal_k else 5
            )
            
            if st.button("Boyut İndirgeme Analizi Yap"):
                from advanced_clustering import dimensionality_reduction_analysis
                
                with st.spinner("Boyut indirgeme analizi yapılıyor..."):
                    st.session_state.dimensionality_reduction_results = dimensionality_reduction_analysis(n_clusters=dr_k)
            
            if 'dimensionality_reduction_results' in st.session_state and st.session_state.dimensionality_reduction_results is not None:
                dr_results = st.session_state.dimensionality_reduction_results
                
                method_names = list(dr_results.keys())
                if method_names:
                    method_tabs = st.tabs(method_names)
                    for i, method_name in enumerate(method_names):
                        with method_tabs[i]:
                            show_matplotlib_figure(dr_results[method_name])
        
        # Tab 5: Çapraz Doğrulama
        with tabs[4]:
            st.write("""
            ### Çapraz Doğrulama Analizi
            Bu analiz, verinin farklı alt kümeleri üzerinde kümeleme yaparak sonuçların
            tutarlılığını değerlendirir. Yüksek silüet skorları, güvenilir kümeleme göstergesidir.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                cv_k = st.slider(
                    "Çapraz Doğrulama için K", 
                    2, 15, 
                    st.session_state.optimal_k if 'optimal_k' in st.session_state and st.session_state.optimal_k else 5
                )
            with col2:
                n_splits = st.slider(
                    "Çapraz Doğrulama Parça Sayısı", 
                    3, 10, 5
                )
            
            if st.button("Çapraz Doğrulama Yap", key="cv_button"):
                st.session_state.cross_validation_results = None  # Önceki sonuçları temizle
                from advanced_clustering import perform_cross_validation_clustering
                
                with st.spinner("Çapraz doğrulama analizi yapılıyor..."):
                    st.session_state.cross_validation_results = perform_cross_validation_clustering(n_splits=n_splits, n_clusters=cv_k)
            
            if 'cross_validation_results' in st.session_state and st.session_state.cross_validation_results is not None:
                cv_results = st.session_state.cross_validation_results
                
                st.success(f"Ortalama Silüet Skoru: **{cv_results.get('avg_silhouette', 0):.4f}**")
                
                if 'fold_results' in cv_results:
                    st.write("#### Parça Bazlı Sonuçlar:")
                    st.table(pd.DataFrame(cv_results['fold_results']).set_index('fold'))
                
                if 'visualization' in cv_results:
                    show_matplotlib_figure(cv_results['visualization'])
        
        # Tab 6: PDF Raporu
        with tabs[5]:
            st.write("""
            ### PDF Raporu Oluştur
            Tüm analizleri içeren detaylı bir PDF raporu oluşturur.
            """)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                report_title = st.text_input("Rapor Başlığı", "SOM Kümeleme Analizi Raporu")
                include_basic = st.checkbox("Temel Analizleri Dahil Et", True)
                include_advanced = st.checkbox("Gelişmiş Analizleri Dahil Et", True)
            
            with col2:
                if st.button("PDF Rapor Oluştur", key="create_pdf_button"):
                    try:
                        from pdf_report import create_pdf_report
                        
                        with st.spinner("PDF raporu oluşturuluyor..."):
                            st.session_state.pdf_report = None  # Eski raporu temizle
                            
                            # Gelişmiş analiz sonuçlarını birleştir
                            if include_advanced:
                                advanced_results = {}
                                
                                # Optimal K analizi
                                if 'optimal_k_results' in st.session_state and st.session_state.optimal_k_results is not None:
                                    advanced_results['optimal_k'] = st.session_state.optimal_k
                                    visualization = st.session_state.optimal_k_results.get('visualization')
                                    if visualization is not None:
                                        if isinstance(visualization, io.BytesIO):
                                            advanced_results['optimal_k_visualization'] = _buffer_to_base64(visualization)
                                        else:
                                            advanced_results['optimal_k_visualization'] = visualization
                                
                                # Kümeleme algoritmaları karşılaştırması
                                if 'alternative_clustering_results' in st.session_state and st.session_state.alternative_clustering_results is not None:
                                    clustering_results = st.session_state.alternative_clustering_results
                                    visualizations = clustering_results.get('visualizations', {})
                                    
                                    # BytesIO nesnelerini base64'e çevir
                                    converted_visualizations = {}
                                    for name, viz in visualizations.items():
                                        if viz is not None:
                                            if isinstance(viz, io.BytesIO):
                                                converted_visualizations[name] = _buffer_to_base64(viz)
                                            else:
                                                converted_visualizations[name] = viz
                                    
                                    advanced_results['clustering_comparison'] = {
                                        'metrics_df': clustering_results.get('metrics'),
                                        'visualizations': converted_visualizations
                                    }
                                
                                # Stabilite analizi
                                if 'stability_results' in st.session_state and st.session_state.stability_results is not None:
                                    stability_results = st.session_state.stability_results.copy()
                                    if 'visualization' in stability_results and stability_results['visualization'] is not None:
                                        if isinstance(stability_results['visualization'], io.BytesIO):
                                            stability_results['visualization'] = _buffer_to_base64(stability_results['visualization'])
                                    advanced_results['stability_analysis'] = stability_results
                                
                                # Boyut indirgeme
                                if 'dimensionality_reduction_results' in st.session_state and st.session_state.dimensionality_reduction_results is not None:
                                    dr_results = st.session_state.dimensionality_reduction_results
                                    converted_dr_results = {}
                                    for method_name, viz in dr_results.items():
                                        if viz is not None:
                                            if isinstance(viz, io.BytesIO):
                                                converted_dr_results[method_name] = _buffer_to_base64(viz)
                                            else:
                                                converted_dr_results[method_name] = viz
                                    
                                    advanced_results['dimensionality_reduction'] = {
                                        'visualizations': converted_dr_results
                                    }
                                
                                # Çapraz doğrulama
                                if 'cross_validation_results' in st.session_state and st.session_state.cross_validation_results is not None:
                                    cv_results = st.session_state.cross_validation_results
                                    cv_visualization = cv_results.get('visualization')
                                    if cv_visualization is not None and isinstance(cv_visualization, io.BytesIO):
                                        cv_visualization = _buffer_to_base64(cv_visualization)
                                    
                                    advanced_results['cross_validation'] = {
                                        'mean_silhouette': cv_results.get('avg_silhouette'),
                                        'std_silhouette': cv_results.get('std_silhouette', 0),
                                        'visualization': cv_visualization
                                    }
                                
                                # Gelişmiş sonuçları session state'e kaydet
                                st.session_state.advanced_analysis_results = advanced_results
                            
                            pdf_output = create_pdf_report(
                                title=report_title,
                                include_basic=include_basic,
                                include_advanced=include_advanced
                            )
                            
                            if pdf_output is not None:
                                st.session_state.pdf_report = pdf_output
                                st.success("PDF raporu başarıyla oluşturuldu!")
                            else:
                                st.error("PDF raporu oluşturulurken bir hata oluştu.")
                    except Exception as e:
                        st.error(f"PDF rapor oluşturma hatası: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())
                
            if 'pdf_report' in st.session_state and st.session_state.pdf_report is not None:
                st.download_button(
                    "PDF Raporunu İndir",
                    st.session_state.pdf_report,
                    "coraza_log_som_report.pdf",
                    "application/pdf"
                )
    
    except Exception as e:
        st.error(f"Gelişmiş analiz sırasında bir hata oluştu: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
