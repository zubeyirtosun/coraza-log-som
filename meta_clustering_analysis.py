import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import io
import base64

# Matplotlib font ayarları - emoji uyarılarını azaltmak için
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def perform_direct_meta_clustering(X, n_clusters=5, random_state=42):
    """
    Doğrudan meta kümeleme (K-means) uygular
    """
    try:
        # K-means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        # Metrikleri hesapla
        silhouette = silhouette_score(X, labels)
        
        return {
            'labels': labels,
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_
        }
    except Exception as e:
        st.error(f"Meta kümeleme hatası: {str(e)}")
        return None

@st.cache_data
def visualize_meta_clusters(X, labels, n_clusters, cluster_centers):
    """
    Meta kümeleme sonuçlarını görselleştirir
    """
    try:
        # PCA ile 2 boyuta indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Görselleştirme
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Kümeleme görselleştirmesi
        plt.subplot(221)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c=labels,
                            cmap='viridis', alpha=0.6)
        plt.title('Meta Kümeleme Sonuçları (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, label='Küme')
        
        # 2. Küme boyutları
        plt.subplot(222)
        cluster_sizes = pd.Series(labels).value_counts()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title('Küme Boyutları')
        plt.xlabel('Küme')
        plt.ylabel('Veri Sayısı')
        
        # 3. Silüet analizi
        plt.subplot(223)
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 10
        
        for i in range(n_clusters):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        plt.title('Silüet Analizi')
        plt.xlabel('Silüet Skoru')
        plt.ylabel('Küme')
        
        # 4. Küme merkezleri
        plt.subplot(224)
        centers_2d = pca.transform(cluster_centers)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.title('Küme Merkezleri (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Görselleştirme hatası: {str(e)}")
        return None

@st.cache_data
def compare_clustering_approaches(X, som_neurons, meta_labels):
    """
    SOM ve meta kümeleme sonuçlarını karşılaştırır
    """
    try:
        if som_neurons is None or meta_labels is None:
            st.error("Karşılaştırma için hem SOM hem de meta kümeleme sonuçları gerekli!")
            return None
            
        # SOM etiketlerini al
        som_labels = st.session_state.som_labels
        
        # Veri boyutlarını kontrol et
        if len(som_labels) != len(meta_labels):
            st.error(f"Veri boyutları uyumsuz: SOM ({len(som_labels)}) vs Meta ({len(meta_labels)})")
            return None
            
        # Adjusted Rand Index hesapla
        ari = adjusted_rand_score(som_labels, meta_labels)
        
        # Silüet skorlarını hesapla
        som_silhouette = silhouette_score(X, som_labels)
        meta_silhouette = silhouette_score(X, meta_labels)
        
        return {
            'adjusted_rand_index': ari,
            'som_silhouette': som_silhouette,
            'meta_silhouette': meta_silhouette
        }
    except Exception as e:
        st.error(f"Karşılaştırma hatası: {str(e)}")
        return None

def show_meta_clustering_analysis():
    # CSS stilleri - sadece bu sayfa için
    st.markdown("""
    <style>
        .meta-control-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .meta-control-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: #1e40af;
        }
        
        .meta-control-card h4 {
            margin: 0;
            color: #0f172a !important;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }
        
        .meta-hero {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .meta-hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1e40af, #3b82f6);
        }
        
        .meta-hero h1 {
            color: #0f172a !important;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            font-family: 'Inter', sans-serif;
        }
        
        .meta-hero p {
            color: #475569;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        
        .status-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .status-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Session state başlangıç değerleri - her çağrıda kontrol et
    defaults = {
        "meta_clustering_done": False,
        "direct_meta_labels": None,
        "direct_meta_metrics": None,
        "last_n_clusters": 5,
        "direct_meta_cluster_centers": None,
        "comparison_type": "Doğrudan K-means vs SOM",
        "meta_results_visible": False,
        "comparison_results": None
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Hero section
    st.markdown("""
        <div class="meta-hero">
            <h1>Meta Kümeleme Analizi</h1>
            <p>K-means algoritmasıyla doğrudan kümeleme ve SOM karşılaştırması</p>
        </div>
    """, unsafe_allow_html=True)

    # Ana kontrol paneli
    st.markdown("### Kontrol Paneli")
    
    control_col1, control_col2 = st.columns([1, 1])
    
    with control_col1:
        st.markdown("""
            <div class="meta-control-card">
                <h4>Kümeleme Ayarları</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Küme sayısı seçimi
        n_clusters = st.slider(
            "Küme Sayısı",
            min_value=2,
            max_value=10,
            value=st.session_state.last_n_clusters,
            help="Meta kümeleme için kullanılacak küme sayısı",
            key="meta_n_clusters_slider"
        )
        
        # Slider değeri değiştiğinde session state'i güncelle
        if n_clusters != st.session_state.last_n_clusters:
            st.session_state.last_n_clusters = n_clusters

        # Meta kümeleme butonu
        if st.button("Meta Kümeleme Başlat", type="primary", use_container_width=True, key="meta_apply_button"):
            with st.spinner("Meta kümeleme uygulanıyor..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("K-means algoritması çalışıyor...")
                progress_bar.progress(25)
                
                results = perform_direct_meta_clustering(st.session_state.X, n_clusters)
                
                if results:
                    progress_bar.progress(75)
                    status_text.text("Sonuçlar kaydediliyor...")
                    
                    st.session_state.direct_meta_labels = results['labels']
                    st.session_state.direct_meta_metrics = {
                        'silhouette_score': results['silhouette_score']
                    }
                    st.session_state.meta_clustering_done = True
                    st.session_state.last_n_clusters = n_clusters
                    st.session_state.direct_meta_cluster_centers = results['cluster_centers']
                    st.session_state.meta_results_visible = True
                    
                    progress_bar.progress(100)
                    status_text.success("Meta kümeleme tamamlandı!")
                    
                    st.balloons()
                    st.rerun()

    with control_col2:
        st.markdown("""
            <div class="meta-control-card">
                <h4>Karşılaştırma Ayarları</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Karşılaştırma türü seçimi
        comparison_options = ["Doğrudan K-means vs SOM", "SOM vs SOM Meta Kümeleme"]
        
        selected_comparison = st.selectbox(
            "Karşılaştırma Türü",
            options=comparison_options,
            index=comparison_options.index(st.session_state.comparison_type),
            help="Hangi kümeleme yaklaşımlarını karşılaştırmak istediğinizi seçin",
            key="meta_comparison_select"
        )
        
        # Seçim değiştiğinde session state'i güncelle
        if selected_comparison != st.session_state.comparison_type:
            st.session_state.comparison_type = selected_comparison

        # Karşılaştırma butonu
        if st.button("Karşılaştır", type="secondary", use_container_width=True, key="meta_compare_button"):
            if not st.session_state.get("som_done", False):
                st.error("Karşılaştırma için önce SOM analizi yapılmalı!")
            elif not st.session_state.meta_clustering_done:
                st.error("Karşılaştırma için önce meta kümeleme yapılmalı!")
            else:
                with st.spinner("Karşılaştırma yapılıyor..."):
                    comparison = compare_clustering_approaches(
                        st.session_state.X,
                        st.session_state.meta_clusters,
                        st.session_state.direct_meta_labels
                    )
                    if comparison:
                        st.session_state.comparison_results = comparison
                        st.success("Karşılaştırma tamamlandı!")
                        st.rerun()

    # Durum kartları
    if st.session_state.som_done or st.session_state.meta_clustering_done:
        st.markdown("### Durum Özeti")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            som_status = "Tamamlandı" if st.session_state.get("som_done", False) else "Beklemede"
            st.metric("SOM Analizi", som_status)
        
        with status_col2:
            meta_status = "Tamamlandı" if st.session_state.meta_clustering_done else "Beklemede"
            st.metric("Meta Kümeleme", meta_status)
        
        with status_col3:
            comp_status = "Tamamlandı" if st.session_state.comparison_results else "Beklemede"
            st.metric("Karşılaştırma", comp_status)

    # Sonuçları göster
    if st.session_state.meta_clustering_done and st.session_state.direct_meta_labels is not None:
        st.markdown("---")
        st.markdown("### Meta Kümeleme Sonuçları")
        
        # Ana metrikler
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Silüet Skoru", 
                f"{st.session_state.direct_meta_metrics['silhouette_score']:.3f}",
                help="Kümeleme kalitesi göstergesi (0-1 arası, yüksek=iyi)"
            )
        
        with metric_col2:
            cluster_count = len(np.unique(st.session_state.direct_meta_labels))
            st.metric("Küme Sayısı", cluster_count)
        
        with metric_col3:
            data_count = len(st.session_state.direct_meta_labels)
            st.metric("Veri Sayısı", data_count)
        
        with metric_col4:
            avg_cluster_size = data_count // cluster_count
            st.metric("Ort. Küme Boyutu", avg_cluster_size)

        # Görselleştirmeler
        with st.expander("Görselleştirmeler", expanded=True):
            viz_col1, viz_col2 = st.columns([2, 1])
            
            with viz_col1:
                meta_plot = visualize_meta_clusters(
                    st.session_state.X,
                    st.session_state.direct_meta_labels,
                    st.session_state.last_n_clusters,
                    st.session_state.direct_meta_cluster_centers
                )
                if meta_plot:
                    st.pyplot(meta_plot)
            
            with viz_col2:
                # Küme dağılımı
                cluster_sizes = pd.Series(st.session_state.direct_meta_labels).value_counts().sort_index()
                st.markdown("**Küme Dağılımı:**")
                st.bar_chart(cluster_sizes)
                
                # Küme bilgileri
                st.markdown("**Küme Detayları:**")
                for i, size in cluster_sizes.items():
                    percentage = (size / len(st.session_state.direct_meta_labels)) * 100
                    st.write(f"Küme {i}: {size} veri ({percentage:.1f}%)")

        # Küme özeti tablosu
        try:
            cluster_means = pd.DataFrame(st.session_state.X).groupby(st.session_state.direct_meta_labels).mean()
            feature_names = [f"Özellik {i+1}" for i in range(st.session_state.X.shape[1])]
            cluster_means.columns = feature_names
            
            with st.expander("Küme Özeti Tablosu"):
                st.dataframe(
                    cluster_means.round(3), 
                    use_container_width=True,
                    height=300
                )
        except Exception as e:
            st.warning("Küme özeti oluşturulurken bir hata oluştu.")

    # Karşılaştırma sonuçları
    if st.session_state.comparison_results is not None:
        st.markdown("---")
        st.markdown("### Karşılaştırma Sonuçları")
        
        comparison = st.session_state.comparison_results
        
        # Karşılaştırma metrikleri
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            ari_color = "green" if comparison['adjusted_rand_index'] > 0.5 else "orange" if comparison['adjusted_rand_index'] > 0.2 else "red"
            st.metric(
                "Adjusted Rand Index", 
                f"{comparison['adjusted_rand_index']:.3f}",
                help="Kümeleme benzerliği (0-1 arası, yüksek=benzer)"
            )
        
        with comp_col2:
            st.metric(
                "SOM Silüet Skoru", 
                f"{comparison['som_silhouette']:.3f}",
                help="SOM kümeleme kalitesi"
            )
        
        with comp_col3:
            st.metric(
                "Meta Silüet Skoru", 
                f"{comparison['meta_silhouette']:.3f}",
                help="Meta kümeleme kalitesi"
            )
        
        # Görsel karşılaştırma
        with st.expander("Görsel Karşılaştırma", expanded=True):
            st.markdown("Aşağıdaki grafikler SOM ve Meta Kümeleme sonuçlarını yan yana göstermektedir:")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # PCA ile 2 boyuta indir
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(st.session_state.X)
            
            # SOM sonuçları
            scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], 
                                 c=st.session_state.som_labels,
                                 cmap='viridis', alpha=0.7, s=50)
            ax1.set_title('SOM Kümeleme', fontsize=14, fontweight='bold')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter1, ax=ax1, label='Küme')
            ax1.grid(True, alpha=0.3)
            
            # Meta kümeleme sonuçları
            scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                                 c=st.session_state.direct_meta_labels,
                                 cmap='viridis', alpha=0.7, s=50)
            ax2.set_title('Meta Kümeleme', fontsize=14, fontweight='bold')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter2, ax=ax2, label='Küme')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

    # Eğer hiç analiz yapılmamışsa bilgilendirici mesaj
    if not st.session_state.meta_clustering_done:
        st.markdown("---")
        st.info("""
        **Başlamak için:** Yukarıdaki kontrol panelinden küme sayısını ayarlayın ve "Meta Kümeleme Başlat" butonuna tıklayın.
        
        **Not:** Meta kümeleme analizi için SOM modelinin eğitilmiş olması gerekmektedir.
        """) 