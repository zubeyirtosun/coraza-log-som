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
def perform_direct_meta_clustering(X, n_clusters=5, random_state=42, init_method='k-means++', n_init=10, max_iter=300):
    """
    Doğrudan meta kümeleme (K-means) uygular - gelişmiş parametre kontrolü
    """
    try:
        # K-means kümeleme - daha fazla parametre kontrolü
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            algorithm='lloyd'  # Lloyd algoritması daha deterministik
        )
        labels = kmeans.fit_predict(X)
        
        # Metrikleri hesapla
        silhouette = silhouette_score(X, labels)
        
        return {
            'labels': labels,
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
    except Exception as e:
        st.error(f"Meta kümeleme hatası: {str(e)}")
        return None

@st.cache_data
def perform_alternative_clustering(X, algorithm='spectral', n_clusters=5, random_state=42):
    """
    SOM'dan farklı sonuçlar için alternatif kümeleme algoritmaları
    """
    try:
        if algorithm == 'spectral':
            from sklearn.cluster import SpectralClustering
            clustering = SpectralClustering(
                n_clusters=n_clusters, 
                random_state=random_state,
                affinity='rbf',
                gamma=1.0
            )
        elif algorithm == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            # DBSCAN için eps değerini otomatik hesapla
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            eps = np.percentile(distances, 95)
            
            clustering = DBSCAN(eps=eps, min_samples=5)
        else:  # Varsayılan: K-means random init
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                init='random',  # Random initialization
                n_init=20,
                algorithm='lloyd'
            )
        
        labels = clustering.fit_predict(X)
        
        # DBSCAN için gürültü noktalarını işle
        if algorithm == 'dbscan':
            # -1 (gürültü) etiketlerini 0'a çevir
            unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
            labels = np.where(labels == -1, unique_labels, labels)
            n_clusters = unique_labels + 1
        
        # Silüet skoru hesapla (en az 2 küme varsa)
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0.0
        
        return {
            'labels': labels,
            'silhouette_score': silhouette,
            'algorithm': algorithm,
            'n_clusters_found': len(set(labels))
        }
        
    except Exception as e:
        st.error(f"Alternatif kümeleme hatası ({algorithm}): {str(e)}")
        return None

@st.cache_data
def visualize_meta_clusters(X, labels, n_clusters, cluster_centers):
    """
    Meta kümeleme sonuçlarını görselleştirir - gelişmiş hata kontrolü ile
    """
    try:
        # Veri kontrolü - NaN değerleri temizle
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            st.warning("Veri setinde NaN veya sonsuz değerler tespit edildi, temizleniyor...")
            # NaN ve sonsuz değerleri temizle
            finite_mask = np.all(np.isfinite(X), axis=1)
            X_clean = X[finite_mask]
            labels_clean = labels[finite_mask]
            
            if len(X_clean) == 0:
                st.error("Tüm veriler geçersiz, görselleştirme yapılamıyor!")
                return None
        else:
            X_clean = X
            labels_clean = labels
        
        # Küme sayısını gerçek veriden hesapla
        unique_labels = len(np.unique(labels_clean))
        actual_n_clusters = min(n_clusters, unique_labels)
        
        # PCA ile 2 boyuta indir - özel kontrol
        if X_clean.shape[1] == 1:
            # Tek özellik varsa özel işlem
            X_2d = np.column_stack([X_clean.flatten(), np.zeros(len(X_clean))])
            explained_variance = [1.0, 0.0]
        elif X_clean.shape[0] < 2:
            st.error("Görselleştirme için en az 2 veri noktası gerekli!")
            return None
        else:
            # Standart PCA
            pca = PCA(n_components=min(2, X_clean.shape[1]))
            X_2d = pca.fit_transform(X_clean)
            explained_variance = pca.explained_variance_ratio_
            
            # Tek boyutlu sonuç varsa ikinci boyutu sıfır ekle
            if X_2d.shape[1] == 1:
                X_2d = np.column_stack([X_2d, np.zeros(len(X_2d))])
                explained_variance = np.append(explained_variance, 0.0)
        
        # Görselleştirme
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Kümeleme görselleştirmesi
        plt.subplot(221)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c=labels_clean,
                            cmap='viridis', alpha=0.6)
        plt.title('Meta Kümeleme Sonuçları (PCA)')
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.colorbar(scatter, label='Küme')
        
        # 2. Küme boyutları
        plt.subplot(222)
        cluster_sizes = pd.Series(labels_clean).value_counts().sort_index()
        if len(cluster_sizes) > 0:
            bars = plt.bar(cluster_sizes.index, cluster_sizes.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes))))
            plt.title('Küme Boyutları')
            plt.xlabel('Küme')
            plt.ylabel('Veri Sayısı')
            # Çubukların üzerine değerleri yazdır
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'Küme bulunamadı', ha='center', va='center')
            plt.title('Küme Boyutları')
        
        # 3. Silüet analizi
        plt.subplot(223)
        try:
            from sklearn.metrics import silhouette_samples
            if len(np.unique(labels_clean)) > 1:
                silhouette_vals = silhouette_samples(X_clean, labels_clean)
                y_lower = 10
                
                for i in range(actual_n_clusters):
                    cluster_silhouette_vals = silhouette_vals[labels_clean == i]
                    if len(cluster_silhouette_vals) > 0:
                        cluster_silhouette_vals.sort()
                        
                        size_cluster_i = len(cluster_silhouette_vals)
                        y_upper = y_lower + size_cluster_i
                        
                        color = plt.cm.viridis(float(i) / actual_n_clusters)
                        plt.fill_betweenx(np.arange(y_lower, y_upper),
                                         0, cluster_silhouette_vals,
                                         facecolor=color, edgecolor=color, alpha=0.7)
                        
                        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        y_lower = y_upper + 10
                
                plt.title('Silüet Analizi')
                plt.xlabel('Silüet Skoru')
                plt.ylabel('Küme')
            else:
                plt.text(0.5, 0.5, 'Silüet analizi için\nen az 2 küme gerekli', 
                        ha='center', va='center')
                plt.title('Silüet Analizi')
        except Exception as e:
            plt.text(0.5, 0.5, f'Silüet analizi hatası:\n{str(e)[:50]}...', 
                    ha='center', va='center')
            plt.title('Silüet Analizi')
        
        # 4. Küme merkezleri (sadece K-means için)
        plt.subplot(224)
        if cluster_centers is not None and len(cluster_centers) > 0:
            try:
                # Küme merkezlerini de PCA ile dönüştür
                if X_clean.shape[1] == 1:
                    centers_2d = np.column_stack([cluster_centers.flatten(), np.zeros(len(cluster_centers))])
                else:
                    if 'pca' in locals():
                        centers_2d = pca.transform(cluster_centers)
                        if centers_2d.shape[1] == 1:
                            centers_2d = np.column_stack([centers_2d, np.zeros(len(centers_2d))])
                    else:
                        centers_2d = cluster_centers[:, :2] if cluster_centers.shape[1] >= 2 else np.column_stack([cluster_centers, np.zeros(len(cluster_centers))])
                
                plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                           c='red', marker='x', s=200, linewidths=3)
                plt.title('Küme Merkezleri (PCA)')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
            except Exception as e:
                plt.text(0.5, 0.5, f'Merkez görselleştirme hatası:\n{str(e)[:50]}...', 
                        ha='center', va='center')
                plt.title('Küme Merkezleri')
        else:
            plt.text(0.5, 0.5, 'Bu algoritma için\nküme merkezleri mevcut değil\n(Spectral, Agglomerative, DBSCAN)', 
                    ha='center', va='center')
            plt.title('Küme Merkezleri')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Görselleştirme hatası: {str(e)}")
        import traceback
        st.error(f"Detay: {traceback.format_exc()}")
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

@st.cache_data
def perform_neuron_based_meta_clustering(neuron_weights, n_clusters=5, random_state=42):
    """
    SOM nöronlarını kümeleyip logları bu kümelere atar
    """
    try:
        # Nöron ağırlıklarını kümeleme
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            init='k-means++',
            n_init=10,
            algorithm='lloyd'
        )
        neuron_labels = kmeans.fit_predict(neuron_weights)
        
        # Silüet skoru hesapla
        if len(set(neuron_labels)) > 1:
            neuron_silhouette = silhouette_score(neuron_weights, neuron_labels)
        else:
            neuron_silhouette = 0.0
        
        return {
            'neuron_labels': neuron_labels,
            'neuron_silhouette': neuron_silhouette,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
    except Exception as e:
        st.error(f"Nöron bazında meta kümeleme hatası: {str(e)}")
        return None

@st.cache_data  
def map_logs_to_neuron_clusters(bmu_coordinates, neuron_labels, grid_size):
    """
    Logları nöron kümelerine eşler
    """
    try:
        # BMU koordinatlarını nöron indekslerine çevir
        bmu_indices = bmu_coordinates[:, 0] * grid_size + bmu_coordinates[:, 1]
        
        # Log etiketlerini nöron etiketlerine eşle
        log_labels = np.array([neuron_labels[int(idx)] for idx in bmu_indices])
        
        return log_labels
    except Exception as e:
        st.error(f"Log-nöron eşleme hatası: {str(e)}")
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
        
        # Algoritma seçimi
        algorithm_choice = st.selectbox(
            "Kümeleme Algoritması",
            options=['K-means (Standart)', 'K-means (Random Init)', 'Spectral Clustering', 'Agglomerative', 'DBSCAN'],
            help="Farklı algoritmalar farklı kümeleme sonuçları verebilir"
        )
        
        # Kümeleme tipi seçimi
        clustering_type = st.radio(
            "Kümeleme Tipi",
            options=["Log Bazında", "Nöron Bazında"],
            help="Log bazında: Ham log verilerini kümeler | Nöron bazında: SOM nöronlarını kümeler, logları eşler"
        )
        
        if clustering_type == "Nöron Bazında":
            st.info("🧠 **Nöron Bazında Kümeleme**: SOM nöronları kümelenir, sonra loglar nöron kümelerine atanır. Bu yöntem SOM'un öğrendiği yapıyı kullanır.")
        else:
            st.info("📊 **Log Bazında Kümeleme**: Ham log verileri doğrudan kümelenir. SOM'dan bağımsız kümeleme yapar.")
        
        # Küme sayısı seçimi
        n_clusters = st.slider(
            "Küme Sayısı",
            min_value=2,
            max_value=25,
            value=st.session_state.last_n_clusters,
            help="Meta kümeleme için kullanılacak küme sayısı (DBSCAN için etkisiz)",
            disabled=(algorithm_choice == 'DBSCAN'),
            key="meta_n_clusters_slider"
        )
        
        # Gelişmiş K-means parametreleri
        if algorithm_choice.startswith('K-means'):
            with st.expander("Gelişmiş Parametreler"):
                random_state = st.number_input(
                    "Random State", 
                    min_value=1, 
                    max_value=9999, 
                    value=42,
                    help="Farklı değerler farklı başlangıç noktaları verir"
                )
                
                n_init = st.slider(
                    "Başlangıç Deneme Sayısı",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Daha fazla deneme daha iyi sonuç verebilir"
                )
                
                max_iter = st.slider(
                    "Maksimum İterasyon",
                    min_value=100,
                    max_value=1000,
                    value=300,
                    help="Yakınsama için maksimum iterasyon sayısı"
                )
        else:
            random_state = 42
            n_init = 10
            max_iter = 300
        
        # Slider değeri değiştiğinde session state'i güncelle
        if n_clusters != st.session_state.last_n_clusters:
            st.session_state.last_n_clusters = n_clusters

        # Meta kümeleme butonu
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Standart Meta Kümeleme", use_container_width=True, key="meta_apply_standard"):
                with st.spinner(f"{clustering_type} meta kümeleme uygulanıyor..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if clustering_type == "Nöron Bazında":
                        # Nöron bazında kümeleme
                        status_text.text("SOM nöronları kümeleniyor...")
                        progress_bar.progress(25)
                        
                        # SOM nöron ağırlıklarını al
                        if hasattr(st.session_state, 'som') and st.session_state.som is not None:
                            som_weights = st.session_state.som.get_weights()
                            # Reshape: (grid_x, grid_y, features) -> (grid_x*grid_y, features)
                            neuron_weights = som_weights.reshape(-1, som_weights.shape[-1])
                            
                            # Nöron kümeleme
                            neuron_results = perform_neuron_based_meta_clustering(
                                neuron_weights, n_clusters, random_state
                            )
                            
                            if neuron_results:
                                progress_bar.progress(60)
                                status_text.text("Loglar nöron kümelerine eşleniyor...")
                                
                                # BMU koordinatlarını al
                                bmu_coords = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
                                grid_size = int(np.sqrt(len(neuron_weights)))
                                
                                # Logları nöron kümelerine eşle
                                log_labels = map_logs_to_neuron_clusters(
                                    bmu_coords, neuron_results['neuron_labels'], grid_size
                                )
                                
                                if log_labels is not None:
                                    progress_bar.progress(85)
                                    status_text.text("Sonuçlar kaydediliyor...")
                                    
                                    # Log silüet skorunu hesapla
                                    if len(set(log_labels)) > 1:
                                        log_silhouette = silhouette_score(st.session_state.X, log_labels)
                                    else:
                                        log_silhouette = 0.0
                                    
                                    st.session_state.direct_meta_labels = log_labels
                                    st.session_state.direct_meta_metrics = {
                                        'silhouette_score': log_silhouette,
                                        'neuron_silhouette': neuron_results['neuron_silhouette'],
                                        'algorithm': f"Nöron Bazında {algorithm_choice}",
                                        'clustering_type': 'neuron_based'
                                    }
                                    st.session_state.direct_meta_cluster_centers = neuron_results['cluster_centers']
                                    st.session_state.meta_clustering_done = True
                                    st.session_state.last_n_clusters = n_clusters
                                    st.session_state.meta_results_visible = True
                                    
                                    progress_bar.progress(100)
                                    status_text.success("Nöron bazında kümeleme tamamlandı!")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("Log-nöron eşleme başarısız!")
                            else:
                                st.error("Nöron kümeleme başarısız!")
                        else:
                            st.error("SOM modeli bulunamadı! Önce SOM analizi yapın.")
                    
                    else:
                        # Log bazında kümeleme (mevcut işlem)
                        status_text.text("Loglar kümeleniyor...")
                        progress_bar.progress(25)
                        
                        if algorithm_choice == 'K-means (Standart)':
                            results = perform_direct_meta_clustering(
                                st.session_state.X, n_clusters, random_state, 'k-means++', n_init, max_iter
                            )
                        elif algorithm_choice == 'K-means (Random Init)':
                            results = perform_direct_meta_clustering(
                                st.session_state.X, n_clusters, random_state, 'random', n_init, max_iter
                            )
                        else:
                            # Alternatif algoritmalar
                            algo_map = {
                                'Spectral Clustering': 'spectral',
                                'Agglomerative': 'agglomerative', 
                                'DBSCAN': 'dbscan'
                            }
                            results = perform_alternative_clustering(
                                st.session_state.X, algo_map[algorithm_choice], n_clusters, random_state
                            )
                        
                        if results:
                            progress_bar.progress(75)
                            status_text.text("Sonuçlar kaydediliyor...")
                            
                            st.session_state.direct_meta_labels = results['labels']
                            st.session_state.direct_meta_metrics = {
                                'silhouette_score': results['silhouette_score'],
                                'algorithm': f"Log Bazında {algorithm_choice}",
                                'clustering_type': 'log_based'
                            }
                            
                            # Küme merkezleri sadece K-means için mevcut
                            if 'cluster_centers' in results:
                                st.session_state.direct_meta_cluster_centers = results['cluster_centers']
                            else:
                                st.session_state.direct_meta_cluster_centers = None
                                
                            st.session_state.meta_clustering_done = True
                            st.session_state.last_n_clusters = n_clusters
                            st.session_state.meta_results_visible = True
                            
                            progress_bar.progress(100)
                            status_text.success(f"Log bazında {algorithm_choice} tamamlandı!")
                            
                            st.balloons()
                            st.rerun()
        
        with col2:
            if st.button("🔀 Farklı Seed Dene", use_container_width=True, key="meta_apply_random"):
                with st.spinner(f"Farklı başlangıç ile {clustering_type} kümeleme..."):
                    # Rastgele bir seed kullan
                    import time
                    random_seed = int(time.time()) % 10000
                    
                    if clustering_type == "Nöron Bazında":
                        # Nöron bazında kümeleme
                        if hasattr(st.session_state, 'som') and st.session_state.som is not None:
                            som_weights = st.session_state.som.get_weights()
                            neuron_weights = som_weights.reshape(-1, som_weights.shape[-1])
                            
                            # Nöron kümeleme
                            neuron_results = perform_neuron_based_meta_clustering(
                                neuron_weights, n_clusters, random_seed
                            )
                            
                            if neuron_results:
                                # BMU koordinatlarını al
                                bmu_coords = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
                                grid_size = int(np.sqrt(len(neuron_weights)))
                                
                                # Logları nöron kümelerine eşle
                                log_labels = map_logs_to_neuron_clusters(
                                    bmu_coords, neuron_results['neuron_labels'], grid_size
                                )
                                
                                if log_labels is not None:
                                    # Log silüet skorunu hesapla
                                    if len(set(log_labels)) > 1:
                                        log_silhouette = silhouette_score(st.session_state.X, log_labels)
                                    else:
                                        log_silhouette = 0.0
                                    
                                    st.session_state.direct_meta_labels = log_labels
                                    st.session_state.direct_meta_metrics = {
                                        'silhouette_score': log_silhouette,
                                        'neuron_silhouette': neuron_results['neuron_silhouette'],
                                        'algorithm': f"Nöron Bazında {algorithm_choice} (Seed: {random_seed})",
                                        'clustering_type': 'neuron_based'
                                    }
                                    st.session_state.direct_meta_cluster_centers = neuron_results['cluster_centers']
                                    st.session_state.meta_clustering_done = True
                                    st.session_state.meta_results_visible = True
                                    
                                    st.success(f"Nöron bazında Seed {random_seed} ile farklı sonuç elde edildi!")
                                    st.rerun()
                                else:
                                    st.error("Log-nöron eşleme başarısız!")
                            else:
                                st.error("Nöron kümeleme başarısız!")
                        else:
                            st.error("SOM modeli bulunamadı! Önce SOM analizi yapın.")
                    
                    else:
                        # Log bazında kümeleme (mevcut işlem)
                        if algorithm_choice.startswith('K-means'):
                            results = perform_direct_meta_clustering(
                                st.session_state.X, n_clusters, random_seed, 'random', n_init, max_iter
                            )
                        else:
                            algo_map = {
                                'Spectral Clustering': 'spectral',
                                'Agglomerative': 'agglomerative',
                                'DBSCAN': 'dbscan'
                            }
                            results = perform_alternative_clustering(
                                st.session_state.X, algo_map[algorithm_choice], n_clusters, random_seed
                            )
                        
                        if results:
                            st.session_state.direct_meta_labels = results['labels']
                            st.session_state.direct_meta_metrics = {
                                'silhouette_score': results['silhouette_score'],
                                'algorithm': f"Log Bazında {algorithm_choice} (Seed: {random_seed})",
                                'clustering_type': 'log_based'
                            }
                            
                            if 'cluster_centers' in results:
                                st.session_state.direct_meta_cluster_centers = results['cluster_centers']
                            else:
                                st.session_state.direct_meta_cluster_centers = None
                                
                            st.session_state.meta_clustering_done = True
                            st.session_state.meta_results_visible = True
                            
                            st.success(f"Log bazında Seed {random_seed} ile farklı sonuç elde edildi!")
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
            
            # Kullanılan algoritma bilgisi
            if 'algorithm' in st.session_state.direct_meta_metrics:
                st.caption(f"🔧 Algoritma: {st.session_state.direct_meta_metrics['algorithm']}")
        
        with metric_col2:
            # Nöron bazında kümeleme yapıldıysa nöron silüet skorunu göster
            if (st.session_state.direct_meta_metrics.get('clustering_type') == 'neuron_based' and 
                'neuron_silhouette' in st.session_state.direct_meta_metrics):
                st.metric(
                    "Nöron Silüet Skoru", 
                    f"{st.session_state.direct_meta_metrics['neuron_silhouette']:.3f}",
                    help="Nöron kümeleme kalitesi (nöronların ne kadar iyi kümelendiği)"
                )
            else:
                cluster_count = len(np.unique(st.session_state.direct_meta_labels))
                st.metric("Küme Sayısı", cluster_count)
        
        with metric_col3:
            cluster_count = len(np.unique(st.session_state.direct_meta_labels))
            if not (st.session_state.direct_meta_metrics.get('clustering_type') == 'neuron_based' and 
                    'neuron_silhouette' in st.session_state.direct_meta_metrics):
                data_count = len(st.session_state.direct_meta_labels)
                st.metric("Veri Sayısı", data_count)
            else:
                st.metric("Küme Sayısı", cluster_count)
        
        with metric_col4:
            data_count = len(st.session_state.direct_meta_labels)
            if (st.session_state.direct_meta_metrics.get('clustering_type') == 'neuron_based' and 
                'neuron_silhouette' in st.session_state.direct_meta_metrics):
                st.metric("Veri Sayısı", data_count)
            else:
                avg_cluster_size = data_count // len(np.unique(st.session_state.direct_meta_labels))
                st.metric("Ort. Küme Boyutu", avg_cluster_size)
        
        # Nöron bazında kümeleme için ek bilgiler
        if (st.session_state.direct_meta_metrics.get('clustering_type') == 'neuron_based' and 
            'neuron_silhouette' in st.session_state.direct_meta_metrics):
            
            st.markdown("### 🧠 Nöron Bazında Kümeleme Detayları")
            
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                avg_cluster_size = data_count // len(np.unique(st.session_state.direct_meta_labels))
                st.metric("Ort. Küme Boyutu", avg_cluster_size)
            
            with detail_col2:
                # Toplam nöron sayısı
                if hasattr(st.session_state, 'som') and st.session_state.som is not None:
                    som_weights = st.session_state.som.get_weights()
                    total_neurons = som_weights.shape[0] * som_weights.shape[1]
                    st.metric("Toplam Nöron", total_neurons)
            
            with detail_col3:
                # Aktif nöron sayısı (logların kullandığı)
                if hasattr(st.session_state, 'df') and 'bmu_x' in st.session_state.df.columns:
                    unique_neurons = len(st.session_state.df[['bmu_x', 'bmu_y']].drop_duplicates())
                    st.metric("Aktif Nöron", unique_neurons)
            
            st.info("🔍 **Nöron Bazında Kümeleme**: Önce SOM nöronları kümelenir, sonra her log en yakın nöronun kümesine atanır. Bu yaklaşım SOM'un öğrendiği topolojik yapıyı korur.")

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
        
        # Metrik yorumları
        st.markdown("### 📊 Metrik Yorumları:")
        
        # ARI yorumu
        ari = comparison['adjusted_rand_index']
        if ari > 0.7:
            ari_interpretation = "🟢 **Çok Yüksek Benzerlik**: İki yöntem neredeyse aynı kümelemeyi yapıyor"
        elif ari > 0.5:
            ari_interpretation = "🟡 **Orta-Yüksek Benzerlik**: İki yöntem benzer ama farklı kümelemeler yapıyor"
        elif ari > 0.2:
            ari_interpretation = "🟠 **Orta Benzerlik**: İki yöntem bazı noktalarda benzer kümelemeler yapıyor"
        else:
            ari_interpretation = "🔴 **Düşük Benzerlik**: İki yöntem oldukça farklı kümelemeler yapıyor"
        
        # Silüet skorları karşılaştırması
        som_sil = comparison['som_silhouette']
        meta_sil = comparison['meta_silhouette']
        sil_diff = meta_sil - som_sil
        
        if abs(sil_diff) < 0.05:
            sil_interpretation = "⚖️ **Benzer Kalite**: İki yöntem de benzer kümeleme kalitesi gösteriyor"
        elif sil_diff > 0.1:
            sil_interpretation = "📈 **Meta Kümeleme Daha İyi**: Meta kümeleme daha yüksek kalite gösteriyor"
        elif sil_diff < -0.1:
            sil_interpretation = "📉 **SOM Daha İyi**: SOM daha yüksek kümeleme kalitesi gösteriyor"
        else:
            sil_interpretation = "🔄 **Hafif Fark**: Bir yöntem diğerinden biraz daha iyi performans gösteriyor"
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**ARI Analizi:** {ari_interpretation}")
        with col2:
            st.info(f"**Kalite Karşılaştırması:** {sil_interpretation}")
        
        # Özet öneri
        st.markdown("### 💡 Öneri:")
        if ari > 0.7 and abs(sil_diff) < 0.05:
            recommendation = "Her iki yöntem de benzer sonuçlar veriyor. SOM'un yorumlanabilirlik avantajını kullanabilirsiniz."
        elif meta_sil > som_sil + 0.1:
            recommendation = "Meta kümeleme daha iyi kalite gösteriyor. Daha hassas analiz için meta kümelemeyi tercih edin."
        elif som_sil > meta_sil + 0.1:
            recommendation = "SOM daha iyi kümeleme kalitesi gösteriyor. Veri yapısı SOM'a daha uygun olabilir."
        elif ari < 0.3:
            recommendation = "İki yöntem farklı kümeleme stratejileri uyguluyor. Her ikisini de analiz ederek daha kapsamlı görüş elde edebilirsiniz."
        else:
            recommendation = "Her iki yöntem de geçerli sonuçlar veriyor. Analiz amacınıza göre birini seçebilirsiniz."
        
        st.success(f"🎯 {recommendation}")
        
        # Görsel karşılaştırma
        with st.expander("Görsel Karşılaştırma", expanded=True):
            st.markdown("**Karşılaştırma Modu Seçin:**")
            comparison_mode = st.radio(
                "Hangi karşılaştırmayı görmek istiyorsunuz?",
                ["Aynı Koordinat Sistemi (Adil Karşılaştırma)", "Farklı Koordinat Sistemleri (Detaylı Analiz)"],
                help="Aynı koordinat sistemi kümeleme kalitesini karşılaştırmak için, farklı sistemler her metodun kendi özelliklerini görmek için"
            )
            
            st.markdown("---")
            
            if comparison_mode == "Aynı Koordinat Sistemi (Adil Karşılaştırma)":
                st.markdown("**📊 Aynı PCA koordinatlarında her iki kümeleme sonucu:**")
                st.info("Bu görünüm her iki algoritmanın aynı veri üzerinde nasıl kümeleme yaptığını görsel olarak karşılaştırmanızı sağlar.")
            else:
                st.markdown("**🔍 Her algoritmanın kendi doğal koordinat sisteminde sonuçları:**")
                st.info("SOM: Nöron grid koordinatları | Meta: PCA koordinatları")
            
            try:
                # Veri kontrolü - NaN değerleri kontrol et
                X_data = st.session_state.X
                if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
                    st.warning("Veri setinde NaN değerler tespit edildi, temizleniyor...")
                    finite_mask = np.all(np.isfinite(X_data), axis=1)
                    X_clean = X_data[finite_mask]
                    som_labels_clean = st.session_state.som_labels[finite_mask]
                    meta_labels_clean = st.session_state.direct_meta_labels[finite_mask]
                    
                    # DataFrame'den BMU koordinatlarını al
                    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
                        df_clean = st.session_state.df.iloc[finite_mask].copy()
                    else:
                        st.error("SOM BMU koordinatları bulunamadı!")
                        return
                else:
                    X_clean = X_data
                    som_labels_clean = st.session_state.som_labels
                    meta_labels_clean = st.session_state.direct_meta_labels
                    df_clean = st.session_state.df.copy()
                
                if len(X_clean) == 0:
                    st.error("Temizleme sonrası veri kalmadı!")
                    return
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                if comparison_mode == "Aynı Koordinat Sistemi (Adil Karşılaştırma)":
                    # Her iki grafik için de aynı PCA koordinatlarını kullan
                    if X_clean.shape[1] == 1:
                        # Tek özellik varsa özel işlem
                        coords_x = X_clean.flatten()
                        coords_y = np.zeros(len(X_clean))
                        explained_variance = [1.0, 0.0]
                    elif X_clean.shape[0] < 2:
                        st.error("Görselleştirme için en az 2 veri noktası gerekli!")
                        return
                    else:
                        # Standart PCA
                        pca = PCA(n_components=min(2, X_clean.shape[1]))
                        pca_coords = pca.fit_transform(X_clean)
                        explained_variance = pca.explained_variance_ratio_
                        
                        # Tek boyutlu sonuç varsa ikinci boyutu sıfır ekle
                        if pca_coords.shape[1] == 1:
                            coords_x = pca_coords[:, 0]
                            coords_y = np.zeros(len(pca_coords))
                            explained_variance = np.append(explained_variance, 0.0)
                        else:
                            coords_x = pca_coords[:, 0]
                            coords_y = pca_coords[:, 1]
                    
                    # SOM sonuçları - PCA koordinatları
                    scatter1 = ax1.scatter(coords_x, coords_y, 
                                         c=som_labels_clean,
                                         cmap='viridis', alpha=0.7, s=50)
                    ax1.set_title('SOM Kümeleme', fontsize=14, fontweight='bold')
                    ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    plt.colorbar(scatter1, ax=ax1, label='SOM Küme')
                    ax1.grid(True, alpha=0.3)
                    
                    # Meta kümeleme sonuçları - Aynı PCA koordinatları
                    scatter2 = ax2.scatter(coords_x, coords_y, 
                                         c=meta_labels_clean,
                                         cmap='viridis', alpha=0.7, s=50)
                    ax2.set_title('Meta Kümeleme', fontsize=14, fontweight='bold')
                    ax2.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    ax2.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    plt.colorbar(scatter2, ax=ax2, label='Meta Küme')
                    ax2.grid(True, alpha=0.3)
                    
                    # Küme örtüşme analizi
                    st.markdown("### 🔗 Küme Örtüşme Analizi:")
                    try:
                        # Örtüşme matrisi oluştur
                        overlap_matrix = pd.crosstab(
                            som_labels_clean, 
                            meta_labels_clean, 
                            normalize='index'
                        ).round(3)
                        overlap_matrix.index.name = 'SOM Küme'
                        overlap_matrix.columns.name = 'Meta Küme'
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Örtüşme Matrisi (Satır Yüzdesi):**")
                            st.dataframe(
                                overlap_matrix.style.background_gradient(cmap='RdYlBu_r'),
                                use_container_width=True
                            )
                            st.caption("Her hücre: SOM kümesindeki logların ne kadarlık kısmının belirtilen Meta kümede olduğunu gösterir")
                        
                        with col2:
                            # En yüksek örtüşmeleri bul
                            max_overlaps = []
                            for som_cluster in overlap_matrix.index:
                                max_meta = overlap_matrix.loc[som_cluster].idxmax()
                                max_value = overlap_matrix.loc[som_cluster, max_meta]
                                max_overlaps.append({
                                    'SOM': som_cluster,
                                    'Meta': max_meta, 
                                    'Örtüşme': f"{max_value:.1%}"
                                })
                            
                            overlap_df = pd.DataFrame(max_overlaps)
                            st.markdown("**En Güçlü Eşleşmeler:**")
                            st.dataframe(overlap_df, hide_index=True)
                            
                            # Ortalama örtüşme
                            avg_overlap = np.diagonal(overlap_matrix.values).mean()
                            st.metric("Ort. Diagonal Örtüşme", f"{avg_overlap:.1%}")
                            
                    except Exception as e:
                        st.warning(f"Örtüşme analizi yapılamadı: {str(e)}")
                    
                else:
                    # Farklı koordinat sistemleri (önceki versiyon)
                    # SOM için BMU koordinatlarını kullan
                    if 'bmu_x' in df_clean.columns and 'bmu_y' in df_clean.columns:
                        som_x = df_clean['bmu_x'].values
                        som_y = df_clean['bmu_y'].values
                    else:
                        st.error("BMU koordinatları (bmu_x, bmu_y) bulunamadı!")
                        return
                    
                    # Meta kümeleme için PCA koordinatlarını kullan
                    if X_clean.shape[1] == 1:
                        # Tek özellik varsa özel işlem
                        meta_x = X_clean.flatten()
                        meta_y = np.zeros(len(X_clean))
                        explained_variance = [1.0, 0.0]
                    elif X_clean.shape[0] < 2:
                        st.error("Görselleştirme için en az 2 veri noktası gerekli!")
                        return
                    else:
                        # Standart PCA
                        pca = PCA(n_components=min(2, X_clean.shape[1]))
                        meta_coords = pca.fit_transform(X_clean)
                        explained_variance = pca.explained_variance_ratio_
                        
                        # Tek boyutlu sonuç varsa ikinci boyutu sıfır ekle
                        if meta_coords.shape[1] == 1:
                            meta_x = meta_coords[:, 0]
                            meta_y = np.zeros(len(meta_coords))
                            explained_variance = np.append(explained_variance, 0.0)
                        else:
                            meta_x = meta_coords[:, 0]
                            meta_y = meta_coords[:, 1]
                    
                    # SOM sonuçları - BMU grid koordinatları
                    scatter1 = ax1.scatter(som_x, som_y, 
                                         c=som_labels_clean,
                                         cmap='viridis', alpha=0.7, s=50)
                    ax1.set_title('SOM Kümeleme (BMU Grid)', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('BMU X Koordinatı')
                    ax1.set_ylabel('BMU Y Koordinatı')
                    plt.colorbar(scatter1, ax=ax1, label='SOM Küme')
                    ax1.grid(True, alpha=0.3)
                    
                    # Meta kümeleme sonuçları - PCA koordinatları
                    scatter2 = ax2.scatter(meta_x, meta_y, 
                                         c=meta_labels_clean,
                                         cmap='viridis', alpha=0.7, s=50)
                    ax2.set_title('Meta Kümeleme (PCA)', fontsize=14, fontweight='bold')
                    ax2.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    ax2.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    plt.colorbar(scatter2, ax=ax2, label='Meta Küme')
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Karşılaştırma görselleştirme hatası: {str(e)}")
                import traceback
                st.error(f"Detay: {traceback.format_exc()}")
                
                # Hata durumunda basit metin çıktısı
                st.info("Görselleştirme başarısız, metrik bilgiler:")
                st.write(f"- SOM Küme Sayısı: {len(np.unique(st.session_state.som_labels))}")
                st.write(f"- Meta Küme Sayısı: {len(np.unique(st.session_state.direct_meta_labels))}")
                st.write(f"- Veri Boyutu: {st.session_state.X.shape}")
                st.write(f"- ARI Skoru: {comparison['adjusted_rand_index']:.3f}")

    # Eğer hiç analiz yapılmamışsa bilgilendirici mesaj
    if not st.session_state.meta_clustering_done:
        st.markdown("---")
        st.info("""
        **Başlamak için:** Yukarıdaki kontrol panelinden küme sayısını ayarlayın ve "Meta Kümeleme Başlat" butonuna tıklayın.
        
        **Not:** Meta kümeleme analizi için SOM modelinin eğitilmiş olması gerekmektedir.
        """) 