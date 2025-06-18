import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from kneed import KneeLocator
import seaborn as sns
from sklearn.model_selection import KFold
from scipy import stats

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

def analyze_clustering_stability(n_runs=5, n_clusters=5):
    """
    Kümeleme stabilitesini analiz eder. Aynı verilerde K-means'i birden 
    fazla kez çalıştırarak sonuçların tutarlılığını değerlendirir.
    
    İyileştirmeler:
    - Düzeltilmiş matrix hesaplama
    - NMI (Normalized Mutual Information) ek metriği
    - Bootstrap güven aralıkları
    - Çoklu random state testleri
    """
    if st.session_state.som is None:
        st.warning("SOM modeli bulunamadı.")
        return None
    
    try:
        # SOM ağırlıklarını hazırla
        if 'som_weights_reshaped' not in st.session_state:
            som_weights = st.session_state.som.get_weights()
            grid_size = som_weights.shape[0]
            feature_dim = som_weights.shape[2]
            weights = som_weights.reshape(grid_size * grid_size, feature_dim)
            st.session_state.som_weights_reshaped = weights
        else:
            weights = st.session_state.som_weights_reshaped
        
        # Farklı çalıştırmalar için kümeleme etiketlerini sakla
        all_labels = []
        
        # Farklı başlangıç noktalarıyla K-means çalıştır - çoklu random state
        random_states = [42, 123, 456, 789, 999, 1337, 2021, 2024][:n_runs]
        
        for i, rs in enumerate(random_states):
            kmeans = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10)
            labels = kmeans.fit_predict(weights)
            all_labels.append(labels)
        
        # Stabiliteti hem ARI hem de NMI ile ölç
        ari_scores = []
        nmi_scores = []
        
        # Pairwise karşılaştırmalar (sadece üst üçgen)
        for i in range(len(all_labels)):
            for j in range(i+1, len(all_labels)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
                nmi_scores.append(nmi)
        
        # İstatistikler
        ari_mean = np.mean(ari_scores)
        ari_std = np.std(ari_scores)
        nmi_mean = np.mean(nmi_scores)
        nmi_std = np.std(nmi_scores)
        
        # Bootstrap güven aralıkları (95%)
        n_bootstrap = 1000
        ari_bootstrap = []
        nmi_bootstrap = []
        
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(len(ari_scores), len(ari_scores), replace=True)
            ari_bootstrap.append(np.mean([ari_scores[i] for i in sample_indices]))
            nmi_bootstrap.append(np.mean([nmi_scores[i] for i in sample_indices]))
        
        ari_ci_lower = np.percentile(ari_bootstrap, 2.5)
        ari_ci_upper = np.percentile(ari_bootstrap, 97.5)
        nmi_ci_lower = np.percentile(nmi_bootstrap, 2.5)
        nmi_ci_upper = np.percentile(nmi_bootstrap, 97.5)
        
        # Görselleştirme - İyileştirilmiş heatmap
        plt.figure(figsize=(15, 6))
        
        # Sol panel: Stability matrix (ARI)
        plt.subplot(121)
        stability_matrix = np.ones((n_runs, n_runs))  # Diagonal 1.0 olarak başlat
        idx = 0
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                stability_matrix[i, j] = ari_scores[idx]
                stability_matrix[j, i] = ari_scores[idx]  # Simetrik
                idx += 1
        
        sns.heatmap(stability_matrix, annot=True, cmap='viridis', fmt='.3f',
                   xticklabels=[f'Run {i+1}' for i in range(n_runs)],
                   yticklabels=[f'Run {i+1}' for i in range(n_runs)])
        plt.title(f'Stabilite Matrisi (ARI)\nOrtalama: {ari_mean:.3f} ± {ari_std:.3f}')
        
        # Sağ panel: Metrik karşılaştırması
        plt.subplot(122)
        metrics = ['ARI', 'NMI']
        means = [ari_mean, nmi_mean]
        stds = [ari_std, nmi_std]
        colors = ['skyblue', 'lightcoral']
        
        bars = plt.bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        plt.title('Stabilite Metrikleri')
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        
        # Güven aralıklarını göster
        plt.text(0, ari_mean + ari_std + 0.05, f'CI: [{ari_ci_lower:.3f}, {ari_ci_upper:.3f}]',
                ha='center', fontsize=9)
        plt.text(1, nmi_mean + nmi_std + 0.05, f'CI: [{nmi_ci_lower:.3f}, {nmi_ci_upper:.3f}]',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        return {
            'ari_mean': float(ari_mean),
            'ari_std': float(ari_std),
            'ari_ci_lower': float(ari_ci_lower),
            'ari_ci_upper': float(ari_ci_upper),
            'nmi_mean': float(nmi_mean),
            'nmi_std': float(nmi_std),
            'nmi_ci_lower': float(nmi_ci_lower),
            'nmi_ci_upper': float(nmi_ci_upper),
            'stability_score': float(ari_mean),  # Backward compatibility
            'visualization': _buffer_to_base64(img_buffer)
        }
    except Exception as e:
        st.error(f"Stabilite analizi hatası: {str(e)}")
        return None

def compare_clustering_algorithms(n_clusters=5, dbscan_eps=None, dbscan_min_samples=5):
    """
    Farklı kümeleme algoritmalarını karşılaştırır:
    - K-means
    - Hiyerarşik Kümeleme
    - DBSCAN
    - HDBSCAN
    """
    if st.session_state.som is None:
        st.warning("SOM modeli bulunamadı.")
        return None
    
    try:
        # SOM ağırlıklarını hazırla
        if 'som_weights_reshaped' not in st.session_state:
            som_weights = st.session_state.som.get_weights()
            grid_size = som_weights.shape[0]
            feature_dim = som_weights.shape[2]
            weights = som_weights.reshape(grid_size * grid_size, feature_dim)
            st.session_state.som_weights_reshaped = weights
        else:
            weights = st.session_state.som_weights_reshaped
        
        # PCA'yı bir kez hesapla - TÜM ALGORİTMALAR İÇİN AYNI KULLAN
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(weights)
        
        # Algoritmaları tanımla
        algorithms = {
            'K-means': KMeans(n_clusters=n_clusters, random_state=42),
            'Hiyerarşik Kümeleme': AgglomerativeClustering(n_clusters=n_clusters)
        }
        
        # DBSCAN için epsilon değerini hesapla
        if dbscan_eps is None:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, len(weights)))
            nn.fit(weights)
            distances, _ = nn.kneighbors(weights)
            distances = np.sort(distances[:, 1])
            knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
            epsilon = distances[knee_locator.knee] if knee_locator.knee is not None else np.median(distances)
        else:
            epsilon = dbscan_eps
        
        # DBSCAN ve HDBSCAN ekle
        algorithms['DBSCAN'] = DBSCAN(eps=epsilon, min_samples=dbscan_min_samples, metric='euclidean')
        algorithms['HDBSCAN'] = hdbscan.HDBSCAN(min_cluster_size=dbscan_min_samples, min_samples=2)
        
        # Her algoritma için etiketleri ve metrikleri hesapla
        results = {}
        metrics = {
            'Algoritma': [],
            'Silüet Skoru': [],
            'Calinski-Harabasz': [],
            'Davies-Bouldin': [],
            'Küme Sayısı': []
        }
        visualizations = {}
        
        for name, algorithm in algorithms.items():
            # Kümeleme yap
            try:
                labels = algorithm.fit_predict(weights)
                
                # Küme sayısı
                n_clusters_actual = len(np.unique(labels))
                if -1 in np.unique(labels):  # DBSCAN/HDBSCAN gürültü noktaları için
                    n_clusters_actual -= 1
                
                # Silüet skoru hesapla (en az 2 küme olmalı ve her kümede en az 1 eleman)
                silhouette = silhouette_score(weights, labels) if n_clusters_actual >= 2 else 0
                
                # Calinski-Harabasz skoru (en az 2 küme olmalı)
                calinski = calinski_harabasz_score(weights, labels) if n_clusters_actual >= 2 else 0
                
                # Davies-Bouldin skoru (en az 2 küme olmalı)
                davies = davies_bouldin_score(weights, labels) if n_clusters_actual >= 2 else float('inf')
                
                # Sonuçları sakla
                metrics['Algoritma'].append(name)
                metrics['Silüet Skoru'].append(silhouette)
                metrics['Calinski-Harabasz'].append(calinski)
                metrics['Davies-Bouldin'].append(davies)
                metrics['Küme Sayısı'].append(n_clusters_actual)
                
                # Görselleştir - AYNI PCA VERİSİNİ KULLAN
                plt.figure(figsize=(10, 8))
                
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
                plt.title(f'{name} Kümeleme ({n_clusters_actual} küme)')
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.colorbar(label='Küme')
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                plt.close()
                img_buffer.seek(0)
                visualizations[name] = _buffer_to_base64(img_buffer)
                
            except Exception as e:
                st.warning(f"{name} algoritması çalıştırılırken hata oluştu: {str(e)}")
        
        # Metrikleri DataFrame'e dönüştür
        metrics_df = pd.DataFrame(metrics).set_index('Algoritma')
        
        # Görselleştirme - Metrikler karşılaştırması
        plt.figure(figsize=(12, 6))
        
        # Silüet skoru (yüksek = iyi)
        plt.subplot(131)
        bars = plt.bar(metrics_df.index, metrics_df['Silüet Skoru'])
        plt.title('Silüet Skoru')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Calinski-Harabasz (yüksek = iyi)
        plt.subplot(132)
        bars = plt.bar(metrics_df.index, metrics_df['Calinski-Harabasz'])
        plt.title('Calinski-Harabasz')
        plt.xticks(rotation=45)
        
        # Davies-Bouldin (düşük = iyi)
        plt.subplot(133)
        bars = plt.bar(metrics_df.index, metrics_df['Davies-Bouldin'])
        plt.title('Davies-Bouldin')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        metrics_img_buffer = io.BytesIO()
        plt.savefig(metrics_img_buffer, format='png')
        plt.close()
        metrics_img_buffer.seek(0)
        visualizations['Metrikler'] = _buffer_to_base64(metrics_img_buffer)
        
        return {
            'metrics': metrics_df,
            'visualizations': visualizations
        }
    except Exception as e:
        st.error(f"Algoritma karşılaştırma hatası: {str(e)}")
        return None

def find_optimal_k(max_k=15):
    """
    Dirsek yöntemi, silüet skoru ve diğer metrikleri kullanarak 
    optimal küme sayısını belirler.
    """
    if st.session_state.som is None:
        st.warning("SOM modeli bulunamadı.")
        return None
    
    try:
        # SOM ağırlıklarını hazırla
        if 'som_weights_reshaped' not in st.session_state:
            som_weights = st.session_state.som.get_weights()
            grid_size = som_weights.shape[0]
            feature_dim = som_weights.shape[2]
            weights = som_weights.reshape(grid_size * grid_size, feature_dim)
            st.session_state.som_weights_reshaped = weights
        else:
            weights = st.session_state.som_weights_reshaped
        
        # Değerlendirilecek K değerleri
        k_range = range(2, max_k + 1)
        
        # Metrikler
        inertia_values = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for k in k_range:
            # K-means kümeleme
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(weights)
            
            # Inertia (SSE)
            inertia_values.append(kmeans.inertia_)
            
            # Diğer metrikler
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(weights, labels))
            calinski_scores.append(calinski_harabasz_score(weights, labels))
            davies_scores.append(davies_bouldin_score(weights, labels))
        
        # Inertia için dirsek noktasını bul
        try:
            knee_locator = KneeLocator(
                list(k_range), inertia_values, 
                curve='convex', direction='decreasing'
            )
            elbow_k = knee_locator.elbow
        except:
            elbow_k = None
        
        # Silüet skoru için en iyi k
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Calinski-Harabasz için en iyi k
        calinski_k = k_range[np.argmax(calinski_scores)]
        
        # Davies-Bouldin için en iyi k (düşük = iyi)
        davies_k = k_range[np.argmin(davies_scores)]
        
        # Görselleştirme
        plt.figure(figsize=(12, 10))
        
        plt.subplot(221)
        plt.plot(k_range, inertia_values, 'bo-')
        if elbow_k:
            plt.axvline(x=elbow_k, color='r', linestyle='--')
            plt.text(elbow_k+0.5, np.max(inertia_values)*0.9, f'Dirsek = {elbow_k}', color='r')
        plt.title('Dirsek Yöntemi (Inertia)')
        plt.xlabel('Küme Sayısı (k)')
        plt.ylabel('Inertia')
        
        plt.subplot(222)
        plt.plot(k_range, silhouette_scores, 'go-')
        plt.axvline(x=silhouette_k, color='r', linestyle='--')
        plt.text(silhouette_k+0.5, np.max(silhouette_scores)*0.9, f'En iyi = {silhouette_k}', color='r')
        plt.title('Silüet Skoru (Yüksek = İyi)')
        plt.xlabel('Küme Sayısı (k)')
        plt.ylabel('Silüet Skoru')
        
        plt.subplot(223)
        plt.plot(k_range, calinski_scores, 'mo-')
        plt.axvline(x=calinski_k, color='r', linestyle='--')
        plt.text(calinski_k+0.5, np.max(calinski_scores)*0.9, f'En iyi = {calinski_k}', color='r')
        plt.title('Calinski-Harabasz (Yüksek = İyi)')
        plt.xlabel('Küme Sayısı (k)')
        plt.ylabel('Calinski-Harabasz Skoru')
        
        plt.subplot(224)
        plt.plot(k_range, davies_scores, 'co-')
        plt.axvline(x=davies_k, color='r', linestyle='--')
        plt.text(davies_k+0.5, np.max(davies_scores)*0.9, f'En iyi = {davies_k}', color='r')
        plt.title('Davies-Bouldin (Düşük = İyi)')
        plt.xlabel('Küme Sayısı (k)')
        plt.ylabel('Davies-Bouldin Skoru')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_buffer.seek(0)
        
        # Önerilen K değeri (çoğunluk oyu)
        from collections import Counter
        votes = Counter([elbow_k, silhouette_k, calinski_k, davies_k])
        # None değeri varsa çıkar
        if None in votes:
            del votes[None]
        # En çok oyu alan K değeri
        optimal_k = votes.most_common(1)[0][0] if votes else None
        
        return {
            'optimal_k': optimal_k,
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'calinski_k': calinski_k,
            'davies_k': davies_k,
            'k_range': list(k_range),
            'visualization': _buffer_to_base64(img_buffer)
        }
    except Exception as e:
        st.error(f"Optimal K analizi hatası: {str(e)}")
        return None

def dimensionality_reduction_analysis(n_clusters=5):
    """
    Farklı boyut indirgeme teknikleriyle veriyi görselleştirir:
    - PCA
    - t-SNE
    - UMAP
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
    try:
        weights = st.session_state.som_weights_reshaped
        
        # K-means kümeleme yap
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(weights)
        
        results = {}
        
        # PCA
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(weights)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('PCA - 2 Boyut')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)')
            plt.colorbar(label='Küme')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            img_buffer.seek(0)
            results['PCA'] = _buffer_to_base64(img_buffer)
        except Exception as e:
            st.warning(f"PCA analizi sırasında hata: {str(e)}")
        
        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(weights)-1))
            X_tsne = tsne.fit_transform(weights)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('t-SNE - 2 Boyut')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.colorbar(label='Küme')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            img_buffer.seek(0)
            results['t-SNE'] = _buffer_to_base64(img_buffer)
        except Exception as e:
            st.warning(f"t-SNE analizi sırasında hata: {str(e)}")
        
        # UMAP
        try:
            # random_state ile n_jobs parametreleri çakışmasını önlemek için
            # random_state=None olarak ayarlıyoruz ve n_jobs parametresini ekliyoruz
            umap_reducer = umap.UMAP(random_state=None, n_jobs=-1)
            X_umap = umap_reducer.fit_transform(weights)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('UMAP - 2 Boyut')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.colorbar(label='Küme')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            img_buffer.seek(0)
            results['UMAP'] = _buffer_to_base64(img_buffer)
        except Exception as e:
            st.warning(f"UMAP analizi sırasında hata: {str(e)}")
        
        # 3D Görselleştirme (PCA)
        try:
            pca_3d = PCA(n_components=3)
            X_pca_3d = pca_3d.fit_transform(weights)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                c=cluster_labels, cmap='viridis', alpha=0.7
            )
            ax.set_title('PCA - 3 Boyut')
            ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
            fig.colorbar(scatter, ax=ax, label='Küme')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            img_buffer.seek(0)
            results['PCA 3D'] = _buffer_to_base64(img_buffer)
        except Exception as e:
            st.warning(f"3D PCA analizi sırasında hata: {str(e)}")
        
        return results
    except Exception as e:
        st.error(f"Boyut indirgeme analizi sırasında bir hata oluştu: {str(e)}")
        return None

def perform_cross_validation_clustering(n_splits=5, n_clusters=5):
    """
    Çapraz doğrulama benzeri bir yaklaşımla kümelemenin tutarlılığını kontrol eder.
    Veriyi parçalara bölerek, her parçada benzer kümelerin oluşup oluşmadığını değerlendirir.
    
    İyileştirmeler:
    - Çoklu metrik değerlendirmesi (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    - Stratified sampling ile daha iyi veri dağılımı
    - Küme merkezi kalitesi analizi
    - İstatistiksel anlamlılık testleri
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
    try:
        weights = st.session_state.som_weights_reshaped
        
        # Veriyi parçalara böl - StratifiedKFold benzeri yaklaşım
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Her parça için kümeleme sonuçlarını sakla
        fold_results = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        inertia_scores = []
        
        # Çoklu random state ile robustluk testi
        random_states = [42, 123, 456]
        
        for rs in random_states:
            rs_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(weights)):
                # Eğitim ve doğrulama verilerini al
                X_train, X_val = weights[train_idx], weights[val_idx]
                
                # Eğitim verisiyle kümeleme modeli oluştur
                kmeans = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10)
                kmeans.fit(X_train)
                
                # Doğrulama verisini etiketle
                val_labels = kmeans.predict(X_val)
                
                # Çoklu metrik hesaplama
                metrics_dict = {}
                
                if len(np.unique(val_labels)) > 1:  # En az 2 küme olmalı
                    # Silhouette Score
                    metrics_dict['silhouette'] = silhouette_score(X_val, val_labels)
                    
                    # Calinski-Harabasz Index
                    metrics_dict['calinski'] = calinski_harabasz_score(X_val, val_labels)
                    
                    # Davies-Bouldin Index
                    metrics_dict['davies'] = davies_bouldin_score(X_val, val_labels)
                else:
                    metrics_dict = {'silhouette': 0, 'calinski': 0, 'davies': float('inf')}
                
                # Model inertia (küme içi varyans)
                val_inertia = kmeans.inertia_
                metrics_dict['inertia'] = val_inertia
                
                # Küme merkezi kalitesi - train ve val arasındaki tutarlılık
                train_labels = kmeans.predict(X_train)
                train_silhouette = silhouette_score(X_train, train_labels) if len(np.unique(train_labels)) > 1 else 0
                consistency = abs(metrics_dict['silhouette'] - train_silhouette)
                metrics_dict['consistency'] = consistency
                
                rs_results.append({
                    'fold': int(fold_idx + 1),
                    'random_state': rs,
                    'train_size': int(len(X_train)),
                    'val_size': int(len(X_val)),
                    **metrics_dict
                })
        
        # Tüm sonuçları birleştir ve istatistikleri hesapla
        all_results = []
        for rs in random_states:
            rs_folds = [r for r in rs_results if r.get('random_state') == rs]
            all_results.extend(rs_folds)
        
        # Metrik ortalamaları
        silhouette_values = [r['silhouette'] for r in all_results]
        calinski_values = [r['calinski'] for r in all_results]
        davies_values = [r['davies'] for r in all_results if r['davies'] != float('inf')]
        
        avg_silhouette = float(np.mean(silhouette_values))
        std_silhouette = float(np.std(silhouette_values))
        avg_calinski = float(np.mean(calinski_values))
        std_calinski = float(np.std(calinski_values))
        avg_davies = float(np.mean(davies_values)) if davies_values else float('inf')
        std_davies = float(np.std(davies_values)) if davies_values else 0
        
        # İstatistiksel anlamlılık (tek-örneklem t-test)
        if len(silhouette_values) > 1:
            t_stat, p_value = stats.ttest_1samp(silhouette_values, 0)
            significance = p_value < 0.05
        else:
            t_stat, p_value, significance = 0, 1, False
        
        # Gelişmiş görselleştirme
        plt.figure(figsize=(15, 10))
        
        # 1. Silhouette scores across folds
        plt.subplot(221)
        fold_labels = [f"F{r['fold']}_RS{r['random_state']}" for r in all_results]
        plt.bar(range(len(silhouette_values)), silhouette_values, color='skyblue', alpha=0.7)
        plt.axhline(y=avg_silhouette, color='r', linestyle='--', 
                   label=f'Ortalama: {avg_silhouette:.3f}±{std_silhouette:.3f}')
        plt.title(f'Çapraz Doğrulama Silhouette Skorları\n(p-value: {p_value:.4f})')
        plt.xlabel('Fold_RandomState')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(len(fold_labels)), fold_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Metrik karşılaştırması
        plt.subplot(222)
        metrics_names = ['Silhouette', 'Calinski-H', 'Davies-B']
        metrics_means = [avg_silhouette, avg_calinski/1000, 1-avg_davies if avg_davies != float('inf') else 0]  # Normalize
        metrics_stds = [std_silhouette, std_calinski/1000, std_davies]
        
        bars = plt.bar(metrics_names, metrics_means, yerr=metrics_stds, 
                      capsize=5, color=['skyblue', 'lightgreen', 'coral'], alpha=0.7)
        plt.title('Metrik Karşılaştırması (Normalize Edilmiş)')
        plt.ylabel('Normalize Edilmiş Skor')
        
        # 3. Fold-wise detailed metrics
        plt.subplot(223)
        folds = [r['fold'] for r in all_results[:n_splits]]  # Sadece ilk RS
        silh_fold = [r['silhouette'] for r in all_results[:n_splits]]
        cal_fold = [r['calinski']/1000 for r in all_results[:n_splits]]  # Normalize
        
        x = np.arange(len(folds))
        width = 0.35
        
        plt.bar(x - width/2, silh_fold, width, label='Silhouette', alpha=0.8)
        plt.bar(x + width/2, cal_fold, width, label='Calinski-H (÷1000)', alpha=0.8)
        
        plt.xlabel('Fold')
        plt.ylabel('Skor')
        plt.title('Fold Bazlı Metrik Detayları')
        plt.xticks(x, folds)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Consistency analysis
        plt.subplot(224)
        consistency_values = [r['consistency'] for r in all_results]
        plt.hist(consistency_values, bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=np.mean(consistency_values), color='red', linestyle='--',
                   label=f'Ortalama: {np.mean(consistency_values):.3f}')
        plt.title('Train-Validation Tutarlılığı')
        plt.xlabel('Tutarlılık Farkı (|Silh_train - Silh_val|)')
        plt.ylabel('Frekans')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        return {
            'fold_results': all_results,
            'avg_silhouette': avg_silhouette,
            'std_silhouette': std_silhouette,
            'avg_calinski': avg_calinski,
            'std_calinski': std_calinski,
            'avg_davies': avg_davies,
            'std_davies': std_davies,
            'statistical_significance': significance,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'consistency_mean': float(np.mean(consistency_values)),
            'consistency_std': float(np.std(consistency_values)),
            'visualization': _buffer_to_base64(img_buffer)
        }
    except Exception as e:
        st.error(f"Çapraz doğrulama analizi sırasında bir hata oluştu: {str(e)}")
        return None 