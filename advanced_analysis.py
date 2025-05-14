import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from kneed import KneeLocator
import seaborn as sns
from sklearn.model_selection import KFold

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
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
    weights = st.session_state.som_weights_reshaped
    
    # Farklı çalıştırmalar için kümeleme etiketlerini sakla
    all_labels = []
    
    # Farklı başlangıç noktalarıyla K-means çalıştır
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=i*10)
        labels = kmeans.fit_predict(weights)
        all_labels.append(labels)
    
    # Çalıştırmalar arası benzerliği hesapla (Adjusted Rand Index kullanarak)
    n_runs = len(all_labels)
    stability_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(n_runs):
            if i != j:
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                stability_matrix[i, j] = ari
    
    # Ortalama stabilite skoru
    stability_score = np.mean(stability_matrix)
    
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(stability_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Kümeleme Stabilite Matrisi (Ortalama Skor: {stability_score:.4f})')
    plt.xlabel('Çalıştırma Numarası')
    plt.ylabel('Çalıştırma Numarası')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    
    return {
        'stability_matrix': stability_matrix.tolist(),  # NumPy dizilerini list'e çevir
        'stability_score': float(stability_score),      # NumPy float'ı Python float'a çevir 
        'visualization': _buffer_to_base64(img_buffer)
    }

def compare_clustering_algorithms(n_clusters=5):
    """
    Farklı kümeleme algoritmalarını karşılaştırır:
    - K-means
    - Hiyerarşik Kümeleme
    - DBSCAN
    - HDBSCAN
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
    weights = st.session_state.som_weights_reshaped
    
    # Algoritmaları tanımla
    algorithms = {
        'K-means': KMeans(n_clusters=n_clusters, random_state=42),
        'Hiyerarşik Kümeleme': AgglomerativeClustering(n_clusters=n_clusters)
    }
    
    # DBSCAN için epsilon değerini hesapla
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(weights)
    distances, _ = nn.kneighbors(weights)
    distances = np.sort(distances[:, 1])
    knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    epsilon = distances[knee_locator.knee] if knee_locator.knee else np.median(distances)
    
    # DBSCAN ve HDBSCAN ekle
    algorithms['DBSCAN'] = DBSCAN(eps=epsilon, min_samples=5)
    algorithms['HDBSCAN'] = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    
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
            
            # Görselleştir
            plt.figure(figsize=(10, 8))
            
            # İki boyutlu görselleştirme için PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(weights)
            
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

def find_optimal_k(max_k=15):
    """
    Dirsek yöntemi, silüet skoru ve diğer metrikleri kullanarak 
    optimal küme sayısını belirler.
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
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
    
    # Silüet skoruna göre en iyi k
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    # Calinski-Harabasz'a göre en iyi k
    calinski_k = k_range[np.argmax(calinski_scores)]
    
    # Davies-Bouldin'e göre en iyi k (düşük = iyi)
    davies_k = k_range[np.argmin(davies_scores)]
    
    # Görselleştirme - Dirsek yöntemi
    plt.figure(figsize=(10, 8))
    
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
    
    # BytesIO nesnesini base64 string olarak kaydet
    img_base64 = _buffer_to_base64(img_buffer)
    
    return {
        'optimal_k': optimal_k,
        'elbow_k': elbow_k,
        'silhouette_k': silhouette_k,
        'calinski_k': calinski_k,
        'davies_k': davies_k,
        'inertia_values': [float(x) for x in inertia_values],  # NumPy değerlerini Python'a çevir
        'silhouette_scores': [float(x) for x in silhouette_scores],
        'calinski_scores': [float(x) for x in calinski_scores],
        'davies_scores': [float(x) for x in davies_scores],
        'k_range': list(k_range),
        'visualization_base64': img_base64,  # BytesIO yerine base64 string
        'visualization': img_buffer  # Geriye dönük uyumluluk için
    }

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
        umap_reducer = umap.UMAP(random_state=42)
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

def perform_cross_validation_clustering(n_splits=5, n_clusters=5):
    """
    Çapraz doğrulama benzeri bir yaklaşımla kümelemenin tutarlılığını kontrol eder.
    Veriyi parçalara bölerek, her parçada benzer kümelerin oluşup oluşmadığını değerlendirir.
    """
    if st.session_state.som is None or 'som_weights_reshaped' not in st.session_state:
        st.warning("SOM modeli veya ağırlıkları bulunamadı.")
        return None
    
    weights = st.session_state.som_weights_reshaped
    
    # Veriyi parçalara böl
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Her parça için kümeleme sonuçlarını sakla
    fold_results = []
    silhouette_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(weights)):
        # Eğitim ve doğrulama verilerini al
        X_train, X_val = weights[train_idx], weights[val_idx]
        
        # Eğitim verisiyle kümeleme modeli oluştur
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        
        # Doğrulama verisini etiketle
        val_labels = kmeans.predict(X_val)
        
        # Silüet skorunu hesapla (doğrulama verisi üzerinde)
        if len(np.unique(val_labels)) > 1:  # En az 2 küme olmalı
            silhouette = silhouette_score(X_val, val_labels)
        else:
            silhouette = 0
        
        silhouette_scores.append(silhouette)
        
        # Sonuçları sakla
        fold_results.append({
            'fold': int(fold_idx + 1),
            'train_size': int(len(X_train)),
            'val_size': int(len(X_val)),
            'silhouette': float(silhouette)
        })
    
    # Ortalama silüet skoru
    avg_silhouette = float(np.mean(silhouette_scores))
    
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 6))
    
    plt.bar(
        [f"Parça {i+1}" for i in range(n_splits)],
        silhouette_scores,
        color='skyblue'
    )
    plt.axhline(y=avg_silhouette, color='r', linestyle='--', label=f'Ortalama: {avg_silhouette:.4f}')
    
    plt.title('Çapraz Doğrulama Silüet Skorları')
    plt.xlabel('Veri Parçası')
    plt.ylabel('Silüet Skoru')
    plt.ylim(0, 1)
    plt.legend()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    
    return {
        'fold_results': fold_results,
        'silhouette_scores': [float(x) for x in silhouette_scores],
        'avg_silhouette': avg_silhouette,
        'visualization': _buffer_to_base64(img_buffer)
    } 