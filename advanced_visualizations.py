import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

@st.cache_data
def create_decision_boundary_plot(X, som_labels, meta_labels):
    """
    Küme sınırlarını ve karar bölgelerini görselleştiren plot oluşturur
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Grid oluştur karar sınırları için
        h = 0.02  # Mesh step size
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Meta kümeleme karar sınırları
        # K-means modeli oluştur
        n_clusters = len(np.unique(meta_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Mesh noktaları için tahmin yap
        mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
        mesh_points = pca.inverse_transform(mesh_points_2d)
        meta_mesh_labels = kmeans.predict(mesh_points)
        meta_mesh_labels = meta_mesh_labels.reshape(xx.shape)
        
        # Karar bölgelerini çiz
        ax1.contourf(xx, yy, meta_mesh_labels, alpha=0.3, cmap='tab10')
        
        # Veri noktalarını üzerine çiz
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=som_labels, 
                             cmap='tab10', edgecolors='black', linewidth=0.5, s=50, alpha=0.8)
        ax1.set_title('SOM Kümeleme\n(Meta Karar Sınırları Üzerinde)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter1, ax=ax1, label='SOM Küme')
        ax1.grid(True, alpha=0.3)
        
        # Meta kümeleme karar sınırları
        ax2.contourf(xx, yy, meta_mesh_labels, alpha=0.3, cmap='tab10')
        
        # Veri noktalarını üzerine çiz
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=meta_labels, 
                             cmap='tab10', edgecolors='black', linewidth=0.5, s=50, alpha=0.8)
        ax2.set_title('Meta Kümeleme\n(Karar Sınırları)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter2, ax=ax2, label='Meta Küme')
        ax2.grid(True, alpha=0.3)
        
        # Küme merkezlerini göster
        if hasattr(kmeans, 'cluster_centers_'):
            centers_2d = pca.transform(kmeans.cluster_centers_)
            ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='x', s=300, linewidths=3, label='Küme Merkezleri')
            ax2.legend()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Karar sınırları görselleştirme hatası: {str(e)}")
        return None

@st.cache_data  
def create_large_points_plot(X, som_labels, meta_labels):
    """
    Büyük noktalar ile net küme görselleştirmesi
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # SOM Kümeleme - Büyük noktalar
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=som_labels, 
                             cmap='tab10', s=150, alpha=0.8, edgecolors='black', linewidth=1)
        ax1.set_title('SOM Kümeleme\n(Büyük Noktalar)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter1, ax=ax1, label='SOM Küme')
        ax1.grid(True, alpha=0.3)
        
        # Meta Kümeleme - Büyük noktalar  
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=meta_labels, 
                             cmap='tab10', s=150, alpha=0.8, edgecolors='black', linewidth=1)
        ax2.set_title('Meta Kümeleme\n(Büyük Noktalar)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter2, ax=ax2, label='Meta Küme')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Büyük noktalar görselleştirme hatası: {str(e)}")
        return None

@st.cache_data
def create_separate_clusters_plot(X, som_labels, meta_labels):
    """
    Her kümeyi ayrı subplot'ta gösteren görselleştirme
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Küme sayılarını al
        som_unique = np.unique(som_labels)
        meta_unique = np.unique(meta_labels)
        max_clusters = max(len(som_unique), len(meta_unique))
        
        # Alt grafik sayısını hesapla
        cols = min(4, max_clusters)
        rows = (max_clusters + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 5*rows))
        
        # SOM kümeleri
        for i, cluster in enumerate(som_unique):
            plt.subplot(2*rows, cols, i+1)
            mask = som_labels == cluster
            
            # Tüm noktaları açık gri göster
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=30, alpha=0.3)
            
            # Bu kümeyi vurgula
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=f'C{cluster}', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.title(f'SOM Küme {cluster}\n({np.sum(mask)} nokta)', fontsize=12, fontweight='bold')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.grid(True, alpha=0.3)
        
        # Meta kümeleri
        for i, cluster in enumerate(meta_unique):
            plt.subplot(2*rows, cols, len(som_unique)+cols+i+1)
            mask = meta_labels == cluster
            
            # Tüm noktaları açık gri göster
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=30, alpha=0.3)
            
            # Bu kümeyi vurgula
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=f'C{cluster}', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.title(f'Meta Küme {cluster}\n({np.sum(mask)} nokta)', fontsize=12, fontweight='bold')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Ayrı kümeler görselleştirme hatası: {str(e)}")
        return None

@st.cache_data
def create_label_differences_plot(X, som_labels, meta_labels):
    """
    Etiket farklarını gösteren görselleştirme
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # 1. SOM Kümeleme
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=som_labels, 
                             cmap='tab10', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax1.set_title('SOM Kümeleme', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter1, ax=ax1, label='SOM Küme')
        ax1.grid(True, alpha=0.3)
        
        # 2. Meta Kümeleme
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=meta_labels, 
                             cmap='tab10', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax2.set_title('Meta Kümeleme', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter2, ax=ax2, label='Meta Küme')
        ax2.grid(True, alpha=0.3)
        
        # 3. Fark Analizi
        # Etiket farkları (aynı küme=0, farklı küme=1)
        diff_labels = (som_labels != meta_labels).astype(int)
        
        scatter3 = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=diff_labels, 
                             cmap='RdYlBu_r', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax3.set_title('Etiket Farkları\n(Kırmızı=Farklı, Mavi=Aynı)', fontsize=14, fontweight='bold')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter3, ax=ax3, label='Fark (0=Aynı, 1=Farklı)')
        ax3.grid(True, alpha=0.3)
        
        # İstatistikler
        same_count = np.sum(diff_labels == 0)
        diff_count = np.sum(diff_labels == 1)
        total_count = len(diff_labels)
        
        ax3.text(0.02, 0.98, f'Aynı: {same_count} ({same_count/total_count:.1%})\nFarklı: {diff_count} ({diff_count/total_count:.1%})', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Etiket farkları görselleştirme hatası: {str(e)}")
        return None

@st.cache_data
def create_convex_hull_plot(X, som_labels, meta_labels):
    """
    Küme sınırlarını konveks hull ile gösteren görselleştirme
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # SOM Hull'ları
        unique_som = np.unique(som_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_som)))
        
        for i, cluster in enumerate(unique_som):
            mask = som_labels == cluster
            points = X_2d[mask]
            
            # Noktaları çiz
            ax1.scatter(points[:, 0], points[:, 1], 
                       c=[colors[i]], s=80, alpha=0.7, label=f'SOM Küme {cluster}')
            
            # ConvexHull çiz (en az 3 nokta gerekli)
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax1.plot(points[simplex, 0], points[simplex, 1], 
                               color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
                except:
                    pass
        
        ax1.set_title('SOM Kümeleme\n(Konveks Hull Sınırları)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Meta Hull'ları  
        unique_meta = np.unique(meta_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_meta)))
        
        for i, cluster in enumerate(unique_meta):
            mask = meta_labels == cluster
            points = X_2d[mask]
            
            # Noktaları çiz
            ax2.scatter(points[:, 0], points[:, 1], 
                       c=[colors[i]], s=80, alpha=0.7, label=f'Meta Küme {cluster}')
            
            # ConvexHull çiz
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax2.plot(points[simplex, 0], points[simplex, 1], 
                               color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
                except:
                    pass
        
        ax2.set_title('Meta Kümeleme\n(Konveks Hull Sınırları)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Konveks hull görselleştirme hatası: {str(e)}")
        return None

@st.cache_data
def create_comprehensive_comparison_plot(X, som_labels, meta_labels):
    """
    Kapsamlı karşılaştırma görselleştirmesi (6 panel)
    """
    try:
        # PCA ile 2D'ye indir
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. SOM Kümeleme - Standart
        plt.subplot(3, 3, 1)
        scatter1 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=som_labels, 
                             cmap='tab10', s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        plt.title('SOM Kümeleme\n(Standart)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter1, label='SOM Küme')
        plt.grid(True, alpha=0.3)
        
        # 2. Meta Kümeleme - Standart
        plt.subplot(3, 3, 2)
        scatter2 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=meta_labels, 
                             cmap='tab10', s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        plt.title('Meta Kümeleme\n(Standart)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter2, label='Meta Küme')
        plt.grid(True, alpha=0.3)
        
        # 3. Fark Analizi
        plt.subplot(3, 3, 3)
        diff_labels = (som_labels != meta_labels).astype(int)
        scatter3 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=diff_labels, 
                             cmap='RdYlBu_r', s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        plt.title('Etiket Farkları\n(Kırmızı=Farklı)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter3, label='Fark')
        plt.grid(True, alpha=0.3)
        
        # 4. SOM - Büyük noktalar
        plt.subplot(3, 3, 4)
        scatter4 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=som_labels, 
                             cmap='tab10', s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
        plt.title('SOM Kümeleme\n(Büyük Noktalar)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        # 5. Meta - Büyük noktalar
        plt.subplot(3, 3, 5)
        scatter5 = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=meta_labels, 
                             cmap='tab10', s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
        plt.title('Meta Kümeleme\n(Büyük Noktalar)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        # 6. İstatistikler
        plt.subplot(3, 3, 6)
        
        # Küme sayıları
        som_n_clusters = len(np.unique(som_labels))
        meta_n_clusters = len(np.unique(meta_labels))
        
        # Etiket benzerlikleri
        same_count = np.sum(som_labels == meta_labels)
        total_count = len(som_labels)
        similarity_ratio = same_count / total_count
        
        # Küme boyutu dağılımları
        som_sizes = pd.Series(som_labels).value_counts().sort_index()
        meta_sizes = pd.Series(meta_labels).value_counts().sort_index()
        
        # Text bilgileri
        info_text = f"""
        KARŞILAŞTIRMA İSTATİSTİKLERİ
        
        Küme Sayıları:
        • SOM: {som_n_clusters} küme
        • Meta: {meta_n_clusters} küme
        
        Etiket Benzerliği:
        • Aynı: {same_count} log ({similarity_ratio:.1%})
        • Farklı: {total_count-same_count} log ({1-similarity_ratio:.1%})
        
        SOM Küme Boyutları:
        {chr(10).join([f"• Küme {i}: {size} log" for i, size in som_sizes.items()])}
        
        Meta Küme Boyutları:
        {chr(10).join([f"• Küme {i}: {size} log" for i, size in meta_sizes.items()])}
        """
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        # 7. SOM Hull
        plt.subplot(3, 3, 7)
        unique_som = np.unique(som_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_som)))
        
        for i, cluster in enumerate(unique_som):
            mask = som_labels == cluster
            points = X_2d[mask]
            plt.scatter(points[:, 0], points[:, 1], c=[colors[i]], s=60, alpha=0.7)
            
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], 
                               color=colors[i], alpha=0.8, linewidth=2)
                except:
                    pass
        
        plt.title('SOM\n(Konveks Hull)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        # 8. Meta Hull
        plt.subplot(3, 3, 8)
        unique_meta = np.unique(meta_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_meta)))
        
        for i, cluster in enumerate(unique_meta):
            mask = meta_labels == cluster
            points = X_2d[mask]
            plt.scatter(points[:, 0], points[:, 1], c=[colors[i]], s=60, alpha=0.7)
            
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], 
                               color=colors[i], alpha=0.8, linewidth=2)
                except:
                    pass
        
        plt.title('Meta\n(Konveks Hull)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        
        # 9. Karşılaştırma metrikleri grafiği
        plt.subplot(3, 3, 9)
        
        # Küme boyutu varyansı
        som_var = np.var(list(som_sizes.values))
        meta_var = np.var(list(meta_sizes.values))
        
        metrics = ['Küme Sayısı', 'Benzerlik Oranı', 'Boyut Varyansı']
        som_values = [som_n_clusters/max(som_n_clusters, meta_n_clusters), 
                     similarity_ratio, 
                     1 - som_var/max(som_var, meta_var)]
        meta_values = [meta_n_clusters/max(som_n_clusters, meta_n_clusters), 
                      similarity_ratio, 
                      1 - meta_var/max(som_var, meta_var)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, som_values, width, label='SOM', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, meta_values, width, label='Meta', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Metrikler')
        plt.ylabel('Normalize Değer')
        plt.title('Metrik Karşılaştırması', fontsize=12, fontweight='bold')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Kapsamlı karşılaştırma görselleştirme hatası: {str(e)}")
        return None
