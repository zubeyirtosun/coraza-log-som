# CORAZA-LOG-SOM

Coraza Web Application Firewall (WAF) log kayıtlarını Self-Organizing Map (SOM) tekniği ile analiz eden ve görselleştiren bir web uygulaması.

## Proje Hakkında

Bu proje, Coraza WAF'ın ürettiği JSON formatındaki log kayıtlarını analiz ederek anormal davranışları ve potansiyel güvenlik tehditlerini tespit etmeyi amaçlamaktadır. Self-Organizing Map (SOM) algoritması kullanılarak loglar kümelenir ve görselleştirilir, böylece benzer özelliklere sahip kayıtlar ve potansiyel anomaliler kolayca belirlenebilir.

## Özellikler

### Temel Özellikler
- JSON formatındaki log dosyalarını yükleme ve işleme
- Self-Organizing Map ile log kayıtlarını kümeleme ve görselleştirme
- Meta-kümeleme ile daha büyük davranış modellerini tespit etme
- Anomali tespiti ve vurgulama
- Nöron bazında detaylı analiz
- Zaman serisi analizleri
- SOM ve kümeleme kalitesini değerlendiren doğrulama metrikleri

### Gelişmiş Analizler (Yeni Eklenen)
- **Kümeleme Stabilitesi Analizi**: K-means'i farklı başlangıç noktalarıyla birden fazla çalıştırarak kümeleme sonuçlarının tutarlılığını ölçme
- **Alternatif Kümeleme Algoritmaları**: K-means, Hiyerarşik Kümeleme, DBSCAN ve HDBSCAN algoritmaları ile karşılaştırmalı analiz
- **Boyut İndirgeme**: Her kümedeki veri noktalarını PCA, t-SNE ve UMAP ile 2D/3D görselleştirme
- **Çapraz Doğrulama Benzeri Yöntem**: Veriyi parçalara bölerek, her parçada benzer kümeler oluşup oluşmadığını kontrol etme
- **Otomatik Optimal Küme Sayısı Belirleme**: Dirsek yöntemi, silüet skoru ve diğer metrikler kullanılarak en uygun küme sayısı önerisi
- **PDF Rapor Oluşturma**: Tüm analizlerin sonuçlarını içeren indirilebilir PDF raporu

## Kurulum

1. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```bash
streamlit run main.py
```

## Kullanım

1. **Log Dosyası Yükleme**: JSON formatındaki Coraza log dosyanızı yükleyin.
2. **Veri İşleme**: Eksik veri işleme yöntemini seçin ve veriyi işleyin.
3. **SOM Parametre Ayarları**: Gerekliyse SOM parametrelerini ayarlayın ve modeli eğitin.
4. **Temel Analizler**: Ana analizleri inceleyerek anomalileri ve kalıpları tespit edin.
5. **Gelişmiş Analizler**: 
   - Optimal küme sayısını belirleyin
   - Boyut indirgeme ile veri yapısını keşfedin
   - Farklı kümeleme algoritmalarını karşılaştırın
   - Kümeleme stabilitesini değerlendirin
   - Veri parçalarında tutarlılığı doğrulayın
   - Tüm analizleri içeren bir PDF raporu indirin

## Bileşenler

- **main.py**: Ana uygulama, arayüz ve uygulama akışını yönetir
- **data_processing.py**: Veri işleme, dönüştürme ve SOM eğitimini yönetir
- **visualizations.py**: SOM sonuçlarını görselleştiren grafikler ve tablolar
- **session_state.py**: Streamlit oturum durumunu yöneten yardımcı fonksiyonlar
- **text_content.py**: Uygulamadaki metin içeriklerini barındırır
- **advanced_clustering.py**: Gelişmiş kümeleme ve boyut indirgeme fonksiyonları
- **pdf_report.py**: PDF rapor oluşturma fonksiyonları

## SOM Nedir?

Self-Organizing Map (SOM), yüksek boyutlu verileri 2B ızgarada görselleştiren bir yapay sinir ağıdır. Log analizinde benzer davranışları gruplamak ve anormallikleri tespit etmek için kullanılır. SOM, her log kaydını en uygun nörona (Best Matching Unit - BMU) atar ve niceleme hatası ile anomalileri belirler.

## Gelişmiş Analizler Detayları

### Kümeleme Stabilitesi
Kümeleme sonuçları, algoritmaların başlangıç koşullarına bağlı olarak değişebilir. Stabilite analizi, aynı algoritmanın farklı çalıştırmalarında ne kadar tutarlı sonuçlar ürettiğini ölçer. Yüksek stabilite, kümeleme sonuçlarının daha güvenilir olduğunu gösterir.

### Alternatif Kümeleme Algoritmaları
Her kümeleme algoritmasının farklı güçlü yönleri vardır:
- **K-means**: Küresel kümelerde iyi çalışır, hızlıdır
- **Hiyerarşik Kümeleme**: Küme hiyerarşisini belirlemeye yardımcı olur
- **DBSCAN**: Düzensiz şekilli kümeleri ve gürültüyü tespit edebilir
- **HDBSCAN**: Farklı yoğunluktaki kümeleri tespit edebilir

### Boyut İndirgeme
Yüksek boyutlu verileri 2D/3D'de görselleştirerek veri yapısını daha iyi anlamaya yardımcı olur:
- **PCA**: Varyansı korur, global yapıyı gösterir
- **t-SNE**: Yerel yapıyı korur, kümeleri ayırmada iyidir
- **UMAP**: Hem yerel hem de global yapıyı dengeler, t-SNE'den daha hızlıdır

### Optimal Küme Sayısı Belirleme
Doğru küme sayısını belirlemek için birden fazla metrik kullanılır:
- **Dirsek Yöntemi**: İnertia'nın düşüş hızının azaldığı nokta
- **Silüet Skoru**: Kümelerin birbirinden ayrılma derecesi
- **Calinski-Harabasz**: Küme içi kompaktlık ve kümeler arası ayrılık
- **Davies-Bouldin**: Küme içi benzerlik ve küme dışı ayrılık ölçüsü

## Son Güncellemeler

### 2023-10-15 Yapılan İyileştirmeler

1. **Veri İşleme Geliştirmeleri**:
   - Eksik sütun hatası için veri işleme mekanizması güçlendirildi
   - Zorunlu sütunlar için varsayılan değerlerle doldurma mekanizması eklendi
   - Hataların daha açıklayıcı şekilde gösterilmesi sağlandı

2. **Arayüz Geliştirmeleri**:
   - Gelişmiş analiz bölümü expander'lar yerine tab'lar kullanacak şekilde yeniden tasarlandı
   - Meta kümeleme doğrulama bölümü sekme tabanlı bir yapıya dönüştürüldü
   - SOM Model Doğrulama bölümü daha iyi bir görsel tasarıma kavuşturuldu
   - Nöron Detayları bölümünde yazım ve gösterim sorunları düzeltildi

3. **Zaman Serisi Analizi İyileştirmeleri**:
   - Zaman serisi analizi sorunları giderildi
   - Saat bazlı çizelgelere ek görselleştirmeler eklendi
   - Eksik zaman sütunları için otomatik dönüştürme eklendi

4. **Optimal Küme Sayısı Analizi**:
   - Gelişmiş analizlerde kaybolmuş olan "Optimal Küme Sayısı Analizi" bölümü yeniden düzenlendi
   - Daha iyi görselleştirme ve daha iyi anlaşılan sonuçlar eklendi

5. **Dayanıklılık İyileştirmeleri**:
   - Çeşitli hata durumlarına karşı daha dayanıklı hale getirildi
   - Veri tutarsızlıkları için daha fazla kontrol eklendi
   - Base64 ve BytesIO nesneleri arasında dönüşüm için yardımcı fonksiyonlar eklendi

### Kullanım

Uygulamayı çalıştırmak için:

```bash
streamlit run main.py
```

Coraza WAF loglarınızı yükleyin ve analiz edin. İnteraktif veri önişleme, SOM kümeleme, anomali tespiti ve gelişmiş analiz özelliklerini kullanabilirsiniz.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.