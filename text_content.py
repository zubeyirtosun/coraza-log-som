def get_main_description():
    return """
Bu uygulama, JSON formatındaki log dosyalarını analiz eder, SOM ile kümeler
ve anomalileri tespit eder.
"""

def get_som_description():
    return """
**Self-Organizing Map (SOM)** yüksek boyutlu verileri 2B ızgarada görselleştiren
bir yapay sinir ağıdır. Log analizinde benzer davranışları gruplamak ve
anormallikleri tespit etmek için kullanılır.

SOM, her log kaydını en uygun nörona (Best Matching Unit - BMU) atar ve niceleme hatası ile anomalileri belirler.
Yüksek niceleme hatası değerleri, potansiyel anomalileri işaret eder.

Daha fazla bilgi için: [Analytics Vidhya SOM Kılavuzu](https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/)
"""

def get_user_gains():
    return """
### Kazanımlar
- **Neden**: Log analizinde desen keşfi ve anomali tespiti
- **Ne Öğrendim**: SOM uygulama, veri ön işleme, görselleştirme
- **Sonuç**: Etkili log analizi ve güvenlik tehdidi tespiti
"""

def get_summary_table_description():
    return """
Bu tablo, SOM ızgarasındaki her nöronun (BMU) özet istatistiklerini gösterir. Her nöron, benzer logları temsil eder. 
Tablo, nöronun koordinatlarını, engellenmiş istek oranını, en sık URI'yi, ortalama niceleme hatasını ve log sayısını içerir.
- **Yüksek engellenmiş oranı**: Potansiyel güvenlik tehditlerini işaret edebilir.
- **Yüksek niceleme hatası**: Anormal davranışları gösterebilir.
"""

def get_scatter_plot_description():
    return """
Bu grafik, log verilerinin SOM ızgarasındaki dağılımını gösterir. Her nokta, bir log kaydını temsil eder ve en uygun nörona (BMU) atanmıştır. 
Renkler, niceleme hatasını gösterir:
- **Mavi (düşük hata)**: Normal davranışları temsil eder.
- **Kırmızı (yüksek hata)**: Potansiyel anomalileri işaret eder.
Üzerine gelindiğinde, logun client port, URI ve engellenme durumu gibi detayları görünür.
"""

def get_error_distribution_description():
    return """
Bu histogram, logların niceleme hatası değerlerinin dağılımını gösterir. 
Yüksek niceleme hataları (sağda) potansiyel anomalileri temsil eder. 
Bu grafik, anomalilerin ne kadar yaygın olduğunu anlamanıza yardımcı olur.
"""

def get_meta_clustering_description():
    return """
Meta-kümeleme, SOM nöronlarını K-means algoritmasıyla daha büyük kümelere ayırır. 
Bu, log verilerindeki geniş davranış modellerini tespit etmeye yardımcı olur. 
Her meta-küme, benzer özelliklere sahip logları temsil eder.
Örnek: Bir meta-küme, `/login` endpoint'ine yönelik şüpheli istekleri içerebilir 
ve yüksek engellenmiş istek oranıyla dikkat çekebilir.
"""

def get_neuron_details_description():
    return """
Bu bölüm, seçilen bir SOM nöronundaki logların detaylarını gösterir. 
Bir nöron seçerek, o nörona atanan logların özelliklerini (örneğin, engellenmiş istek sayısı, en sık URI) 
inceleyebilirsiniz. Vurgulanan kırmızı 'X', seçilen nöronun ızgaradaki konumunu gösterir.
"""

def get_anomaly_detection_description():
    return """
Bu bölüm, yüksek niceleme hatasına sahip logları anomaliler olarak tespit eder. 
Yüzdebirlik eşiğini ayarlayarak, hangi logların anormal olduğunu belirleyebilirsiniz. 
Daha yüksek bir eşik, daha az ama daha belirgin anomaliler gösterir.
Örnek: `/WEB-INF/web.xml` gibi hassas dosyalara erişim girişimleri genellikle 
yüksek niceleme hatası ile anormal olarak işaretlenir.
"""
