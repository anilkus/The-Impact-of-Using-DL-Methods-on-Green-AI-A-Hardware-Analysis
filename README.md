# The Impact of Using Deep Learning Methods on Green AI: A Hardware Analysis

Bu proje, farklı model optimizasyon tekniklerinin (8-bit quantization) enerji tüketimi ve performans üzerindeki etkisini analiz eder.

## 📊 Proje Amaçları

- Derin öğrenme modellerinin enerji tüketimini ölçmek
- 8-bit quantization'ın etkinliğini analiz etmek
- Green AI pratiklerini test etmek
- Donanım verimliliğini değerlendirmek

## 📁 Proje Yapısı

```
├── src/                          # Ana kaynak kodları
│   ├── __init__.py
│   ├── data_preparation.py      # Veri yükleme ve işleme
│   ├── model_utils.py           # Model yükleme ve optimizasyon
│   ├── evaluation.py            # Model değerlendirme
│   └── metrics.py               # Sonuç raporlama
│
├── experiments/                  # Deney scriptleri
│   ├── __init__.py
│   └── run_experiment.py        # Ana deney scripti
│
├── notebooks/                    # Analiz ve görselleştirmeler
│
├── data/                        # Veri klasörü
│
├── logs/                        # Log dosyaları
│
├── requirements.txt             # Bağımlılıklar
└── README.md                    # Bu dosya
```

## 🚀 Kurulum

### 1. Repository'i klonla
```bash
git clone https://github.com/anilkus/The-Impact-of-Using-DL-Methods-on-Green-AI-A-Hardware-Analysis.git
cd The-Impact-of-Using-DL-Methods-on-Green-AI-A-Hardware-Analysis
```

### 2. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 3. CUDA desteği (opsiyonel)
GPU kullanmak için CUDA yüklü olması gerekir:
```bash
# NVIDIA CUDA Toolkit kurulumu için: https://developer.nvidia.com/cuda-toolkit
```

## 🏃 Kullanım

### Deneyi Çalıştır
```bash
cd experiments
python run_experiment.py
```

## 📊 Metrikler

Proje aşağıdaki metrikleri ölçer:

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Model doğruluğu (0-1 arası) |
| **Süre (sn)** | Inference süresi (saniye) |
| **CO2 (kg)** | Enerji tüketimi (kg CO2 equivalenti) |
| **Green Score** | Accuracy/Emissions oranı (yüksek = daha iyi) |

## 📋 Modüller

### `data_preparation.py`
- `DataPreparation` sınıfı
  - `load_and_process_dataset()`: Dataset yükleme ve filtreleme
  - `prepare_dataloader()`: DataLoader hazırlama
  - `tokenize_function()`: Metin tokenizasyonu

### `model_utils.py`
- `ModelLoader` sınıfı
  - `load_normal_model()`: Normal modeli yükle
  - `load_quantized_model()`: Quantized modeli yükle
  - `cleanup()`: GPU belleğini temizle

### `evaluation.py`
- `Evaluator` sınıfı
  - `run_experiment()`: Model testini çalıştır
  - `_calculate_metrics()`: Metrikleri hesapla

### `metrics.py`
- `MetricsReporter` sınıfı
  - `print_results()`: Sonuçları tablo halinde göster

## 🔧 Konfigürasyon

`experiments/run_experiment.py` dosyasında aşağıdaki parametreleri özelleştirebilirsiniz:

```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
batch_size = 32
test_samples = 1000
```

## 📈 Örnek Çıktı

```
======================================================================
METRİK          | NORMAL MODEL    | 8-BIT MODEL     | DEĞİŞİM (%)
----------------------------------------------------------------------
Accuracy        |         0.8567  |         0.8534  |      -0.39%
Süre (sn)       |        45.2300  |        32.1500  |     -28.88%
CO2 (kg)        |         0.0450  |         0.0285  |     -36.67%
Green Score     |        19.0267  |        26.5701  |     +39.67%
======================================================================
```

## 🧪 Desteklenen Modeller

- `distilbert-base-uncased-finetuned-sst-2-english`
- Başka modelleri eklemek için `model_name` parametresini değiştir

## 📚 Kaynaklar

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CodeCarbon - Enerji Ölçme](https://codecarbon.io/)
- [BitsAndBytes - Model Quantization](https://github.com/TimDettmers/bitsandbytes)

## 🤝 Katkıda Bulunma

Lütfen pull request göndererek katkıda bulun. Büyük değişiklikler için önce bir issue açın.

## 📝 Lisans

Bu proje [MIT](LICENSE) lisansı altında lisanslanmıştır.

## ✉️ İletişim

Sorularınız için [Anil Kus](https://github.com/anilkus) ile iletişime geçin.

---

**Son Güncelleme:** 2026-03-25