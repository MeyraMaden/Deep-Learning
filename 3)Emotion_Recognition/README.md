# 😊 Gerçek Zamanlı Duygu Tanıma — CNN ile FER-2013

FER-2013 veri seti kullanılarak geliştirilen CNN tabanlı bir yüz duygu tanıma sistemidir. Model 7 farklı duygu sınıfını sınıflandırmakta olup OpenCV ile entegre edilerek gerçek zamanlı webcam testi desteklenmektedir.

---

## 🎯 Proje Hakkında

Bu proje, insan yüz ifadelerinden duygu tanıma yapabilen derin öğrenme tabanlı bir sistemdir. Google Colab ortamında TensorFlow/Keras kullanılarak geliştirilmiş, eğitilen model gerçek zamanlı kamera akışına uygulanmıştır.

---

## 🗂️ Veri Seti

**FER-2013** (Facial Expression Recognition 2013)  
Kaynak: [Kaggle – FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

| Duygu     | Eğitim Görüntüsü |
|-----------|-----------------|
| angry     | 3.995           |
| disgust   | 436             |
| fear      | 4.097           |
| happy     | 7.215           |
| neutral   | 4.965           |
| sad       | 4.830           |
| surprise  | 3.171           |

- Görüntü boyutu: **48×48 piksel, gri tonlamalı**
- Toplam 7 duygu sınıfı

---

## 🏗️ Model Mimarisi

3 blokluk CNN mimarisi, BatchNormalization ve Dropout ile aşırı öğrenmeye karşı düzenlenmiştir.

```
Giriş: (48, 48, 1)
│
├── Blok 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
├── Blok 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Blok 3: Conv2D(128) → BN → MaxPool → Dropout(0.25)
│
├── Flatten
├── Dense(256) → BN → Dropout(0.5)
└── Dense(7, softmax)

Optimizer : Adam
Loss       : Categorical Crossentropy
```

---

## ⚙️ Eğitim Detayları

| Parametre     | Değer                          |
|---------------|-------------------------------|
| Epoch         | 30 (EarlyStopping ile)        |
| Batch Size    | 64                            |
| Image Size    | 48×48                         |
| Validation    | %20 ayrım (ImageDataGenerator)|
| Callbacks     | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

**Veri artırma (augmentation):**
- Rotation (±10°), Width/Height Shift (0.1), Horizontal Flip

---

## 📊 Değerlendirme

Model performansı test verisi üzerinde Confusion Matrix ve Classification Report ile değerlendirilmiştir.

---

## 📸 Gerçek Zamanlı Webcam Testi

Eğitilen model OpenCV ve Haar Cascade yüz dedektörü ile entegre edilerek webcam görüntüsü üzerinden gerçek zamanlı duygu tahmini yapılmıştır.

- Yüz tespiti: `haarcascade_frontalface_default.xml`
- Tahmin çıktısı: Duygu etiketi + güven skoru (%)
- Colab ortamında JavaScript tabanlı kamera yakalama ile test edilmiştir.

---

## 🛠️ Teknolojiler

| Teknoloji     | Kullanım Amacı                      |
|---------------|-------------------------------------|
| Python 3      | Ana programlama dili                |
| TensorFlow 2.20 / Keras | Model geliştirme ve eğitim  |
| OpenCV        | Yüz tespiti ve görüntü işleme       |
| NumPy         | Sayısal işlemler                    |
| Matplotlib / Seaborn | Görselleştirme              |
| scikit-learn  | Confusion matrix, classification report |
| Google Colab  | Eğitim ortamı (T4 GPU)              |

---

## 🚀 Kullanım

1. Notebook'u Google Colab'da açın.
2. FER-2013 veri setini Drive'ınıza yükleyin.
3. Hücreleri sırasıyla çalıştırın.
4. Son hücrede kamera açılarak gerçek zamanlı test yapılır.

---

## 📁 Dosyalar

```
├── Emotion_recognition.ipynb   # Ana notebook
├── emotion_model.keras         # Kaydedilen model (Drive'da)
└── confusion_matrix.png        # Test sonucu görselleştirme
```

---

## 👩‍💻 Geliştirici

**Hümeyra Hasmaden**  
Yazılım Mühendisliği — Karadeniz Teknik Üniversitesi  
[github.com/MeyraMaden](https://github.com/MeyraMaden)
