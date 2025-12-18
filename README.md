# 🖼️ Image Segmentation - Edge Detection Methods Comparison

Program untuk membandingkan performa berbagai metode **edge detection** (deteksi tepi) pada citra grayscale, termasuk citra yang mengandung noise.

---

## 📋 Deskripsi

Program ini mengimplementasikan dan membandingkan **4 metode deteksi tepi** klasik dalam pengolahan citra:

| Metode | Kernel Size | Karakteristik |
|--------|-------------|---------------|
| **Roberts** | 2×2 | Sederhana, sensitif terhadap noise |
| **Prewitt** | 3×3 | Lebih smooth, mengurangi noise |
| **Sobel** | 3×3 | Weighted gradient, paling populer |
| **Frei-Chen** | 3×3 | Basis optimal, robust |

Program akan memproses 4 citra input (2 landscape + 2 portrait, masing-masing clean & noisy) dan menghasilkan: 
- ✅ Hasil deteksi tepi untuk setiap metode
- 📊 Tabel perbandingan MSE (Mean Squared Error)
- 📈 Grafik visualisasi perbandingan
- 🎯 Analisis performa metode

---

## 🚀 Fitur Utama

### 1. **Edge Detection Multi-Method**
- Implementasi manual konvolusi 2D
- Support 4 operator klasik (Roberts, Prewitt, Sobel, Frei-Chen)
- Normalisasi otomatis hasil ke range 0-255

### 2. **Analisis Kuantitatif**
- **MSE (Mean Squared Error)**:  Mengukur perbedaan hasil antara citra clean vs noisy
- **PSNR (Peak Signal-to-Noise Ratio)**: Metrik kualitas gambar dalam dB
- Perbandingan performa antar operator

### 3. **Visualisasi Komprehensif**
- **Panel per Operator**: Menampilkan hasil satu metode pada 4 citra
- **Panel Perbandingan**: Grid semua metode dan semua citra
- **Grafik MSE**: Bar chart dengan nilai MSE
- **Tabel Summary**:  Tabel metrik lengkap (MSE & PSNR)

### 4. **Output Lengkap**
- Ekspor ke CSV untuk analisis lanjut
- High-resolution PNG (150 DPI)
- Logging detail di console

---

## 📦 Instalasi

### Requirements
- Python 3.7 atau lebih tinggi
- Libraries: OpenCV, NumPy, Pandas, Matplotlib

### Langkah Instalasi

1. **Clone atau download repository ini**
   ```bash
   git clone <repository-url>
   cd image-segmentation-edge-detection
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy pandas matplotlib
   ```

   Atau menggunakan `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### File `requirements.txt`
```text
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
```

---

## 📂 Struktur Input

Siapkan **4 gambar grayscale** dengan nama berikut di root directory:

```
project/
├── landscape_clean.png      # Citra landscape tanpa noise
├── landscape_noisy.png      # Citra landscape dengan noise
├── portrait_clean.png       # Citra portrait tanpa noise
├── portrait_noisy.png       # Citra portrait dengan noise
└── edge_detection. py        # Script utama
```

> **💡 Tips**:  
> - Gunakan citra grayscale (1 channel)
> - Jika citra RGB, akan otomatis dikonversi ke grayscale
> - Resolusi bebas (akan diproses sesuai ukuran asli)

---

## ▶️ Cara Penggunaan

### Menjalankan Program

```bash
python edge_detection.py
```

### Output yang Dihasilkan

Semua hasil akan disimpan di folder `output_segmentasi/`:

```
output_segmentasi/
├── 📁 Panel Berdasarkan Operator
│   ├── panel_roberts.png         # Roberts pada 4 citra
│   ├── panel_prewitt.png         # Prewitt pada 4 citra
│   ├── panel_sobel.png           # Sobel pada 4 citra
│   └── panel_frei-chen. png       # Frei-Chen pada 4 citra
│
├── 📁 Panel Perbandingan
│   └── comparison_panel_all.png  # Grid semua operator & citra
│
├── 📁 Grafik & Tabel
│   ├── mse_comparison_chart.png  # Bar chart MSE
│   ├── summary_table.png         # Tabel metrik lengkap
│   └── mse_comparison_table.csv  # Data CSV
│
├── 📁 Input Images (Copy)
│   ├── input_landscape_clean.png
│   ├── input_landscape_noisy. png
│   ├── input_portrait_clean.png
│   └── input_portrait_noisy.png
│
└── 📁 Individual Results (16 files)
    ├── landscape_clean_Roberts_mag.png
    ├── landscape_clean_Prewitt_mag.png
    ├── landscape_clean_Sobel_mag.png
    ├── landscape_clean_Frei-Chen_mag.png
    ├── landscape_noisy_Roberts_mag.png
    └── ...  (dan seterusnya)
```

---

## 📊 Interpretasi Hasil

### 1. **MSE (Mean Squared Error)**
```
MSE = (1/n) * Σ(clean - noisy)²
```
- **Nilai rendah** → Operator robust terhadap noise
- **Nilai tinggi** → Operator sensitif terhadap noise

### 2. **PSNR (Peak Signal-to-Noise Ratio)**
```
PSNR = 20 * log₁₀(MAX / √MSE)
```
- **PSNR tinggi** (> 30 dB) → Kualitas bagus
- **PSNR rendah** (< 20 dB) → Kualitas buruk

### 3. **Contoh Output Console**

```
==================================================================
 ANALISIS & KESIMPULAN
==================================================================

1. PERFORMA OPERATOR PADA LANDSCAPE:
   ➤ Terbaik (MSE terendah)  :  Sobel (MSE = 245.67)
   ➤ Terburuk (MSE tertinggi): Roberts (MSE = 389.12)

2. PERFORMA OPERATOR PADA PORTRAIT:
   ➤ Terbaik (MSE terendah)  : Frei-Chen (MSE = 198.45)
   ➤ Terburuk (MSE tertinggi): Roberts (MSE = 356.78)

3. RATA-RATA MSE PER OPERATOR (keseluruhan):
   • Sobel       :    221.56
   • Frei-Chen   :   234.89
   • Prewitt     :   278.34
   • Roberts     :   372.95

4. KESIMPULAN UMUM:
   • Operator paling robust:  Sobel
   • Operator paling sensitif: Roberts
   • MSE rendah = edge detection lebih konsisten terhadap noise
```

---

## 🔬 Detail Teknis

### Kernel yang Digunakan

#### 1. Roberts (2×2)
```
Gx = [ 1   0]    Gy = [ 0   1]
     [ 0  -1]         [-1   0]
```

#### 2. Prewitt (3×3)
```
Gx = [-1  0  1]    Gy = [-1 -1 -1]
     [-1  0  1]         [ 0  0  0]
     [-1  0  1]         [ 1  1  1]
```

#### 3. Sobel (3×3)
```
Gx = [-1  0  1]    Gy = [-1 -2 -1]
     [-2  0  2]         [ 0  0  0]
     [-1  0  1]         [ 1  2  1]
```

#### 4. Frei-Chen (3×3)
```
Gx = [ 1   √2   1]    Gy = [ 1   0  -1]
     [ 0    0   0]         [√2   0  -√2]
     [-1  -√2  -1]         [ 1   0  -1]
```

### Algoritma Edge Detection

```python
# Pseudocode
1. Load citra grayscale
2. Untuk setiap operator:
   a. Konvolusi dengan kernel Gx → gradient horizontal
   b. Konvolusi dengan kernel Gy → gradient vertikal
   c. Magnitude = √(Gx² + Gy²)
   d. Normalisasi ke range 0-255
3. Hitung MSE antara hasil clean vs noisy
4. Visualisasi dan export hasil
```

---

## 🎨 Kustomisasi

### Mengganti Input Images

Edit bagian `INPUT_IMAGES` di script:

```python
INPUT_IMAGES = {
    "landscape_clean": "path/to/your/landscape_clean.png",
    "landscape_noisy": "path/to/your/landscape_noisy.png",
    "portrait_clean": "path/to/your/portrait_clean.png",
    "portrait_noisy": "path/to/your/portrait_noisy.png"
}
```

### Menambahkan Operator Baru

```python
# Tambahkan kernel baru
custom_gx = np.array([[... ]])
custom_gy = np.array([[...]])

# Daftarkan di OPERATORS
OPERATORS = {
    "Roberts": (roberts_gx, roberts_gy),
    "Prewitt": (prewitt_gx, prewitt_gy),
    "Sobel": (sobel_gx, sobel_gy),
    "Frei-Chen": (freichen_gx, freichen_gy),
    "Custom": (custom_gx, custom_gy)  # Operator baru
}
```

### Mengubah Output Directory

```python
OUTPUT_DIR = "hasil_segmentasi"  # Ganti nama folder output
```

---

## 🧪 Contoh Penggunaan

### Scenario 1: Analisis Noise Resistance

**Tujuan**: Menguji operator mana yang paling tahan terhadap Gaussian noise

```bash
# 1. Buat citra noisy menggunakan tools eksternal atau Python
# 2. Jalankan program
python edge_detection.py

# 3. Lihat tabel MSE - nilai terendah = paling robust
```

### Scenario 2: Perbandingan Visual

**Tujuan**: Bandingkan kualitas edge detection secara visual

```bash
# 1. Jalankan program
python edge_detection.py

# 2. Buka file panel_*. png untuk melihat hasil per operator
# 3. Operator dengan edge paling jelas = performa terbaik
```

### Scenario 3: Ekspor Data untuk Penelitian

```python
# Load hasil CSV
import pandas as pd
df = pd.read_csv('output_segmentasi/mse_comparison_table.csv')

# Analisis statistik
print(df.groupby('Operator')['MSE'].describe())
```

---

## 📖 Referensi

1. **Roberts Cross**:  Roberts, L. G.  (1963). *Machine Perception of Three-Dimensional Solids*
2. **Prewitt Operator**: Prewitt, J. M. (1970). *Object Enhancement and Extraction*
3. **Sobel Operator**: Sobel, I., Feldman, G. (1968). *A 3x3 Isotropic Gradient Operator*
4. **Frei-Chen Masks**: Frei, W., Chen, C. (1977). *Fast Boundary Detection*

### Bacaan Lebih Lanjut
- [Edge Detection - Stanford CS231A](http://web.stanford.edu/class/cs231a/)
- [OpenCV Edge Detection Tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Digital Image Processing - Gonzalez & Woods](https://www.imageprocessingplace.com/)

---

## ❓ FAQ

### Q: Program error "File tidak ditemukan"? 
**A**: Pastikan 4 file input (landscape_clean.png, landscape_noisy.png, portrait_clean.png, portrait_noisy.png) ada di folder yang sama dengan script. 

### Q: Hasil deteksi tepi terlalu gelap/terang?
**A**: Ini normal. Program otomatis melakukan normalisasi.  Nilai asli magnitude disimpan dalam variabel `mag_float`.

### Q: Bisa pakai citra RGB?
**A**: Bisa.  Program otomatis mengkonversi ke grayscale. 

### Q: Bagaimana cara membuat citra noisy?
**A**: Gunakan code berikut:
```python
import cv2
import numpy as np

img = cv2.imread('clean. png', 0)
noise = np.random.normal(0, 25, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
cv2.imwrite('noisy.png', noisy)
```

### Q: MSE bernilai 0, kenapa PSNR infinity?
**A**: Jika hasil clean dan noisy identik (MSE=0), PSNR secara matematis tak hingga. Ini menandakan tidak ada perbedaan. 

---

## 🤝 Kontribusi

Kontribusi sangat diterima!  Silakan: 
1. Fork repository ini
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

### Ide Pengembangan
- [ ] Tambahkan operator Laplacian
- [ ] Implementasi Canny edge detection
- [ ] Support batch processing untuk banyak gambar
- [ ] GUI menggunakan Tkinter/PyQt
- [ ] Real-time processing dari webcam
- [ ] Hyperparameter tuning otomatis

---

## 📄 License

Distributed under the MIT License.  See `LICENSE` for more information.

---

## 👨‍💻 Author

**Nama Anda**
- GitHub: [@bagaspng](https://github.com/bagaspng)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- OpenCV Team untuk library image processing yang powerful
- Komunitas Python untuk ekosistem yang luar biasa
- Para peneliti yang mengembangkan algoritma edge detection klasik

---

## 📞 Support

Jika ada pertanyaan atau issues: 
1. Buka [GitHub Issues](https://github.com/bagaspng/repo-name/issues)
2. Email: your.email@example.com
3. Diskusi:  [GitHub Discussions](https://github.com/bagaspng/repo-name/discussions)

---

<div align="center">

**⭐ Jangan lupa beri star jika project ini bermanfaat!  ⭐**

Made with ❤️ using Python & OpenCV

</div>
