"""
Tugas Segmentasi Citra - Perbandingan Metode Edge Detection
Metode: Roberts, Prewitt, Sobel, Frei-Chen
Input: 4 gambar grayscale (2 landscape, 2 portrait - sudah termasuk yang berisi noise)
Output: Hasil segmentasi + Tabel MSE + Grafik perbandingan
----------------------------------------------------------------
Requirements: pip install opencv-python numpy pandas matplotlib
"""

import cv2
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def ensure_dir(p):
    """Pastikan folder output ada."""
    os.makedirs(p, exist_ok=True)

def load_gray(p):
    """Load gambar sebagai grayscale."""
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def normalize_to_uint8(imgf):
    """Normalisasi float image ke uint8 (0-255)."""
    mn, mx = float(imgf.min()), float(imgf.max())
    if mx - mn < 1e-8:
        return np.zeros_like(imgf, dtype=np.uint8)
    norm = (imgf - mn) / (mx - mn) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def mse(a, b):
    """Hitung Mean Squared Error antara dua array."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b)**2))

def psnr(mse_val):
    """Hitung PSNR dari MSE."""
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse_val))

def convolve2d_gray(img, kernel):
    """Konvolusi manual 2D untuk grayscale image."""
    imgf = img.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = cv2.copyMakeBorder(imgf, ph, ph, pw, pw, borderType=cv2.BORDER_REPLICATE)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return out

# ============================================================
#  EDGE DETECTION KERNELS
# ============================================================

# Roberts (2x2)
roberts_gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)

# Prewitt (3x3)
prewitt_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

# Sobel (3x3)
sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Frei-Chen (3x3)
s2 = math.sqrt(2.0)
freichen_gx = np.array([[1, s2, 1], [0, 0, 0], [-1, -s2, -1]], dtype=np.float32)
freichen_gy = np.array([[1, 0, -1], [s2, 0, -s2], [1, 0, -1]], dtype=np.float32)

OPERATORS = {
    "Roberts": (roberts_gx, roberts_gy),
    "Prewitt": (prewitt_gx, prewitt_gy),
    "Sobel": (sobel_gx, sobel_gy),
    "Frei-Chen": (freichen_gx, freichen_gy)
}

# ============================================================
#  PROCESSING FUNCTIONS
# ============================================================

def compute_edge_magnitude(img, kx, ky):
    """Hitung magnitude edge dari konvolusi dengan kernel gx dan gy."""
    gx = convolve2d_gray(img, kx)
    gy = convolve2d_gray(img, ky)
    mag = np.sqrt(gx**2 + gy**2)
    return mag

def process_image(img, operator_name, kx, ky, out_dir, img_tag):
    """Proses satu gambar dengan satu operator dan simpan hasilnya."""
    mag = compute_edge_magnitude(img, kx, ky)
    mag_u8 = normalize_to_uint8(mag)
    
    # Simpan hasil magnitude
    out_path = os.path.join(out_dir, f"{img_tag}_{operator_name}_mag.png")
    cv2.imwrite(out_path, mag_u8)
    
    return mag, mag_u8

# ============================================================
#  VISUALIZATION FUNCTIONS
# ============================================================

def create_panel_by_operator(mag_results, operator_name, images_dict, out_dir):
    """
    Buat panel untuk satu operator dengan 4 hasil (landscape_clean, landscape_noisy, 
    portrait_clean, portrait_noisy) dengan keterangan nama gambar.
    
    Layout: 2x2 grid
    """
    # Urutan gambar di panel
    image_order = ["landscape_clean", "landscape_noisy", "portrait_clean", "portrait_noisy"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Operator {operator_name} - Edge Detection Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Warna label berdasarkan tipe (clean/noisy)
    label_colors = {
        "landscape_clean": "#2E7D32",     # Hijau gelap
        "landscape_noisy": "#FF6F00",     # Orange
        "portrait_clean": "#1565C0",      # Biru gelap
        "portrait_noisy": "#C62828"       # Merah gelap
    }
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for pos, img_tag in zip(positions, image_order):
        ax = axes[pos]
        mag_u8 = mag_results[img_tag][operator_name][1]
        
        ax.imshow(mag_u8, cmap='gray')
        ax.axis('off')
        
        # Format label
        label_parts = img_tag.split('_')
        label_type = label_parts[0].title()  # Landscape / Portrait
        label_quality = label_parts[1].upper()  # CLEAN / NOISY
        label_text = f"{label_type}\n{label_quality}"
        
        # Tambahkan label dengan background
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor=label_colors[img_tag], 
                         edgecolor='white', linewidth=2, alpha=0.9)
        ax.text(0.5, -0.05, label_text, transform=ax.transAxes, 
               fontsize=12, fontweight='bold', color='white',
               ha='center', va='top', bbox=bbox_props)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"panel_{operator_name.lower()}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Panel {operator_name} disimpan: panel_{operator_name.lower()}.png")

def create_all_operator_panels(mag_results, operators_list, images_dict, out_dir):
    """Buat panel untuk semua operator."""
    print("\n[Membuat Panel Berdasarkan Operator]")
    print("-"*70)
    for op_name in operators_list:
        create_panel_by_operator(mag_results, op_name, images_dict, out_dir)

def create_comparison_panel(images_dict, mag_results, operators, out_path):
    """
    Buat panel perbandingan semua operator pada satu gambar (layout: operators x images).
    """
    img_tags = ["landscape_clean", "landscape_noisy", "portrait_clean", "portrait_noisy"]
    n_imgs = len(img_tags)
    n_ops = len(operators)
    
    fig, axes = plt.subplots(n_imgs, n_ops, figsize=(n_ops*4, n_imgs*3.5))
    fig.suptitle('Perbandingan Semua Operator pada Semua Citra', 
                 fontsize=16, fontweight='bold')
    
    if n_imgs == 1:
        axes = axes.reshape(1, -1)
    if n_ops == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img_tag in enumerate(img_tags):
        for j, op in enumerate(operators):
            ax = axes[i, j]
            mag_u8 = mag_results[img_tag][op][1]
            ax.imshow(mag_u8, cmap='gray')
            ax.axis('off')
            
            if i == 0:
                ax.set_title(op, fontsize=12, fontweight='bold', pad=10)
            if j == 0:
                label_text = img_tag.replace('_', '\n').title()
                ax.set_ylabel(label_text, fontsize=11, fontweight='bold', 
                            rotation=0, ha='right', va='center', labelpad=20)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Panel perbandingan semua operator disimpan: comparison_panel_all.png")

def plot_mse_comparison(df, out_path):
    """Buat bar chart perbandingan MSE."""
    pivot = df.pivot_table(index='Operator', columns='Comparison', values='MSE')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, width=0.75, colormap='Set2')
    
    ax.set_ylabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator Deteksi Tepi', fontsize=12, fontweight='bold')
    ax.set_title('Perbandingan MSE: Edge Detection pada Citra Clean vs Noisy', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Perbandingan', fontsize=10, title_fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=0)
    
    # Tambahkan nilai MSE di atas bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Grafik MSE disimpan: mse_comparison_chart.png")

def create_summary_table(df, out_path):
    """Buat tabel summary dengan statistik tambahan."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format tabel
    table_data = []
    table_data.append(['Operator', 'Comparison', 'MSE', 'PSNR (dB)'])
    
    for _, row in df.iterrows():
        psnr_val = psnr(row['MSE']) if row['MSE'] > 0 else float('inf')
        psnr_str = f"{psnr_val:.2f}" if psnr_val != float('inf') else "∞"
        table_data.append([
            row['Operator'],
            row['Comparison'],
            f"{row['MSE']:.2f}",
            psnr_str
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.4, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(4):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Tabel Perbandingan Metrik Edge Detection', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Tabel summary disimpan: summary_table.png")

# ============================================================
#  MAIN PROCESSING
# ============================================================

if __name__ == "__main__":
    # ---- KONFIGURASI INPUT ----
    INPUT_IMAGES = {
        "landscape_clean": "landscape_clean.png",
        "landscape_noisy": "landscape_noisy.png",
        "portrait_clean": "portrait_clean.png",
        "portrait_noisy": "portrait_noisy.png"
    }
    
    OUTPUT_DIR = "output_segmentasi"
    ensure_dir(OUTPUT_DIR)
    
    print("="*70)
    print(" SEGMENTASI CITRA - PERBANDINGAN METODE EDGE DETECTION")
    print("="*70)
    print("\nMetode yang digunakan:")
    print("  1. Roberts   - Operator 2x2 (sederhana, sensitif noise)")
    print("  2. Prewitt   - Operator 3x3 (lebih smooth)")
    print("  3. Sobel     - Operator 3x3 (weighted, populer)")
    print("  4. Frei-Chen - Operator 3x3 (basis optimal)")
    print("="*70)
    
    # ---- LOAD IMAGES ----
    images = {}
    missing_files = []
    
    print("\n[1/4] LOADING INPUT IMAGES...")
    for tag, path in INPUT_IMAGES.items():
        try:
            img = load_gray(path)
            images[tag] = img
            copy_path = os.path.join(OUTPUT_DIR, f"input_{tag}.png")
            cv2.imwrite(copy_path, img)
            print(f"  ✓ {tag:20s} : {img.shape} - {path}")
        except FileNotFoundError:
            print(f"  ✗ {tag:20s} : FILE NOT FOUND - {path}")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n⚠ ERROR: {len(missing_files)} file(s) tidak ditemukan!")
        print("  File yang hilang:")
        for f in missing_files:
            print(f"    - {f}")
        print("\n  Pastikan file-file berikut ada di direktori:")
        for tag, path in INPUT_IMAGES.items():
            print(f"    - {path}")
        exit(1)
    
    # ---- PROCESS ALL IMAGES WITH ALL OPERATORS ----
    print(f"\n[2/4] PROSES KONVOLUSI & EDGE DETECTION...")
    print("-"*70)
    
    mag_results = {}
    
    for img_tag, img in images.items():
        mag_results[img_tag] = {}
        print(f"\n→ {img_tag.upper()}")
        
        for op_name, (kx, ky) in OPERATORS.items():
            print(f"  • Konvolusi {op_name}...", end=' ')
            mag_float, mag_u8 = process_image(img, op_name, kx, ky, OUTPUT_DIR, img_tag)
            mag_results[img_tag][op_name] = (mag_float, mag_u8)
            print(f"✓ (min={mag_float.min():.1f}, max={mag_float.max():.1f})")
    
    # ---- COMPUTE MSE COMPARISONS ----
    print(f"\n[3/4] MENGHITUNG MSE (MEAN SQUARED ERROR)...")
    print("-"*70)
    
    results = []
    
    # Perbandingan 1: Landscape clean vs noisy
    print("\n→ LANDSCAPE: Clean vs Noisy")
    for op_name in OPERATORS.keys():
        mag_clean, _ = mag_results["landscape_clean"][op_name]
        mag_noisy, _ = mag_results["landscape_noisy"][op_name]
        mse_val = mse(mag_clean, mag_noisy)
        psnr_val = psnr(mse_val)
        print(f"  • {op_name:12s} : MSE={mse_val:8.2f}  PSNR={psnr_val:6.2f} dB")
        results.append({
            "Operator": op_name,
            "Comparison": "Landscape (Clean vs Noisy)",
            "MSE": round(mse_val, 2),
            "PSNR": round(psnr_val, 2) if psnr_val != float('inf') else psnr_val
        })
    
    # Perbandingan 2: Portrait clean vs noisy
    print("\n→ PORTRAIT: Clean vs Noisy")
    for op_name in OPERATORS.keys():
        mag_clean, _ = mag_results["portrait_clean"][op_name]
        mag_noisy, _ = mag_results["portrait_noisy"][op_name]
        mse_val = mse(mag_clean, mag_noisy)
        psnr_val = psnr(mse_val)
        print(f"  • {op_name:12s} : MSE={mse_val:8.2f}  PSNR={psnr_val:6.2f} dB")
        results.append({
            "Operator": op_name,
            "Comparison": "Portrait (Clean vs Noisy)",
            "MSE": round(mse_val, 2),
            "PSNR": round(psnr_val, 2) if psnr_val != float('inf') else psnr_val
        })
    
    # ---- CREATE DATAFRAME & SAVE ----
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(OUTPUT_DIR, "mse_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Tabel MSE disimpan: {csv_path}")
    
    # ---- PRINT SUMMARY ----
    print(f"\n{'='*70}")
    print(" TABEL HASIL PERBANDINGAN MSE")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    
    # ---- ANALISIS KESIMPULAN ----
    print(f"\n{'='*70}")
    print(" ANALISIS & KESIMPULAN")
    print(f"{'='*70}")
    
    landscape_df = df[df['Comparison'].str.contains('Landscape')]
    best_landscape = landscape_df.loc[landscape_df['MSE'].idxmin()]
    worst_landscape = landscape_df.loc[landscape_df['MSE'].idxmax()]
    
    portrait_df = df[df['Comparison'].str.contains('Portrait')]
    best_portrait = portrait_df.loc[portrait_df['MSE'].idxmin()]
    worst_portrait = portrait_df.loc[portrait_df['MSE'].idxmax()]
    
    print("\n1. PERFORMA OPERATOR PADA LANDSCAPE:")
    print(f"   ➤ Terbaik (MSE terendah)  : {best_landscape['Operator']} (MSE = {best_landscape['MSE']:.2f})")
    print(f"   ➤ Terburuk (MSE tertinggi): {worst_landscape['Operator']} (MSE = {worst_landscape['MSE']:.2f})")
    
    print("\n2. PERFORMA OPERATOR PADA PORTRAIT:")
    print(f"   ➤ Terbaik (MSE terendah)  : {best_portrait['Operator']} (MSE = {best_portrait['MSE']:.2f})")
    print(f"   ➤ Terburuk (MSE tertinggi): {worst_portrait['Operator']} (MSE = {worst_portrait['MSE']:.2f})")
    
    avg_mse = df.groupby('Operator')['MSE'].mean().sort_values()
    print("\n3. RATA-RATA MSE PER OPERATOR (keseluruhan):")
    for op, mse_val in avg_mse.items():
        print(f"   • {op:12s} : {mse_val:8.2f}")
    
    print("\n4. KESIMPULAN UMUM:")
    print(f"   • Operator paling robust (MSE terendah rata-rata): {avg_mse.index[0]}")
    print(f"   • Operator paling sensitif (MSE tertinggi rata-rata): {avg_mse.index[-1]}")
    print("   • MSE rendah = edge detection lebih konsisten terhadap noise")
    print("   • MSE tinggi = operator lebih sensitif terhadap perubahan noise")
    
    # ---- CREATE VISUALIZATIONS ----
    print(f"\n[4/4] MEMBUAT VISUALISASI...")
    print("-"*70)
    
    # Buat panel berdasarkan operator (BARU)
    create_all_operator_panels(mag_results, list(OPERATORS.keys()), images, OUTPUT_DIR)
    
    # Buat panel perbandingan semua operator
    panel_path = os.path.join(OUTPUT_DIR, "comparison_panel_all.png")
    create_comparison_panel(images, mag_results, list(OPERATORS.keys()), panel_path)
    
    # Buat grafik MSE
    mse_chart_path = os.path.join(OUTPUT_DIR, "mse_comparison_chart.png")
    plot_mse_comparison(df, mse_chart_path)
    
    # Buat tabel summary
    summary_table_path = os.path.join(OUTPUT_DIR, "summary_table.png")
    create_summary_table(df, summary_table_path)
    
    # ---- FINAL SUMMARY ----
    print(f"\n{'='*70}")
    print(" PROSES SELESAI!")
    print(f"{'='*70}")
    print(f"\nSemua hasil disimpan di folder: {OUTPUT_DIR}/")
    print("\nFile Output:")
    print("  [Panel Berdasarkan Operator]")
    print("    - panel_roberts.png       : Roberts pada 4 citra")
    print("    - panel_prewitt.png       : Prewitt pada 4 citra")
    print("    - panel_sobel.png         : Sobel pada 4 citra")
    print("    - panel_frei-chen.png     : Frei-Chen pada 4 citra")
    print("\n  [Panel Perbandingan Semua Operator]")
    print("    - comparison_panel_all.png    : Semua operator pada semua citra")
    print("\n  [Grafik & Tabel]")
    print("    - mse_comparison_chart.png    : Bar chart perbandingan MSE")
    print("    - summary_table.png           : Tabel metrik lengkap (MSE & PSNR)")
    print("    - mse_comparison_table.csv    : Data CSV untuk analisis lanjut")
    print("\n  [Input Images (Copy)]")
    print("    - input_landscape_clean.png")
    print("    - input_landscape_noisy.png")
    print("    - input_portrait_clean.png")
    print("    - input_portrait_noisy.png")
    print("\n  [Individual Edge Detection Results]")
    print("    - *_mag.png : Hasil edge detection individual (16 file)")
    print(f"\n{'='*70}\n")
