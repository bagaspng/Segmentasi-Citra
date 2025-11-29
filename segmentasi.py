"""
Tugas Segmentasi Citra (Kelompok)
Metode: Roberts, Prewitt, Sobel, Frei-Chen
Implementasi manual (tanpa cv2.Sobel, cv2.filter2D, dll.)
----------------------------------------------------------------
Kebutuhan:
    pip install opencv-python numpy

Cara pakai (di terminal):
    python segmentasi_kelompok.py
"""

import cv2
import numpy as np
import os

def ensure_dir(path: str):
    """Pastikan folder output ada."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_as_gray(path: str) -> np.ndarray:
    """
    Membaca citra dan mengembalikan versi grayscale (uint8).
    Jika citra sudah grayscale, langsung dikembalikan.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar: {path}")
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalisasi citra float ke rentang 0–255 dan ubah ke uint8.
    """
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - min_val) / (max_val - min_val) * 255.0
    return norm.astype(np.uint8)

# ============================================================
#  KONVOLUSI MANUAL 2D (UNTUK GRAYSCALE)
# ============================================================

def convolve2d_gray(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Konvolusi manual antara citra grayscale (2D) dengan kernel 2D.
    Padding menggunakan BORDER_REPLICATE (edge).
    Output berupa float32.
    """
    img = img.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = cv2.copyMakeBorder(
        img, ph, ph, pw, pw,
        borderType=cv2.BORDER_REPLICATE
    )

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)

    # Flip kernel untuk konvolusi
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)

    return out

# ============================================================
#  OPERATOR SEGMENTASI
# ============================================================

def edge_roberts(img_gray: np.ndarray) -> np.ndarray:
    """
    Deteksi tepi dengan operator Roberts.
    Kernel 2x2.
    """
    gx_kernel = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    gy_kernel = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)

    gx = convolve2d_gray(img_gray, gx_kernel)
    gy = convolve2d_gray(img_gray, gy_kernel)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)


def edge_prewitt(img_gray: np.ndarray) -> np.ndarray:
    """
    Deteksi tepi dengan operator Prewitt (3x3).
    """
    gx_kernel = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    gy_kernel = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)

    gx = convolve2d_gray(img_gray, gx_kernel)
    gy = convolve2d_gray(img_gray, gy_kernel)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)


def edge_sobel(img_gray: np.ndarray) -> np.ndarray:
    """
    Deteksi tepi dengan operator Sobel (3x3).
    """
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32)

    gx = convolve2d_gray(img_gray, gx_kernel)
    gy = convolve2d_gray(img_gray, gy_kernel)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)


def edge_freichen(img_gray: np.ndarray) -> np.ndarray:
    """
    Deteksi tepi dengan operator Frei-Chen.
    Menggunakan 4 kernel edge utama dan menggabungkan responnya.
    """
    s2 = np.sqrt(2.0)

    k1 = np.array([[1,      s2,  1],
                   [0,      0,   0],
                   [-1, -s2, -1]], dtype=np.float32)  # vertikal

    k2 = np.array([[1,  0, -1],
                   [s2, 0, -s2],
                   [1,  0, -1]], dtype=np.float32)    # horizontal

    k3 = np.array([[0,     -1,   s2],
                   [1,      0,  -1],
                   [-s2,    1,   0]], dtype=np.float32)  # diagonal 1

    k4 = np.array([[s2,  -1,  0],
                   [-1,  0,  1],
                   [0,   1, -s2]], dtype=np.float32)     # diagonal 2

    r1 = convolve2d_gray(img_gray, k1)
    r2 = convolve2d_gray(img_gray, k2)
    r3 = convolve2d_gray(img_gray, k3)
    r4 = convolve2d_gray(img_gray, k4)

    mag = np.sqrt(r1**2 + r2**2 + r3**2 + r4**2)
    return normalize_to_uint8(mag)

# ============================================================
#  PANEL VISUALISASI (OPSIONAL, TAPI SANGAT MEMBANTU LAPORAN)
# ============================================================

def add_label_bar(img_gray: np.ndarray, text: str) -> np.ndarray:
    """
    Menambahkan bar teks di bawah gambar grayscale (jadi 3-channel BGR).
    """
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_color.shape[:2]
    bar_h = 35
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)

    cv2.putText(
        bar, text,
        (10, bar_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 1, cv2.LINE_AA
    )

    return np.vstack([img_color, bar])


def save_edge_panel(orig: np.ndarray,
                    rob: np.ndarray,
                    prew: np.ndarray,
                    sob: np.ndarray,
                    frei: np.ndarray,
                    out_path: str,
                    title_prefix: str = ""):
    """
    Menyimpan panel 2x3:
    [ Original | Roberts | Prewitt ]
    [ Sobel    | Frei-Chen | (kosong / logo) ]
    """
    t0 = f"{title_prefix}Original"
    t1 = f"{title_prefix}Roberts"
    t2 = f"{title_prefix}Prewitt"
    t3 = f"{title_prefix}Sobel"
    t4 = f"{title_prefix}Frei-Chen"

    tiles = [
        add_label_bar(orig, t0),
        add_label_bar(rob,  t1),
        add_label_bar(prew, t2),
        add_label_bar(sob,  t3),
        add_label_bar(frei, t4),
    ]

    # bikin kotak putih kosong untuk slot ke-6
    h, w = tiles[0].shape[:2]
    empty = np.full((h, w, 3), 255, dtype=np.uint8)
    tiles.append(add_label_bar(cv2.cvtColor(
        cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), ""))

    row1 = cv2.hconcat(tiles[0:3])
    row2 = cv2.hconcat(tiles[3:6])
    panel = cv2.vconcat([row1, row2])

    cv2.imwrite(out_path, panel)

# ============================================================
#  PROSES UNTUK SATU CITRA
# ============================================================

def process_single_image(img_path: str, tag: str, out_root: str = "output"):
    """
    Memproses satu citra: Roberts, Prewitt, Sobel, Frei-Chen.
    Menyimpan hasil per-metode dan satu panel gabungan.
    """
    print(f"\n=== Memproses: {tag} ({img_path}) ===")

    img_gray = load_as_gray(img_path)

    # folder khusus untuk citra ini
    out_dir = os.path.join(out_root, tag)
    ensure_dir(out_dir)

    # simpan grayscale dasar
    base_gray_path = os.path.join(out_dir, f"{tag}_gray.png")
    cv2.imwrite(base_gray_path, img_gray)

    # hitung masing-masing metode
    edge_rob = edge_roberts(img_gray)
    edge_pre = edge_prewitt(img_gray)
    edge_sob = edge_sobel(img_gray)
    edge_fre = edge_freichen(img_gray)

    # simpan masing-masing
    cv2.imwrite(os.path.join(out_dir, f"{tag}_roberts.png"), edge_rob)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_prewitt.png"), edge_pre)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_sobel.png"), edge_sob)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_freichen.png"), edge_fre)

    # simpan panel 2x3
    panel_path = os.path.join(out_dir, f"{tag}_panel_edges.png")
    save_edge_panel(
        img_gray, edge_rob, edge_pre, edge_sob, edge_fre,
        panel_path,
        title_prefix=f"{tag} - "
    )

    print(f"✔ Hasil disimpan di folder: {out_dir}")
    print(f"   - Grayscale  : {base_gray_path}")
    print(f"   - Roberts    : {tag}_roberts.png")
    print(f"   - Prewitt    : {tag}_prewitt.png")
    print(f"   - Sobel      : {tag}_sobel.png")
    print(f"   - Frei-Chen  : {tag}_freichen.png")
    print(f"   - Panel      : {tag}_panel_edges.png")

# ============================================================
#  MAIN PROGRAMs
# ============================================================

if __name__ == "__main__":
    # GANTI path ini sesuai 4 citra dari tugas restorasi-mu
    # Misal:
    #   1) citra grayscale asli
    #   2) citra dengan derau salt & pepper
    #   3) citra dengan derau gaussian
    #   4) citra hasil restorasi / citra lain yang relevan
    IMAGE_LIST = [
        ("gray_clean",   "potrait_gray.png"),
        ("noise_sp",     "potrait_gray_sp_8.png.png"),
        ("noise_gauss",  "potrait_gray_gauss_25.png.png"),
        ("restored",     "potrait.png"),
    ]

    out_root = "output_segmentasi"
    ensure_dir(out_root)

    for tag, path in IMAGE_LIST:
        if not os.path.isfile(path):
            print(f"[PERINGATAN] File tidak ditemukan: {path} (tag: {tag})")
            continue
        process_single_image(path, tag, out_root=out_root)

    print("\n=== SELESAI SEMUA CITRA ===")
