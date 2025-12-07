# Simple edge segmentation comparison script
# Runs in notebook, reads provided image, applies Roberts, Prewitt, Sobel, Frei-Chen
# on clean and noisy versions, computes MSE between clean-edge and noisy-edge magnitudes,
# displays results table and a bar chart.
# Requirements: opencv-python, numpy, pandas, matplotlib (available in this env).

import cv2, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

# --- helpers ---
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_gray(p):
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def normalize_to_uint8(imgf):
    mn, mx = float(imgf.min()), float(imgf.max())
    if mx - mn < 1e-8:
        return np.zeros_like(imgf, dtype=np.uint8)
    norm = (imgf - mn) / (mx - mn) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b)**2))

def add_salt_pepper(img, prob=0.1):
    out = img.copy()
    h,w = img.shape[:2]
    rnd = np.random.rand(h,w)
    salt = rnd < prob/2
    pepper = rnd > 1 - prob/2
    if img.ndim==2:
        out[salt] = 255
        out[pepper] = 0
    else:
        out[salt,:] = 255
        out[pepper,:] = 0
    return out

def add_gaussian(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def convolve2d_gray(img, kernel):
    imgf = img.astype(np.float32)
    kh,kw = kernel.shape
    ph,pw = kh//2, kw//2
    padded = cv2.copyMakeBorder(imgf, ph, ph, pw, pw, borderType=cv2.BORDER_REPLICATE)
    h,w = img.shape
    out = np.zeros((h,w), dtype=np.float32)
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * k)
    return out

# --- kernels ---
# Roberts (two 2x2 kernels for gx, gy)
roberts_gx = np.array([[1, 0],
                       [0,-1]], dtype=np.float32)
roberts_gy = np.array([[0, 1],
                       [-1,0]], dtype=np.float32)

# Prewitt (3x3)
prew_gx = np.array([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]], dtype=np.float32)
prew_gy = np.array([[-1,-1,-1],
                    [ 0, 0, 0],
                    [ 1, 1, 1]], dtype=np.float32)

# Sobel (3x3)
sobel_gx = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]], dtype=np.float32)
sobel_gy = np.array([[-1,-2,-1],
                     [ 0, 0, 0],
                     [ 1, 2, 1]], dtype=np.float32)

# Frei-Chen pair (approximation)
s2 = math.sqrt(2.0)
freigx = np.array([[1, s2, 1],
                   [0, 0, 0],
                   [-1,-s2,-1]], dtype=np.float32)
freigy = np.array([[1, 0, -1],
                   [s2,0,-s2],
                   [1, 0, -1]], dtype=np.float32)

OPERATORS = {
    "Roberts": (roberts_gx, roberts_gy),
    "Prewitt": (prew_gx, prew_gy),
    "Sobel": (sobel_gx, sobel_gy),
    "Frei-Chen": (freigx, freigy)
}

# --- main processing ---
img_path = "/mnt/data/0825f1ab-400e-426f-9325-23b49b309a88.png"
img_gray = load_gray(img_path)

# create noisy versions
sp_prob = 0.15   # salt & pepper 15%
gauss_sigma = 25  # gaussian sigma
img_sp = add_salt_pepper(img_gray, prob=sp_prob)
img_gauss = add_gaussian(img_gray, sigma=gauss_sigma)

# compute edge magnitudes for clean and noisy for each operator
results = []
out_dir = "/mnt/data/segment_results_simple"
ensure_dir(out_dir)

# store images for visualization (magnitude normalized)
mag_images = {"clean":{}, "sp":{}, "gauss":{}}

for name, (kx, ky) in OPERATORS.items():
    gx_clean = convolve2d_gray(img_gray, kx)
    gy_clean = convolve2d_gray(img_gray, ky)
    mag_clean = np.sqrt(gx_clean**2 + gy_clean**2)
    mag_images["clean"][name] = normalize_to_uint8(mag_clean)
    cv2.imwrite(os.path.join(out_dir, f"{name}_mag_clean.png"), mag_images["clean"][name])

    gx_sp = convolve2d_gray(img_sp, kx)
    gy_sp = convolve2d_gray(img_sp, ky)
    mag_sp = np.sqrt(gx_sp**2 + gy_sp**2)
    mag_images["sp"][name] = normalize_to_uint8(mag_sp)
    cv2.imwrite(os.path.join(out_dir, f"{name}_mag_sp.png"), mag_images["sp"][name])

    gx_g = convolve2d_gray(img_gauss, kx)
    gy_g = convolve2d_gray(img_gauss, ky)
    mag_g = np.sqrt(gx_g**2 + gy_g**2)
    mag_images["gauss"][name] = normalize_to_uint8(mag_g)
    cv2.imwrite(os.path.join(out_dir, f"{name}_mag_gauss.png"), mag_images["gauss"][name])

    # compute MSE between clean magnitude (float) and noisy magnitude (float)
    mse_sp = mse(mag_clean, mag_sp)
    mse_gauss = mse(mag_clean, mag_g)
    results.append({"Operator": name, "Noise": "Salt&Pepper", "MSE": mse_sp})
    results.append({"Operator": name, "Noise": "Gaussian", "MSE": mse_gauss})

# build DataFrame table
df = pd.DataFrame(results)
display_dataframe_to_user("MSE Comparison (edge magnitude)", df)

# Save a CSV
csv_path = os.path.join(out_dir, "mse_comparison.csv")
df.to_csv(csv_path, index=False)

# --- Plot bar chart grouped by operator ---
operators = list(OPERATORS.keys())
mse_sp_vals = [df[(df.Operator==op) & (df.Noise=="Salt&Pepper")]["MSE"].values[0] for op in operators]
mse_gauss_vals = [df[(df.Operator==op) & (df.Noise=="Gaussian")]["MSE"].values[0] for op in operators]

x = np.arange(len(operators))
width = 0.35

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(x - width/2, mse_sp_vals, width, label='Salt&Pepper')
ax.bar(x + width/2, mse_gauss_vals, width, label='Gaussian')
ax.set_ylabel('MSE (magnitude)')
ax.set_title('MSE between clean-edge and noisy-edge (per operator)')
ax.set_xticks(x)
ax.set_xticklabels(operators, rotation=10)
ax.legend()
plt.tight_layout()
plt_path = os.path.join(out_dir, "mse_bar.png")
plt.savefig(plt_path)
plt.show()

# show small montage of magnitude images for quick visual
def make_row(names, kind):
    imgs = [mag_images[kind][n] for n in names]
    labels = [n for n in names]
    rows = []
    for im,l in zip(imgs, labels):
        lab = np.full((30, im.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(lab, l, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1, cv2.LINE_AA)
        imc = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        rows.append(np.vstack([imc, lab]))
    return cv2.hconcat(rows)

row_clean = make_row(operators, "clean")
row_sp = make_row(operators, "sp")
row_gauss = make_row(operators, "gauss")

cv2.imwrite(os.path.join(out_dir, "panel_clean_row.png"), row_clean)
cv2.imwrite(os.path.join(out_dir, "panel_sp_row.png"), row_sp)
cv2.imwrite(os.path.join(out_dir, "panel_gauss_row.png"), row_gauss)

print(f"\nImages and CSV saved under: {out_dir}")
print(f" - CSV: {csv_path}")
print(f" - Bar chart: {plt_path}")
print(" - Panels: panel_clean_row.png, panel_sp_row.png, panel_gauss_row.png")
