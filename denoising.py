import numpy as np
import matplotlib.pyplot as plt
from czifile import imread
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --------- SETTINGS ---------
czi_path = "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"  # Input 2D image file
sigma = 25  # Noise standard deviation (tune for better denoising)
R = 255  # Maximum possible pixel value (for 8-bit images)
# ----------------------------

# --- Load the image ---
print(f"📂 Opening image: {czi_path}")
raw = imread(czi_path)
img = np.squeeze(raw)[0]  # Assuming the image is in the first channel (grayscale)
print(f"✅ Final image shape: {img.shape}")

# --- Visualize the Original Image ---
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()

# --- Calculate PSNR and SSIM for the Original Image ---
psnr_original = psnr(img, img)  # PSNR between original and itself (should be infinity or very high)
ssim_original = ssim(img, img, data_range=img.max() - img.min())  # SSIM for original image

# --- Denoising using BM3D ---
print("🔍 Denoising using BM3D...")
denoised_img = bm3d.bm3d(img, sigma_psd=sigma**2)  # sigma is the noise standard deviation

# --- Visualize the Denoised Image ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(denoised_img, cmap='gray')
axes[1].set_title("Denoised Image (BM3D)")
axes[1].axis("off")

plt.show()

# --- Calculate PSNR and SSIM for the Denoised Image ---
psnr_denoised = psnr(img, denoised_img)  # PSNR for denoised image
ssim_denoised = ssim(img, denoised_img, data_range=denoised_img.max() - denoised_img.min())  # SSIM for denoised image

# --- Print Results ---
print(f"\n🔧 PSNR of the Original Image: {psnr_original:.2f} dB")
print(f"🔧 SSIM of the Original Image: {ssim_original:.2f}")
print(f"🔧 PSNR of the Denoised Image: {psnr_denoised:.2f} dB")
print(f"🔧 SSIM of the Denoised Image: {ssim_denoised:.2f}")

# --- Conclusion based on Metrics ---
if psnr_denoised > psnr_original and ssim_denoised > ssim_original:
    print("\n✅ The denoising process improved the image quality!")
elif psnr_denoised < psnr_original and ssim_denoised < ssim_original:
    print("\n❌ The denoising process worsened the image quality (introducing artifacts).")
else:
    print("\n⚠️ The denoising process had minimal effect or was unnecessary.")

