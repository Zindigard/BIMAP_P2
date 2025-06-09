import tifffile
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Load TIFF
tif_path = r"C:\Users\zindi\PycharmProjects\PythonProject\train_brightness\WT_NADA_RADA_HADA_NHS_40min_ROI3_SIM.tif"
img_data = tifffile.imread(tif_path)
print(img_data.shape)
img_rgb = img_data.transpose(1, 2, 0)

img_normalized = img_data.astype('float32') / 65535.0

# Transpose to (height, width, channels) and plot
plt.imshow(img_normalized.transpose(1, 2, 0))
plt.axis('off')
plt.show()
if img_rgb.dtype == np.uint16:
    # Option A: Normalize to 8-bit (0-255)
    img_8bit = (img_rgb / 256).astype(np.uint8)

    # Option B: Preserve dynamic range with percentile scaling
    p2, p98 = np.percentile(img_rgb, [2, 98])
    img_scaled = np.clip((img_rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
else:
    img_8bit = img_rgb.astype(np.uint8)

# 3. Convert to PIL Image and save
Image.fromarray(img_8bit).save('output2.png')

print("Successfully converted to PNG!")
img = Image.open('output.png')
img_array = np.array(img)  # Convert to numpy array

# Check image shape and channels
print("Image shape:", img_array.shape)  # (height, width, channels)

# Extract individual channels
if img_array.ndim == 3 and img_array.shape[2] >= 3:  # RGB or RGBA
    red_channel = img_array[:, :, 0]  # Red channel
    green_channel = img_array[:, :, 1]  # Green channel
    blue_channel = img_array[:, :, 2]  # Blue channel

    if img_array.shape[2] == 4:  # If there's an alpha channel
        alpha_channel = img_array[:, :, 3]  # Transparency channel
else:
    # Grayscale image (single channel)
    gray_channel = img_array


# Plot all channels
fig, axes = plt.subplots(1, 4 if img_array.shape[2] == 4 else 3, figsize=(15, 5))

# Plot RGB channels
axes[0].imshow(red_channel, cmap='Reds')
axes[0].set_title('Red Channel')
axes[0].axis('off')

axes[1].imshow(green_channel, cmap='Greens')
axes[1].set_title('Green Channel')
axes[1].axis('off')

axes[2].imshow(blue_channel, cmap='Blues')
axes[2].set_title('Blue Channel')
axes[2].axis('off')

# Plot alpha channel if exists
if img_array.shape[2] == 4:
    axes[3].imshow(alpha_channel, cmap='gray')
    axes[3].set_title('Alpha Channel')
    axes[3].axis('off')

plt.tight_layout()
plt.show()