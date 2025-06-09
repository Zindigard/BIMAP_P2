import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from cellpose import models, core, io, plot
from pathlib import Path
from cellpose import utils
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
from skimage.io import imsave

def image_segmentation():
    io.logger_setup()
    model = models.CellposeModel(gpu=False)

    dir_path = r"C:\Users\zindi\PycharmProjects\P2\train data"
    dir = Path(dir_path)

    output_dir = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
    output_dir.mkdir(parents=True, exist_ok=True)

    brightness_dir = r"C:\Users\zindi\PycharmProjects\P2\train_brightness"
    brightness_path = Path(brightness_dir)

    image_ext = ".tif"

    files = natsorted([f for f in dir.glob(f"*{image_ext}")
                       if "_masks" not in f.name and "_flows" not in f.name])

    if len(files) == 0:
        raise FileNotFoundError("No image files found, did you specify the correct folder and extension?")
    else:
        print(f"{len(files)} images in folder:")

    for f in files:
        print(f.name)

    for file in files:
        original_name = file.stem
        outlines_file_path = output_dir / f"{original_name}_outlined.tif"
        mask_file_path = output_dir / f"{original_name}_binary.tif"

        # Check exists
        if outlines_file_path.exists() and mask_file_path.exists():
            print(f"\nSegmentation file already exists for {file.name}, skipping...")
            continue

        print(f"\nProcessing {file.name}...")

        img = io.imread(file)

        img_rgb = np.transpose(img, (1, 2, 0))
        if img_rgb.dtype == np.uint16:
            img_rg1 = (img_rgb / 256).astype(np.uint8)
        else:
            img_rg1 = img_rgb.astype(np.uint8)
        print(f"Image shape: {img_rg1.shape}")

        first_channel = '0'
        second_channel = '1'
        third_channel = '2'

        selected_channels = []
        for i, c in enumerate([first_channel, second_channel, third_channel]):
            if c == 'None':
                continue
            if int(c) > img_rg1.ndim:
                assert False, 'invalid channel index, must have index greater or equal to the number of channels'
            if c != 'None':
                selected_channels.append(int(c))

        img_selected_channels = np.zeros_like(img_rg1)
        print('Selected channels:', selected_channels)
        img_selected_channels[:, :, :len(selected_channels)] = img_rg1[:, :, selected_channels]

        flow_threshold = 0.4
        cellprob_threshold = 0.1
        tile_norm_blocksize = 0

        print("Running Cellpose segmentation...")
        masks, flows, styles = model.eval(img_selected_channels, batch_size=8,
                                          flow_threshold=flow_threshold,
                                          cellprob_threshold=cellprob_threshold,
                                          normalize={"tile_norm_blocksize": tile_norm_blocksize})



        vis_img = None
        brightness_file = brightness_path / file.name
        if brightness_file.exists():
            vis_img = io.imread(brightness_file)
            if len(vis_img.shape) == 3:
                vis_img = vis_img.transpose(1, 2, 0)
            vis_img = (vis_img / 256).astype(np.uint8) if vis_img.dtype == np.uint16 else vis_img.astype(np.uint8)
        else:
            vis_img = img_rg1


        binary_mask = (masks > 0).astype(np.uint8) * 255
        binary_mask_path = output_dir / f"{original_name}_binary.tif"
        io.imsave(binary_mask_path, binary_mask)
        print(f"Saved binary mask to: {binary_mask_path}")

        #  save outlined mask
        outlines = utils.outlines_list(masks)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rg1)
        for outline in outlines:
            plt.plot(outline[:, 0], outline[:, 1], 'y', linewidth=1)
        plt.axis('off')
        outlined_mask_path = output_dir / f"{original_name}_outlined.tif"
        plt.savefig(outlined_mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved outlined mask to: {outlined_mask_path}")

        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
        plt.tight_layout()
        plt.show()

        # binary mask
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')
        plt.show()

        #  outlines on original image
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        for outline in outlines:
            plt.plot(outline[:, 0], outline[:, 1], 'y', linewidth=1)
        plt.title('Outlines on Original Image')
        plt.axis('off')
        plt.show()

        npy_save_path = output_dir / f"{original_name}_masks.npy"
        np.save(npy_save_path, masks)
        print(f"Masks saved to: {npy_save_path}")



if __name__ == "__main__":
    image_segmentation()


