import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import tifffile
import argparse


def find_matching_files(sam_folder, brightness_folder):
    """Find matching mask and image file pairs"""
    mask_files = list(sam_folder.glob("*_masks.npy"))
    file_pairs = []

    for mask_file in mask_files:
        original_stem = mask_file.stem.replace('_masks', '')
        tif_file = brightness_folder / f"{original_stem}.tif"

        if tif_file.exists():
            file_pairs.append((mask_file, tif_file))
        else:
            print(f"No matching tif file found for {mask_file}")

    return file_pairs


def load_data(mask_path, image_path):
    """Load mask and image data with proper 16-bit handling"""
    masks = np.load(mask_path)
    image = tifffile.imread(image_path)

    if image.ndim == 3:
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

    return masks, image


def display_image(ax, image):
    """Display image with proper scaling based on dtype"""
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    elif image.dtype in [np.float32, np.float64]:
        display_img = np.clip(image, 0, 1)
    else:
        display_img = image

    ax.imshow(display_img)
    return display_img


def get_user_selection(regions):
    """Prompt user to select cells to analyze"""
    while True:
        try:
            cell_numbers = input(f"Enter cell numbers (1-{len(regions)}, space-separated, or 'all': ")

            if cell_numbers.lower() == 'all':
                return list(range(1, len(regions) + 1))

            selected = [int(num) for num in cell_numbers.split()]
            if all(1 <= num <= len(regions) for num in selected):
                return selected
            print(f"Numbers must be between 1 and {len(regions)}")
        except ValueError:
            print("Please enter numbers separated by spaces")


def calculate_axis_endpoints(centroid, length, angle, is_major=True):
    """Calculate endpoints for axis lines"""
    y0, x0 = centroid
    if is_major:
        x1 = x0 + np.cos(angle) * 0.5 * length
        y1 = y0 - np.sin(angle) * 0.5 * length
        x2 = x0 - np.cos(angle) * 0.5 * length
        y2 = y0 + np.sin(angle) * 0.5 * length
    else:
        x1 = x0 - np.sin(angle) * 0.5 * length
        y1 = y0 - np.cos(angle) * 0.5 * length
        x2 = x0 + np.sin(angle) * 0.5 * length
        y2 = y0 + np.cos(angle) * 0.5 * length

    return (x1, y1), (x2, y2)


def plot_cell_axes(ax, region, cell_num, is_first_cell=False):
    """Plot major and minor axes for a single cell with centroid point"""
    centroid = region.centroid
    minor_len = region.minor_axis_length
    major_len = region.major_axis_length
    orientation = region.orientation

    # Calculate endpoints
    major_p1, major_p2 = calculate_axis_endpoints(centroid, major_len, orientation, True)
    minor_p1, minor_p2 = calculate_axis_endpoints(centroid, minor_len, orientation, False)

    ax.plot([major_p1[0], major_p2[0]], [major_p1[1], major_p2[1]],
            color='red', linewidth=1.5,
            label='Major Axis' if is_first_cell else '')
    ax.plot([minor_p1[0], minor_p2[0]], [minor_p1[1], minor_p2[1]],
            color='blue', linewidth=1.5,
            label='Minor Axis' if is_first_cell else '')

    y0, x0 = centroid
    ax.plot(x0, y0, 'yo', markersize=3)

    return major_len, minor_len


def plot_cell_intensities(image, mask, cell_num):
    """Plot intensity distributions for each channel of a single cell"""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image must be RGB (3 channels)")

    cell_mask = mask == cell_num
    channels = ['Red', 'Green', 'Blue']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Cell {cell_num} Intensity Distributions', fontsize=14)

    for i, channel in enumerate(channels):
        channel_data = image[:, :, i]
        masked_data = channel_data[cell_mask] / 4.5

        axes[i].hist(masked_data.flatten(), bins=50, color=channel.lower(), alpha=0.7)
        axes[i].set_title(f'{channel} Channel')
        axes[i].set_xlabel('Intensity Value')
        axes[i].set_ylabel('Pixel Count')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def analyze_image_pair(mask_path, image_path, pipeline_mode=False):
    """Main analysis function for one image/mask pair"""
    print(f"\nProcessing {mask_path} and {image_path}")

    TIME_INTERVAL = 40  # Time between frames in minutes
    PIXEL_SIZE_NM = 0.03224  # Microscope pixel size in nanometer
    masks, image = load_data(mask_path, image_path)
    regions = regionprops(masks)
    print(f"Found {len(regions)} cells in the mask")

    if pipeline_mode:
        # Predefined cells to analyze in pipeline mode
        predefined_cells = [77, 108, 199]
        selected_cells = [cell for cell in predefined_cells if 1 <= cell <= len(regions)]
        if not selected_cells:
            print(f"No predefined cells found in this image (available: 1-{len(regions)})")
            return
    else:
        # Get user selection in interactive mode
        selected_cells = get_user_selection(regions)

    fig, ax = plt.subplots(figsize=(10, 10))
    display_img = display_image(ax, image)

    for i, cell_num in enumerate(selected_cells):
        region = regions[cell_num - 1]
        major_px, minor_px = plot_cell_axes(ax, region, cell_num, is_first_cell=(i == 0))

        # Calculate half of major axis length (from centroid to endpoint)
        half_major_px = major_px
        half_major_nm = half_major_px * PIXEL_SIZE_NM
        growth_rate = half_major_nm / TIME_INTERVAL  # nm/min

        print(f"\nCell {cell_num}:")
        print(f"  Major Axis: {major_px:.1f} px ({major_px * PIXEL_SIZE_NM:.1f} mm)")
        print(f"  Half Major Axis (centroid to endpoint): {half_major_px:.1f} px ({half_major_nm:.1f} mm)")
        print(f"  Minor Axis: {minor_px:.1f} px")
        print(f"  Growth Rate (half axis): {growth_rate:.2f} mm/min")


        plot_cell_intensities(image, masks, cell_num)

    if len(selected_cells) > 0:
        ax.legend()

    plt.title(f"Analysis of {mask_path.stem.replace('_masks', '')}")
    plt.show()


def main():
    """Main program entry point"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cell analysis script')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run in pipeline mode with predefined cells')
    args = parser.parse_args()

    sam_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
    brightness_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\train_brightness")


    file_pairs = find_matching_files(sam_folder, brightness_folder)

    if not file_pairs:
        print("No matching file pairs found!")
        return

    if args.pipeline:
        for mask_path, image_path in file_pairs:
            analyze_image_pair(mask_path, image_path, pipeline_mode=True)
    else:
        # Free mode - interactive selection
        print("\nFound the following file pairs:")
        for i, (mask_path, image_path) in enumerate(file_pairs, 1):
            print(f"{i}: {mask_path.name} with {image_path.name}")


        while True:
            try:
                selection = input(f"\nEnter which file to process (1-{len(file_pairs)}), or 'all': ")

                if selection.lower() == 'all':

                    for mask_path, image_path in file_pairs:
                        analyze_image_pair(mask_path, image_path)
                    break
                else:

                    selected_idx = int(selection) - 1
                    if 0 <= selected_idx < len(file_pairs):
                        mask_path, image_path = file_pairs[selected_idx]
                        analyze_image_pair(mask_path, image_path)
                        break
                    print(f"Please enter a number between 1 and {len(file_pairs)}")
            except ValueError:
                print("Please enter a valid number or 'all'")


if __name__ == "__main__":
    main()