import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def evaluate_and_visualize_masks(pred_path, gt_path, output_path):
    """
    Evaluate segmentation masks with interactive display and file saving.

    Args:
        pred_path: Path to predicted masks (format: *_binary.tif)
        gt_path: Path to ground truth masks (format: *_mask.tif)
        output_path: Directory to save results

    Returns:
        List of dictionaries containing metrics for each image pair
    """

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    def calculate_metrics(pred, gt):
        """Calculate IoU, Precision, Accuracy with detailed components"""
        tp = np.sum(np.logical_and(pred == 1, gt == 1))  # True positives
        fp = np.sum(np.logical_and(pred == 1, gt == 0))  # False positives
        fn = np.sum(np.logical_and(pred == 0, gt == 1))  # False negatives
        tn = np.sum(np.logical_and(pred == 0, gt == 0))  # True negatives

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'iou': iou,
            'precision': precision,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    def create_visualization(pred, gt, filename, metrics):
        """Create and display comparison plot with metrics"""
        # Create RGB overlay
        overlay = np.zeros((*pred.shape, 3))
        overlay[gt == 1] = [0, 1, 0]  # Green - Ground Truth
        overlay[pred == 1] = [1, 0, 0]  # Red - Prediction
        overlay[np.logical_and(pred, gt)] = [0.5, 0, 0.5]  # Purple - Overlap

        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.title(f"Segmentation Evaluation: {os.path.splitext(filename)[0]}", pad=20)
        plt.axis('off')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=4, label='Ground Truth'),
            plt.Line2D([0], [0], color='red', lw=4, label='Prediction'),
            plt.Line2D([0], [0], color='purple', lw=4, label='Overlap')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Metrics box
        metrics_text = (
            f"Evaluation Metrics:\n"
            f"• IoU (Jaccard): {metrics['iou']:.3f}\n"
            f"• Precision: {metrics['precision']:.3f}\n"
            f"• Accuracy: {metrics['accuracy']:.3f}\n"
            f"\nConfusion Matrix:\n"
            f"TP: {metrics['tp']} | FP: {metrics['fp']}\n"
            f"FN: {metrics['fn']} | TN: {metrics['tn']}"
        )

        plt.gcf().text(0.82, 0.70, metrics_text,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                       fontsize=9, fontfamily='monospace')

        # Save the figure
        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_eval.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=150)

        # Display in interactive window
        plt.show(block=False)
        plt.pause(0.1)
        return output_file

    # Process all images
    results = []
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('_binary.tif')]

    for pred_file in pred_files:
        try:
            # Find matching ground truth
            gt_file = pred_file.replace('_binary', '_mask')
            gt_path_full = os.path.join(gt_path, gt_file)

            if not os.path.exists(gt_path_full):
                print(f"[SKIPPED] No ground truth found for {pred_file}")
                continue

            # Load and binarize masks
            pred = (io.imread(os.path.join(pred_path, pred_file)) > 0).astype(np.uint8)
            gt = (io.imread(gt_path_full) > 0).astype(np.uint8)

            # Calculate metrics
            metrics = calculate_metrics(pred, gt)

            # Generate and display visualization
            viz_path = create_visualization(pred, gt, pred_file, metrics)

            # Store results
            results.append({
                'image': pred_file,
                **metrics,
                'visualization': viz_path
            })

            print(f"[PROCESSED] {pred_file}")
            print(
                f"  IoU: {metrics['iou']:.3f} | Precision: {metrics['precision']:.3f} | Accuracy: {metrics['accuracy']:.3f}")

        except Exception as e:
            print(f"[ERROR] Processing {pred_file}: {str(e)}")

    # Save comprehensive results
    csv_path = os.path.join(output_path, 'segmentation_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("Image,IoU,Precision,Accuracy,TP,FP,FN,TN,Visualization\n")
        for r in results:
            f.write(f"{r['image']},{r['iou']},{r['precision']},{r['accuracy']},"
                    f"{r['tp']},{r['fp']},{r['fn']},{r['tn']},{r['visualization']}\n")

    print(f"\nEvaluation complete. Processed {len(results)} images.")
    print(f"Results saved to: {output_path}")
    return results


# Example usage
if __name__ == "__main__":
    # Set matplotlib backend for interactive display
    plt.switch_backend('TkAgg')  # or 'Qt5Agg' if you have PyQt installed

    results = evaluate_and_visualize_masks(
        pred_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM",
        gt_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Ground",
        output_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Results"
    )

    # Keep plots open until manually closed
    plt.show()