import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

def draw_detections(image, detections, plot_filename):
    """
    Draws bounding boxes from detections on a PIL image using matplotlib.

    Args:
        image (PIL.Image): The input image.
        detections (dict): A dictionary mapping object names to a list of bounding boxes.
                          Each bbox is in (x1, y1, x2, y2) format, in pixels.

    Returns:
        None (displays the image with bounding boxes).
    """
    plt.clf()

    # Convert PIL image to NumPy array for display
    img_array = np.array(image)

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img_array)  # Show the image

    # Define colors for different categories
    color_map = plt.cm.get_cmap("tab10")  # 10 distinct colors

    # Iterate over detections
    for i, (label, bboxes) in enumerate(detections.items()):
        color = color_map(i % 10)  # Get a distinct color for each category

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Create a rectangle patch for the bbox
            rect = patches.Rectangle(
                (x1, y1), bbox_width, bbox_height, linewidth=2,
                edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Text label (Object name + center)
            text_label = '%s(%d,%d)'%(label, (x1+x2)/2, (y1+y2)/2)
            ax.text(
                x1, y1 - 5, text_label, fontsize=10, color="white",
                bbox=dict(facecolor=color, edgecolor=color, boxstyle="round,pad=0.3", alpha=0.7)
            )

    # Remove axes and display
    plt.axis("off")
    plt.savefig(plot_filename)
    plt.clf()
    plt.close()
