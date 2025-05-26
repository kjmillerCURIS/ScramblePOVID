import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


DPI = 100  # Use 100 DPI for figure size estimation
MAX_WIDTH_INCHES = 12
WRAP_CHARS = 100


def display_image_with_caption(image_path, caption, vis_dir):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Calculate image size in inches (preserve resolution)
    width_in = img.width / DPI
    height_in = img.height / DPI

    # Cap width to avoid going off-screen
    width_in = min(width_in, MAX_WIDTH_INCHES)
    height_in = height_in * (width_in / (img.width / DPI))  # maintain aspect ratio

    # Create figure
    plt.clf()
    fig, ax = plt.subplots(figsize=(width_in, height_in + 1))  # +1 for caption space

    #wrappity wrap
    wrapped_caption = "\n".join(textwrap.wrap(caption, WRAP_CHARS))

    # Show caption above image
    ax.set_title(caption, wrap=True)

    # Display image
    ax.imshow(img)
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, os.path.basename(image_path)))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    display_image_with_caption(
        image_path='dog_lying_on_rug.jpg',
        caption='This is a long caption that will automatically wrap across multiple lines if needed, and the image will retain its resolution without blowing up or shrinking weirdly.',
        vis_dir='ohio_rizz'
       )
