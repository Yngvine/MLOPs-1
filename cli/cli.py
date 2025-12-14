"""
CLI implementation for mylib functions.
"""
import click
import cv2
import numpy as np

from mylib import predict_class, crop_image, normalize_image

@click.group()
def cli():
    """CLI for image processing and classification."""
    
@cli.command()
@click.argument("img_path", type=click.Path(exists=True))
def classify(img_path):
    """Classify an image located at IMG_PATH."""
    
    img = cv2.imread(img_path)
    pred_class = predict_class(img)
    click.echo(f"Predicted class: {pred_class}")

@cli.command()
@click.argument("img_path", type=click.Path(exists=True))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.argument("dest_path", type=click.Path())
def crop(img_path, width, height, dest_path):
    """Crop an image located at IMG_PATH to WIDTH x HEIGHT and save it to DEST_PATH."""
    
    img = cv2.imread(img_path)
    cropped_img = crop_image(img, (width, height))
    cv2.imwrite(dest_path, cropped_img)
    click.echo(f"Cropped image saved to: {dest_path}")

@cli.command()
@click.argument("img_path", type=click.Path(exists=True))
@click.argument("dest_path", type=click.Path())
def normalize(img_path, dest_path):
    """Normalize an image located at IMG_PATH and save it to DEST_PATH."""
    
    img = cv2.imread(img_path)
    # get the normalized data and ensure it's a NumPy array with uint8 dtype
    normalized = normalize_image(img)
    normalized_img = np.asarray(normalized, dtype=np.uint8)
    cv2.imwrite(dest_path, normalized_img)
    click.echo(f"Normalized image saved to: {dest_path}")
    

if __name__ == "__main__":
    cli()