import os
from PIL import Image
import numpy as np

# Function to process an image and save vertical slices
def process_image(image_path, output_folder):
    # Load and convert the image to grayscale
    image = Image.open(image_path)
    gray_image = image.convert('L')

    # Convert the grayscale image to a numpy array
    image_array = np.array(gray_image)

    # Threshold the image to create a binary mask
    binary_mask = image_array > 0  # Assuming white lines on a black background

    # Find the vertical slices by detecting columns with content
    columns_with_content = np.any(binary_mask, axis=0)
    column_indices = np.where(columns_with_content)[0]

    # Identify the ranges of columns that correspond to individual objects
    slices = []
    start = None
    for i in range(len(column_indices)):
        if start is None:
            start = column_indices[i]
        if i == len(column_indices) - 1 or column_indices[i + 1] != column_indices[i] + 1:
            slices.append((start, column_indices[i]))
            start = None

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Slice the image for each detected range and save to the output folder
    for idx, (start, end) in enumerate(slices):
        sliced_image = image.crop((start, 0, end + 1, image.height))
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_slice_{idx + 1}.png")
        sliced_image.save(output_path)
    print(f"Reprocessed and saved slices for {image_path}")

# Reprocess the problematic image
def main():
    input_folder = "."  # Folder containing your input images
    output_folder = "./cropped_roots_test"  # Folder to save cropped images

    # Specific image to reprocess
    image_file = "task5_test_image_8_prediction_jeremy_232189.png"
    image_path = os.path.join(input_folder, image_file)

    # Reprocess and save slices
    process_image(image_path, output_folder)

if __name__ == "__main__":
    main()
