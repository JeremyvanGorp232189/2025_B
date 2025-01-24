import os
from PIL import Image
import numpy as np
from scipy.ndimage import label

# Function to process an image using the original slicing logic
def process_image_original(image_path, output_folder):
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
    print(f"Processed and saved slices for {image_path} using original logic")

# Function to process image 8 using the improved slicing logic
def process_image_8(image_path, output_folder):
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

    # Process each slice
    slice_idx = 1
    for start, end in slices:
        # Crop the slice from the image
        slice_image = binary_mask[:, start:end + 1]

        # Label connected components within the slice
        labeled_array, num_features = label(slice_image)

        # Process each connected component as a separate slice
        for component in range(1, num_features + 1):
            component_mask = labeled_array == component

            # Find bounding box of the component
            rows, cols = np.where(component_mask)
            if rows.size > 0 and cols.size > 0:
                col_start = cols.min() + start
                col_end = cols.max() + start

                # Crop the original image for this component
                cropped_component = image.crop((col_start, 0, col_end + 1, image.height))

                # Save the cropped slice
                output_path = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_slice_{slice_idx}.png")
                cropped_component.save(output_path)
                slice_idx += 1

    print(f"Processed and saved slices for {image_path} using improved logic")

# Main script to process all images
def main():
    input_folder = "."  # Folder containing your input images
    output_folder = "./cropped_roots"  # Folder to save cropped images

    # List of all image files
    image_files = [
        "task5_test_image_1_prediction_jeremy_232189.png",
        "task5_test_image_2_prediction_jeremy_232189.png",
        "task5_test_image_3_prediction_jeremy_232189.png",
        "task5_test_image_4_prediction_jeremy_232189.png",
        "task5_test_image_5_prediction_jeremy_232189.png",
        "task5_test_image_6_prediction_jeremy_232189.png",
        "task5_test_image_7_prediction_jeremy_232189.png",
        "task5_test_image_8_prediction_jeremy_232189.png",  # Special case
        "task5_test_image_9_prediction_jeremy_232189.png",
        "task5_test_image_10_prediction_jeremy_232189.png",
        "task5_test_image_11_prediction_jeremy_232189.png",
        "task5_test_image_12_prediction_jeremy_232189.png",
        "task5_test_image_13_prediction_jeremy_232189.png",
        "task5_test_image_14_prediction_jeremy_232189.png",
        "task5_test_image_15_prediction_jeremy_232189.png",
        "task5_test_image_16_prediction_jeremy_232189.png",
        "task5_test_image_17_prediction_jeremy_232189.png",
        "task5_test_image_18_prediction_jeremy_232189.png",
    ]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Use the improved logic for image 8, and original logic for others
        if image_file == "task5_test_image_8_prediction_jeremy_232189.png":
            process_image_8(image_path, output_folder)
        else:
            process_image_original(image_path, output_folder)

if __name__ == "__main__":
    main()
