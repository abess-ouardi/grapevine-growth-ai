import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_images1(prediction_path, colored_path, output_path="afterpr.png"):
    try:
        # Load images
        prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        colored = cv2.imread(colored_path, cv2.IMREAD_COLOR)

        if prediction is None or colored is None:
            raise ValueError("One or both input images could not be loaded. Check file paths and formats.")

        # Resize images normally to 256x56
        target_size = (256, 256)
        prediction = cv2.resize(prediction, target_size)
        colored = cv2.resize(colored, target_size)


        # Apply Otsu's thresholding to binarize prediction.png
        _, binary_prediction = cv2.threshold(prediction, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply Connected Component Analysis on the binarized base image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_prediction, connectivity=8)
        
        # Create a cleaned mask to preserve only large components
        clean_mask = np.zeros_like(binary_prediction)
        min_size = 20  # Threshold for small objects
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                clean_mask[labels == i] = 255
        
        # Apply the cleaned mask to the base image
        binary_prediction = cv2.bitwise_and(binary_prediction, clean_mask)
        
        # Convert binary image back to 3 channels
        prediction = cv2.merge([binary_prediction, binary_prediction, binary_prediction])
        

        # Replace white pixels in colored.png with green (0, 255, 0)
        mask_white = np.all(colored == [255, 255, 255], axis=-1)
        colored[mask_white] = [0, 255, 0]

        # Copy non-black pixels from colored.png to a new copy of binary_prediction
        result = cv2.cvtColor(binary_prediction, cv2.COLOR_GRAY2BGR)  # Convert binary image to 3-channel
        non_black_mask = np.any(colored != [0, 0, 0], axis=-1)  # Find non-black pixels
        result[non_black_mask] = colored[non_black_mask]

        # Save output image as PNG
        cv2.imwrite(output_path, result)

        # Return NumPy array of the result
        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage (replace with actual file paths)
# result_array = process_images("prediction.png", "colored.png")

def expand_green_pixels(image_array, output_path="afterprgr.png"):
    """
    Expands green pixels by setting their 8 neighboring pixels to green as well.
    
    Args:
        image_array (numpy.ndarray): The input image as a NumPy array.
        output_path (str): Path where the processed image will be saved.

    Returns:
        numpy.ndarray: The processed image array.
    """
    try:
        # Define the green color in BGR format
        green = np.array([0, 255, 0])

        # Create a copy of the image to modify
        expanded_image = image_array.copy()

        # Get the height and width of the image
        height, width, _ = image_array.shape

        # Iterate over each pixel to find green ones
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if np.array_equal(image_array[y, x], green):
                    # Set the 8 neighboring pixels to green
                    expanded_image[y - 1:y + 2, x - 1:x + 2] = green

        # Save the modified image
        cv2.imwrite(output_path, expanded_image)

        return expanded_image

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage (assume `result_array` is the previous function's output)
# expanded_array = expand_green_pixels(result_array)


def remove_white_pixels(image_path, threshold=229, output_path="final_black_output.png"):
    """
    Replaces white pixels (above a given threshold) with black in an image.

    Args:
        image_path (str): Path to the input image.
        threshold (int): The pixel intensity threshold to consider as white.
        output_path (str): Path where the processed image will be saved.

    Returns:
        numpy.ndarray: The processed image array.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded. Check the file path.")

        # Convert to grayscale to detect white pixels
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a mask for strictly white pixels
        white_mask = grayscale >= threshold

        # Create a copy of the image to modify
        modified_image = image.copy()

        # Set only the true white pixels to black (0, 0, 0)
        modified_image[white_mask] = [10, 10, 10]

        # Save the modified image
        cv2.imwrite(output_path, modified_image)

        return modified_image

    except Exception as e:
        print(f"Error: {e}")
        return None


def color_low_intensity_pixels(image_array, output_path="blue_image.png"):
    """
    Colors pixels with intensity values between 5 and 20 in blue.

    Args:
        image_array (numpy.ndarray): The input image as a NumPy array.
        output_path (str): Path where the processed image will be saved.

    Returns:
        numpy.ndarray: The processed image array.
    """
    try:
        # Convert the image to grayscale to detect low-intensity pixels
        grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Create a mask for pixels with intensity between 5 and 20
        low_intensity_mask = (grayscale >= 5) & (grayscale <= 20)

        # Create a copy of the image to modify
        modified_image = image_array.copy()

        # Set these pixels to blue (255, 0, 0) in BGR format
        modified_image[low_intensity_mask] = [255, 0, 0]

        # Save the modified image
        cv2.imwrite(output_path, modified_image)

        return modified_image

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage:
