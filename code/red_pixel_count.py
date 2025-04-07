import cv2
import numpy as np

def count_low_intensity_pixels(filepath):
    """
    Reads an image using OpenCV and counts the number of pixels with intensity values in the range [5, 20].

    Parameters:
    filepath (str): Path to the PNG image file.

    Returns:
    int: Number of pixels in the intensity range [5, 20].
    """
    # Load the image in BGR format (default in OpenCV)
    image = cv2.imread(filepath)

    # Ensure the image is loaded correctly
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image at {filepath}")

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where pixel intensity is between 5 and 20
    low_intensity_mask = (grayscale >= 5) & (grayscale <= 20)

    # Count the number of pixels in this range
    low_intensity_pixel_count = np.sum(low_intensity_mask)

    return low_intensity_pixel_count



def assign_red_pixels(num_red_pixels: int, values_list: list, image: np.ndarray) -> list:
    """
    Maps each float value in the given sorted list to a proportional number of red pixels.

    Parameters:
    - num_red_pixels (int): The total number of red pixels.
    - values_list (list of float): A sorted list of 300 float values.
    - image (numpy.ndarray): An image array (unused in this function).

    Returns:
    - list of int: A list where each float value in the input list is assigned a rounded number of red pixels.
    """
    max_value = values_list[-1]  # The last value corresponds to the total red pixels

    # Compute assigned pixels using proportional mapping and round to the nearest integer
    assigned_pixels = [round((value * num_red_pixels) / max_value) for value in values_list]

    return assigned_pixels


def interactive_sinusoidal_coloring(num_target_pixels: int, values_list: list, image: np.ndarray):
    """
    Interactive OpenCV window where a slider controls the number of pixels to be colored.
    The pixels are colored based on a sinusoidal function that shifts upwards until 
    the required number of pixels is reached.

    Features:
    - Uses OpenCV trackbar to adjust the number of pixels.
    - Only pixels in grayscale range [5, 20] and **below** the sinusoidal function are considered.
    - Follows **bottom-to-top row priority** and **left-to-right** if needed.
    - Enlarged visualization for better display.
    - Slider legend changed to "DAYS".
    - Press 's' to save the displayed image.
    - Press 'q' to quit.
    """

    global saved_image  # Variable to store the most recently saved image

    # Compute the assigned pixels based on scaling
    max_value = values_list[-1]  # Maximum value in the list
    assigned_pixels = [round((value * num_target_pixels) / max_value) for value in values_list]

    height, width, _ = image.shape

    # Convert image to grayscale to detect intensity range [5, 20]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sinusoidal function parameters
    A = int(height / 10)  # Amplitude: 1/10th of image height
    frequency = 2  # Number of cycles across the image width

    # Store a copy of the original image for reset purposes
    original_image = image.copy()

    # Create OpenCV window with a larger display
    window_name = "Interactive Sinusoidal Coloring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)  # Set larger display window

    # Trackbar callback (does nothing but required by OpenCV)
    def nothing(x):
        pass

    # Create trackbar (slider from 0 to len(assigned_pixels) - 1) and label it "DAYS"
    cv2.createTrackbar("DAYS", window_name, 0, len(assigned_pixels) - 1, nothing)

    while True:
        # Get slider position
        index = cv2.getTrackbarPos("DAYS", window_name)

        # Reset image before applying new changes
        temp_image = original_image.copy()

        # Number of pixels to color based on the slider position
        pixels_to_color = assigned_pixels[index]

        # Sinusoidal wave starts outside the image at height + A
        shift_y = height + A
        candidate_pixels = []

        while shift_y > -A:  # Shift the sine wave upwards
            y_sine = np.array([shift_y - A * np.sin(2 * np.pi * frequency * (x / width)) for x in range(width)], dtype=int)

            # Collect pixels below the sine curve in grayscale range [5, 20]
            candidate_pixels.clear()
            for y in range(height - 1, -1, -1):  # Bottom to top
                for x in range(width):  # Left to right
                    if y < y_sine[x] and 5 <= grayscale[y, x] <= 20:
                        candidate_pixels.append((y, x))

            if len(candidate_pixels) >= pixels_to_color:
                break  # Stop shifting once enough pixels are collected

            shift_y -= 1  # Move the sine wave upward

        # Apply coloring with bottom-to-top, left-to-right priority
        colored_count = 0
        for y in range(height - 1, -1, -1):  # Bottom to top
            row_pixels = [px for px in candidate_pixels if px[0] == y]

            if colored_count + len(row_pixels) <= pixels_to_color:
                # If adding the whole row doesn't exceed the target, color them all
                for _, x in row_pixels:
                    temp_image[y, x] = [0, 255, 0]  # Set to green
                colored_count += len(row_pixels)
            else:
                # If adding the whole row exceeds the target, color left-to-right
                row_pixels.sort(key=lambda p: p[1])  # Sort by x-coordinate
                for _, x in row_pixels:
                    if colored_count < pixels_to_color:
                        temp_image[y, x] = [0, 255, 0]  # Set to green
                        colored_count += 1
                    else:
                        break
                break  # Stop once exact count is reached

        # Display the updated image
        cv2.imshow(window_name, temp_image)

        # Key event handling
        key = cv2.waitKey(1) & 0xFF

        # Save the displayed image when "s" is pressed
        if key == ord("s"):
            saved_image = temp_image.copy()
            print("Image saved to variable 'saved_image'.")

        # Quit when "q" is pressed
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
