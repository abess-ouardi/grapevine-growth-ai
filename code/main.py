from class_vinemodel import VineyardModel, visualize_dataset
from red_filling import  process_images1, expand_green_pixels, remove_white_pixels, color_low_intensity_pixels
from red_pixel_count import  count_low_intensity_pixels, assign_red_pixels, interactive_sinusoidal_coloring
import pprint
import cv2




obj = VineyardModel()
co2_dict, co2_list = obj.co2_acquisition("co2_file.csv")
temperature_dict, temperature_list = obj.temperature_acquisition("temperature_file.csv")
soil_moisture_dict, soil_moisture_list = obj.soil_moisture_acquisition("soil_moisture_file.csv")
light_intensity_dict, light_intensity_list = obj.light_intensity_acquisition("light_intensity_file.csv")
tracking_data = obj.phenological_stages()
visualize_dataset(tracking_data)
phenology_timeline, biomass_list, branch_biomass, leaf_biomass, fruit_biomass = obj.photosynthesis()
result_array = process_images1("40_fake_B.png","40.png")
def resize_image(image_array, target_size=(1920, 1080)):
    """
    Resizes an input image from 256x256 to 1920x1080.

    Args:
        image_array (numpy.ndarray): The input image as a NumPy array.
        target_size (tuple): The desired output size (width, height).

    Returns:
        numpy.ndarray: The resized image array.
    """
    try:
        # Resize the image
        resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
        return resized_image

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage (assuming `input_array` is a 256x256 NumPy array
resized_array = resize_image(result_array)
expanded_array = expand_green_pixels(resized_array)
final_form = resize_image(expanded_array,(256,256))
# Save output image as PNG
cv2.imwrite("final.png", final_form)
modified_image = remove_white_pixels("final.png", threshold=229)
blue_image = color_low_intensity_pixels(modified_image)



amount_black_pixels = count_low_intensity_pixels("final_black_output.png")
assigned_pixels = assign_red_pixels(amount_black_pixels,branch_biomass,modified_image)
interactive_sinusoidal_coloring(amount_black_pixels, branch_biomass, modified_image)




