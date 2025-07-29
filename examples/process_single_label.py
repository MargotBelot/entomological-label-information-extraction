# Example: Process a single label image using the label processing pipeline
import sys
from label_processing import process_label

# Specify the path to the image and the desired output directory
image_path = "./data/single_label_example.jpg"
output_path = "./output/single_label/"

# Process the label
try:
    process_label(image_path, output_path)
    print("Label processing complete. Output saved to:", output_path)
except Exception as e:
    print("An error occurred while processing the label:", e)
    sys.exit(1)
