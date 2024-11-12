<<<<<<< HEAD
import os
import cv2
import numpy as np

# Define color ranges in BGR for each class
color_ranges = {
    0: ([124, 0, 255], [124, 0, 255]),  # Oil
    1: ([255, 221, 51], [255, 221, 51]),  # Water
    2: ([51, 204, 255], [51, 204, 255])   # Other
}

def yolo_format(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each .png file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            # Load annotated image
            annotated_img_path = os.path.join(input_folder, file_name)
            annotated_img = cv2.imread(annotated_img_path)
            height, width, _ = annotated_img.shape

            # Prepare to collect YOLO annotations for this image
            yolo_data = []

            # Iterate over each class color range
            for class_id, (lower, upper) in color_ranges.items():
                # Convert range to NumPy array
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")

                # Create mask for the class color
                mask = cv2.inRange(annotated_img, lower, upper)

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Process each contour to get bounding boxes
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Convert to YOLO format
                    x_center, y_center, bbox_width, bbox_height = yolo_format(x, y, w, h, width, height)

                    # Append data in YOLO format (class_id, x_center, y_center, width, height)
                    yolo_data.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

            # Save YOLO annotations to corresponding .txt file
            output_file_path = os.path.join(output_folder, file_name.replace('.png', '.txt'))
            with open(output_file_path, "w") as f:
                f.writelines(yolo_data)

            print(f"Processed {file_name} and saved annotations to {output_file_path}")

# Example usage
input_folder = r"S:\IOE Project\dataset\val\masks" # Replace with your folder path
output_folder = r"S:\IOE Project\dataset\val\labels"  # Replace with your folder path
process_images(input_folder, output_folder)
=======
import os
import cv2
import numpy as np

# Define color ranges in BGR for each class
color_ranges = {
    0: ([124, 0, 255], [124, 0, 255]),  # Oil
    1: ([255, 221, 51], [255, 221, 51]),  # Water
    2: ([51, 204, 255], [51, 204, 255])   # Other
}

def yolo_format(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each .png file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            # Load annotated image
            annotated_img_path = os.path.join(input_folder, file_name)
            annotated_img = cv2.imread(annotated_img_path)
            height, width, _ = annotated_img.shape

            # Prepare to collect YOLO annotations for this image
            yolo_data = []

            # Iterate over each class color range
            for class_id, (lower, upper) in color_ranges.items():
                # Convert range to NumPy array
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")

                # Create mask for the class color
                mask = cv2.inRange(annotated_img, lower, upper)

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Process each contour to get bounding boxes
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Convert to YOLO format
                    x_center, y_center, bbox_width, bbox_height = yolo_format(x, y, w, h, width, height)

                    # Append data in YOLO format (class_id, x_center, y_center, width, height)
                    yolo_data.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

            # Save YOLO annotations to corresponding .txt file
            output_file_path = os.path.join(output_folder, file_name.replace('.png', '.txt'))
            with open(output_file_path, "w") as f:
                f.writelines(yolo_data)

            print(f"Processed {file_name} and saved annotations to {output_file_path}")

# Example usage
input_folder = r"S:\IOE Project\dataset\val\masks" # Replace with your folder path
output_folder = r"S:\IOE Project\dataset\val\labels"  # Replace with your folder path
process_images(input_folder, output_folder)
>>>>>>> 60a215889218f5e2fc17543489ae788d1c38dadf
