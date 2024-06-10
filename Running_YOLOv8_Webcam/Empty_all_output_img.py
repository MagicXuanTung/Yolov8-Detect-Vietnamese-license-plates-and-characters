import os
import shutil

# Original image directory
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"

# Cropped
output_images_plates_cropped = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates_cropped"

# (color)
cropped_output_directory_color = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_color"

# (black and white)
cropped_output_directory_bw = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_bw"

# Canny
cropped_images_plates_canny = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_canny"

# Contours
cropped_images_plates_contours = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_contours"

# character_detector
output_directory_character_detector = "./Running_YOLOv8_Webcam/detection_by_picture/output_character_detector/"
# List of directories
directories = [output_directory, output_images_plates_cropped, cropped_output_directory_color, cropped_output_directory_bw,
               cropped_images_plates_canny, cropped_images_plates_contours, output_directory_character_detector]

# Empty all directories
for directory in directories:
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Successfully emptied directory: {directory}")
