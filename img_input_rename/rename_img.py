import os


def rename_files(input_directory, output_directory):
    # Get list of files in input directory
    files = os.listdir(input_directory)

    # Sort files alphabetically
    files.sort()

    # Counter for numbering
    i = 1

    # Iterate over files and rename
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct new file name with number
            new_name = f"{i}.jpg"

            # Full path of current and new file in input and output directories
            old_path = os.path.join(input_directory, file_name)
            new_path = os.path.join(output_directory, new_name)

            # Rename file
            os.rename(old_path, new_path)

            # Increment counter
            i += 1


# Specify input and output directories
input_directory = r'D:\Yolov8-Detect-Vietnamese-license-plates-and-characters\img_input_rename\img'
output_directory = r'D:\Yolov8-Detect-Vietnamese-license-plates-and-characters\Running_YOLOv8_Webcam\detection_by_picture\input_images_license_plate'

# Call the function with input and output directories
rename_files(input_directory, output_directory)
