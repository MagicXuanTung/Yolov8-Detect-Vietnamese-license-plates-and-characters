import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO
import Preprocess

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_ plates.pt")

# Update class name for license plate
classNames = ["license_plate"]

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_license_plate/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

# Output directory for saving cropped images (colored)
cropped_output_directory_color = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_color"
os.makedirs(cropped_output_directory_color, exist_ok=True)

# Output directory for saving cropped images (black and white)
cropped_output_directory_bw = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_bw"
os.makedirs(cropped_output_directory_bw, exist_ok=True)

cropped_images_plates_canny = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_canny"
os.makedirs(cropped_images_plates_canny, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define the set of valid characters for Vietnamese license plates
valid_characters = set("0123456789ABCDEFGHKLMNPSTUVXYZ")

# List of image filenames
image_filenames = os.listdir(image_directory)
# Sort the image filenames for consistent navigation
image_filenames.sort()

# Index to keep track of current image
current_image_index = 0

while current_image_index < len(image_filenames):
    image_filename = image_filenames[current_image_index]
    image_path = os.path.join(image_directory, image_filename)

    # Read the image
    img = cv2.imread(image_path)

    # Resize the image to 1920x1080
    img = cv2.resize(img, dsize=(1920, 1080))

    # Doing detections using YOLOv8 for each image
    results = model_yolo(img)

    # Initialize a counter for cropped images
    cropped_images_count = 0

    # Loop through the results and extract license plate bounding boxes
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a license plate
            cls = int(box.cls[0])
            if classNames[cls] == "license_plate":
                # Magenta color for both bounding box and text
                color = (255, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Define the coordinates for cropping the image
                # Adjust the values to include some margin around the bounding box
                crop_x1 = max(0, x1 - 0)
                crop_y1 = max(0, y1 - 0)

                crop_x2 = min(img.shape[1], x2 + 0)
                crop_y2 = min(img.shape[0], y2 + 0)

                # Crop the image to include only the bounding box and the object inside
                cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

                # cv2.Imgproc.findContours
                # cv2.Imgproc.getPerspectiveTransform
                # cv2.Imgproc.warpPerspective

                # SHOW THE CROPPED IMAGE NORMAL COLOR
                cv2.namedWindow(
                    f"Color {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Color {cropped_images_count + 1}", 640, 360)
                cv2.imshow(
                    f"Color {cropped_images_count + 1}", cropped_img)

                # Save the cropped image (colored)
                cropped_output_filename_color = os.path.join(
                    cropped_output_directory_color, f'{image_filename}_cropped_{i+1}-{j+1}_{cropped_images_count + 1}.jpg')
                cv2.imwrite(cropped_output_filename_color, cropped_img)

                # Perform OCR on the cropped license plate region using EasyOCR
                results = reader.readtext(cropped_img)

                # Extract license plate text from EasyOCR results
                license_plate_text = ""
                for res in results:
                    text = res[1]
                    # Check if each character in the recognized text is valid
                    license_plate_text_valid = "".join(
                        [char if char in valid_characters else "." for char in text])
                    license_plate_text += license_plate_text_valid + "\n"

                # Print the license plate text to the console
                print(f"Color:\n{license_plate_text}")

                # TRANSFORMS THE IMAGE BLACK AND WHITE
                # Preprocess the cropped image
                _, cropped_binary = Preprocess.preprocess(cropped_img)
                cropped_value = Preprocess.extractValue(cropped_img)
                cropped_max_contrast = Preprocess.maximizeContrast(
                    cropped_value)

                # Show the preprocessed image
                cv2.namedWindow(
                    f"Black and white {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Black and white {cropped_images_count + 1}", 640, 360)
                cv2.imshow(
                    f"Black and white {cropped_images_count + 1}", cropped_max_contrast,)

                results = reader.readtext(cropped_max_contrast)

                # Extract license plate text from EasyOCR results
                license_plate_text = ""
                for res in results:
                    text = res[1]
                    # Check if each character in the recognized text is valid
                    license_plate_text_valid = "".join(
                        [char if char in valid_characters else "." for char in text])
                    license_plate_text += license_plate_text_valid + "\n"

                # Print the license plate text to the console
                print(f"Black and white:\n{license_plate_text}")

                # Save the preprocessed image (black and white)
                cropped_output_filename_bw = os.path.join(
                    cropped_output_directory_bw, f'{image_filename}_cropped_{i+1}-{j+1}_{cropped_images_count + 1}.jpg')
                cv2.imwrite(cropped_output_filename_bw,
                            cropped_max_contrast)

                #  SHOW WINDOWS CANNY EDGE
                canny_plate = cv2.Canny(cropped_img, 250, 255)

                cv2.namedWindow(
                    f"Canny edge {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Canny edge {cropped_images_count + 1}", 640, 360)
                cv2.imshow(
                    f"Canny edge {cropped_images_count + 1}", canny_plate)

                results = reader.readtext(canny_plate)

                # Extract license plate text from EasyOCR results
                license_plate_text = ""
                for res in results:
                    text = res[1]
                    # Check if each character in the recognized text is valid
                    license_plate_text_valid = "".join(
                        [char if char in valid_characters else "." for char in text])
                    license_plate_text += license_plate_text_valid + "\n"

                # Print the license plate text to the console
                print(f"Canny edge:\n{license_plate_text}")

                # Save the preprocessed image (canny_plate)
                cropped_output_filename_plates_canny = os.path.join(
                    cropped_images_plates_canny, f'{image_filename}_cropped_{i+1}-{j+1}_{cropped_images_count + 1}.jpg')
                cv2.imwrite(cropped_output_filename_plates_canny,
                            canny_plate)
                # Increment the cropped images count
                cropped_images_count += 1

    # Show the processed image with license plate information
    cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("License Plate Detection", 640, 360)
    cv2.imshow("License Plate Detection", img)

    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

    # Wait for a key press to move to the next image
    key = cv2.waitKey(0)

    # Handle keypress
    if key == ord('n'):  # 'n' for next image
        current_image_index += 1
    elif key == ord('p') and current_image_index > 0:  # 'p' for previous image
        current_image_index -= 1


# Close all windows
cv2.destroyAllWindows()
