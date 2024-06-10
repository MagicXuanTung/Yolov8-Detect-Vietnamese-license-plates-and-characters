import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO
import Preprocess

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

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

# Create windows for displaying images
cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped Image 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped Image 2", cv2.WINDOW_NORMAL)

# Initialize a counter for additional crop windows
crop_windows_count = 0

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define the set of valid characters for Vietnamese license plates
valid_characters = set("0123456789ABCDEFGHKLMNPSTUVXYZ")

for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)

    # Read the image
    img = cv2.imread(image_path)

    # Resize the image to 1920x1080
    img = cv2.resize(img, dsize=(1920, 1080))

    # Doing detections using YOLOv8 for each image
    results = model_yolo(img)

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

                # Increment the crop windows count
                crop_windows_count += 1

                # Convert cropped image to grayscale
                cropped_gray = cv2.GaussianBlur(cropped_img, (5, 5), 100)

                # Perform OCR on the cropped license plate region using EasyOCR
                result = reader.readtext(cropped_gray)

                # Extract license plate text from EasyOCR result
                license_plate_text = result[0][1] if result else "OCR Error"

                # Check if each character in the recognized text is valid
                license_plate_text_valid = "".join(
                    [char if char in valid_characters else "?" for char in license_plate_text])

                # Print the license plate text to the console
                print(f"License Plate Text: {license_plate_text_valid}")

                # Show the first cropped image
                cv2.imshow("Cropped Image 1", cropped_img)

                # Save the first cropped image (colored)
                cropped_output_filename_color = os.path.join(
                    cropped_output_directory_color, f'{image_filename}_cropped_{i+1}-{j+1}_1.jpg')
                cv2.imwrite(cropped_output_filename_color, cropped_img)

                # Preprocess the second cropped image
                _, cropped_binary = Preprocess.preprocess(cropped_img)

                # Perform OCR on the preprocessed image
                result2 = reader.readtext(cropped_binary)

                # Extract license plate text from EasyOCR result
                license_plate_text2 = result2[0][1] if result2 else "OCR Error"

                # Check if each character in the recognized text is valid
                license_plate_text_valid2 = "".join(
                    [char if char in valid_characters else "?" for char in license_plate_text2])

                # Print the license plate text to the console
                print(f"License Plate Text 2: {license_plate_text_valid2}")

                # Show the second cropped image
                cv2.imshow("Cropped Image 2", cropped_binary)

                # Save the second cropped image (black and white)
                cropped_output_filename_bw = os.path.join(
                    cropped_output_directory_bw, f'{image_filename}_cropped_{i+1}-{j+1}_2.jpg')
                cv2.imwrite(cropped_output_filename_bw, cropped_binary)

    # Show the processed image with license plate information
    cv2.imshow("License Plate Detection", img)

    # Wait for a key press to move to the next image
    cv2.waitKey(0)

    # Save the image with license plate information
    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

# Print the total number of crop windows
print(f"Total number of crop windows: {crop_windows_count}")

# Close all windows
cv2.destroyAllWindows()
