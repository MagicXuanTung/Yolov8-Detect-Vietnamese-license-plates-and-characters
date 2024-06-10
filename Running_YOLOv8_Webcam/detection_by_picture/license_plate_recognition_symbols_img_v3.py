import cv2
import os
import numpy as np
import Preprocess
from ultralytics import YOLO
import easyocr

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Update class name for license plate
classNames = ["license_plate"]

# Initialize EasyOCR with the desired language
reader = easyocr.Reader(['en'])

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_license_plate/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

# Output directory for saving cropped images
cropped_output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(cropped_output_directory, exist_ok=True)

# Create windows for displaying images
cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)

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

                # Preprocess the cropped image using the Preprocess module
                _, preprocessed_img = Preprocess.preprocess(cropped_img)

                # Apply Canny edge detection
                edges = cv2.Canny(preprocessed_img, 100, 100)

                # Find contours
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw contours on the preprocessed image
                cv2.drawContours(preprocessed_img,
                                 contours, -1, (0, 255, 0), 2)

                # Perform OCR on the preprocessed cropped license plate region
                results_ocr = reader.readtext(preprocessed_img)

                if results_ocr:
                    # Concatenate lines of the license plate text into a single string
                    license_plate_text = ' '.join(
                        result[1] for result in results_ocr)

                    # Print the license plate text to the console
                    print(f"License Plate Text: {license_plate_text}")

                # Show the preprocessed cropped image with contours
                window_name = f"Cropped Image {i+1}-{j+1}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, preprocessed_img)

                # Save the cropped image
                cropped_output_filename = os.path.join(
                    cropped_output_directory, f'{image_filename}_cropped_{i+1}-{j+1}.jpg')
                cv2.imwrite(cropped_output_filename, preprocessed_img)

    # Show the processed image with license plate information
    cv2.imshow("License Plate Detection", img)

    # Wait for a key press to move to the next image
    cv2.waitKey(0)

    # Save the image with license plate information
    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

# Close all windows
cv2.destroyAllWindows()
