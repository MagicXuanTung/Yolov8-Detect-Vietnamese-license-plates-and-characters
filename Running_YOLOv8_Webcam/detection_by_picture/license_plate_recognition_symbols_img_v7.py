import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO
import Preprocess
import imutils

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Update class name for license plate
classNames = ["license_plate"]

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_license_plate/"

# Original image directory
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

# (color)
cropped_output_directory_color = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_color"
os.makedirs(cropped_output_directory_color, exist_ok=True)

# (black and white)
cropped_output_directory_bw = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_bw"
os.makedirs(cropped_output_directory_bw, exist_ok=True)

# Canny
cropped_images_plates_canny = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_canny"
os.makedirs(cropped_images_plates_canny, exist_ok=True)

# Contours
cropped_images_plates_contours = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_contours"
os.makedirs(cropped_images_plates_contours, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

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
    cropped_output_directory_color = os.path.join(
        image_directory, image_filename)

    # Read the image
    img = cv2.imread(cropped_output_directory_color)

    # Resize the image
    img = cv2.resize(img, (620, 480))

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
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                # Crop the image to include only the bounding box and the object inside
                cropped_img = img[y1:y2, x1:x2]

                # Show the cropped image (color)
                cv2.namedWindow(
                    f"Color {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Color {cropped_images_count + 1}", 620, 480)
                cv2.imshow(
                    f"Color {cropped_images_count + 1}", cropped_img)

                # Save the cropped image (color)
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
                    f"Black and white {cropped_images_count + 1}", 620, 480)
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
                canny_plate = cv2.Canny(cropped_img, 480, 255)

                cv2.namedWindow(
                    f"Canny edge {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Canny edge {cropped_images_count + 1}", 620, 480)
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

                # CONTOUR IMAGE
                # Convert ảnh cropped_img sang ảnh grayscale
                gray_cropped_img = cv2.cvtColor(
                    cropped_img, cv2.COLOR_BGR2GRAY)
                # Dilate để tăng độ dày của các đường biên phát hiện được
                kernel = np.ones((1, 1), np.uint8)
                dilated_canny_plate = cv2.dilate(
                    canny_plate, kernel, iterations=1)

                # Tìm các contour trong ảnh đã xử lý
                contours, _ = cv2.findContours(
                    dilated_canny_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Vẽ contour lên ảnh gốc để kiểm tra
                cv2.drawContours(cropped_img, contours, -1, (255, 255, 0), 1)

                # Hiển thị ảnh với contour
                cv2.namedWindow(
                    f"Contours {cropped_images_count + 1}", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(
                    f"Contours {cropped_images_count + 1}", 620, 480)
                cv2.imshow(
                    f"Contours {cropped_images_count + 1}", cropped_img,)

                results = reader.readtext(cropped_img)
                # Extract license plate text from EasyOCR results
                license_plate_text = ""
                for res in results:
                    text = res[1]
                    # Check if each character in the recognized text is valid
                    license_plate_text_valid = "".join(
                        [char if char in valid_characters else "." for char in text])
                    license_plate_text += license_plate_text_valid + "\n"

                # Save the preprocessed image (contour)
                cropped_output_filename_plates_contours = os.path.join(
                    cropped_images_plates_contours, f'{image_filename}_cropped_{i+1}-{j+1}_{cropped_images_count + 1}.jpg')
                cv2.imwrite(cropped_output_filename_plates_contours,
                            cropped_img)

                # Print the license plate text to the console
                print(f"Contours:\n{license_plate_text}")

                # Increment the cropped images count
                cropped_images_count += 1

    # Show the processed image with license plate information
    cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("License Plate Detection", 620, 480)
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
