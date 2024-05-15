
from ultralytics import YOLO
import cv2
import easyocr
import os
import Preprocess  

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Update class name for license plate
classNames = ["license_plate"]

# Initialize EasyOCR with the desired language (replace 'en' with the appropriate language code)
reader = easyocr.Reader(['vi'])

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
cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)

for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    img = cv2.imread(image_path)

    # Doing detections using YOLOv8 for each image
    results = model_yolo(img)

    # Loop through the results and extract license plate bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a license plate
            cls = int(box.cls[0])
            if classNames[cls] == "license_plate":
                color = (255, 0, 255)  # Magenta color for both bounding box and text

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Define the coordinates for cropping the image
                crop_x1 = max(0, x1 - 10)  # Adjust the values to include some margin around the bounding box
                crop_y1 = max(0, y1 - 10)
                crop_x2 = min(img.shape[1], x2 + 10)
                crop_y2 = min(img.shape[0], y2 + 10)

                # Crop the image to include only the bounding box and the object inside
                cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

                # Preprocess the cropped image using the Preprocess module
                _, preprocessed_img = Preprocess.preprocess(cropped_img)

                # Show the preprocessed cropped image in the "Cropped Image" window
                cv2.imshow("Cropped Image", preprocessed_img)

                # Save the cropped image
                cropped_output_filename = os.path.join(cropped_output_directory, f'{image_filename}_cropped.jpg')
                cv2.imwrite(cropped_output_filename, preprocessed_img)

                # Perform OCR on the cropped license plate region
                results_ocr = reader.readtext(preprocessed_img)

                if results_ocr:
                    # Concatenate lines of the license plate text into a single string
                    license_plate_text = ' '.join(result[1] for result in results_ocr)

                    # Display the entire license plate text on a single line
                    cv2.putText(img, f'License Plate: {license_plate_text}', (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Show the processed image with license plate information
    cv2.imshow("License Plate Detection", img)

    # Wait for a key press and update windows simultaneously
    key = cv2.waitKey(0)
    if key == 27:  # Press 'Esc' key to exit
        break

    # Save the image with license plate information
    output_filename = os.path.join(output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

cv2.destroyAllWindows()
