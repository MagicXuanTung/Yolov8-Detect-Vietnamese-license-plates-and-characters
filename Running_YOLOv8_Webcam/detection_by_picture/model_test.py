from ultralytics import YOLO
import cv2
import os

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/best.pt")

# Update class name for license plate
classNames = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F',
              16: 'G', 17: 'H', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'R', 25: 'S', 26: 'T', 27: 'U', 28: 'V', 29: 'X', 30: 'Y', 31: 'Z'}

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_color/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

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
            if classNames[cls] in classNames.values():
                # Magenta color for bounding box
                color = (255, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Display class name inside the bounding box
                cv2.putText(img, classNames[cls], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Crop the detected license plate region
                plate_region = img[y1:y2, x1:x2]

                # Perform Canny edge detection on the cropped license plate region
                canny_plate = cv2.Canny(plate_region, 100, 200)

                # Draw Canny edge detection on the license plate
                cv2.imshow("Canny Edge Detection", canny_plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # Show the processed image with license plate information
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)

    # Save the image with license plate information
    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

cv2.destroyAllWindows()
