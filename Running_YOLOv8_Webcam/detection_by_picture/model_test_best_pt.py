from ultralytics import YOLO
import cv2
import os

# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/corners.pt")

# Update class name for license plate
classNames = {0: 'corner'}

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates_cropped/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates_cropped/"
os.makedirs(output_directory, exist_ok=True)

for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    img = cv2.imread(image_path)
    img_resize = cv2.resize(img, (620, 480))

    # Use color image for YOLO detection
    results = model_yolo(img_resize)

    # Loop through the results and extract license plate bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a license plate
            cls = int(box.cls[0])
            if classNames[cls] == 'corner':
                continue  # Skip drawing for class 31

            if classNames[cls] in classNames.values():
                # Magenta color for bounding box
                color = (255, 0, 255)
                # confidence = box.conf[0] * 100
                cv2.rectangle(img_resize, (x1, y1), (x2, y2), color, 1)

                # text = classNames[cls]
                # font_scale = 0.8
                # thickness = 1
                # text_size, _ = cv2.getTextSize(
                #     text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # text_w, text_h = text_size

                # text_x = x1 + 5
                # text_y = y1 + 20

                # cv2.putText(img_resize, text, (text_x, text_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                cv2.imshow("img_resize", img_resize)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # output_filename = os.path.join(
    #     output_directory, f'{os.path.splitext(image_filename)[0]}_output.jpg')
    # cv2.imwrite(output_filename, img_resize)

cv2.destroyAllWindows()
