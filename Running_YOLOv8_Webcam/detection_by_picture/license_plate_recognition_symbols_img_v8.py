import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os
import imutils


# Update the path to your YOLO license plate detection model
model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Path to the directory containing images
directory_path = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_license_plate/"
# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is an image (assuming all files in the directory are images)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        image_path = os.path.join(directory_path, filename)
        img_source = cv2.imread(image_path)
        img = cv2.resize(img_source, (1920, 1080))

        # Perform license plate detection using YOLO
        results = model_yolo(img)
        # Process YOLO detection results
        if len(results) == 0 or results[0].boxes is None:
            print("KHÔNG CÓ BIỂN SỐ NÀO")
        else:
            # Extracting boxes from the first result
            detections = results[0].boxes.cpu().numpy()
            for detection in detections:
                # Extracting bounding box coordinates
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cropped_img = img[y1:y2, x1:x2]
                img_resize = cv2.resize(cropped_img, (620, 480))

                cv2.namedWindow("Detected License Plate", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detected License Plate", 620, 480)
                cv2.imshow("Detected License Plate", img_resize)

                mask = np.zeros_like(img)
                cv2.rectangle(mask, (x1, y1),
                              (x2, y2), (255, 255, 255), -1)
                masked_img = cv2.bitwise_and(img, mask)

                # cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("crop", 620, 480)
                # cv2.imshow("crop", masked_img)

########################################################################################################

                # Edge detection
                gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                bilateralFilter = cv2.bilateralFilter(gray, 1, 100, 255)
                edged = cv2.Canny(bilateralFilter, 100, 255)
                kernel = np.ones((1, 1), np.uint8)
                dilated_canny_plate = cv2.dilate(edged, kernel, iterations=1)
                cnts = cv2.findContours(
                    edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

                mask = np.zeros(gray.shape, np.uint8)
                screenCnt = None
                for c in cnts:
                    peri = 0.05 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, peri, True)
                    if len(approx) == 4:
                        screenCnt = approx
                        break

                if screenCnt is None:
                    print("Chưa phát hiện được 4 góc biển")
                else:
                    # Mask the detected license plate
                    pts = np.array([screenCnt[i][0]
                                   for i in range(4)], dtype="int")
                    sorted_pts = pts[np.argsort(pts[:, 1]), :]
                    top_pts = sorted_pts[:2, :]
                    bottom_pts = sorted_pts[2:, :]
                    top_left, top_right = top_pts[np.argsort(top_pts[:, 0]), :]
                    bottom_left, bottom_right = bottom_pts[np.argsort(
                        bottom_pts[:, 0]), :]

                    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
                    new_image = cv2.circle(
                        gray, (top_left[0], top_left[1]), 5, (0, 255, 0), -1)
                    new_image = cv2.circle(
                        gray, (bottom_right[0], bottom_right[1]), 5, (0, 255, 0), -1)
                    new_image = cv2.bitwise_and(
                        masked_img, masked_img, mask=mask)
                    # cv2.namedWindow("Mask Image", cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow("Mask Image", 620, 480)
                    # cv2.imshow('Mask Image', new_image)

##################################################################################################

                    # Perspective transform to crop the plate
                    pts1 = np.float32(
                        [top_left, top_right, bottom_right, bottom_left])
                    width = max(np.linalg.norm(
                        pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
                    height = max(np.linalg.norm(
                        pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
                    pts2 = np.float32(
                        [[0, 0], [width, 0], [width, height], [0, height]])

                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    warp = cv2.warpPerspective(
                        new_image, matrix, (int(width), int(height)))
                    # cv2.namedWindow("Transformed Image", cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow("Transformed Image", 620, 480)
                    # cv2.imshow('Transformed Image', warp)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
