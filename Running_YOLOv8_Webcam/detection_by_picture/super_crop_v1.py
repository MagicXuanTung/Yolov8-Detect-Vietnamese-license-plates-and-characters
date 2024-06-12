import cv2
import numpy as np
from ultralytics import YOLO
import os
import imutils

model_yolo = YOLO("../YOLO-Weights/license_plate_detector.pt")

model_yolo2 = YOLO("../YOLO-Weights/character_detector.pt")
classNames2 = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G',
    17: 'H', 18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}

directory_path = "./Running_YOLOv8_Webcam/detection_by_picture/input_images_license_plate/"

output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates/"
os.makedirs(output_directory, exist_ok=True)

output_directory_character_detector = "./Running_YOLOv8_Webcam/detection_by_picture/output_character_detector/"
os.makedirs(output_directory_character_detector, exist_ok=True)

output_images_plates_cropped = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates_cropped"
os.makedirs(output_images_plates_cropped, exist_ok=True)


def detect_and_crop(image, model):
    results = model(image)
    if results and results[0].boxes is not None:
        detections = results[0].boxes.cpu().numpy()
        crops = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cropped_img = image[y1:y2, x1:x2]
            crops.append((cropped_img, (x1, y1, x2, y2)))
        return crops
    return []


def sortByXmin(point):
    x1, y1, x2, y2 = map(int, point.xyxy[0])
    return x1


# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory_path, filename)
        img_source = cv2.imread(image_path)
        img = cv2.resize(img_source, (1920, 1080))

        crops_1 = detect_and_crop(img, model_yolo)
        if not crops_1:
            print("KHÔNG CÓ BIỂN SỐ NÀO")
            continue

        for crop_1, bbox_1 in crops_1:
            crop_1_resized = cv2.resize(crop_1, (620, 480))

            # Second detection and crop
            crops_2 = detect_and_crop(crop_1_resized, model_yolo)
            if not crops_2:
                print("crop lần 2 lỗi")
                continue

            for crop_2, bbox_2 in crops_2:
                # Third detection and crop
                crops_3 = detect_and_crop(crop_2, model_yolo)
                if not crops_3:
                    print("crop lần 3 lỗi ")
                    continue

                for crop_3, bbox_3 in crops_3:
                    x1_f, y1_f, x2_f, y2_f = bbox_3
                    # Fourth detection and crop
                    crops_4 = detect_and_crop(crop_3, model_yolo)

                    if not crops_4:
                        print("crop lần 4 lỗi")

                        continue

                    for crop_4, bbox_4 in crops_4:
                        gray = cv2.cvtColor(crop_4, cv2.COLOR_BGR2GRAY)
                        bilateralFilter = cv2.bilateralFilter(
                            gray, 1, 10, 10)
                        output_filename = os.path.join(
                            output_images_plates_cropped, f'{os.path.splitext(filename)[0]}_.jpg')
                        cv2.imwrite(output_filename, bilateralFilter)
                        cv2.namedWindow("original_img_got_crop",
                                        cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("original_img_got_crop", 620, 480)
                        cv2.imshow("original_img_got_crop", crop_4)

                        gray = cv2.cvtColor(crop_4, cv2.COLOR_BGR2GRAY)
                        bilateralFilter = cv2.bilateralFilter(
                            gray, 1, 10, 10)
                        _, thresh = cv2.threshold(
                            bilateralFilter, 155, 200, cv2.THRESH_BINARY)
                        edged = cv2.Canny(thresh, 100, 150)
                        kernel = np.ones((3, 3), np.uint8)
                        dilated_canny_plate = cv2.dilate(
                            edged, kernel, iterations=1)
                        cnts = cv2.findContours(
                            dilated_canny_plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                            pts = np.array([screenCnt[i][0]
                                            for i in range(4)], dtype="int")
                            sorted_pts = pts[np.argsort(pts[:, 1]), :]
                            top_pts = sorted_pts[:2, :]
                            bottom_pts = sorted_pts[2:, :]
                            top_left, top_right = top_pts[np.argsort(
                                top_pts[:, 0]), :]
                            bottom_left, bottom_right = bottom_pts[np.argsort(
                                bottom_pts[:, 0]), :]

                            new_image = cv2.drawContours(
                                mask, [screenCnt], 0, 255, -1)
                            new_image = cv2.circle(
                                gray, (top_left[0], top_left[1]), 5, (0, 255, 0), -1)
                            new_image = cv2.circle(
                                gray, (bottom_right[0], bottom_right[1]), 5, (0, 255, 0), -1)
                            new_image = cv2.bitwise_and(
                                crop_4, crop_4, mask=mask)
                            # cv2.namedWindow("Mask Image", cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("Mask Image", 620, 480)
                            # cv2.imshow('Mask Image', new_image)
##################################################################################################
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
                            # cv2.namedWindow("Transformed Image",
                            #                 cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("Transformed Image", 620, 480)
                            # cv2.imshow('Transformed Image', warp)

#######################################################################################################
                            gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                            bilateralFilter = cv2.bilateralFilter(
                                gray, 1, 10, 10)
                            _, thresh = cv2.threshold(
                                bilateralFilter, 145, 170, cv2.THRESH_BINARY)
                            img_resize = cv2.resize(thresh, (620, 480))
                            output_thresh_filename = os.path.join(
                                output_directory, f'{os.path.splitext(filename)[0]}_thresh.jpg')
                            cv2.imwrite(output_thresh_filename, img_resize)

                            thresh_image = cv2.imread(output_thresh_filename)

                            results = model_yolo2(thresh_image)

                            texts = []

                            # tách dòng trên và dưới (nếu là 2 dòng), chỉ lấy dòng trên (nếu có 1 dòng)
                            lineUp = []
                            lineDown = []

                            # midle height
                            midHeight = 480 / 2

                            contour_index = 0
                            textPlatelices = ""

                            # Extract bounding boxes and class names
                            for result in results[0].boxes:
                                x1, y1, x2, y2 = map(int, result.xyxy[0])

                                if y1 < midHeight:
                                    lineUp.append(result)
                                else:
                                    lineDown.append(result)

                                class_id = int(result.cls)
                                class_name = classNames2[class_id]

                                cv2.rectangle(thresh_image, (x1, y1),
                                              (x2, y2), (255, 0, 255), 1)

                                text_x = x1 + 5
                                text_y = y1 + 20
                                contour_index += 1
                                # kết quả nhận diện ký tự của model_yolo2
                                cv2.putText(thresh_image, class_name, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                                # hiển thị trình tự thứ tự index detect boundingbox của model_yolo2
                                # cv2.putText(thresh_image, str(contour_index), (x1, y1 + 25),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            lineUp.sort(key=sortByXmin)
                            for char in lineUp:
                                textPlatelices += classNames2[int(char.cls)]

                            if len(lineDown) > 0:
                                textPlatelices += ""

                                lineDown.sort(key=sortByXmin)
                                for char in lineDown:
                                    textPlatelices += classNames2[int(char.cls)]
                            # Kết quả cuối cùng
                            # cv2.putText(crop_4, textPlatelices, (15, 45),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print(
                                f"Result license plate number: {textPlatelices}")
                            output_filename = os.path.join(
                                output_directory_character_detector, f'{os.path.splitext(filename)[0]}_thresh.jpg')
                            cv2.imwrite(output_filename, crop_4)
                            cv2.namedWindow("Final_Result", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Final_Result", 620, 480)
                            cv2.imshow('Final_Result', crop_4)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
