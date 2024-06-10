import cv2
import imutils
import numpy as np
import easyocr


# Param
max_size = 10000
min_size = 900

reader = easyocr.Reader(['en'], gpu=True)
# Load image
img_source = cv2.imread(
    r'D:\AI_SERVER_DETECTED_IMG\Running_YOLOv8_Webcam\detection_by_picture\input_images_license_plate\9.jpg', cv2.IMREAD_COLOR)

# Resize image
img = cv2.resize(img_source, (620, 480))
cv2.imshow("img", img)

# Edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
bilateralFilter = cv2.bilateralFilter(
    gray, 1, 10, 10)  # Blur to reduce noise
edged = cv2.Canny(bilateralFilter, 50, 100)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1


reader = easyocr.Reader(['en'], gpu=True)
if detected == 1:

    mask = np.zeros(gray.shape, np.uint8)

    p1 = screenCnt[0][0]
    p2 = screenCnt[1][0]
    p3 = screenCnt[2][0]
    p4 = screenCnt[3][0]

    pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [
                   p3[0], p3[1]], [p4[0], p4[1]]], dtype="int")

    # Sắp xếp các điểm theo trục y (cột thứ 1)
    sorted_pts = pts[np.argsort(pts[:, 1]), :]

    # Lấy 2 điểm trên cùng và 2 điểm dưới cùng
    top_pts = sorted_pts[:2, :]
    bottom_pts = sorted_pts[2:, :]

    # Sắp xếp lại theo trục x (cột thứ 0)
    top_left, top_right = top_pts[np.argsort(top_pts[:, 0]), :]
    bottom_left, bottom_right = bottom_pts[np.argsort(bottom_pts[:, 0]), :]

    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

    new_image = cv2.circle(
        gray, (top_left[0], top_left[1]), 5, (255, 255, 255), -1)
    new_image = cv2.circle(
        gray, (bottom_right[0], bottom_right[1]), 5, (255, 255, 255), -1)

    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('mask image123', new_image)

    # Now crop
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    # Tính toán chiều rộng và chiều cao của hình chữ nhật sau khi cắt
    width = max(np.linalg.norm(pts1[0] - pts1[1]),
                np.linalg.norm(pts1[2] - pts1[3]))
    height = max(np.linalg.norm(pts1[0] - pts1[3]),
                 np.linalg.norm(pts1[1] - pts1[2]))

    # Xác định tọa độ của vùng đích (hình chữ nhật) sau khi cắt
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Tạo ma trận chuyển đổi
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Áp dụng phép biến đổi phối cảnh để cắt và biến đổi vùng ảnh
    result = cv2.warpPerspective(new_image, matrix, (int(width), int(height)))
    cv2.imshow('matrix29u37346234', result)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = new_image[topx:bottomx + 1, topy:bottomy + 1]

    # cv2.imshow('Cropped image', Cropped)
    # _, thresh = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)
    # edged = cv2.Canny(thresh, 50, 100)
    # cv2.imshow('edged', edged)

    contours = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    min_x = float('inf')
    min_y = float('inf')

    max_x = 0
    max_y = 0

    # Vẽ đường viền quanh từng ký tự
    for point in contours[8]:
        x = point[0][0]
        y = point[0][1]
        # print(point)
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        # print(point)
        cv2.drawContours(img, [point], -1, (255, 0, 0), 1)
    cv2.circle(img, (min_x, min_y), 5, (0, 255, 0), -1, )
    cv2.circle(img, (max_x, max_y), 5, (0, 255, 0), -1, )

    character = result[min_y-10:max_y+10, min_x-10:max_x+10]

    results = reader.readtext(character)
    print(f"character:\n{results}")
    cv2.rectangle(result, (min_x-10, min_y-10),
                  (max_x+10, max_y+10), (0, 0, 255), 1)
    cv2.imshow('detailed_boxes', result)
    cv2.imshow('character', character)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
