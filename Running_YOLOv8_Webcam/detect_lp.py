import cv2
import imutils
import numpy as np
import skimage.measure

# Param
max_size = 10000
min_size = 900

# Load image
img = cv2.imread('test1.jpg', cv2.IMREAD_COLOR)

# Resize image
img = cv2.resize(img, (620, 480))

# Edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

gray = cv2.bilateralFilter(gray, 1, 10, 10)  # Blur to reduce noise
# cv2.imshow('gray image', gray)
edged = cv2.Canny(gray, 200, 800)  # Perform Edge detection
# cv2.imshow('edged image', edged)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    # print("cv2.contourArea(c): ", cv2.contourArea(c))
    # print("approx: ", approx)
    # print("peri: ", peri)
    # if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

# cv2.waitKey(0)
# cv2.destroyAllWindows()

if detected == 1:
    # cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    p1 = screenCnt[0][0]
    p2 = screenCnt[1][0]
    p3 = screenCnt[2][0]
    p4 = screenCnt[3][0]

    pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype="int")

    # Sắp xếp các điểm theo trục y (cột thứ 1)
    sorted_pts = pts[np.argsort(pts[:, 1]), :]

    # Lấy 2 điểm trên cùng và 2 điểm dưới cùng
    top_pts = sorted_pts[:2, :]
    bottom_pts = sorted_pts[2:, :]

    # Sắp xếp lại theo trục x (cột thứ 0)
    top_left, top_right = top_pts[np.argsort(top_pts[:, 0]), :]
    bottom_left, bottom_right = bottom_pts[np.argsort(bottom_pts[:, 0]), :]
    print(top_left, top_right, bottom_left, bottom_right)
    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

    new_image = cv2.circle(gray, (top_left[0],top_left[1]), 5, (255, 255, 255) , -1)
    new_image = cv2.circle(gray, (bottom_right[0],bottom_right[1]), 5, (255, 255, 255) , -1)

    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('new_image image', new_image)

    # Now crop
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    # Tính toán chiều rộng và chiều cao của hình chữ nhật sau khi cắt
    width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
    height = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))

    # print(width, height, np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]), np.linalg.norm(pts1[0] - pts1[3]),np.linalg.norm(pts1[1] - pts1[2]))

    # Xác định tọa độ của vùng đích (hình chữ nhật) sau khi cắt
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Tạo ma trận chuyển đổi
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # # Áp dụng phép biến đổi phối cảnh để cắt và biến đổi vùng ảnh
    result = cv2.warpPerspective(new_image, matrix, (int(width), int(height)))

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = new_image[topx:bottomx + 1, topy:bottomy + 1]

    cv2.imshow('Cropped image', Cropped)
    cv2.imshow('result image', result)
    cv2.imshow('gray image', gray)

    # print("labels: " ,labels)
    # Display image
    # cv2.imshow('Input image', img)
    # cv2.imshow('License plate', Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()