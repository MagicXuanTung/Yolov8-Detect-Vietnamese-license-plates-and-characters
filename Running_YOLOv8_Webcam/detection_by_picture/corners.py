import cv2
import numpy as np
import os

# Đường dẫn tới thư mục chứa ảnh
input_directory = './Running_YOLOv8_Webcam/detection_by_picture/output_images_plates_cropped/'

# Lấy danh sách tất cả các file ảnh trong thư mục
image_files = [f for f in os.listdir(input_directory) if os.path.isfile(
    os.path.join(input_directory, f))]

# Duyệt qua từng file ảnh
for image_file in image_files:
    # Đọc ảnh
    img_path = os.path.join(input_directory, image_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (620, 480))

    # Kiểm tra xem ảnh có đọc được không
    if img is None:
        print(f"Không thể đọc được ảnh: {image_file}")
        continue

    # Chuyển đổi ảnh sang thang độ xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chuyển đổi ảnh sang kiểu float32
    gray = np.float32(gray)

    # Áp dụng thuật toán Harris Corner Detector
    dst = cv2.cornerHarris(gray, 9, 9, 0.1)

    # Dùng threshold để xác định các góc (option 1)
    dst = cv2.dilate(dst, None)
    img[dst > 0.1 * dst.max()] = [0, 255, 0]

    # Xác định các góc bằng phương pháp Shi-Tomasi (option 2)
    # corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 10)
    # if corners is not None:
    #     corners = np.int0(corners)
    #     for i in corners:
    #         x, y = i.ravel()
    #         cv2.circle(img, (x, y), 5, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow('Corners', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
