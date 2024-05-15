import cv2
import os
import pytesseract

# Set the path to the Tesseract executable (change this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_canny/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

# Load all image filenames from the directory
for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    img = cv2.imread(image_path)

    # Perform OCR on the entire image using Tesseract
    plate_text = pytesseract.image_to_string(img)

    # Display the detected text
    print("Detected Text:", plate_text)

    # Draw bounding boxes around the detected text
    h, w, _ = img.shape
    text_boxes = pytesseract.image_to_boxes(img)
    for bbox in text_boxes.splitlines():
        bbox = bbox.split()
        x, y, w, h = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        print("x, y, w, h:", x, y, w, h)

    # Show the processed image with license plate information
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)

    # Save the image with license plate information
    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

cv2.destroyAllWindows()
