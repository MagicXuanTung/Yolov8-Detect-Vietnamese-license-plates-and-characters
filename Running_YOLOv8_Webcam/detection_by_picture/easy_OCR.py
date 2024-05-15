import cv2
import os
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Directory containing the images for license plate detection
image_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_cropped_images_plates_bw/"

# Output directory for saving images with license plate information
output_directory = "./Running_YOLOv8_Webcam/detection_by_picture/output_images_plates"
os.makedirs(output_directory, exist_ok=True)

for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    img = cv2.imread(image_path)

    # Perform OCR on the entire image using EasyOCR
    results = reader.readtext(img)

    # Extract and display text without drawing bounding boxes
    for result in results:
        text = result[1]
        print("Detected Text:", text)

    # Show the processed image with text information
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)

    # Save the image with text information
    output_filename = os.path.join(
        output_directory, f'{image_filename}_output.jpg')
    cv2.imwrite(output_filename, img)

cv2.destroyAllWindows()
