Setup for project

1. Install Python: https://www.python.org/
2. Install Anaconda ( development environment): https://www.anaconda.com/download
3. Run install file requirements.txt (by Terminal on Project)
   Content inside file “requirements.txt”:

pip install -r requirements.txt

ultralytics

opencv-python

easyoc

numpy

4. ultralytics (The YOLO model is used to detect the location of objects in an image and return bounding boxes for each object)
5. opencv-python (OpenCV is used to read, process and display images)
6. easyocr (Use EasyOCR to read text on license plates)

• Detection license_plate_recognition_symbols

Preprocess.py:

![alt text](imgGrayscalePlusTopHatMinusBlackHat.jpg)

Start detection
Color img :

![alt text](Color.png)

Black and White:

![alt text](BlackWhite.png)

Canny Edge Detection:

![alt text](Canny.png)

Find and Draw Contours:

![alt text](Contours.png)

Font text License plate Vietnamese:

[text](font_bien_so_xe_vn.TTF)
![alt text](font_bien_so_xe_vn.png)

final_v0 (detect-character-by-model + final_result_by EASY OCR)
final_v1 (detect-character-by-model "character_detector.pt")

original_image
![alt text](original_image.jpg)

original_img got crop_detect ( license_plate_detector.pt )
![alt text](original_img_crop_only.png)

Image got detect and transform:
![alt text](<image_got_transform_and got_detect.jpg>)

Final result:
![alt text](Final_result.jpg)

Bonus:

1. You can delete all img in \input_images_license_plate , and replace your img in folder
   File rename_img.py in folder \img_input_rename , replace you input \_path and output_path to rename img to manage it (ex: 1.jpg , 2.jpg) 1.jpg

2. [text](Running_YOLOv8_Webcam/Empty_all_output_img.py) this file to clean up all output images folder
