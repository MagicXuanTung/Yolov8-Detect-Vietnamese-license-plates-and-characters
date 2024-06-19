Setup for project

1. Install Python: https://www.python.org/
2. Install Anaconda ( development environment): https://www.anaconda.com/download
3. Run install file requirements.txt (by Terminal on Project)
   Content inside file “requirements.txt”:

pip install -r requirements.txt

ultralytics==8.2.28
opencv-python==4.10.0
easyocr==1.7.1
numpy==1.24.4
Pillow==10.3.0
imutils==0.5.4


- ultralytics (The YOLO model is used to detect the location of objects in an image and return bounding boxes for each object)
- opencv-python (OpenCV is used to read, process and display images)
- easyocr (Use EasyOCR to read text on license plates)
- numpy (Used for numerical computations and data handling related to image processing, including array operations, linear algebra, etc)
- Pillow (Provides tools to open, manipulate, and save images, necessary for preprocessing images before license plate recognition)
- imutils (Provides convenient functions for image processing tasks like resizing, rotating, cropping, and other transformations)

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


original_image:
![alt text](original_image.jpg)


original_img got crop_detect: ( license_plate_detector.pt )
![alt text](original_img_crop_only.png)


Image got detect and transform:
![alt text](<image_got_transform_and got_detect.jpg>)

Final result:
![alt text](Final_result.jpg)

Bonus:
1. You can delete all img in \input_images_license_plate , and replace your img in folder
   File rename_img.py in folder \img_input_rename , replace you input \_path and output_path to rename img to manage it (ex: 1.jpg , 2.jpg) 1.jpg

2. [text](Running_YOLOv8_Webcam/Empty_all_output_img.py) this file to clean up all output images folder
