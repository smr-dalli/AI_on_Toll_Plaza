import cv2 
import re
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import pytesseract

# Set tesseract path to where the tesseract exe file is located (Edit this path accordingly based on your own settings)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None
        
        
    def detect(self, img_path: str):
        orig = cv2.imread(img_path)
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                cv2.rectangle(img, (x,y), (x + w, y + h), self.color, 15)
                cv2.putText(img, label + ' ' + confidence, (x, y + 20), self.font, 3, (255, 255, 255), 3)
        self.fig_image = img
        self.coordinates = (x, y, w, h)
        return
    
    
    def crop_plate(self):
        x, y, w, h = self.coordinates
        roi = self.img[y:y + h, x:x + w]
        self.roi_image = roi
        return

lpd = LicensePlateDetector(
    pth_weights='YOLO/yolov3_training_last.weights', 
    pth_cfg='YOLO/yolov3_testing.cfg', 
    pth_classes='YOLO/classes.txt'
)

# Detect license plate
lpd.detect('database_images/fgDry.jpg')

# Plot original image with rectangle around the plate
plt.figure(figsize=(24, 24))
plt.imshow(cv2.cvtColor(lpd.fig_image, cv2.COLOR_BGR2RGB))
plt.savefig('saved_images/detected.jpg')
plt.show()

# Crop plate and show cropped plate
lpd.crop_plate()
plt.figure(figsize=(10, 4))
#kernel = np.ones((3,3),np.uint8)
cropped_image = cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB)
#_,thresh = cv2.threshold(cropped_image,127,255,cv2.THRESH_BINARY_INV)
#gaussian_blur = cv2.GaussianBlur(thresh,(3,3),0)
#canny = cv2.Canny(gaussian_blur,100,200)
#dilation = cv2.dilate(canny,kernel)

plt.imshow(cropped_image)


# Display the text extracted from the car plate
extracted_text = pytesseract.image_to_string(cropped_image, 
                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

processed_text = re.findall("[A-Z0-9]",extracted_text)
text = ''.join(processed_text)
print(text)


# # Testing all PSM values
# for i in range(3,14):
#     print(f'PSM: {i}')
#     print(pytesseract.image_to_string(cropped_image, 
#                                       config = f'--psm {i} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

import sqlite3

conn = sqlite3.connect('database/licence_plates.db')

c = conn.cursor()

#c.execute('CREATE TABLE vehicle_number_plate (id integer,plates text)')

#c.execute("INSERT INTO licence_plate VALUES (1,'CZ20FSE')")

# number_plate = [
#     (1,'DL8CAF5030'),
#     (2,'HR26DK8337'),
#     (3,'YX38590'),
#     (4,'KWI3MXR'),
#     (5,'COVID19'),
#     (6,'KA05NB1786'),
#     (7,'MH20DV2366'),
#     ]
# c.executemany("INSERT INTO vehicle_number_plate VALUES (?,?)", number_plate)
c.execute(" SELECT * FROM vehicle_number_plate"); 
#c.execute(" SELECT * FROM vehicle_number_plate where @text LIKE '%'+plates+'%' "); 
#c.execute(" SELECT * FROM vehicle_number_plate where find_in_set(text, plates)"); 
result = c.fetchall()
input_value = text
temp = False
for i in result:
    if input_value in i:
        temp = True

if temp:
    print('The licence plate is active and allowed to travel on roads.')
else:
    print('The licence plate is inactive and it is time to renewal.')

conn.commit()

conn.close()