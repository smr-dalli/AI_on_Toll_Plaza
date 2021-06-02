from flask import Flask, render_template, request
import sqlite3
import os

import io
import sys
import json
from functools import lru_cache
from pathlib import Path
from PIL import Image

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
        
        
    def detect(self, img_path: str,image_name: str):
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
        lpd.crop_plate()
        cv2.imwrite('static/test.jpg',self.fig_image)

        cropped_image = cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB)

        extracted_text = pytesseract.image_to_string(cropped_image, 
                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        processed_text = re.findall("[A-Z0-9]",extracted_text)
        text = ''.join(processed_text)
    
        return text
    
    
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


app = Flask(__name__)

@app.route('/')
def home():
    #os.remove('static/detected.jpg')
    return render_template('index.html' )

@app.route('/number_plate',methods=['POST'])
def number_plate():

    if request.method == 'POST':
        
        # for file in os.listdir('static/'):
        #     if file.endswith('.jpg'):
        #         os.remove(file)

        files = request.files['file_1']
        img_bytes = files.read()

        image = Image.open(io.BytesIO(img_bytes))
        image.save('temp/image.jpg')
        
        # print(type(img))
        image_name = str(os.urandom(10)) + '.jpg'
        extracted_text = lpd.detect('temp/image.jpg', image_name)
        # # Plot original image with rectangle around the plate
        # plt.figure(figsize=(24, 24))
        #user_image= plt.imshow(cv2.cvtColor(lpd.fig_image, cv2.COLOR_BGR2RGB))

        conn = sqlite3.connect('database/licence_plates.db')

        c = conn.cursor()
        c.execute(" SELECT * FROM vehicle_number_plate"); 
        final_statement = ''
        result = c.fetchall()
        input_value = extracted_text
        temp = False
        for i in result:
            if input_value in i:
                temp = True

        if temp:
            print('The licence plate is "ACTIVE" and allowed to travel on the roads. Happy Journey :)')
            final_statement ='The licence plate is "ACTIVE" and allowed to travel on the roads. Happy Journey :)'
        else:
            print('The licence plate is "NOT ACTIVE" and it is time to renewal.')
            final_statement ='The licence plate is "NOT ACTIVE" and it is time to renewal.'

        conn.commit()

        conn.close()




    return render_template('number_plate.html',image_path = 'static/test.jpg', number_palte_text = extracted_text, final_statement = final_statement)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    
    response.headers['Cache-Control'] = 'no-cache, no-store'
    return response 


if __name__ == '__main__':
    app.run(debug=True)