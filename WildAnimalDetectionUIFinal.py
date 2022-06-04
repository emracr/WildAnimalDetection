from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap

from PIL.ImageQt import ImageQt 
from PIL import Image

class Window(QWidget):
    
    uploadedImagePath = ''
    
    def __init__(self):
        super().__init__()

        self.title = "Yabani Hayvan Tespiti"
        self.top = 150
        self.left = 350
        self.width = 400
        self.height = 300
        self.InitWindow()


    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()

        self.btnUploadImage = QPushButton("Resim Yükle")
        self.btnUploadImage.clicked.connect(self.getImage)
        
        self.btnDetect = QPushButton("Hayvanı Tespit Et")
        self.btnDetect.clicked.connect(self.detectAnimal)
        self.btnDetect.hide()

        vbox.addWidget(self.btnUploadImage)
        vbox.addWidget(self.btnDetect)

        self.label = QLabel()
        vbox.addWidget(self.label)

        self.setLayout(vbox)

        self.show()

    
    
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','d:\\Dataset\\', "Image files (*.jpg *.gif)")
        imagePath = fname[0]
        self.uploadedImagePath = imagePath
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        self.btnDetect.show()
        
    def detectAnimal(self):
       
       image = cv2.imread(self.uploadedImagePath)

       #görüntünün yükseklik ve genişliği alındı
       img_width = image.shape[1]
       img_height = image.shape[0]

       #görüntü üzerinde işlem yapabilmek için 4 boyutlu hale gitiriliyor 1/255 yolo standartı, (416,416) ölçeklendirme boyutu yolo'yu config dosyalarını
       #buna göre indirmiştik, swapRB renki RGB görüntüye çeviriyor
       img_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB = True)

       labels = ["Eagle","Elephant","Frog","Giraffe","Squirrel","Tiger","Tortoise","Zebra"]

       colors = ["33, 255, 107","10, 108, 255","242, 255, 0","0, 255, 255"]
       colors = [np.array(color.split(",")).astype("int") for color in colors]
       colors = np.array(colors)
       colors = np.tile(colors,(20,1))

       model = cv2.dnn.readNetFromDarknet("./new_model/yolov4-obj.cfg","./new_model/yolov4-obj_last.weights")

       layers = model.getLayerNames()

       output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

       model.setInput(img_blob)

       detection_layers = model.forward(output_layer)


       ids_list = []
       boxes_list = []
       confidences_list = []


       for detection_layer in detection_layers:
           for object_detection in detection_layer:
               scores = object_detection[5:]
               predicted_id = np.argmax(scores)
               confidence = scores[predicted_id]
               
               if confidence > 0.50:
                   label = labels[predicted_id]
                   bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                   (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                   
                   start_x = int(box_center_x - (box_width / 2))
                   start_y = int(box_center_y - (box_height / 2))
                   
                   
                   ids_list.append(predicted_id)
                   confidences_list.append(float(confidence))
                   boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                   
       max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
                   

       for max_id in max_ids:
           max_class_id = max_id
           box = boxes_list[max_class_id]
                       
           start_x = box[0]
           start_y = box[1]
           box_width = box[2]
           box_height = box[3]
                       
           predicted_id = ids_list[max_class_id]
           confidence = confidences_list[max_class_id]
           label = labels[predicted_id] + " " + format(confidence,".2f")
          
                   
                    
           end_x = start_x + box_width
           end_y = start_y + box_height
                   
           box_color = colors[predicted_id]
                   
           box_color = [int(each) for each in box_color]
                   
           cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color, 2)
           
           font_family = cv2.FONT_HERSHEY_SIMPLEX
           
           (text_width, text_height) = cv2.getTextSize(label, font_family, fontScale=0.5, thickness=1)[0]
           box_coords = ((start_x - 1, start_y), (start_x + text_width + 2, start_y - 5 - text_height - 2))
           
           cv2.rectangle(image, box_coords[0], box_coords[1], box_color, cv2.FILLED)
           cv2.putText(image, label, (start_x, start_y - 5), font_family, fontScale=0.5, color=(35, 35 ,35), thickness=1)      
       

       cv2.imshow("Tespit Ekranı", image)
       

App = QApplication(sys.argv)
window = Window()
window.resize(1200,800)
sys.exit(App.exec())


