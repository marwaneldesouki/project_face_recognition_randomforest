import os
import numpy as np
import cv2
import pickle
from PIL import ImageGrab

name = input("type your name:")
path = "./images/"+name
count = 0
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame= cap.read() #camera
    # frame = np.array(ImageGrab.grab(bbox=(0,40,1200,1200))) #screen
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces =  face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        count +=1
        roi_color = frame[y:y+h,x:x+w]
        if not os.path.exists(path):
            os.makedirs(path)
        img_item = './images/'+name+'/'+str(count)+'.png'
        cv2.imwrite(img_item,roi_color) # save image
        color = (255,0,0) #bgr
        stroke = 2
        end_cordx = x+w
        end_cordy = y+h
        cv2.rectangle(frame,(x,y),(end_cordx,end_cordy),color,stroke) #draw rectangle

    cv2.imshow('frame',frame)#display window
    if cv2.waitKey(20) & 0xFF ==ord('q') or count == 10:
        break
