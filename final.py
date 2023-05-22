import cv2
import os
import numpy as np
import keras
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model
import sys
import time
import multiprocessing
from termcolor import colored
import serial

import smtplib
from email.mime.text import MIMEText

global cap
cap=None

# =============================================================================
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('train.yml')
# cascadePath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascadePath)
# 
# 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# model = keras.models.load_model('model/vlstm_92.h5')
# image_model = VGG16(include_top=True, weights='imagenet')  
# model.summary()  
# #We will use the output of the layer prior to the final
# # classification-layer which is named fc2. This is a fully-connected (or dense) layer.
# transfer_layer = image_model.get_layer('fc2')
# image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
# transfer_values_size = K.int_shape(transfer_layer.output)[1]
# font = cv2.FONT_HERSHEY_SIMPLEX
# # Frame size  
# img_size = 224
# 
# img_size_touple = (img_size, img_size)
# 
# # Number of channels (RGB)
# num_channels = 3
# 
# # Flat frame size
# img_size_flat = img_size * img_size * num_channels
# 
# # Number of classes for classification (Violence-No Violence)
# num_classes = 2
# 
# # Number of files to train
# _num_files_train = 1
# 
# # Number of frames per video
# _images_per_file = 20
# 
# # Number of frames per training set
# _num_images_train = _num_files_train * _images_per_file
# 
# # Video extension
# video_exts = ".avi"
# 
# in_dir = "data"
# alert=0
# 
# #url of video stream
# url = 'http://26.146.143.10:8080/video'
# =============================================================================
# url = 'http://26.72.110.56:8080/video'

# ===================

# 
# if __name__ == "__main__":
# =============================================================================
def send_mail():
    sender_email = "aleenasijoy@gmail.com"
    sender_password = "wgjjcbjhogfwisbt"
    receiver_email = "ethateam@gmail.com"
    message = "Violence Detected"

    # Set up the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    # Send the email
    msg = MIMEText(message)
    msg['Subject'] = 'ALERT!!!!!'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    server.sendmail(sender_email, receiver_email, msg.as_string())

    # Close the connection
    server.quit()
def start(n,filepath):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    model = keras.models.load_model('model/vlstm_92.h5')
    image_model = VGG16(include_top=True, weights='imagenet')    
    #We will use the output of the layer prior to the final
    # classification-layer which is named fc2. This is a fully-connected (or dense) layer.
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Frame size  
    img_size = 224

    img_size_touple = (img_size, img_size)

    # Number of channels (RGB)
    num_channels = 3

    # Flat frame size
    img_size_flat = img_size * img_size * num_channels

    # Number of classes for classification (Violence-No Violence)
    num_classes = 2

    # Number of files to train
    _num_files_train = 1

    # Number of frames per video
    _images_per_file = 20

    # Number of frames per training set
    _num_images_train = _num_files_train * _images_per_file

    # Number of frames per training set
    _num_images_train = _num_files_train * _images_per_file
    if(n==1):
        cap = cv2.VideoCapture(filepath)
    else:
        cap=cv2.VideoCapture(0)
    count = 0 
    images=[]
    shape = (_images_per_file,) + img_size_touple + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    frame_count =0
    while(True):
        ret, frame = cap.read()
        count+=1
        if frame_count == ret:
            break
        else:
            if count <= _images_per_file:
                RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC) 
                images.append(res)
            else:
                resul = np.array(images)
                resul = (resul / 255.).astype(np.float16)
                transfer_shape = (_images_per_file, transfer_values_size)
                transfer_values = np.zeros(shape=transfer_shape, dtype=np.float16)
                transfer_values = image_model_transfer.predict(image_batch)
                print(transfer_values.shape)
                inp = np.array(transfer_values)
                inp = inp[np.newaxis,...]
                print(inp.shape)
                pred = model.predict(inp)
                print(pred)
                res = np.argmax(pred[0])
                count = 0
                images = []
                shape = (_images_per_file,) + img_size_touple + (3,)
                image_batch = np.zeros(shape=shape, dtype=np.float16)
                if res == 0:
                    print("\n\n"+ colored('VIOLENT','red')+" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
                    #print(gray)
                    faces=faceCascade.detectMultiScale(gray, 1.2,5)
                    if len(faces) == 0: # If no faces detected
                        print("No faces detected")
                    else:
                        for(x,y,w,h) in faces:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
                            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                            print(Id)
                            print(conf)
                            if(conf<50):
                                if(Id==1):
                                    print("Aleena")
                                    Id="Aleena"
                                if(Id==2):
                                    print("Deepthi")
                                    Id="Deepthi"
                                if(Id==3):
                                    print("Chaitra")
                                    Id="Chaitra"   
                                if(Id==4):
                                    print("Abhi")
                                    Id="Abhi"
                                alert=1
                            else:
                                Id="unknown"
                                print("unknown user")
                          
                            cv2.putText(frame, str(Id), (x,y-40), font, 1, (255,255,255), 3)
                else:
                    print("\n\n" + colored('NON-VIOLENT','green') +" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
                    #print(gray)
                    faces=faceCascade.detectMultiScale(gray, 1.2,5)
                    if len(faces) == 0: # If no faces detected
                        print("No faces detected")
                    else:
                        for(x,y,w,h) in faces:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
                            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                            print(Id)
                            print(conf)
                            if(conf<50):
                                if(Id==1):
                                    print("Aleena")
                                    Id="Aleena"
                                if(Id==2):
                                    print("Deepthi")
                                    Id="Deepthi"
                                if(Id==3):
                                    print("Chaitra")
                                    Id="Chaitra"   
                                if(Id==4):
                                    print("Abhi")
                                    Id="Abhi"
                                #send_mail()
                            else:
                                Id="unknown"
                                print("unknown user")
                          
                            cv2.putText(frame, str(Id), (x,y-40), font, 1, (255,255,255), 3)
        # showing the video stream
                if frame is not None:
                    cv2.imshow('frame',frame)
                q = cv2.waitKey(1)
                if q == ord("q"):
                    break
#start(0,'nil')
    print('Cleaning Up')
cv2.destroyAllWindows()
    

