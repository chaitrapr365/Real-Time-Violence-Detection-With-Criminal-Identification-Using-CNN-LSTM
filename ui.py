# -*- coding: utf-8 -*-
"""
Created on Mon May 15 01:01:38 2023

@author: chaip
"""
import final as f
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

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import smtplib
from email.mime.text import MIMEText

"""sender_email = "aleenasijoy@gmail.com"
sender_password = "wgjjcbjhogfwisbt"
receiver_email = "allenpaulson19@gmail.com"
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
server.quit()"""

global cap
cap=None

def real_time_detection():
    # TODO: Implement the real-time detection functionality
    print("Real-Time Detection")
    text_label = tk.Label(window, text="REAL TIME")
    # Add the label widget to the main window
    text_label.pack()
    f.start(0,'nil')



def upload_video():
# =============================================================================
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read('train.yml')
#     cascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(cascadePath)
# 
# 
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     model = keras.models.load_model('model/vlstm_92.h5')
#     image_model = VGG16(include_top=True, weights='imagenet')  
#     #model.summary()  
#     #We will use the output of the layer prior to the final
#     # classification-layer which is named fc2. This is a fully-connected (or dense) layer.
#     transfer_layer = image_model.get_layer('fc2')q
#     image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
#     transfer_values_size = K.int_shape(transfer_layer.output)[1]
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     # Frame size  
#     img_size = 224
# 
#     img_size_touple = (img_size, img_size)
# 
#     # Number of channels (RGB)
#     num_channels = 3
# 
#     # Flat frame size
#     img_size_flat = img_size * img_size * num_channels
# 
#     # Number of classes for classification (Violence-No Violence)
#     num_classes = 2
# 
#     # Number of files to train
#     _num_files_train = 1
# 
#     # Number of frames per video
#     _images_per_file = 20
# 
#     # Number of frames per training set
#     _num_images_train = _num_files_train * _images_per_file
# 
#     # Video extension
#     video_exts = ".avi"
# 
#     in_dir = "data"
# =============================================================================
    # Open a file dialog to select a video file
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

    if filepath:
        # TODO: Implement the video upload functionality
        print("Uploaded Video:", filepath)
        filename=filepath.split('/')
        fname=filename[-1]
        text_label = tk.Label(window, text=fname+" Uploaded")
        # Add the label widget to the main window
        text_label.pack()
        #f.start(1,filepath)
        detect(1,filepath)
        
def detect(n,filepath):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    model = keras.models.load_model('model/vlstm_new.h5')
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
                image_batch = resul
                shape = (_images_per_file, transfer_values_size)
                transfer_values = np.zeros(shape=shape, dtype=np.float16)
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

def resize_background(event):
    canvas.delete("gradient")
    canvas.create_rectangle(0, 0, event.width, event.height, fill=gradient_color1, width=0, tags="gradient")
    for i in range(event.height):
        canvas.create_line(0, i, event.width, i, fill=gradient_color2, width=1, tags="gradient")

# Create the main window
window = tk.Tk()
window.title("VIOLENCE DETECTION")

# Create a canvas widget for the background gradient
canvas = tk.Canvas(window)
canvas.pack(fill="both", expand=True)

# Define the gradient colors
gradient_color1 = "#8ED6FF"  # Light blue
gradient_color2 = "#004CB5"  # Dark blue

# Apply a custom style with light colors for buttons and labels
style = ttk.Style()
style.configure("Custom.TButton", background="#D6E6F5", foreground="#333333", font=("Helvetica", 12, "bold"), relief="flat", borderwidth=0)
style.configure("Custom.TLabel", background="#D6E6F5", foreground="#333333", font=("Helvetica", 14))

# Create a frame to hold the buttons and labels
frame = tk.Frame(window, bg="#D6E6F5")
frame.place(relx=0.5, rely=0.5, anchor="center")

# Create a label widget with a custom style
label = ttk.Label(frame, text="Choose an option:", style="Custom.TLabel")
label.pack(pady=10)

# Create a frame to hold the buttons
button_frame = tk.Frame(frame, bg="#D6E6F5")
button_frame.pack()

# Create a button for real-time detection with a custom style
realtime_button = ttk.Button(button_frame, text="Real-Time Detection", command=real_time_detection, style="Custom.TButton")
realtime_button.pack(side="left", padx=5)

# Create a button for video upload with a custom style
upload_button = ttk.Button(button_frame, text="Upload Video", command=upload_video, style="Custom.TButton")
upload_button.pack(side="left", padx=5)

# Bind the resizing event of the canvas to adjust the background gradient
canvas.bind("<Configure>", resize_background)

# Start the main event loop
window.mainloop()