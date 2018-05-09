#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:56:37 2018

@author: reynaldyhardiyanto
"""

import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier('/Users/reynaldyhardiyanto/anaconda3/pkgs/opencv-3.3.1-py35hb620dcb_1/share/OpenCV/lbpcascades/lbpcascade_frontalface_improved.xml')


for i in range(28):
   for j in range(3):
        # Capture frame-by-frame
        src = '/Users/reynaldyhardiyanto/Desktop/Data Gender/'
        src = src + str(i+1) + '/00' + str(j+1) + '.png'
        frame = cv2.imread(src)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            faceImage = frame[y:y+h,x:x+w]
        path = "/Users/reynaldyhardiyanto/Desktop/Hasil Data Gender/"+str(i+1)
        cv2.imwrite(os.path.join(path,str(j+1)+'.jpg'), faceImage)
cv2.waitKey(0)
# When everything is done, release the capture
