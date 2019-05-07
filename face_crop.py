#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:56:37 2018

@author: reynaldyhardiyanto
"""

import cv2
import os
import numpy as np
import random
import fnmatch
import argparse
import time

faceCascade = cv2.CascadeClassifier('/Users/reynaldyhardiyanto/anaconda3/pkgs/opencv-3.3.1-py35hb620dcb_1/share/OpenCV/lbpcascades/lbpcascade_frontalface_improved.xml')

def get_args():
    parser = argparse.ArgumentParser(
        description="Face Crop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--raw_images_dir",
        type=str,
        required=True,
        help='input raw images filepath'
    )
    parser.add_argument(
        "--outpath",
        type=float,
        required=True,
        help='result image filepath'
    )
    args = parser.parse_args()
    return args


def crop_faces(
        raw_images_dir,
        outpath,
        image_ext_pattern_list=['jpg']):
    list_images = []
    # print(val_files)
    for root, dirs, files in os.walk(raw_images_dir):
        for filename in files:
            print(filename)
            image_path = os.path.join(root, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
            # Draw a rectangle around the faces
            for i,face in enumerate(faces):
                for (x, y, w, h) in face:
                    faceImage = img[y:y+h,x:x+w]
                    cv2.imwrite(os.path.join(outpath,filename,str(i)), faceImage)
            
                   
if __name__ == "__main__":
    args = get_args()
    raw_images_dir = args.raw_images_dir
    outpath = args.outpath
    crop_faces(raw_images_dir,outpath)
