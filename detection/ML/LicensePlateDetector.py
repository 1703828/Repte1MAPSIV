# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025

@author: debora
"""
# import the necessary packages
#from collections import namedtuple
# from skimage.filters import threshold_local
# from skimage import segmentation
# from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

SHOW=1
minPlateW=100
minPlateH=30

def detectPlates(image):
        imHeight, imWidth = image.shape[:2]

        # if the width is greater than 640 pixels, then resize the image
        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)
            
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions

        
        # Structuring Element first rectangular shape (width 15 hight 5)  second square shape
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  
        # list of potential regions with a license plate
        regions = []
        

        # convert the image to grayscale, and apply the blackhat operation to emphasize narrow regions with dark gray level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel,iterations =3) #rectKernel
        
        if (SHOW):
            plt.figure()
            plt.imshow(blackhat,cmap='gray')
            plt.title("Black Top Hat ")

        # numbers have vertical changes in gradient
        gradX = cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gradient X")
        
        # gaussian blur with a 5 x 5 kernel to smooth detail and noise
        gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel,iterations = 2)
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gausian Gx")
            
        
        # el valor de corte se fija como el 40% del màximo
        ThrValue= (0.40)*np.max(gradX)
        ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]
        if (SHOW):
            plt.figure()
            plt.imshow(ThrGradX,cmap='gray')
            plt.title("Threshold Gx")
        

        # some morphological operation to join parts first opening to remove small spots second to grow the area of license plate
        thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel,iterations = 4 )
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        if(SHOW):
            plt.figure()
            plt.imshow(thresh,cmap='gray')
            plt.title("Possible license plates")
                  
        # find contours in the thresholded image
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and
            # aspect ratio
            (x,y,w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspectRatio = w / float(h)
            if (SHOW):
                print("BLOB ANALYSIS ->",x,y,w,h,aspectRatio,area)
            
            # condition of not touching the border of the image
            NotouchBorder = x!=0 and y!=0 and x+w!=imWidth and y+h!=imHeight

            
                
            # dimension conditions of a license plate
            keepArea = area > 3400 and area < 8000
            keepWidth = w > minPlateW and w <= 250
            keepHeight = h > minPlateH and h <= 60
            keepAspectRatio = 2.5<w/h<7

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            if all((NotouchBorder,keepAspectRatio,keepWidth,keepHeight,keepArea)):
                # compute the rotated bounding box of the region: 
                
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)

                regions.append(box)
                if (SHOW):
                    print("REGION BOX ACCEPTED->",box)
        return regions


    