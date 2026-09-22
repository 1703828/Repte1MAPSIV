"""
This is the pipeline for the introduction to data properties exploration. 

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""

##### PYTHON PACKAGES


# import the necessary packages
import numpy as np
import cv2
import glob
import os
#from imutils import perspective
from matplotlib import pyplot as plt

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
#from license_plate.LicensePlateDetector import detectPlates
from LicensePlateDetector import detectPlates


    

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir = '/Users/elenaa/Documents/UNI/QUART/MAPSIV/Challenge 1/real_plates'
Views=['Frontal','Lateral']

plateArea={}
plateAngle={}
imageColor={}
imageIlluminance={}
imageSaturation={}

#### COMPUTE PROPERTIES FOR EACH VIEW
for View in Views:
    
    ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
    plateArea[View]=[]
    plateAngle[View]=[]
    imageColor[View]=[]
    imageIlluminance[View]=[]
    imageSaturation[View]=[]
    # loop over the images
    for imagePath in ImageFiles:
        # load the image
        image = cv2.imread(imagePath)
        # Image Color and Illuminance properties
        imageColor[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0].flatten()))
        imageIlluminance[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2].flatten()))
        imageSaturation[View].append(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1].flatten()))
        # Image ViewPoint (orientation with respect frontal view and focal distance)
        regions=detectPlates(image)
        for reg in regions:
            # Region Properties
            rect = cv2.minAreaRect(reg)
            plateArea[View].append(np.prod(rect[1]))
            # Due to the way cv2.minAreaRect computes the sides of the rectangle
            # (https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/)
            # Depending on view point, the estimated rectangle has not
            # the largest side along the horizontal axis. This cases are corrected to ensure that orientations
            # are always with respect the largest side 
            if (rect[1][0]<rect[1][1]):
                plateAngle[View].append(rect[2]-90)
            else:
                plateAngle[View].append(rect[2])
            
#### VISUALLY EXPLORE PROPERTIES DISTRIBUTION FOR EACH VIEW
## Color Distribution
# Histograms
co=['b','r']
plt.figure()  
for k in np.arange(len(Views)):
    plt.hist(imageColor[Views[k]],bins=20,edgecolor='k',color=co[k],alpha=1-0.5*k)  
plt.title('Color Distribution')
plt.legend(Views)   
# BoxPlots
x=[]   

for k in np.arange(len(Views)):
    x.append(imageColor[Views[k]])    
plt.figure()
plt.boxplot(x,labels=Views)
plt.title('Color Distribution')

## Brightness
# Histograms
co=['b','r']
plt.figure()  
for k in np.arange(len(Views)):
    plt.hist(imageIlluminance[Views[k]],bins=20,edgecolor='k',color=co[k],alpha=1-0.5*k)  
plt.title('Brightness Distribution')
plt.legend(Views)   
# BoxPlots
x=[]   

for k in np.arange(len(Views)):
    x.append(imageIlluminance[Views[k]])    
plt.figure()
plt.boxplot(x,labels=Views)
plt.title('Brightness Distribution')

# Camera ViewPoint
co=['b','r']
plt.figure()  
for k in np.arange(len(Views)):
    plt.hist(plateAngle[Views[k]],bins=20,edgecolor='k',color=co[k],alpha=1-0.5*k)  
plt.title('View Point Distribution')
plt.legend(Views)   
# BoxPlots
x=[]   

for k in np.arange(len(Views)):
    x.append(plateAngle[Views[k]])    
plt.figure()
plt.boxplot(x,labels=Views)
plt.title('View Point Distribution')
plt.show()
