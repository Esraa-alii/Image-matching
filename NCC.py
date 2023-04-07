# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Normalized_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor=np.sum(roi*target)
    nor = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
    return cor / nor


def template_matching(image, target):
    # initial parameter
    img_c = cv2.imread(image)
    target = cv2.imread(target,0) #Reading sub image in grey mode
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) 
    height,width=img.shape
    tar_height,tar_width=target.shape
    (max_Y,max_X)=(0, 0)
    MaxValue = 0

    # Set image, target and result value matrix
    img=np.array(img, dtype="int")
    target=np.array(target, dtype="int")
    NccValue=np.zeros((height-tar_height,width-tar_width))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0,height-tar_height):
        for x in range(0,width-tar_width):
            # image roi
            roi=img[y:y+tar_height,x:x+tar_width]
            # calculate ncc value
            NccValue[y,x] = Normalized_Cross_Correlation(roi,target)
            # find the most match area
            if NccValue[y,x]>MaxValue:
                MaxValue=NccValue[y,x]
                (max_Y,max_X) = (y,x)

    return (max_X,max_Y)