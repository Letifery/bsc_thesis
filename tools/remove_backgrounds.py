import os, random, shutil, math
from rembg import remove 
from PIL import Image 
import cv2
import numpy as np
import skimage.exposure

DATA_PATH = r"C:\Users\Letifery\Desktop\Bachelorarbeit\tools"
#SAVE_INPLACE = True

i = 0
#Source code from https://stackoverflow.com/questions/63526621/remove-background-from-image-python

#Important!!! It will still create a transparent background which still has to be removed in another step or simply implemented here (i.e. simply via rembg), I still
#needed the script to leave a transparent background behind for other tasks.

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        img = cv2.imread(root+"/"+file)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) #erosion -> erosion -> dilation -> dilation, morphologyEx works differently than simply using MORPH.OPEN
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        contour = np.zeros_like(gray)
        cv2.drawContours(contour, [big_contour], 0, 255, -1)

        blur = cv2.GaussianBlur(contour, (5,5), sigmaX=0, sigmaY=0, borderType = cv2.BORDER_DEFAULT)

        mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))

        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask
        
        cv2.imwrite('ex_image_thresh.png', thresh)
        cv2.imwrite('ex_image_morph.png', morph)
        cv2.imwrite('ex_image_contour.png', contour)
        cv2.imwrite('ex_image_mask.png', mask)
        cv2.imwrite('ex_image_antialiased.png', result)
        
        cv2.imwrite(root+"/"+file, result)