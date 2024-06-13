import traceback

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2


PATH_DATA = "dataset_without_background"
SPLIT_FACTOR = 20
REMOVE_BACKGROUND = (False, False)  #1: If background should be removed, 2: If only the object should persist, 3: If the image should be saved
SAMPLE_SAVE_FLAG = True
WITHOUT_PREFIX_CHECK = False
SKIP_RATIO = 0.05
SKIP_CLR_AVG = 50

data = pd.read_excel("labels.xlsx")

tdata = data[~data[["filename", "label"]]["label"].isnull()]
tdata = tdata[tdata["usable"]==1.0]
tdata = tdata[tdata["label"]!="-"]
for index, row in tdata.iterrows():
    print(index)
    try:
        Path("labeled_dataset"+"/"+row["label"]).mkdir(parents=False, exist_ok=True)
        img_path = PATH_DATA+(row["filename"].partition("_")[-1]) if WITHOUT_PREFIX_CHECK else PATH_DATA+row["filename"]
        if REMOVE_BACKGROUND[0]:        #Older variant, the background-removal tool in the tools folder should be used
            img = cv2.imread(img_path, 1)
            original = img.copy()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.threshold(blur,25,255,cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            dilate = cv2.dilate(thresh, kernel, iterations=1)
            cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            obj_index = cnts.index(max(cnts, key=len))
            contour_img = cv2.drawContours(img, cnts, obj_index, (0,255,0), 3)
                
            x,y,w,h = cv2.boundingRect(cnts[0])
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            if SAMPLE_SAVE_FLAG:
                cv2.imwrite("gray.png", gray)
                cv2.imwrite("blur.png", blur)
                cv2.imwrite("thresh.png", thresh)
                cv2.imwrite("dilated.png", dilate)
                cv2.imwrite("end.png", ROI)
                SAMPLE_SAVE_FLAG = False
            if REMOVE_BACKGROUND[1]:
                Path("bb_imgs").mkdir(parents=False, exist_ok=True)
                cv2.imwrite("bb_imgs/"+str(index)+row["filename"], ROI)
                
        img = np.array(Image.open(img_path)) if not REMOVE_BACKGROUND[0] else np.array(ROI)
        
        M = img.shape[0]//SPLIT_FACTOR
        N = img.shape[1]//SPLIT_FACTOR
        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
        for i, tile in enumerate(tiles):
            #Will check if the average color of the image is less than SKIP_CLR_AVG or if the image contains more black pixels (percentage) than SKIP_RATIO, skip if true
            if sum(np.average(np.average(tile, axis=0), axis=0)) <= SKIP_CLR_AVG or (np.count_nonzero(np.all(tile==[0]*tile.shape[2],axis=2))/(tile.shape[0]*tile.shape[1])) >= SKIP_RATIO:
                continue
            tile = Image.fromarray(tile)
            tile.save("labeled_dataset"+"/"+row["label"]+"/"+row["filename"]+"_"+str(i)+".png") 
    except (AttributeError, FileNotFoundError):
        print("[INFO] Couldn't find " + row["filename"])
        print(traceback.format_exc())
        continue
    except ValueError:
        if REMOVE_BACKGROUND[0]:
            Path("log").mkdir(parents=False, exist_ok=True)
            cv2.imwrite("log/gray.png", gray)
            cv2.imwrite("log/blur.png", blur)
            cv2.imwrite("log/thresh.png", thresh)
            cv2.imwrite("log/dilated.png", dilate)
        print("[INFO] Couldn't process"+row["filename"]+" correctly")
        print(traceback.format_exc())
        
    except:
        print("[INFO] Couldn't process"+row["filename"]+" correctly")
        print(traceback.format_exc())
        continue