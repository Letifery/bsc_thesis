import cv2
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

#sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
#sr_model = r"models\LapSRN_x4"
#sr.readModel(sr_model+".pb" if sr_model[-3:] != ".pb" else "")
#sr.setModel("lapsrn",4)

INPUT_PATH = r"C:\Users\Letifery\Desktop\Bachelorarbeit\data\_fabricnet_datasetsplit\base_dataset\satin\\"
OUTPUT_PATH = r"C:\Users\Letifery\Desktop\Bachelorarbeit\data\_fabricnet_datasetsplit\base_dataset_unsharpen\satin\\"

SUPERRESOLUTION_FLAG, OPENCV_RESCALE_FLAG, KERNEL_SHARPEN_FLAG, UNSHARP_MASK_FLAG = False, False, False, True

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
interpolations = [0,1,2,3,4,5,6]                                #7,8,16 nicht nutzbar atm weils keine Interpolationen sind sondern zus. flags o.ä. 
interpolations_name_map = {0:["INTER_NEAREST", cv2.INTER_NEAREST], 
                          1:["INTER_LINEAR",cv2.INTER_LINEAR],
                          2:["INTER_CUBIC",cv2.INTER_CUBIC],
                          3:["INTER_AREA",cv2.INTER_AREA],
                          4:["INTER_LANCZOS4",cv2.INTER_LANCZOS4], 
                          5:["INTER_LINEAR_EXACT",cv2.INTER_LINEAR_EXACT], 
                          6:["INTER_NEAREST_EXACT",cv2.INTER_NEAREST_EXACT],
                          7:["INTER_MAX",cv2.INTER_MAX],
                          8:["WARP_FILL_OUTLIERS",cv2.WARP_FILL_OUTLIERS],
                          16:["WARP_INVERSE_MAP",cv2.WARP_INVERSE_MAP]}

def unsharp_masking(image, kernel_size=(5, 5), sigma = 1.5, amount = 2.5, threshold = 0.25, repetitions = 1):
    for x in range(repetitions):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        image = sharpened
    return [sharpened,("k%s-s%s-a%s-t%s-r%s" % (kernel_size, sigma, amount, threshold, repetitions))]

for root, _, folder in os.walk(INPUT_PATH):
    for c, file in enumerate(folder):
        print("\n[%s] <%s/%s> : %s" % (root, c, len(folder), file))
        img_name = OUTPUT_PATH+pathlib.Path(file).stem
        img = cv2.imread(INPUT_PATH+file)
        if OPENCV_RESCALE_FLAG:
            img_list, name_list = [img]*len(interpolations), [img_name]*len(interpolations)
            for i, image in enumerate(img_list):
                img_list[i] = cv2.resize(image,None,fx=4, fy=4, interpolation = interpolations_name_map[interpolations[i]][1])
                name_list[i] += "_"+interpolations_name_map[interpolations[i]][0]   
        else:
            img_list, name_list = [img],[img_name]
        if SUPERRESOLUTION_FLAG:
            img_list[0] = sr.upsample(img_list[0])[:,:,::-1]
            name_list[0] += "_"+sr_model
        if KERNEL_SHARPEN_FLAG:
            for i, image in enumerate(img_list):
                img_list[i] = cv2.filter2D(image, -1, kernel)
                name_list[i] += "_ksharpened"
        if UNSHARP_MASK_FLAG:
            for i, image in enumerate(img_list):
                t_image, parameter_string = unsharp_masking(image)
                img_list[i] = t_image
                name_list[i] += "_unsharp_mask"+parameter_string
        for i, image in enumerate(img_list):
            cv2.imwrite(name_list[i]+".png", image)