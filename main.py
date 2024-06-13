import torch
import torch.nn as nn
import numpy as np
import copy
from torchvision import models
from torchvision.models import ViT_H_14_Weights, ResNet50_Weights, ViT_L_16_Weights, ResNet18_Weights
from torchvision.models.quantization import ResNet50_QuantizedWeights

from framework import visualisation
from framework import dataloader
from framework import model

from tools import createdataset

from transformers import ViTImageProcessor, ViTModel

DATA_PATHS = [r"C:\Users\Letifery\Desktop\Bachelorarbeit\data\_clean_tiles\dataset_without_background_unsharp"]
            
TVT_PATH = r"C:\Users\Letifery\Desktop\Bachelorarbeit\data\\"
CLASS_NUMBER = 2
EPOCHS = 50
FOLDS = 5

#basemodel = models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
#basemodel = models.convnext_large(pretrained=True)
#models.quantization.resnet50(weights=ResNet50_QuantizedWeights.DEFAULT, quantize = True),

basemodels = [#models.resnet50(),
              #models.resnet50(weights=ResNet50_Weights.DEFAULT), 
              #models.resnet18(weights=ResNet18_Weights.DEFAULT),
              models.vit_l_16(weights=ViT_L_16_Weights.DEFAULT),
             ]

#There sadly is a pretty nasty bug in the current implementation: using multiple basemodels in a sequence will break the architectures, resulting in either low validation
#accuracies or in RuntimeErrors concerning the second basemodel and onwards. The first basemodel seems to work just fine

for DATA_PATH in DATA_PATHS:
    dl = dataloader.DataLoader(TVT_PATH = TVT_PATH, DATA_PATH = DATA_PATH)
    for i, basemodel in enumerate(basemodels):
        basemodel = basemodel.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        SAVE_NAME = basemodel.__class__.__name__+str(CLASS_NUMBER)+"_"+str(EPOCHS)+DATA_PATH.split("\\")[-2]+"-"+DATA_PATH.split("\\")[-1]+"_"+str(i)#+("nopretrain" if not i else "")
        print(SAVE_NAME)
        
        for param in basemodel.parameters():
            param.requires_grad = False
        if not i:
            try:
                fc_inputs = basemodel.fc.in_features
                basemodel.fc = nn.Sequential(
                    nn.Linear(fc_inputs, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, CLASS_NUMBER),
                    nn.LogSoftmax(dim=1)
                )
            except:
                assert(hasattr(basemodel.heads, "head") and isinstance(basemodel.heads.head, nn.Linear))
                basemodel.heads.head = nn.Sequential(
                    nn.Linear(basemodel.heads.head.in_features, CLASS_NUMBER), 
                    nn.LogSoftmax(dim=1)
                )

#        loss_func = nn.CrossEntropyLoss()
        loss_func = nn.NLLLoss()
        optimizer = torch.optim.Adam(basemodel.parameters(), lr=0.003) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3)
        trained_model, bstats, wstats = model.train_and_validate(basemodel, loss_func, optimizer, EPOCHS, SAVE_NAME, CLASS_NUMBER, dl, FOLDS, scheduler)

        torch.save(bstats, "models//"+SAVE_NAME+'_beststats.pt')
        torch.save(wstats, "models//"+SAVE_NAME+'_worststats.pt')

        visualisation.plot_loss([[z[0],z[1], z[2], z[3]] for z in bstats], "visuals/")
        visualisation.plot_acc([[z[0],z[1], z[2], z[3]] for z in wstats], "visuals/")



