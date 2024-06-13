import torch
from torchvision import datasets
from torchvision.transforms import v2

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255. 

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)

class DataLoader:
    def __init__(self, TVT_PATH, DATA_PATH):
        self.TVT_PATH = TVT_PATH
        self.DATA_PATH = DATA_PATH
        self.image_transforms = { 
            'train': v2.Compose([
                v2.RandomRotation(degrees=90),
                v2.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(0.25),#
                v2.RandomVerticalFlip(0.25),
                v2.RandomCrop(size=(224,224)),
#                v2.CenterCrop(size=128),
                v2.ToTensor(),
#                v2.Grayscale(num_output_channels=3),
                v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
#                ThresholdTransform(thr_255=160),
            ]),
            'test': v2.Compose([
                v2.Resize(size=256),
                v2.CenterCrop(size=224),
                v2.ToTensor(),
#                v2.Grayscale(),
                v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ]),
            'val': v2.Compose([
                v2.Resize(size=int(256)),
                v2.CenterCrop(size=224),
#                v2.RandomHorizontalFlip(0.25),#
#                v2.RandomVerticalFlip(0.25),
#                v2.RandomCrop(size=(128,128)),
                v2.ToTensor(),
#                v2.Grayscale(num_output_channels=3),
                v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
#                ThresholdTransform(thr_255=160),
            ])}
    
    def get_data(self, include_test = True):
        data = {'train': datasets.ImageFolder(root=self.TVT_PATH+"\\train", transform=self.image_transforms['train']),
                'val': datasets.ImageFolder(root=self.TVT_PATH+"\\val", transform=self.image_transforms['val'])}
        data |= {'test': datasets.ImageFolder(root=self.TVT_PATH+"\\test", transform=self.image_transforms['test'])} if include_test else {}
        return(data)
        
    def get_iterators(self, data, bsize = 32, include_test = True):
        iterators = [torch.utils.data.DataLoader(data['train'], batch_size=bsize, shuffle=True), 
                    torch.utils.data.DataLoader(data['val'], batch_size=bsize, shuffle=True)]
        iterators += [torch.utils.data.DataLoader(data['test'], batch_size=bsize, shuffle=True)] if include_test else []
        return (iterators)