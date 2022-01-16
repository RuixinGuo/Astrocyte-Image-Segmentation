import segmentation_models_pytorch as smp
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

transform = transforms.Compose([transforms.Resize((512, 512)),
                                 ])
#012_5, 015_3, 017_2
img = Image.open('../train/data2/cells9/a009_2.jpg')
img = transform(img)
img.save('input1.jpg')

img = Image.open('../train/data2/mask_cells9/a009_2.tif')
img = transform(img)
img.save('true_mask1.tif')
