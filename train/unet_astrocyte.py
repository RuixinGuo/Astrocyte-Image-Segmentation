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

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = smp.Unet(
    encoder_name="vgg19", #"resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation="sigmoid"
)

#print(model)
model.eval()

transform = transforms.Compose([transforms.Resize((512, 512)),
                                 transforms.ToTensor()])

#dataset = datasets.ImageFolder('./data', transform=transform)
dataset = datasets.ImageFolder('./data/', transform=transform)
test_loader = DataLoader(dataset, batch_size=1)

data_iter = iter(test_loader)
test_x, test_label = data_iter.next()
test_x, test_label = data_iter.next()
test_x, test_label = data_iter.next()
test_x, test_label = data_iter.next()
#print(test_x[25:26, :, :, :])
#print(test_label[0])
#print(test_x.size())
#print(model.encoder.layer1[0].conv1.weight)

img = test_x[0]
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
plt.imshow(img)
plt.savefig('input_img.png')
#plt.show()

#model.eval()
print(test_x)
with torch.no_grad():
    output = model(test_x)
print(output[0][0])
print(output.size())

out_threshold=0.5
#full_mask = torch.sigmoid(output)[0]
#print(full_mask[0][0])
#print(full_mask[0][0].size())

#plt.imshow((output[0][0] > out_threshold).numpy(), cmap = "gray")
#plt.show()
plt.imshow(output[0][0].numpy(), cmap = "gray")
plt.savefig('output_img.png')
plt.show()

