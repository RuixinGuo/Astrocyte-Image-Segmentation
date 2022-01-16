import segmentation_models_pytorch as smp
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch import optim
import matplotlib.pyplot as plt
import matplotlib
import random
from tqdm import tqdm
from pathlib import Path
from data_loading import BasicDataset
from evaluate import evaluate
from dice_score import dice_loss

#def train_val_dataset(dataset, val_split=0.25):
#    shuffled_dataset = random.shuffle(list(range(len(dataset))))
#    train_idx, val_idx = train_test_split(shuffled_dataset, test_size=val_split)
#    datasets = {}
#    datasets['train'] = Subset(dataset, train_idx)
#    datasets['val'] = Subset(dataset, val_idx)
#    return datasets

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#print(device)
val_percent = 0.2
batch_size = 10
learning_rate = 0.0001
epochs = 50
save_checkpoint = True

#print(torch.cuda.device_count())

'''
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''

model_ft = smp.MAnet(
    encoder_name="mobilenet_v2", #"resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    activation="softmax",
)
print(model_ft)
model_ft.to(device=device, dtype=torch.float32)

ct = 0
for name, param in model_ft.named_parameters():
    ct += 1
    #param.requires_grad = False  #Freeze all layers
    print(str(ct) + ' ' + name) #Show the layers that contains parameters
    #if ct >= 25 and ct <= 44:
    if ct >= 1 and ct <= 1000:
        param.requires_grad = True  # Fine tune these layers
    else:
        param.requires_grad = False  # Freeze these layers
#print(ct)

#transform = transforms.Compose([transforms.Resize((512, 512)),
#                                 transforms.ToTensor()])

dir_img = Path('./data2/cells9')
dir_mask = Path('./data2/mask_cells9')
dir_checkpoint = Path('./checkpoints/')
img_scale = 1.0
dataset = BasicDataset(dir_img, dir_mask, img_scale)
#torch.set_printoptions(profile="full")
#print(dataset[0])
#print(len(dataset))
#print(dataset['mask'])

#dataset = datasets.ImageFolder('./data/', transform=transform)
#print(len(dataset))

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

#img = next(iter(train_loader))
#print(img)
#plt.imshow(img['image'][0][0])
#plt.show()
#plt.imshow(img['mask'][0])
#plt.show()

#print(train_set)
#print(len(train_set))
#print(img['image'].size())
#masks_pred = model_ft(img['image'].to(device))

#optimizer = optim.RMSprop(model_ft.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
optimizer = optim.RMSprop(model_ft.parameters(), lr=learning_rate, weight_decay=1e-7, momentum=0.9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
criterion = nn.CrossEntropyLoss()
global_step = 0


for epoch in range(epochs):
    model_ft.train()
    epoch_loss = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']
            #true_masks = torch.unsqueeze(true_masks, dim=1)
        
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=False):
                #print(images.size())
                masks_pred = model_ft(images)
                #masks_pred = torch.squeeze(masks_pred, dim=1)
                #print(masks_pred)
                #print(masks_pred.size())
                #print(true_masks.size())
                #loss = criterion(masks_pred, true_masks) \
                #     + dice_loss(nn.functional.softmax(masks_pred, dim=1).float(),
                #            nn.functional.one_hot(true_masks, num_classes = 1).permute(0, 3, 1, 2).float(),
                #            multiclass=True)

                #print(nn.functional.one_hot(true_masks, num_classes = 1).size())

                #loss = dice_loss(nn.functional.softmax(masks_pred, dim=1).float(),
                loss = dice_loss(masks_pred.float(),
                            nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2).float(),
                            multiclass=True)
                #print(nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2).float()[0][0])
                #print(nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2).float()[0][1])
                
                '''
                print(nn.functional.softmax(masks_pred, dim=1)[0][0].cpu().detach().numpy())
                print(nn.functional.softmax(masks_pred, dim=1)[0][1].cpu().detach().numpy())
                plt.imshow(nn.functional.softmax(masks_pred, dim=1)[0][1].cpu().detach().numpy())
                plt.show()
                print(nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2)[0][0].cpu().detach().numpy())
                print(nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2)[0][1].cpu().detach().numpy())
                plt.imshow(nn.functional.one_hot(true_masks, num_classes = 2).permute(0, 3, 1, 2)[0][1].cpu().detach().numpy())
                plt.show()
                '''

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            #division_step = (n_train // (10 * batch_size))
            division_step = (n_train // (5 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score = evaluate(model_ft, val_loader, device)
                    print("val_score: " + str(val_score))

    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save(model_ft.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        #logging.info(f'Checkpoint {epoch + 1} saved!')
        


'''
dataloaders = {x:DataLoader(datasets[x], batch_size=1, shuffle = True) for x in ['train','val']}
img, label = next(iter(dataloaders['train']))
'''
'''
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
'''
