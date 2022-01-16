import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from data_loading import BasicDataset
import segmentation_models_pytorch as smp
from utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    '''
    img1 = img.detach().cpu().numpy()
    img1 = img1[0][0]
    img1 = Image.fromarray((img1 * 255).astype(np.uint8))
    img1.save('input1.jpg')
    '''
    with torch.no_grad():
        output = net(img)

        #if net.n_classes > 1:
        #print(output.size())
        probs = output[0] # F.softmax(output, dim=1)[0]
        #print(probs)
        #else:
        #    probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        #print(full_mask)
        #print(full_mask.size())
#    if net.n_classes == 1:
#        return (full_mask > out_threshold).numpy()
#    else:
#        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        #print(full_mask.argmax(dim=0))
        #print(F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy())
        #print(F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy().shape)
        return F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
        #return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


class EnsembledModel(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5, model6, model7):
        super(EnsembledModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.conv0 = nn.Conv2d(7, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv1 = nn.Conv2d(7, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)
        pred4 = self.model4(x)
        pred5 = self.model5(x)
        pred6 = self.model6(x)
        pred7 = self.model7(x)
        #pred = (pred1 + pred2 + pred3) / 3
        pred_0 = torch.cat((pred1[:, 0:1, :, :], pred2[:, 0:1, :, :], pred3[:, 0:1, :, :], pred4[:, 0:1, :, :], pred5[:, 0:1, :, :], pred6[:, 0:1, :, :], pred7[:, 0:1, :, :]), 1)
        pred_1 = torch.cat((pred1[:, 1:2, :, :], pred2[:, 1:2, :, :], pred3[:, 1:2, :, :], pred4[:, 1:2, :, :], pred5[:, 1:2, :, :], pred6[:, 1:2, :, :], pred7[:, 1:2, :, :]), 1)
        pred_0 = self.conv0(pred_0)
        pred_1 = self.conv1(pred_1)
        #pred_0 = pred_0 * pred_0
        #pred_1 = pred_1 * pred_1
        #pred_0 = torch.mean(pred_0, dim = 1, keepdim = True)
        #pred_1 = torch.mean(pred_1, dim = 1, keepdim = True)
        pred = torch.cat((pred_0, pred_1), 1)
        pred = self.softmax(pred)
        return pred


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)


    model_ft1 = smp.Unet(
    encoder_name="inceptionv4", #"resnet34",                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                                              # model output channels (number of classes in your dataset)
    activation="softmax",
    )

    model_ft2 = smp.Unet(
    encoder_name="timm-resnest101e", #"resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    activation="softmax"
    )

    model_ft3 = smp.MAnet(
    encoder_name="resnet152", #"resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    activation="softmax"
    )

    model_ft4 = smp.UnetPlusPlus(
        encoder_name="resnet152", #"resnet34",          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                                              # model output channels (number of classes in your dataset)
        activation="softmax",
    )

    model_ft5 = smp.Unet(
        encoder_name="mobilenet_v2", #"resnet34",          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                                              # model output channels (number of classes in your dataset)
        activation="softmax",
    )

    model_ft6 = smp.Unet(
        encoder_name="resnet50", #"resnet34",          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                                              # model output channels (number of classes in your dataset)
        activation="softmax",
    )

    model_ft7 = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2", #"resnet34",          # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                                              # model output channels (number of classes in your dataset)
        activation="softmax",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model_ft = EnsembledModel(model_ft1, model_ft2, model_ft3, model_ft4, model_ft5, model_ft6, model_ft7)
    model_ft.to(device=device)
    model_ft.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=model_ft,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        #print(mask.ndim)
        #print(mask.shape[0])

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
