import torch
import os
import argparse
import time

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import utils

from network import normal
from network import slim
from dataset import IDataset
from vgg import Vgg16


def trainer(args):
    print_args(args)

    # if cuda is available, GPU will be used. On the contrary, CPU will be used.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imgs_path = args.dataset
    style_image = args.style_image

    size = (args.image_width, args.image_height)

    # define network
    if args.model_mode != "slim":
        image_transformer = normal.ImageTransformNet(size).to(device)
    else:
        image_transformer = slim.ImageTransformNet(size).to(device)

    # set optimizer
    optimizer = Adam(image_transformer.parameters(), args.learning_rate)

    # define loss function
    loss_mse = nn.MSELoss()

    # load vgg network
    vgg = Vgg16(args.VGG_path).to(device)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Resize(size),                # scale shortest side to image_size
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = IDataset(imgs_path, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(style_image)
    style = style_transform(style)
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).to(device)
    style_name = os.path.split(style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    print("Start training. . .")
    best_style_loss = 1e9
    best_content_loss = 1e9
    for e in range(args.epoch):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, x in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).to(device)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.style_weight * style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = args.content_weight * loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = args.tv_weight * (diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if (batch_num + 1) % 50 == 0:
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  " \
                         "agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} "\
                    .format(
                            time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                            aggregate_style_loss/(batch_num+1.0),
                            aggregate_content_loss/(batch_num+1.0),
                            aggregate_tv_loss/(batch_num+1.0), style_loss.item(),
                            content_loss.item(), tv_loss.item())
                print(status)

        # save model
        image_transformer.eval()

        model_folder = args.model_folder
        if not os.path.exists("models/{}" .format(model_folder)):
            os.makedirs("models/{}" .format(model_folder))
        num = len(train_dataset) / args.batch_size

        aggregate_style_loss /= num
        aggregate_content_loss /= num
        aggregate_tv_loss /= num

        filename = "models/{}/{}_epoch={}_style={:.4f}_content={:.4f}_tv={:.4f}.pth" .format(
            model_folder, style_name, e + 1, aggregate_style_loss,
            aggregate_content_loss, aggregate_tv_loss)

        if aggregate_style_loss < best_style_loss or aggregate_content_loss < best_content_loss:
            torch.save(image_transformer.state_dict(), filename)

        if aggregate_style_loss < best_style_loss:
            best_style_loss = aggregate_style_loss
        if aggregate_content_loss < best_content_loss:
            best_content_loss = aggregate_content_loss

def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')

    parser.add_argument("--model_mode", type=str, default="slim",
                          help="select the pattern of model, normal or slim")

    parser.add_argument("--dataset", type=str, default="./dataset", help="path to a dataset")
    parser.add_argument("--style_image", type=str, default="./style_imgs/udnie.jpg",
                          help="path to a style image to train")

    parser.add_argument("--VGG_path", type=str, default=None, help="VGG save path")

    parser.add_argument("--image_height", type=int, default=256,
                        help="image's height, which will be fed into model")
    parser.add_argument("--image_width", type=int, default=256,
                        help="image's width, which will be fed into model")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                          help="a hyperparameter which determines to what extent newly "
                               "acquired information overrides old information")
    parser.add_argument("--batch_size", type=int, default=16,
                          help="the number of training examples in one forward/backward pass")
    parser.add_argument("--epoch", type=int, default=10,
                        help="one cycle through the full training dataset")

    parser.add_argument("--style_weight", type=float, default=1e5,
                        help="hyperparameter, control style learning")
    parser.add_argument("--content_weight", type=float, default=1e0,
                        help="hyperparameter, bring into some correspondence with content image")
    parser.add_argument("--tv_weight", type=float, default=1e-7,
                        help="make result image smooth")

    parser.add_argument("--model_folder", type=str, default="./models/style-transfer",
                        help="path to save model")

    args = parser.parse_args()

    trainer(args)


def print_args(args):
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    print("\n" + "* " * 30)
    print()

    print("model mode: {}" .format(args.model_mode))
    print("dataset path: {}" .format(args.dataset))
    print("style image path: {}" .format(args.style_image))
    print("VGG path: {}" .format(args.VGG_path))
    print("image size: ({}, {})" .format(args.image_height, args.image_width))
    print("learning rate: {}" .format(args.learning_rate))
    print("batch size: {}" .format(args.batch_size))
    print("epoch: {}" .format(args.epoch))
    print("style weight: {}" .format(args.style_weight))
    print("content weight: {}" .format(args.content_weight))
    print("tv weight: {}" .format(args.tv_weight))
    print("model folder: {}" .format(args.model_folder))

    print("\n" + "* " * 30 + "\n")

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    main()
