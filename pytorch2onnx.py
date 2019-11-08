import argparse
import torch

from torch import onnx

from network import normal
from network import slim


def convert(args):
    # if cuda is available, GPU will be used. On the contrary, CPU will be used.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = (args.image_width, args.image_height)

    # define network
    if args.model_mode != "slim":
        model = normal.ImageTransformNet(size).to(device)
    else:
        model = slim.ImageTransformNet(size).to(device)

    # load weights
    model.load_state_dict(torch.load(args.pth_file))
    model.eval()

    inputs = torch.Tensor(1, 3, args.image_width, args.image_height).to(device)

    onnx.export(model, inputs, args.onnx_name, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_mode", type=str, default="slim",
                        help="select the pattern of model, normal or slim")

    parser.add_argument("--image_height", type=int, default=512,
                        help="image's height, which will be fed into model")
    parser.add_argument("--image_width", type=int, default=512,
                        help="image's width, which will be fed into model")

    parser.add_argument("--pth_file", type=str, required=True, help="path to a pth file")
    parser.add_argument("--onnx_name", type=str, default="./onnx/model.onnx")

    args = parser.parse_args()

    convert(args)
