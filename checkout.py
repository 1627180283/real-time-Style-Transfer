import argparse
import cv2
import torch

import numpy as np
import onnxruntime as rt

from network import normal
from network import slim


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_onnx", type=bool, default=False,
                        help="clear test model is onnx or not")

    parser.add_argument("--model_mode", type=str, default="slim",
                        help="select the pattern of model, normal or slim")

    parser.add_argument("--image_height", type=int, default=512,
                        help="image's height, which will be fed into model")
    parser.add_argument("--image_width", type=int, default=512,
                        help="image's width, which will be fed into model")
    parser.add_argument("--image_path", type=str, default="./content_imgs/maine.jpg",
                        help="path to a content image")

    parser.add_argument("--model_path", type=str, required=True,
                        help="path to model file")

    parser.add_argument("--save_path", type=str, default=None,
                        help="if this argument is None, the result will not be saved"
                             "if you want to save the result to the specified location, "
                             "fill in the path")

    args = parser.parse_args()

    process(args)


def process(args):
    size = (args.image_width, args.image_height)
    img = load_image(args.image_path, size)

    if args.is_onnx:
        out = onnx_infer(args.model_path, img)
    else:
        out = pytorch_infer(args.model_mode, args.model_path, img)

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imshow("out", out)
    cv2.waitKey(0)

    if args.save_path is not None:
        cv2.imwrite(args.save_path, out)


def load_image(path, size):
    # define scale factor
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # load image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    for i in range(3):
        img[i] -= mean[i]
        img[i] /= std[i]

    return img


def pytorch_infer(mode, path, img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = img.shape[1:]

    # define network
    if mode != "slim":
        model = normal.ImageTransformNet(size).to(device)
    else:
        model = slim.ImageTransformNet(size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # ndarray to tensor
    img = torch.from_numpy(np.array([img])).float().to(device)

    # inference
    out = model(img)
    if torch.cuda.is_available():
        out = out.cpu()

    out = out.detach().numpy()
    out = np.squeeze(out)
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    out = ((out * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")

    return out


def onnx_infer(path, img):
    sess = rt.InferenceSession(path)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: img.astype(np.float32)})[0]

    return


if __name__ == '__main__':
    run()
