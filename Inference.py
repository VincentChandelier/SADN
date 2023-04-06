
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time
import csv
# import cv2



from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision

import compressai

from compressai.zoo import load_state_dict
from compressai.zoo import models as pretrained_models
from compressai.zoo.image import model_architectures as architectures
from scipy.io import savemat
from utils import *

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)



def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f).replace('\\', '/')
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

def img_window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def img_window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

@torch.no_grad()
def inference(model, x, f, outputpath, patch, s, factor, factormode):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath).replace('\\', '/')
    csvfile = '/'.join(imgpath[:-1]).replace('\\', '/') + '/'+outputpath+'_result.csv'
    print('decoding img: {}'.format(f))
########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()
    x_padded = img_window_partition(x_padded, p)
    Rec = torch.zeros_like(x_padded)
    enc_time = 0
    dec_time = 0
    bpp = 0
    for i in range(x_padded.size(0)):
        start = time.time()
        out_enc = model.compress(x_padded[i:i+1, :, :, :], s, factor)
        enc_time += time.time() - start

        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s, factor)
        dec_time += time.time() - start
        for k in out_enc["strings"]:
            for j in k:
                bpp += len(j)
        Rec[i:i+1, :, :, :] = out_dec["x_hat"]

    Rec = img_window_reverse(Rec, p, height, width)  # 反变换(N,C,Hpatch,Wpatch)->（1,C,H,W)
    Rec = F.pad(
        Rec, (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)

    bpp *= 8.0 / num_pixels  # 计算bpp

    torchvision.utils.save_image(Rec, imgPath, nrow=1)
    PSNR = psnr(x, Rec)
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, F.mse_loss(x, Rec).item()*255**2,
               PSNR,  ms_ssim(x, Rec, data_range=1.0).item(), enc_time, dec_time]
        write = csv.writer(f)
        write.writerow(row)
    print('bpp:{}, PSNR: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, enc_time, dec_time))
    return {
        "psnr": PSNR,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_entropy_estimation(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath).replace('\\', '/')
    csvfile = '/'.join(imgpath[:-1]).replace('\\', '/') + '/'+outputpath+'_result.csv'
    print('decoding img: {}'.format(f))
    ########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()
    x_padded = img_window_partition(x_padded, p)
    B, _, _, _ = x_padded.size()
    Rec = torch.zeros_like(x_padded)
    bpp = 0
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    start = time.time()
    out_net = model.forward(x_padded)
    # for i in range(B):
    #     out_net = model.forward(x_padded[i:i + 1, :, :, :])
    #     Rec[i:i + 1, :, :, :] = out_net["x_hat"]
    #     bpp += sum(
    #         (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
    #         for likelihoods in out_net["likelihoods"].values()
    #     )

    elapsed_time = time.time() - start
    Rec = img_window_reverse(Rec, p, height, width)  # 反变换(N,C,Hpatch,Wpatch)->（1,C,H,W)
    Rec = F.pad(
        Rec, (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    torchvision.utils.save_image(Rec, imgPath, nrow=1)
    PSNR = psnr(x, Rec)
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp.item() * num_pixels, num_pixels, bpp.item(), F.mse_loss(x, Rec).item() * 255 ** 2,
            PSNR, ms_ssim(x, out_net["x_hat"], data_range=1.0).item(), elapsed_time / 2.0, elapsed_time / 2.0]
        write = csv.writer(f)
        write.writerow(row)
    return {
        "psnr": PSNR,
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, entropy_estimation=False, half=False, outputpath='Recon', patch=832, s=2, factor=0, factormode=False):
    device = next(model.parameters()).device
    print("variable rate s:{}".format(s))
    metrics = defaultdict(float)
    imgdir = filepaths[0].split('/')
    imgdir[-2] = outputpath
    imgDir = '/'.join(imgdir[:-1]).replace('\\', '/')
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)
    csvfile = imgDir + '/'+outputpath+'_result.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'mse', 'psnr(dB)', 'ms-ssim', 'enc_time', 'dec_time']
        write = csv.writer(f)
        write.writerow(row)
    for f in filepaths:

        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f, outputpath, patch, s, factor, factormode)
        else:
            rv = inference_entropy_estimation(model, x, f, outputpath, patch)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument(
        "--output_path",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )

    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--factormode",
        type=int,
        default=0,
        help="weather to use factor mode",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.0,
        help="choose the value of factor",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=832,
        help="padding patch size (default: %(default)s)",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=2,
        help="select the scale factor",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    filepaths = sorted(filepaths)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)


    model = SADN(N=48, M=48, angRes=13, n_blocks=1)
    checkpoint = torch.load(args.paths, map_location="cpu")
    if "state_dict" in checkpoint:
        ckpt = checkpoint["state_dict"]
    else:
        ckpt = checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if
                       k in model_dict.keys() and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    # net.load_state_dict(model_dict)
    # state_dict = load_state_dict(torch.load(args.paths))
    model_cls = SADN(N=48, M=48, angRes=13, n_blocks=1)
    model = model_cls.from_state_dict(model_dict).eval()
    model.update(force=True)
    results = defaultdict(list)

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    if args.factormode and args.factor != 0:
        for factor in [0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                       2.9, 3.1, 3.3, 3.5, 3.7, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 6.0, 6.4, 6.8, 7.2,
                       7.6, 8.0, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 12.0, 12.4, 12.8, 13.2, 13.6, 14.0]:
            metrics = eval_model(model, filepaths, args.entropy_estimation, args.half,
                                 args.output_path + '_factor_' + str(factor),
                                 args.patch, s=2, factor=factor, factormode=args.factormode)
            for k, v in metrics.items():
                results[k].append(v)

    for s in range(model.levels):
        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.output_path + '_s_' + str(s),
                             args.patch, s, factor=0, factormode=0)
        for k, v in metrics.items():
            results[k].append(v)


    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
