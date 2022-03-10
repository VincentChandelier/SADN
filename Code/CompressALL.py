
"""
Update the CDFs parameters of a trained model.

To be called on a model checkpoint after training. This will update the internal
CDFs related buffers required for entropy coding.

And I embeding the test algorithm into the same code for better usage
"""
import hashlib
import argparse
import json
import math
import os
import sys
import time
import csv

from pathlib import Path
from typing import Dict

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



from Network import LFContext

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

def sha256_file(filepath: Path, len_hash_prefix: int = 8) -> str:
    # from pytorch github repo
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        while True:
            buf = f.read(8192)
            if len(buf) == 0:
                break
            sha256.update(buf)
    digest = sha256.hexdigest()

    return digest[:len_hash_prefix]


def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
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

@torch.no_grad()
def inference(model, x, f):
    x = x.unsqueeze(0)

    imgpath = f.split('/')
    imgpath[-2] = 'Recon'
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))
# ########original padding
#     h, w = x.size(2), x.size(3)
#     p = 64  # maximum 6 strides of 2
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
####### ReplicationPad2d
    h, w = x.size(2), x.size(3)
    p = 832  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ReplicationPad2d(padding=(padding_left, padding_right, padding_top, padding_bottom))
    x_padded = pad(x)

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    # save_image_tensor2cv2(out_dec["x_hat"], imgPath)
    torchvision.utils.save_image(out_dec["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_dec["x_hat"])
    MSSSIM = ms_ssim(x, out_dec["x_hat"], data_range=1.0).item()
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, F.mse_loss(x, out_dec["x_hat"]).item()*255**2, MSSSIM,
               -10 * math.log10(1 - MSSSIM), psnr(x, out_dec["x_hat"]), enc_time, dec_time]
        write = csv.writer(f)
        write.writerow(row)
    print('bpp:{}, PSNR: {}, SSIM: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, MSSSIM, enc_time, dec_time))
    return {
        "psnr": PSNR,
        "ms-ssim": MSSSIM,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, f):
    x = x.unsqueeze(0)

    imgpath = f.split('/')
    imgpath[-2] = 'Recon'
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/EstimateResult.csv'
    print('decoding img: {}'.format(f))

    ####### ReplicationPad2d
    h, w = x.size(2), x.size(3)
    p = 832  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ReplicationPad2d(padding=(padding_left, padding_right, padding_top, padding_bottom))
    x_padded = pad(x)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    # save_image_tensor2cv2(out_dec["x_hat"], imgPath)
    torchvision.utils.save_image(out_net["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_net["x_hat"])
    MSSSIM = ms_ssim(x, out_net["x_hat"], data_range=1.0).item()
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, F.mse_loss(x, out_net["x_hat"]).item() * 255 ** 2,
               MSSSIM,
               -10 * math.log10(1 - MSSSIM), psnr(x, out_net["x_hat"]), elapsed_time / 2.0, elapsed_time / 2.0]
        write = csv.writer(f)
        write.writerow(row)
    print('bpp:{}, PSNR: {}, SSIM: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, MSSSIM, elapsed_time / 2.0,
                                                                                    elapsed_time / 2.0))
    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }



def eval_model(model, filepaths, entropy_estimation=False, half=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    imgdir = filepaths[0].split('/')
    imgdir[-2] = 'Recon'
    imgDir = '/'.join(imgdir[:-1])
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)
    if not entropy_estimation:
        csvfile = imgDir + '/result.csv'
    else:
        csvfile = imgDir + '/EstimateResult.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'mse', 'ssim', 'logssim', 'psnr', 'enc_time', 'dec_time']
        write = csv.writer(f)
        write.writerow(row)
    for f in filepaths:

        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f)
        else:
            rv = inference_entropy_estimation(model, x, f)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

description = """
Export a trained model to a new checkpoint with an updated CDFs parameters and a
hash prefix, so that it can be loaded later via `load_state_dict_from_url`.
""".strip()



def setup_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )

    parser.add_argument(
        "--no-update",
        action="store_true",
        default=False,
        help="Do not update the model CDFs parameters.",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset path"
    )
    parser.add_argument(
         "-f", "--filepath", type=str, default="checkpoint_best_loss.pth.tar", help="checkpoint path"
    )
    parser.add_argument(
        "--channels", type=int, default=128,
        help="Setting the network channels.")
    parser.add_argument(
        "--angRes", type=int, default=13,
        help="Setting e angular resolution of the MacPi.")
    parser.add_argument(
        "--n_blocks", type=int, default=2,
        help="Setting the mber of the inter-blocks.")
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Setting the number of the layers of inter-blocks.")
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
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    filepath = Path(args.filepath).resolve()
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)
    state_dict = load_checkpoint(filepath)

    model_cls = LFContext(N=args.channels, angRes=args.angRes, n_blocks=args.n_blocks, n_layers=args.n_layers)
    model = model_cls.from_state_dict(state_dict)

    if not args.no_update:
        model.update(force=True)

    results = defaultdict(list)
    model.eval()
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")
    metrics = eval_model(model, filepaths, args.entropy_estimation, args.half)
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
