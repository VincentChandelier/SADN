# SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation 
  This repository contains the code for reproducing the results with trained models, in the following paper:
  Official code o f paper：[SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation.arxiv](https://arxiv.org/abs/2202.10837), ICASSP 2022
  
Kedeng Tong, Xin Jin, Chen Wang, Fan Jiang 

[Poster](https://drive.google.com/file/d/1HCDC7KsgLDEpkr0ZjK7UcliMVnwWK5v0/view?usp=sharing) is available.
[Video presentation](https://drive.google.com/file/d/18-lm0xE4gofmfLGLOyv5mW2XAPV87WaQ/view?usp=sharing) is available.

# Paper Summury:
Light field image becomes one of the most promising media types for immersive video applications. In this paper, we propose a novel end-to-end spatial-angular-decorrelated network (SADN) for high-efficiency light field image compression. Different from the existing methods that exploit either spatial or angular consistency in the light field image, SADN decouples the angular and spatial information by dilation convolution and stride convolution in spatialangular interaction, and performs feature fusion to compress spatial and angular information jointly. To train a stable and robust algorithm, a large-scale dataset consisting of 7549 light field images is proposed and built. The proposed method provides 2.137 times and 2.849 times higher compression efficiency relative to H.266/VVC and H.265/HEVC inter coding, respectively. It also outperforms the end-to-end image compression networks by an average of 79.6% bitrate saving with much higher subjective quality and light field consistency. 

# Available data
 Data |  Link                                                                                              |
| ----|---------------------------------------------------------------------------------------------------|
| PINet | [PINet](https://pan.baidu.com/s/1RqEZFrR1Kt4BFCd1hX6iSA?pwd=12t5)    |
| Packaged PINet | [Packaged PINet](https://pan.baidu.com/s/1TDHZnBAF5K1kR51awYsV2A?pwd=8ofs)    |
| PINet central SAIs and registered depth maps| [PINet central SAIs and registered depth maps](https://pan.baidu.com/s/1gIf1liNR47PtdDyUbrexHQ?pwd=mcr6)     |
| LFtoolbox0.4 | [LFtoolbox0.4](https://pan.baidu.com/s/1HdzENydi1WKvJ0jL6wS2TA?pwd=v0xj) |
| Training patches | [Training patches](https://pan.baidu.com/s/1wPLjhdjUY0A8xdLAEdqzEA?pwd=0uow)    |
| Full-resolution test images | [Full-resolution test images](https://pan.baidu.com/s/14LdMV7ybwEiSauR4DlfiQA?pwd=gf66) |
| Variable Rate models   | [SADN+QVRF](https://github.com/VincentChandelier/SADN-QVRF)|

# PINet： A large scale image dataset of a hand-held light Field camera is proposed
Description: We propose a new LF image database called “PINet” inheriting the hierarchical structure from WordNet. It 
consists of 7549 LIs captured by Lytro Illum, which is much larger than the existing databases. The images are manually annotated to 178 categories according to WordNet, such as cat, camel, bottle, fans, etc. The registered depth maps are also provided. Each image is generated by processing the raw LI from the camera by Light Field Toolbox v0.4 for demosaicing and devignetting. 

## Some central SAIs and registerd depth maps of LF images in the dataset. 

.![](https://github.com/VincentChandelier/SADN/blob/main/PINet/PINet.png)

[The central-SAIs and registered depth maps](https://cloud.tsinghua.edu.cn/d/d47ad68552ec408eac94/  ) are available for previewing.

The raw light field images supposed to be decoded by LFtoolbox are available to download. (https://pan.baidu.com/s/1ebpt2K6F7-DOB42_Y2ZaDg Extraction code：d4l8)

The decoding code with LFtoolbox0.4 and related camera parameters is available to download. (https://pan.baidu.com/s/1DsTL2ftKBrnp-nKnJQkilQ  Extraction code：5nxt)

If you want to explore the PINet or to produce your own training data, please follow the readme in the folder. 

For testing, the light field raw images and macro images are provided. (https://pan.baidu.com/s/1JLv5oAax8j9xrzqFEY5__Q 
Extraction code：a832)

The training patches I used for training is available (https://pan.baidu.com/s/1wPLjhdjUY0A8xdLAEdqzEA?pwd=0uow access code：0uow)

# SADN
## Network
.![](https://github.com/VincentChandelier/SADN/blob/main/RDdata/Network.png)


## Usage
This code is based on the [CompressAI](https://github.com/InterDigitalInc/CompressAI).
### Installation
  
   ```
   conda create -n SADN python=3.8
   conda activate SADN
   pip install compressai==1.1.5
   pip install thop
   pip install ptflops
   pip install tensorboardX
   ```
### Train Usage
First stage with noise quantization
   ```
   cd Code
   python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 70 -lr 1e-4 -n 8  --lambda 3e-3 --batch-size 8  --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --gpu-id  2,3 --savepath   ./checkpoint0003   
   ```
Second stage with straight-through-estimation quantization by loading the lastest checkpoint of stage 1
```
   python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 10 -lr 1e-4 -n 8 --lambda 3e-3 --batch-size 8  --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --gpu-id 2,3 --savepath   ./checkpoint0003ste --pretrained  --checkpoint ./checkpoint0003/checkpoint_last_69.pth.tar  --ste 1
```
   The training patches I used for training is available (https://pan.baidu.com/s/1wPLjhdjUY0A8xdLAEdqzEA?pwd=0uow access code：0uow)

### Update the entropy model
```
python updata.py checkpoint_path -n checkpoint
```
### Test 
Since the full test images are too large, I only upload a patch of the test image in Code/dataset/test. I re-trained the re-implementation algorithm in PyTorch with lambda=0.003, and the checkpoint is saved as the Code/checkpoint.pth.tar. 

```
python Inference.py --dataset/test --output_path Result_dir -p checkpoint.pth.tar
```

## Evaluation Results
.![](https://github.com/VincentChandelier/SADN/blob/main/RDdata/RD.png)

# Notes
This implementation is not the original code of our ICASSP2022 paper, because the original code is based on Tensorflow 2.4.0 wihch many features have been removed in the latest tensorflow version. This repo is a re-implementation, but the core codes are the same. 
## original RD data
Our original RD data in the paper is contained in the folder ./RDdata/.
## Retrained RD data
Since our original proposed method is trained on Nvidia V100, Tensorflow 2.4, we retrained our algorithm on Nvidia RTX3090, PyTorch 2.3 using lambda belonging to {0.0001, 0.00015, 0.0003, 0.0006, 0.001, 0.003} for comparison.  
The retrained results are obtained by variable rate model [SADN+QVRF](https://github.com/VincentChandelier/SADN-QVRF)  
The retrained results are saved in [./RDdata/SADN_Pytorch_RTX3090.txt](https://github.com/VincentChandelier/SADN/blob/main/RDdata/SADN_Pytorch_RTX3090.txt).

The Retrained Evaluation Results
.![](https://github.com/VincentChandelier/SADN/blob/main/RDdata/RTX3090_Result.png)

If you have any problem, please contact me: tkd20@mails.tsinghua.edu.cn

If you think it is useful for your reseach, please cite our ICASSP2022 paper. 
```
@inproceedings{tong2022sadn,
  title={SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation},
  author={Tong, Kedeng and Jin, Xin and Wang, Chen and Jiang, Fan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1870--1874},
  year={2022},
  organization={IEEE}
}
```

## EPLF dataset
[EPLF](https://www.epfl.ch/labs/mmspg/downloads/epfl-light-field-image-dataset/)
```
@inproceedings{rerabek2016new,
  title={New light field image dataset},
  author={Rerabek, Martin and Ebrahimi, Touradj},
  booktitle={8th International Conference on Quality of Multimedia Experience (QoMEX)},
  number={CONF},
  year={2016}
}
```
## ICME 12 LF dataset
```
@article{rerabek2016icme,
  title={Icme 2016 grand challenge: Light-field image compression},
  author={Rerabek, Martin and Bruylants, Tim and Ebrahimi, Touradj and Pereira, Fernando and Schelkens, Peter},
  journal={Call for proposals and evaluation procedure},
  year={2016}
}
```
