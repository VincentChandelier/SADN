# SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation 
  This repository contains the code for reproducing the results with trained models, in the following paper:
  Official code o f paper：SADN: Learned Light Field Image Compression with Spatial-Angular Decorrelation.[arxiv](https://arxiv.org/abs/2202.10837), ICASSP2020
  
Kedeng Tong, Xin Jin, Chen Wang, Fan Jiang 

# Paper Summury:
Light field image becomes one of the most promising media types for immersive video applications. In this paper, we propose a novel end-to-end spatial-angular-decorrelated network (SADN) for high-efficiency light field image compression. Different from the existing methods that exploit either spatial or angular consistency in the light field image, SADN decouples the angular and spatial information by dilation convolution and stride convolution in spatialangular interaction, and performs feature fusion to compress spatial and angular information jointly. To train a stable and robust algorithm, a large-scale dataset consisting of 7549 light field images is proposed and built. The proposed method provides 2.137 times and 2.849 times higher compression efficiency relative to H.266/VVC and H.265/HEVC inter coding, respectively. It also outperforms the end-to-end image compression networks by an average of 79.6% bitrate saving with much higher subjective quality and light field consistency. 
# Notes
This implementations are not original codes of our ICASSP2022 paper, because original code is based on Tensorflow 2.4.0 and many features have been removed. This repo is a re-implementation, but the core codes are almost the same and results are also consistent with original results. 

If you think it is useful for your reseach, please cite our ICASSP2020 paper. Our original RD data in the paper is contained in the folder RDdata/.

