
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info
from compressai.layers.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    conv1x1,
)
from compressai.entropy_models import GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import ste_round
from compressai.models.utils import update_registered_buffers

from compressai.layers import MaskedConv2d
from compressai.models.priors import CompressionModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
eps = 1e-9

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class InterNet(nn.Module):
    def __init__(self, channels, angRes, n_blocks, analysis=True, *args, **kwargs):
        super(InterNet, self).__init__(*args, **kwargs)
        # Feature Extraction
        if analysis:
            self.AngFE = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False))
            self.SpaFE = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False))
        else:
            self.AngFE = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False))
            self.SpaFE = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False))
        # Spatial-Angular Interaction
        self.CascadeInterBlock = CascadeInterBlock(angRes, n_blocks, channels)
        # Fusion and Reconstruction
        self.BottleNeck = BottleNeck(angRes, n_blocks, channels)

    def forward(self, x):
        xa = self.AngFE(x)
        xs = self.SpaFE(x)
        buffer_a, buffer_s = self.CascadeInterBlock(xa, xs)
        buffer_out = self.BottleNeck(buffer_a, buffer_s) + xs
        return buffer_out


class make_chains(nn.Module):
    def __init__(self, angRes, channels):
        super(make_chains, self).__init__()

        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.AngConvSq = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.SpaConvSq = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))
        buffer_spa1 = xs
        buffer_spa2 = self.Ang2Spa(xa)
        buffer_a = torch.cat((buffer_ang1, buffer_ang2), 1)
        buffer_s = torch.cat((buffer_spa1, buffer_spa2), 1)
        out_a = self.ReLU(self.AngConvSq(buffer_a)) + xa
        out_s = self.ReLU(self.SpaConvSq(buffer_s)) + xs
        return out_a, out_s


class CascadeInterBlock(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(CascadeInterBlock, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(make_chains(angRes, channels))
        self.body = nn.Sequential(*body)
    def forward(self, buffer_a, buffer_s):
        out_a = []
        out_s = []
        for i in range(self.n_blocks):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck, self).__init__()

        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                    padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, self.Ang2Spa(xa)), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SADN(CompressionModel):
    """
    Args:
        N (int): Number of channels
    """

    def __init__(self, N=48, M=48, angRes=13, n_blocks=1, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            InterNet(N, angRes, n_blocks),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
            conv3x3(N, M),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, M),
            nn.LeakyReLU(inplace=True),
        )

        self.g_s = nn.Sequential(
            conv3x3(M, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            InterNet(N, angRes, n_blocks, analysis=False),
            conv3x3(N, 3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 2, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, 640, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, M * 2, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.quantizer = Quantizer()

        self.lmbda = [0.0001, 0.00015, 0.0003, 0.0006, 0.001, 0.003]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000,  1.2247,  1.7321,  2.4495,  3.1623,  5.4772]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noise=False, stage=3, s=1):
        if stage > 1:
            if s != 0:
                scale = self.Gain[s]
            else:
                s = 0
                scale = self.Gain[s].detach()
        else:
            scale = self.Gain[0].detach()

        rescale = 1.0 / scale.clone().detach()
        if noise:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(y*scale, "noise" if self.training else "dequantize") * rescale
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y*scale - means_hat*scale, scales_hat*scale)
            x_hat = self.g_s(y_hat)
        else:
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            params = self.h_s(z_hat)
            kernel_size = 5  # context prediction kernel size
            padding = (kernel_size - 1) // 2
            y_hat = F.pad(y, (padding, padding, padding, padding))
            y_hat, y_likelihoods = self._stequantization(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)

            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _stequantization(self, y_hat, params, height, width, kernel_size, padding, scale, rescale):
        y_likelihoods = torch.zeros([y_hat.size(0), y_hat.size(1), height, width]).to(scale.device)
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size].clone()
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_crop = y_crop[:, :, padding, padding]
                _, y_likelihoods[:, :, h: h + 1, w: w + 1] = self.gaussian_conditional(
                    ((y_crop - means_hat) * scale).unsqueeze(2).unsqueeze(3),
                    (scales_hat * scale).unsqueeze(2).unsqueeze(3))
                y_q = self.quantizer.quantize((y_crop - means_hat.detach()) * scale,
                                        "ste") * rescale + means_hat.detach()
                y_hat[:, :, h + padding, w + padding] = y_q
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat, y_likelihoods

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, s, inputscale=0):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
                scale,
                rescale,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding, scale, rescale,):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize((y_crop - means_hat)* scale, "symbols")
                y_hat[:, :, h + padding, w + padding] = (y_q) * rescale + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
                scale,
                rescale,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding, scale, rescale
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat* scale)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv)*rescale + means_hat

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv

if __name__ == "__main__":
    model = SADN(N=48, M=48, angRes=13, n_blocks=1)
    input = torch.Tensor(1, 3, 832, 832)
    print(model)
    out = model(input)
    flops, params = get_model_complexity_info(model, (3, 832, 832), as_strings=True, print_per_layer_stat=True, flops_units=" ", param_units=" ")
    print('flops: ', flops, 'params: ', params)
    # flops, params = profile(model, (input,))
    # print('flops: ', flops, 'params: ', params)
