import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from collections import OrderedDict




# ============================================================
#                SincConv Layer
# ============================================================
class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1,
                 sample_rate=16000, stride=1, padding=0, dilation=1,
                 bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports 1 input channel (got {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # force odd kernel
        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias")
        if groups > 1:
            raise ValueError("SincConv does not support groups")

        # mel scale init
        NFFT = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        mel_min, mel_max = fmel.min(), fmel.max()
        fmel_space = np.linspace(mel_min, mel_max, out_channels + 1)

        self.mel = self.to_hz(fmel_space)
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]

            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )

            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)


# ============================================================
#               Residual Block
# ============================================================
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])

        self.lrelu = nn.LeakyReLU(0.3)

        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], 1)
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


# ============================================================
#                   RawNet  (modified)
# ============================================================
class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        self.device = device

        # Sinc + CNN
        self.Sinc_conv = SincConv(
            device=device,
            out_channels=d_args['filts'][0],
            kernel_size=d_args['first_conv'],
            in_channels=d_args['in_channels']
        )

        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])
        self.selu = nn.SELU()

        # Residual blocks
        self.block0 = nn.Sequential(Residual_block(d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(d_args['filts'][2]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Attention FCs
        self.fc_attention0 = self._make_attention_fc(d_args['filts'][1][-1], d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(d_args['filts'][1][-1], d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(d_args['filts'][2][-1], d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(d_args['filts'][2][-1], d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(d_args['filts'][2][-1], d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(d_args['filts'][2][-1], d_args['filts'][2][-1])

        # GRU
        self.bn_before_gru = nn.BatchNorm1d(d_args['filts'][2][-1])
        self.gru = nn.GRU(
            input_size=d_args['filts'][2][-1],
            hidden_size=d_args['gru_node'],
            num_layers=d_args['nb_gru_layer'],
            batch_first=True
        )

        # ★★ Embedding layer ★★
        self.fc1_gru = nn.Linear(d_args['gru_node'], d_args['nb_fc_node'])

        # classifier
        self.fc2_gru = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    # ============================================================
    #                 forward() with embedding
    # ============================================================
    def forward(self, x, return_embedding=False):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]

        x = x.view(nb_samp, 1, len_seq)

        # CNN + Sinc
        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        # helper
        def apply_block(x, block, att):
            xb = block(x)
            y = att(self.avgpool(xb).view(xb.size(0), -1))
            y = self.sig(y).view(y.size(0), y.size(1), 1)
            return xb * y + y

        # Residual + attention
        x = apply_block(x, self.block0, self.fc_attention0)
        x = apply_block(x, self.block1, self.fc_attention1)
        x = apply_block(x, self.block2, self.fc_attention2)
        x = apply_block(x, self.block3, self.fc_attention3)
        x = apply_block(x, self.block4, self.fc_attention4)
        x = apply_block(x, self.block5, self.fc_attention5)

        # GRU
        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (B,T,C)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        # Embedding
        embedding = self.fc1_gru(x)   # shape = [B, 512]

        # Classification head
        logits = self.fc2_gru(embedding)
        output = self.logsoftmax(logits)

        if return_embedding:
            return output, embedding

        return output

    # ============================================================
    def _make_attention_fc(self, in_features, l_out_features):
        return nn.Sequential(nn.Linear(in_features, l_out_features))

