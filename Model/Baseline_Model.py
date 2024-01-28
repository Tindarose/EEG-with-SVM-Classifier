import math
import torch
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class CCNN(torch.nn.Module):
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.5):
        super(CCNN, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = torch.nn.Sequential(torch.nn.ZeroPad2d((1, 2, 1, 2)), torch.nn.Conv2d(self.in_channels, 64, kernel_size = 4, stride = 1),
                                   torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.ZeroPad2d((1, 2, 1, 2)), torch.nn.Conv2d(64, 128, kernel_size = 4, stride = 1), torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.ZeroPad2d((1, 2, 1, 2)), torch.nn.Conv2d(128, 256, kernel_size = 4, stride = 1), torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(torch.nn.ZeroPad2d((1, 2, 1, 2)), torch.nn.Conv2d(256, 64, kernel_size = 4, stride = 1), torch.nn.ReLU())

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            torch.nn.SELU(), # Not mentioned in paper
            # torch.nn.Dropout2d(self.dropout),
            torch.nn.Dropout(self.dropout)
        )
        self.lin2 = torch.nn.Linear(1024, self.num_classes)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.conv1(mock_eeg)
            mock_eeg = self.conv2(mock_eeg)
            mock_eeg = self.conv3(mock_eeg)
            mock_eeg = self.conv4(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim = 1)

            return mock_eeg.shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim = 1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

class EEG_Net(torch.nn.Module):
    def __init__(self, chunk_size: int = 128, num_electrodes: int = 32,
                 F1: int = 8, F2: int = 16, D: int = 2, num_classes: int = 2,
                 kernel_1: int = 64, kernel_2: int = 16, dropout: float = 0.5):
        super(EEG_Net, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            torch.nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            torch.nn.Conv2d(self.F1, self.F1 * self.D, (self.num_electrodes, 1), stride = 1, padding = (0, 0), groups = self.F1, bias=False),
            torch.nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            torch.nn.ELU(), torch.nn.AvgPool2d((1, 4), stride=4), torch.nn.Dropout(p=dropout))

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            torch.nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            torch.nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), torch.nn.ELU(), torch.nn.AvgPool2d((1, 8), stride=8),
            torch.nn.Dropout(p=dropout))

        self.lin = torch.nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x

class TSCeption(torch.nn.Module):
    def __init__(self, num_electrodes: int = 28, num_T: int = 15, num_S: int = 15,
                 in_channels: int = 1, hid_channels: int = 32, num_classes: int = 2,
                 sampling_rate: int = 128, dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(TSCeption, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = in_channels, out_channels = num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, int(self.pool * 0.25)), stride=(1, int(self.pool * 0.25))))
        self.Tception2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = in_channels, out_channels = num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, int(self.pool * 0.25)), stride=(1, int(self.pool * 0.25))))
        self.Tception3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = in_channels, out_channels = num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, int(self.pool * 0.25)), stride=(1, int(self.pool * 0.25))))

        self.Sception1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = num_T, out_channels = num_S, kernel_size=(int(num_electrodes), 1), stride=1),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, int(self.pool * 0.25)), stride=(1, int(self.pool * 0.25))))
        self.Sception2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = num_T, out_channels = num_S, kernel_size=(int(num_electrodes * 0.5), 1), stride=(int(num_electrodes * 0.5), 1)),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, int(self.pool * 0.25)), stride=(1, int(self.pool * 0.25))))

        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = num_S, out_channels = num_S, kernel_size=(3, 1), stride=1),
            torch.nn.LeakyReLU(), torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.BN_t = torch.nn.BatchNorm2d(num_T)
        self.BN_s = torch.nn.BatchNorm2d(num_S)
        self.BN_fusion = torch.nn.BatchNorm2d(num_S)

        self.fc = torch.nn.Sequential(torch.nn.Linear(num_S, hid_channels), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                torch.nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out

class PatchEmbedding(torch.nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = torch.nn.Sequential(
            torch.nn.Conv2d(1, 40, (1, 25), (1, 1)),
            torch.nn.Conv2d(40, 40, (22, 1), (1, 1)),
            torch.nn.BatchNorm2d(40),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            torch.nn.Dropout(0.5),
        )

        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = torch.nn.Linear(emb_size, emb_size)
        self.queries = torch.nn.Linear(emb_size, emb_size)
        self.values = torch.nn.Linear(emb_size, emb_size)
        self.att_drop = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(torch.nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            torch.nn.Linear(emb_size, expansion * emb_size),
            torch.nn.GELU(),
            torch.nn.Dropout(drop_p),
            torch.nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(torch.nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                torch.nn.Dropout(drop_p)
            )),
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                torch.nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(torch.nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(torch.nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # global average pooling
        self.clshead = torch.nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, n_classes)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(880, 256),
            torch.nn.ELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 32),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.contiguous().reshape(x.size(0), -1)
        out = self.fc(x)
        return x, out

class EEG_Conformer(torch.nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__()
        self.layer = torch.nn.Sequential(PatchEmbedding(emb_size), \
        TransformerEncoder(depth, emb_size), \
        ClassificationHead(emb_size, n_classes))

    def forward(self, input):
        return self.layer(input)