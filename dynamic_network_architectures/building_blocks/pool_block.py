from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list


class PoolBlock(nn.Module):
    def __init__(self,
                 input_channel : int,
                 output_channel : int,
                 stride : int,
                 reduced_input_size : Tuple[int, ...],
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ): 
        super(PoolBlock, self).__init__()

        self.nonlin = nonlin(**nonlin_kwargs)
        self.pool_block = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=2, stride=1,padding=1, bias=False),
            nn.MaxPool3d(kernel_size=2, stride=stride),  # 첫 번째 Pooling
            self.nonlin,
            nn.LayerNorm((output_channel, *reduced_input_size))
        )

    def forward(self, x):
        return self.pool_block(x)
    