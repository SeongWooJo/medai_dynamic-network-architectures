import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

class ConvDiscriminator(nn.Module):
    def __init__(self,
                 n_stages: int,
                 input_stage : int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 kernel_size : Union[int, List[int], Tuple[int, ...]],
                 strides : Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False
                 ):
        super(ConvDiscriminator, self).__init__()
        """
        Parameters:
            n_stage : 원래값 그대로
            input_stage : 시작되는 stage(skips 위치) + 1로 넣어주기 맨위가 1 stage 내려갈때마다 +1씩 추가된다고 생각하면됨

        Returns:
            _type_: _description_
        """
        self.input_stage = input_stage
        self.strides = []
        for stride in strides:
            if isinstance(stride, int):
                stride = [stride] * 3
            else:
                stride = stride
            self.strides.append(stride)
        self.kernel_size = []
        for kernel in kernel_size:
            if isinstance(kernel, int):
                #kernel = [kernel] * 3
                kernel = [3] * 3
            else:
                kernel = [3,3,3]
                #kernel = kernel
            self.kernel_size.append(kernel)

        def discriminator_block(in_filters, out_filters, kernel_size, stride, conv_bias, normalized=True):
            paddings = [(i - 1) // 2 for i in kernel_size]
            block = [nn.Conv3d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=paddings, bias=conv_bias),\
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout3d(0.25)]
            if normalized:
                block.append(nn.InstanceNorm3d(out_filters, 0.8))
            return block
        
        dcgan_list = []
        for s in range(input_stage, n_stages):
            block = discriminator_block(features_per_stage[s-1], features_per_stage[s], self.kernel_size[s-1], self.strides[s], conv_bias, normalized=True)
            dcgan_list.append(nn.Sequential(*block))

        self.pool_layer = nn.Sequential(*dcgan_list)
        self.adv_layer = None
        self.first_layer = None

    def make_last_block(self, input_size, features_per_stage, num_domains):
        skip_len = 1
        skip_sizes = []
        if self.input_stage > 1:
            for s in range(0, self.input_stage):
                input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        for s in range(self.input_stage, len(self.strides)):
            skip_sizes.append([i // j for i, j in zip(input_size, self.strides[s])])
            input_size = skip_sizes[-1]
        for elem in input_size:
            skip_len *= elem
        self.adv_layer = nn.Sequential(
            nn.Linear(features_per_stage[-1] * skip_len, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_domains)
        )

    def forward(self, input):
        if self.first_layer is not None:
            input = self.first_layer(input)
        pool_output = self.pool_layer(input)
        flatten_output = torch.flatten(pool_output, start_dim=1)
        #print(flatten_output.shape)
        output = self.adv_layer(flatten_output)

        return output