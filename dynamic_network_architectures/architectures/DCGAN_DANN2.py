from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.building_blocks.reversalblock import GradientReversalLayer
from dynamic_network_architectures.building_blocks.ConvDiscriminator import ConvDiscriminator
from dynamic_network_architectures.initialization.dann_weight_init import dann_InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class DCGAN_DANN2(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 input_stage: int = 0,
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.input_stage = input_stage
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        
        self.classifier_list = []
        for input in range(1, n_stages + 1):
            self.classifier_list.append(ConvDiscriminator(n_stages=n_stages, input_stage=input, features_per_stage=features_per_stage,\
            kernel_size=kernel_sizes, strides=strides, conv_bias=conv_bias))
        
        paddings = [(i - 1) // 2 for i in kernel_sizes[-1]]
        base_filters = features_per_stage[-1]
        
        self.classifier_list[-2].first_layer = nn.Sequential(
            nn.Conv3d(in_channels=base_filters, out_channels=base_filters, kernel_size=kernel_sizes[-1], stride=1, padding=paddings, bias=conv_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.InstanceNorm3d(base_filters, 0.8),
        )

        self.classifier_list[-1].first_layer = nn.Sequential(
            nn.Conv3d(in_channels=base_filters, out_channels=base_filters, kernel_size=kernel_sizes[-1], stride=1, padding=paddings, bias=conv_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.InstanceNorm3d(base_filters, 0.8),
            nn.Conv3d(in_channels=base_filters, out_channels=base_filters, kernel_size=kernel_sizes[-1], stride=1, padding=paddings, bias=conv_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.InstanceNorm3d(base_filters, 0.8)
        )
        self.classifier = nn.ModuleList(self.classifier_list)
        self.grl = GradientReversalLayer.apply
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        
        
    def forward(self, x):
        skips = self.encoder(x)
        for idx, skip in enumerate(skips):
            if torch.isnan(skip).any():
                print(f"{idx} skips have nan!")
                print(skips[idx])
        result_list = []
        reversed_skips = []
        for skip in skips:
            reversed_skips.append(self.grl(skip))  # GRL 적용
        for idx, block in enumerate(self.classifier_list):
            result_list.append(block(reversed_skips[idx]))
            
        return self.decoder(skips), result_list
        #return self.classifier(skips[self.input_stage])
    
    def make_classifier(self, input_size, features_per_stage, num_domains):
        for block in self.classifier_list:
            block.make_last_block(input_size, features_per_stage, num_domains)
    
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        dann_InitWeights_He(1e-2)(module)
