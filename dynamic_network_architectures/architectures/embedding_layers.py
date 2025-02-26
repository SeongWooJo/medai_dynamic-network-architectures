import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.embedding_layer import Embedding_Layer
class Embedding_Layers(nn.Module):
    def __init__(self,
                 input_size,
                 feature_nums,
                 strides,
                 n_stages):
        super().__init__()

        block_list = []
        self.n_stages = n_stages
        for s in range(n_stages - 1):
            skip_size = [i // j for i, j in zip(input_size, strides[s])]
            feature_num = feature_nums[s]
            block_list.append(Embedding_Layer(skip_size, feature_num))
            input_size = skip_size
        self.embedding = nn.ModuleList(block_list)

    def forward(self, features):
        ret = []
        for idx in range(len(self.embedding)):
            ret.append(self.embedding[idx](features[idx]))
        return ret