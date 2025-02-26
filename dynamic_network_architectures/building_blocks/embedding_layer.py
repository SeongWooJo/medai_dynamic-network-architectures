import torch
import torch.nn as nn

class Embedding_Layer(nn.Module):
    def __init__(self,
                 input_volume_size,
                 feature_nums):
        super().__init__()

        self.embedding_layer = nn.Sequential(
            nn.Conv3d(feature_nums, feature_nums // 2, 1, 1, 0),  # 첫 번째 Linear Layer (입력: 512 → 출력: 256)
            nn.ReLU(),            # 비선형 활성화 함수
            nn.LayerNorm(normalized_shape=(feature_nums // 2, *input_volume_size)),    # Layer Normalization 적용
            nn.Conv3d(feature_nums // 2, feature_nums // 4, 1, 1, 0),  # 두 번째 Linear Layer (출력: 128)
            nn.LayerNorm(normalized_shape=(feature_nums // 4, *input_volume_size)),    # 최종 Layer Normalization
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.embedding_layer(x)
        
        
class Embedding_Layer2(nn.Module):
    def __init__(self,
                 input_volume_size,
                 feature_nums):
        super().__init__()

        self.embedding_layer = nn.Sequential(
            nn.Conv3d(feature_nums, feature_nums - 8, 1, 1, 0),  # 첫 번째 Linear Layer (입력: 512 → 출력: 256)
            nn.ReLU(),            # 비선형 활성화 함수
            nn.LayerNorm(normalized_shape=(feature_nums - 8, *input_volume_size)),    # Layer Normalization 적용
            nn.Conv3d(feature_nums - 8, feature_nums - 16, 1, 1, 0),  # 두 번째 Linear Layer (출력: 128)
            nn.LayerNorm(normalized_shape=(feature_nums - 16, *input_volume_size)),    # 최종 Layer Normalization
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.embedding_layer(x)
        