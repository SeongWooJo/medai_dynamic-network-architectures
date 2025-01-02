import torch
from torch import nn
from dynamic_network_architectures.architectures.DANN_unet import DANNConvUNet

def test_cbam_unet():
    # Model parameters from 3d_fullres configuration
    input_channels = 1  # CT 데이터라 가정
    n_stages = 6
    features_per_stage = [32, 64, 128, 256, 320, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ]
    strides = [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2]
    ]
    n_conv_per_stage = [2, 2, 2, 2, 2, 2]
    num_classes = 4
    n_conv_per_stage_decoder = [2, 2, 2, 2, 2]
    conv_bias = True
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    dropout_op = None
    dropout_op_kwargs = None
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {"inplace": True}
    deep_supervision = False
    nonlin_first = False

    # Model instantiation
    model = DANNConvUNet(
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        conv_bias,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
        deep_supervision,
        nonlin_first
    )
    print("DANNConvUNet 모델 생성 성공!")

    # Testing input
    input_shape = (2, input_channels, 96, 160, 160)  # batch_size=2 from configuration
    input_tensor = torch.randn(input_shape)

    # Forward pass
    model.make_classifier((input_shape[2],input_shape[3],input_shape[4]), features_per_stage, 2)
    output, domain_output = model(input_tensor)

    # Output shape
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output shape: {domain_output.shape}")

if __name__ == "__main__":
    test_cbam_unet()
