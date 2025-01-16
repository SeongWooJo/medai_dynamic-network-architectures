import torch
import torch.nn as nn
import torch.autograd as autograd

class GradientReversalLayer(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #return input.clone()  # input을 그대로 반환 (forward pass에서 변화 없음)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 그래디언트의 부호를 반전시킴
        grad_input = -0.1 * grad_output  # 여기서 -를 곱해 그래디언트 방향을 반대로
        return grad_input