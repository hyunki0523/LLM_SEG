import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        # 길이 맞추기: args[0]와 weights 중 더 짧은 쪽까지만 사용
        n = min(len(args[0]), len(weights))

        return sum(
            weights[i] * self.loss(*inputs)
            for i, inputs in enumerate(zip(*(a[:n] for a in args)))
            if weights[i] != 0.0
        )