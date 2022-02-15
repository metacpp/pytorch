import torch
import numbers
# from . import masking
# 
# from . import creation
# 
# import nestedtensor
# import warnings



class NestedTensor(torch.Tensor):
    # The attributes must match across all constiuents
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # data must be a list of Tensors or NestedTensors
    #
    # Attributes:
    #     dim()
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned()
    # Neighbors may share data, maybe all share data.
    # Levels of contiguity

    def __init__(self, impl):
        return None
