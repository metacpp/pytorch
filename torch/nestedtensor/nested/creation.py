import torch
import numbers
import warnings

# from . import nested
# import nestedtensor


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False, channels_last=False):
    """
    Arguments match torch.tensor
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if channels_last is None:
        channels_last = False
    if channels_last:
        raise RuntimeError("NestedTensor currently does not support the channels last format. Please open a feature request if you need it.")
    if requires_grad:
        raise RuntimeError("NestedTensor currently does not support autograd. Please open a feature request if you need it.")
    return torch.NestedTensor(torch.nested_tensor_constructor(data, dtype, device, pin_memory, channels_last))


def as_nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    # TODO: Needs tests to check failure cases
    if not isinstance(data, torch.NestedTensor):
        return nested_tensor(data, dtype, device, requires_grad, pin_memory)
    return data