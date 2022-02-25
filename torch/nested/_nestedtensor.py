import torch
from functools import wraps


@wraps(torch._nested_tensor)
def nested_tensor(*args, **kwargs):
    return NestedTensor(torch._nested_tensor(*args, **kwargs))


class NestedTensor:
    # data is a torch.Tensor backed by a NestedTensorImpl

    def __init__(self, impl):
        self._impl = impl

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._impl.requires_grad

    def stride(self):
        """
        The stride of ```self``` NestedTensor.
        """
        return self._impl.stride()

    def size(self):
        """
        The size of ```self``` NestedTensor.
        """
        return self._impl.size()

    def dim(self):
        """
        The dimension of ```self``` NestedTensor.
        """
        return self._impl.dim()

    def numel(self):
        """
        The number of elements of ```self``` NestedTensor.
        """
        return self._impl.numel()

    def is_contiguous(self):
        """
        Returns true if ```self``` NestedTensor is contiguous.
        """
        return self._impl.is_contiguous()

    def __str__(self):
        def _str(x, indent=0, tab="  "):
            s = indent * tab + "[\n"
            strs = list(map(str, x.unbind()))
            strs = list(
                map(
                    lambda xi: "\n".join(
                        map(lambda xij: (indent + 1) * tab + xij, xi.split("\n"))
                    ),
                    strs,
                )
            )
            s += ",\n".join(strs)
            s += "\n" + indent * tab + "]"
            return s

        return "nested_tensor(" + _str(self) + ")"

    def __repr__(self):
        return self.__str__()

    def unbind(self, dim=None):
        if self._impl.dim() == 0 and dim is None:
            return ()
        if dim is None:
            dim = 0
        return torch.ops.aten.unbind.int(self._impl, dim)
