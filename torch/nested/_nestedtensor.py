import torch
import numbers
from functools import wraps


def _filter_impl(args, kwargs):
    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        if isinstance(a, torch.NestedTensor):
            impl_args.append(a._impl)
        elif torch.is_tensor(a):
            impl_args.append(a)
        elif isinstance(a, list):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(a_impl)
        elif isinstance(a, tuple):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(tuple(a_impl))
        else:
            impl_args.append(a)
    impl_kwargs = {
        k: v._impl if isinstance(v, torch.NestedTensor) else v
        for (k, v) in kwargs.items()
    }
    return impl_args, impl_kwargs


def _wrap_result(result):
    if isinstance(result, list):
        return list(_wrap_result(r) for r in result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    return (
        torch.NestedTensor(result)
        if torch.is_tensor(result) and torch.is_nt_impl(result)
        else result
    )

@wraps(torch._nested_tensor)
def nested_tensor(*args, **kwargs):
    return NestedTensor(torch._nested_tensor(*args, **kwargs))


class NestedTensor(torch.Tensor):
    # data is a torch.Tensor backed by a NestedTensorImpl

    @staticmethod
    def __new__(cls, impl):
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {}
        kwargs["device"] = impl.device
        kwargs["dtype"] = impl.dtype
        kwargs["layout"] = impl.layout
        kwargs["requires_grad"] = impl.requires_grad
        size = tuple([1] * impl.dim())
        return torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)

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

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print("func: ", func)
        print("args: ", type(args[0]))
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        print("impl_args: ", type(impl_args[0]))
        return _wrap_result(func(*impl_args, **impl_kwargs))
