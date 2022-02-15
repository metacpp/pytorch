import torch
import numbers
# from . import masking
# 
# from . import creation
# 
# import nestedtensor
# import warnings

def nt_multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 bias_k,
                                 bias_v,
                                 add_zero_attn,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None,
                                 need_weights=True,
                                 attn_mask=None,
                                 use_separate_proj_weight=False,
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None
                                 ):
    assert isinstance(query, torch.NestedTensor)
    assert isinstance(key, torch.NestedTensor)
    assert isinstance(value, torch.NestedTensor)
    assert torch.is_tensor(out_proj_weight)
    assert torch.is_tensor(out_proj_bias)
    # Self-attention only
    # CUDA only
    assert query is key and key is value and in_proj_weight.is_cuda

    # TODO: Explicitly unsupported flags
    assert not use_separate_proj_weight
    assert attn_mask is None
    assert key_padding_mask is None
    assert bias_k is None
    assert bias_v is None
    assert static_k is None
    assert static_v is None
    assert not add_zero_attn
    # assert not need_weights

    # bsz, tgt_len, embed_dim = query.size()
    embed_dim = query.size(2)
    assert embed_dim == embed_dim_to_check

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    return torch.ops.nestedtensor.bt_min_mha(num_heads,
                                             head_dim,
                                             0.5,
                                             False,
                                             query,
                                             query,
                                             query,
                                             in_proj_weight,
                                             in_proj_bias,
                                             scaling,
                                             out_proj_weight,
                                             in_proj_bias), None

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
        k: v._impl if isinstance(v, torch.NestedTensor) else v for (k, v) in kwargs.items()
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

class NestedTensorMeta(type):
    def __getattr__(cls, name):
        if getattr(torch.Tensor, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(impl_args[0], name)(
                    *(impl_args[1:]), **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return cls.__dict__[name]



class NestedTensor(metaclass=NestedTensorMeta):
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
        self._impl = impl

    def __getattr__(self, name):
        if hasattr(self._impl, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(self._impl, name)(*impl_args, **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return self.__dict__[name]

    def nested_dim(self):
        """
        The nested dimension of ```self``` NestedTensor.
        The nested dimension is defined as the level of indexing required
        to reach a Tensor constiuent.
        """
        # This NT only supports nesting of 1.
        return 1

    def size(self, dim):
        return torch.nested_tensor_size_int(self._impl, dim)

    def __str__(self):
        def _str(x, indent=0, tab="  "):
            if x.nested_dim() == 0:
                return ""
            s = indent*tab + "[\n"
            if x.nested_dim() == 1:
                strs = list(map(str, x.unbind()))
                strs = list(map(lambda xi: "\n".join(
                    map(lambda xij: (indent + 1)*tab + xij, xi.split("\n"))), strs))
                s += ",\n".join(strs)
            else:
                s += ",\n".join(list(map(
                    lambda xi: _str(xi, indent + 1), x.unbind())))
            s += "\n" + indent * tab + "]"
            return s
        return "nested_tensor(" + _str(self) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        print("HEE: func", func)
        if func is torch.nn.functional.multi_head_attention_forward:
            return _wrap_result(nt_multi_head_attention_forward(*args, **kwargs))
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        return _wrap_result(func(*impl_args, **impl_kwargs))

