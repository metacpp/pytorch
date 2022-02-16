#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/nestedtensor/nested_tensor_impl.h>
#include <ATen/native/nestedtensor/nested_node_functions.h>
#include <c10/core/DispatchKey.h>

namespace at {
namespace native {

using namespace torch::nested_tensor;
using namespace c10;

TensorNode _unbind_tensors(TensorNode structure) {
  std::vector<TensorNode> result_nodes;
  if (structure.is_leaf()) {
    for (at::Tensor tensor : structure.payload().unbind()) {
      result_nodes.emplace_back(TensorNode(std::move(tensor)));
    }
  } else {
    for (TensorNode child : structure.unbind()) {
      result_nodes.emplace_back(_unbind_tensors(child));
    }
  }
  return TensorNode(std::move(result_nodes));
}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       EfficientSizeNode nested_size,
       EfficientSizeNode nested_stride)
    : TensorImpl(
          c10::DispatchKeySet({NestedTensorKey}),
          buffer.dtype(),
          buffer.device()),
      _buffer(buffer),
      _nested_size(nested_size),
      _nested_stride(nested_stride),
      _is_pinned(_buffer.is_pinned()),
      _is_contiguous(torch::nested_tensor::impl::storage_is_contiguous(
          _buffer,
          _nested_size,
          _nested_stride)),
      _is_contiguous_channels_last(torch::nested_tensor::impl::storage_is_contiguous_channels_last(
          _buffer,
          _nested_size,
          _nested_stride)) {
  remove_autograd_key();
  key_set_ = key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       EfficientSizeNode nested_size)
  : NestedTensorImpl(std::move(buffer),
                     nested_size,
                     torch::nested_tensor::impl::_cont_stride(nested_size)) {}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       SizeNode nested_size,
       SizeNode nested_stride)
  : NestedTensorImpl(std::move(buffer),
                     EfficientSizeNode(nested_size),
                     EfficientSizeNode(nested_stride)) {}

NestedTensorImpl::NestedTensorImpl(at::Tensor&& buffer,
       SizeNode nested_size)
  : NestedTensorImpl(std::move(buffer),
                     EfficientSizeNode(nested_size)) {}

NestedTensorImpl::NestedTensorImpl(TensorNode structure)
  : NestedTensorImpl(
             torch::nested_tensor::impl::pack(structure),
             EfficientSizeNode(
               map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                 structure))) {}


inline TensorNode _squeeze_nested_dim(TensorNode structure, int64_t dim) {
  return squeeze(structure, dim, false);
}

int64_t NestedTensor_size_int(const Tensor& self, int64_t dim) {
  std::vector<c10::optional<int64_t>> size =
      get_nested_tensor_impl(self)->opt_sizes();
  TORCH_CHECK(size[dim], "NestedTensor is not regular at dimension ", dim, ".");
  return *(size[dim]);
}

int64_t nt_size(Tensor tensor, int64_t dim) {
  auto impl = get_nested_tensor_impl(tensor);
  std::vector<c10::optional<int64_t>> size = impl->opt_sizes();
  if (size[dim]) {
    return *(size[dim]);
  }
  throw std::runtime_error(
      "NestedTensor size at dim is not Tensor shape compliant.");
}

at::Tensor wrap_tensor_node(TensorNode&& result) {
  if (result.is_leaf()) {
    return result.payload();
  }
  return at::detail::make_tensor<NestedTensorImpl>(result);
}

std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode> input) {
  std::vector<at::Tensor> result;
  for (size_t i = 0; i < input.size(); i++) {
    result.push_back(wrap_tensor_node(std::move(input[i])));
  }
  return result;
}

at::Tensor wrap_buffer(at::Tensor&& buffer, SizeNode nested_size) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  if (nested_size.is_leaf()) {
    return buffer.reshape(IntArrayRef(nested_size.payload()));
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), nested_size);
}

at::Tensor wrap_buffer(
    at::Tensor&& buffer,
    EfficientSizeNode efficient_nested_size,
    EfficientSizeNode efficient_nested_stride) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  TORCH_CHECK(
      efficient_nested_size.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  TORCH_CHECK(
      efficient_nested_stride.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      efficient_nested_size,
      efficient_nested_stride);
}

at::Tensor wrap_buffer(
    at::Tensor&& buffer,
    EfficientSizeNode efficient_nested_size) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  TORCH_CHECK(
      efficient_nested_size.height() > 0,
      "Internal error: expected nested_size of non-zero height.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      efficient_nested_size);
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  auto _data = get_nested_tensor_impl(self);
  dim = at::maybe_wrap_dim(dim, get_dim(self));
  auto node = _data->get_structure();
  if (dim == 0) {
    return wrap_tensor_node(node.unbind());
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (auto child : node.unbind()) {
    std::vector<at::Tensor> tmp =
        at::unbind(wrap_tensor_node(std::move(child)), dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (j >= unbound.size()) {
        unbound.resize(j + 1);
      }
      unbound[j].push_back(TensorNode(std::move(tmp[j])));
    }
  }
  std::vector<TensorNode> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    result.push_back(TensorNode(std::move(unbound[i])));
  }
  return wrap_tensor_node(result);
}

bool is_nt_impl(const Tensor& tensor) {
  return is_nested_tensor_impl(tensor);
}

}
}
