#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/nestedtensor/nested_tensor_impl.h>
#include <c10/core/DispatchKey.h>

namespace at {
namespace native {

using namespace torch::nested_tensor;
using namespace c10;

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
  auto esizes = get_efficient_nested_size(self).sizes();
  auto buffer = get_buffer(self);
  auto esizes_chunks = esizes.unbind(0);
  std::vector<int64_t> splits;
  for (int64_t i = 0; i < esizes_chunks.size(); i++) {
    splits.push_back(esizes_chunks[i].prod().item<int64_t>());
  }
  // TODO: This will fail if one of the Tensors has numel 0.
  auto buffer_chunks = at::split_with_sizes(buffer, IntArrayRef(splits));
  std::vector<at::Tensor> result_tensors;
  for (int64_t i = 0; i < buffer_chunks.size(); i++) {
    auto esize_chunk = esizes_chunks[i];
    std::vector<int64_t> esize_vector(esize_chunk.data_ptr<int64_t>(),
                                      esize_chunk.data_ptr<int64_t>() + esize_chunk.numel());
    result_tensors.push_back(buffer_chunks[i].view(IntArrayRef(esize_vector)));
  }
  return result_tensors;
}

bool is_nt_impl(const Tensor& tensor) {
  return is_nested_tensor_impl(tensor);
}

}
}
