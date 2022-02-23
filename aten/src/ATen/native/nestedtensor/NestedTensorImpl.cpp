#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/nestedtensor/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>

namespace at {
namespace native {

NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor)
    : TensorImpl(
          c10::DispatchKeySet({DispatchKey::NestedTensor}),
          buffer.dtype(),
          buffer.device()),
      _buffer(buffer),
      _nested_size_tensor(nested_size_tensor) {
  remove_autograd_key();
  key_set_ =
      key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
}

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), nested_size_tensor);
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");
  auto esizes = get_nested_size_tensor(self);
  auto buffer = get_buffer(self);
  std::vector<at::Tensor> result_tensors;
  if (esizes.dim() == 0) {
    return result_tensors;
  }
  auto esizes_chunks = esizes.unbind(0);
  std::vector<int64_t> splits;
  for (int64_t i = 0; i < esizes_chunks.size(); i++) {
    splits.push_back(esizes_chunks[i].prod().item<int64_t>());
  }
  auto buffer_chunks = at::split_with_sizes(buffer, IntArrayRef(splits));
  for (int64_t i = 0; i < buffer_chunks.size(); i++) {
    auto esize_chunk = esizes_chunks[i];
    std::vector<int64_t> esize_vector(
        esize_chunk.data_ptr<int64_t>(),
        esize_chunk.data_ptr<int64_t>() + esize_chunk.numel());
    result_tensors.push_back(buffer_chunks[i].view(IntArrayRef(esize_vector)));
  }
  return result_tensors;
}

bool is_nt_impl(const Tensor& tensor) {
  return is_nested_tensor_impl(tensor);
}

/*
 * This result of this function cannot be used by itself. The result needs to
 * be wrapped in torch.nested.NestedTensor.
 */
Tensor _nested_tensor(
    TensorList list,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  std::vector<Tensor> sizes;
  std::vector<Tensor> flat_tensors;
  for (size_t i = 0; i < list.size(); i++) {
    flat_tensors.push_back(list[i].reshape(-1).contiguous());
    sizes.push_back(tensor(c10::IntArrayRef(list[i].sizes())));
  }
  if (flat_tensors.size() == 0) {
    return wrap_buffer(ones({0}), ones({}));
  }

  TensorOptions options = flat_tensors[0].options().merge_in(options_);

  return wrap_buffer(
      at::native::cat(at::TensorList(flat_tensors)).to(options),
      at::native::stack(at::TensorList(sizes)));
}

} // namespace native
} // namespace at
