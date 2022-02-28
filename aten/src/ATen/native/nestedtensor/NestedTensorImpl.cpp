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
          // TODO: This doesn't properly report is_cpu/is_cuda for NestedTensor.
          // The intended resolution is that once #72827 lands we will be able to
          // allocate separate dispatch keys for CPUNestedTensor (and any other
          // hypothetical device backends for NestedTensor); then we will be
          // able to derive this directly.  If you need this to work before then,
          // make sure you add CPU to this dispatch key set
          c10::DispatchKeySet({DispatchKey::NestedTensor}),
          buffer.dtype(),
          buffer.device()),
      buffer_(std::move(buffer)),
      nested_size_tensor_(std::move(nested_size_tensor)) {
  TORCH_INTERNAL_ASSERT(nested_size_tensor_.is_contiguous());
  int64_t size_dim = nested_size_tensor_.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  remove_autograd_key();
  key_set_ =
      key_set_ - c10::DispatchKeySet({c10::DispatchKey::ADInplaceOrView});
}

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor) {
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_size_tensor));
}

// CPU only!
// TODO: The algorithm here can be optimized, right now it involves a lot of
// small tensor manipulations
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
  std::vector<at::Tensor> result_tensors;
  if (esizes.dim() == 0) {
    return result_tensors;
  }
  auto esizes_chunks = esizes.unbind(0);
  std::vector<int64_t> splits;
  for (const auto i : c10::irange(esizes_chunks.size())) {
    splits.push_back(esizes_chunks[i].prod().item<int64_t>());
  }
  auto buffer_chunks = at::split_with_sizes(get_buffer(self), splits);
  for (const auto i : c10::irange(buffer_chunks.size())) {
    const auto& esize_chunk = esizes_chunks[i];
    result_tensors.push_back(buffer_chunks[i].view(IntArrayRef(
        esize_chunk.data_ptr<int64_t>(),
        esize_chunk.data_ptr<int64_t>() + esize_chunk.numel())));
  }
  return result_tensors;
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

  if (list.size() == 0) {
    return wrap_buffer(ones({0}), ones({}));
  }
  std::vector<Tensor> sizes;
  std::vector<Tensor> flat_tensors;
  for (const auto i : c10::irange(list.size())) {
    if (i > 0) {
      int64_t dim_i = list[i].dim();
      int64_t dim_prev = list[i - 1].dim();
      TORCH_CHECK(
          dim_i == dim_prev,
          "All Tensors given to nested_tensor must have the same dimension. ",
          "Found dimension ",
          dim_i,
          " for Tensor at index ",
          i,
          " and dimension ",
          dim_prev,
          " for Tensor at index ",
          i - 1,
          ".");
    }
    // TODO: Remove call to contiguous once we support strides.
    flat_tensors.push_back(list[i].reshape(-1).contiguous());
    sizes.push_back(tensor(c10::IntArrayRef(list[i].sizes())));
  }

  TensorOptions options = flat_tensors[0].options().merge_in(options_);

  return wrap_buffer(
      at::native::cat(flat_tensors).to(options), at::native::stack(sizes));
}

} // namespace native
} // namespace at
