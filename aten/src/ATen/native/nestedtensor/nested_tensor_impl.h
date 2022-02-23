#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <c10/core/MemoryFormat.h>

namespace at {
namespace native {

bool is_nested_tensor_impl(at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::NestedTensor);
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor);

  int64_t dim() const override {
    return _nested_size_tensor.dim() > 0 ? 1 + _nested_size_tensor.size(1) : 1;
  }

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  int64_t numel() const override {
    TORCH_CHECK(
        false, "numel is disabled. These methods are not virtual in fbcode.");
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    TORCH_CHECK(
        false,
        "is_contiguous is disabled. These methods are not virtual in fbcode.");
  }
#endif
  Tensor get_nested_size_tensor() {
    return _nested_size_tensor;
  }
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    std::vector<int64_t> sizes;
    return IntArrayRef(sizes);
  }
#endif
#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  IntArrayRef strides() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
    std::vector<int64_t> strides;
    return IntArrayRef(strides);
  }
#endif

  const at::Tensor& get_buffer() const {
    return _buffer;
  }

  at::Tensor& get_buffer() {
    return _buffer;
  }

 private:
  at::Tensor _buffer;
  const at::Tensor _nested_size_tensor;
};

inline at::native::NestedTensorImpl* get_nested_tensor_impl(
    const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::native::NestedTensorImpl*>(
      tensor.unsafeGetTensorImpl());
}

inline at::Tensor get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

inline const at::Tensor get_nested_size_tensor(
    const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_size_tensor();
}

at::Tensor wrap_buffer(
    at::Tensor,
    at::Tensor nested_size_tensor);

} // namespace native
} // namespace at
