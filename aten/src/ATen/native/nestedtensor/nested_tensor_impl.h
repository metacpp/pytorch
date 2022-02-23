#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <c10/core/MemoryFormat.h>

namespace at {
namespace native {

struct EfficientSizeNode {
  explicit EfficientSizeNode(int64_t structure, const at::Tensor& sizes)
      : _num_entries(structure),
        _sizes(sizes) {}

  int64_t height() const {
    return 1;
  }
  int64_t degree() const {
    if (_sizes.dim() == 0) {
      return 0;
    }
    return _sizes.size(0);
  }
  int64_t dim() const {
    return _sizes.dim() > 0 ? 1 + _sizes.size(1) : 1;
  }
  const at::Tensor& sizes() const {
    return _sizes;
  }
  int64_t numel() const {
    if (_sizes.dim() == 0 && _num_entries > 0) {
      return _num_entries;
    }
    if (_sizes.dim() > 0) {
      if (_sizes.numel() == 0) {
        return 0;
      }
      at::Tensor nt_sizes = at::native::narrow(
          _sizes, 1 /* dim */, 0 /* start */, 1 /* length */);
      for (int64_t i = 1; i < _sizes.size(1); i++) {
        at::Tensor tmp = at::native::narrow(
            _sizes, 1 /* dim */, i /* start */, 1 /* length */);
        nt_sizes = nt_sizes * tmp;
      }
      return nt_sizes.sum().item<int64_t>();
    }
    return 0;
  }

 private:
  int64_t _num_entries;
  const at::Tensor _sizes;
};

bool is_nested_tensor_impl(at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::NestedTensor);
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      at::Tensor&& buffer,
      EfficientSizeNode nested_size);

  int64_t dim() const override {
    return _nested_size.dim();
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
  EfficientSizeNode get_nested_size() {
    return _nested_size;
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
  const EfficientSizeNode _nested_size;
  bool _is_pinned;
  const bool _is_contiguous;
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

inline const EfficientSizeNode get_efficient_nested_size(
    const at::Tensor& tensor) {
  TORCH_CHECK(
      is_nested_tensor_impl(tensor), "Given tensor must be NestedTensor.");
  return get_nested_tensor_impl(tensor)->get_nested_size();
}

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_buffer(
    at::Tensor&&,
    EfficientSizeNode efficient_nested_size);

} // namespace native
} // namespace at
