#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <c10/core/MemoryFormat.h>

namespace at {
namespace native {

namespace impl {

inline std::vector<c10::optional<int64_t>> construct_efficient_size(
    int64_t out,
    const at::Tensor& sizes) {
  std::vector<c10::optional<int64_t>> result;
  result.push_back(out);
  size_t nested_dim = result.size();
  if (sizes.dim() > 0) {
    int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
    result.resize(nested_dim + sizes.size(1));
    for (int64_t i = 0; i < sizes.size(1); i++) {
      result[nested_dim + i] = sizes_ptr[i];
    }
    for (int64_t j = 0; j < sizes.size(1); j++) {
      for (int64_t i = 0; i < sizes.size(0); i++) {
        if (result[nested_dim + j] &&
            (result[nested_dim + j] != sizes_ptr[i * sizes.size(1) + j])) {
          result[nested_dim + j] = c10::nullopt;
        }
      }
    }
  }
  return result;
}

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(int64_t structure, const at::Tensor& sizes)
      : _structure(structure),
        _sizes(sizes),
        _opt_sizes(impl::construct_efficient_size(_structure, _sizes)) {}

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
  const std::vector<c10::optional<int64_t>>& opt_sizes() const {
    return _opt_sizes;
  }
  void refresh_opt_sizes() {
    _opt_sizes = impl::construct_efficient_size(_structure, _sizes);
  }
  const at::Tensor& sizes() const {
    return _sizes;
  }
  int64_t structure() const {
    return _structure;
  }
  EfficientSizeNode clone() const {
    return EfficientSizeNode(_structure, _sizes.clone());
  }
  int64_t numel() const {
    if (_sizes.dim() == 0 && _structure > 0) {
      return _structure;
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
  int64_t _structure;
  const at::Tensor _sizes;
  bool _opt_sizes_set = false;
  std::vector<c10::optional<int64_t>> _opt_sizes;
};

constexpr auto NestedTensorKey = DispatchKey::NestedTensor;

struct NestedTensorImpl;

template <class A>
bool is_nested_tensor_impl(A tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::NestedTensor);
}

template <class A, class B>
bool is_nested_tensor_impl(A first, B other) {
  return is_nested_tensor_impl(first) && is_nested_tensor_impl(other);
}

template <class A, class B, class... C>
bool is_nested_tensor_impl(A first, B second, C... other) {
  return is_nested_tensor_impl(first, second) &&
      is_nested_tensor_impl(other...);
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
//   EfficientSizeNode get_nested_stride() {
//     return _nested_stride;
//   }
  int64_t nested_dim() const {
    return _nested_size.height();
  }
  bool is_pinned() const {
    return _buffer.is_pinned();
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const {
    return _nested_size.opt_sizes();
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

  bool get_is_cuda() const {
    return _buffer.is_cuda();
  }

  bool get_is_contiguous(at::MemoryFormat memory_format) const {
    if (memory_format == at::MemoryFormat::Contiguous) {
      return _is_contiguous;
    }
    TORCH_CHECK(
        false, "is_contiguous does not support memory format ", memory_format);
    return false;
  }

  bool get_is_pinned() const {
    return _is_pinned;
  }

 private:
  at::Tensor _buffer;
  const EfficientSizeNode _nested_size;
  bool _is_pinned;
  const bool _is_contiguous;
};

int64_t nt_size(Tensor tensor, int64_t dim);

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
