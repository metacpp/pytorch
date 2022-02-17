#pragma once
#include <ATen/native/nestedtensor/EfficientSizeNode.h>
#include <c10/core/MemoryFormat.h>

namespace torch {
namespace nested_tensor {
namespace impl {

inline std::vector<int64_t> _cont_stride(std::vector<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return std::vector<int64_t>(stride);
}

inline std::vector<int64_t> _cont_stride(int64_t* size_ptr, int64_t size) {
  std::vector<int64_t> size_vector(size_ptr, size_ptr + size);
  return _cont_stride(size_vector);
}

inline EfficientSizeNode _cont_stride(const EfficientSizeNode& nested_size) {
  auto nested_stride = map_efficient_size(
      [](int64_t* size_ptr, int64_t size) {
        auto cont_stride = _cont_stride(size_ptr, size);
        for (int64_t i = 0; i < size; i++) {
          size_ptr[i] = cont_stride[i];
        }
      }, nested_size);
  return nested_stride;
}

inline bool _is_cont_stride(int64_t* size, int64_t* stride, size_t length) {
  int64_t p = 1;
  size_t p_i = length;
  for (size_t i = 0; i < length; i++) {
    p_i--;
    if (p != stride[p_i]) {
      return false;
    }
    p *= size[p_i];
  }
  return true;
}

inline bool storage_is_contiguous(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size,
    const EfficientSizeNode& nested_stride) {
  if (!buffer.is_contiguous()) {
    return false;
  }
  if (buffer.numel() == 0) {
    return true;
  }
  const at::Tensor& sizes_sizes = nested_size.sizes();
  const at::Tensor& strides_sizes = nested_stride.sizes();
  int64_t* sizes_sizes_ptr = sizes_sizes.data_ptr<int64_t>();
  int64_t* strides_sizes_ptr = strides_sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes_sizes.size(0); i++) {
    if (!_is_cont_stride(
            sizes_sizes_ptr + i * sizes_sizes.size(1),
            strides_sizes_ptr + i * strides_sizes.size(1),
            sizes_sizes.size(1))) {
      return false;
    }
  }
  return true;
}

inline bool storage_is_contiguous_channels_last(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size,
    const EfficientSizeNode& nested_stride) {
  if (!buffer.is_contiguous()) {
    return false;
  }
  if (buffer.numel() == 0) {
    return true;
  }
  if (nested_size.dim() != 4) {
    return false;
  }
  const at::Tensor& sizes_sizes = nested_size.sizes();
  const at::Tensor& strides_sizes = nested_stride.sizes();
  int64_t* sizes_sizes_ptr = sizes_sizes.data_ptr<int64_t>();
  int64_t* strides_sizes_ptr = strides_sizes.data_ptr<int64_t>();
  std::vector<int64_t> sizes(4, 0);
  std::vector<int64_t> strides(4, 0);
  for (int64_t i = 0; i < sizes_sizes.size(0); i++) {
    sizes[0] = 1;
    sizes[1] = sizes_sizes_ptr[i * 3 + 0];
    sizes[2] = sizes_sizes_ptr[i * 3 + 1];
    sizes[3] = sizes_sizes_ptr[i * 3 + 2];
    strides[0] = sizes_sizes_ptr[i * 3 + 0] *
                 sizes_sizes_ptr[i * 3 + 1] *
                 sizes_sizes_ptr[i * 3 + 2];
    strides[1] = strides_sizes_ptr[i * 3 + 0];
    strides[2] = strides_sizes_ptr[i * 3 + 1];
    strides[3] = strides_sizes_ptr[i * 3 + 2];
    if (!c10::is_channels_last_strides_2d(c10::IntArrayRef(sizes), c10::IntArrayRef(strides))) {
      return false;
    }
  }
  return true;
}

} // namespace impl
} // namespace nested_tensor
} // namespace torch
