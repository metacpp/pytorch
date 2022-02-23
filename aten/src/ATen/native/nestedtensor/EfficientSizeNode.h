#pragma once

namespace torch {
namespace nested_tensor {

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

} // namespace nested_tensor
} // namespace torch
