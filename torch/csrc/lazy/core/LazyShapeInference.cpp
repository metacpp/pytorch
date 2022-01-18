/**
 * This is a handwritten file that accompanies codegenerated header
 * LazyShapeDtype.h
 *
 * The purpose of these shape/dtype inference methods are to fill gaps
 * where we do not yet have structured kernels in pytorch core.  Ops
 * for which there _are_ structured kernels can use meta::op() to infer
 * shape/dtype, and codegen makes use of this.  Ops for which there are not
 * yet structured kernels can still be used with lazy_tensor codegen, but require
 * manual intervention to implement compute_shape_{op} and compute_dtype_{op}.
 *
 * READ THIS!
 *
 * 1. Beware: Tech Debt!
 * ---------------------
 * These functions are tech debt.  We want to delete them all and use structured
 * kernels instead, but it's a lot faster to write these so we're decoupling the
 * two efforts to move fast for adding support for codegenned Lazy Tensor ops.
 *
 * Codegenned Lazy Tensor ops with handwritten shape formulae are still better than
 * fully handwritten Lazy Tensor ops (which also have handwritten shape formulae).
 *
 * 2. Structured Kernels For The Win
 * ---------------------------------
 * Long term, more and more ops should be supported as 'structured kernels'.  Consider
 * doing your part and porting an op.  As ops get ported over, the codegen will automatically
 * notice and stop generating declarations for these shape formulae, so we'll need to
 * manually clean up the unused functions in this file, or somehow automate that.
 *
 * https://dev-discuss.pytorch.org/t/slides-from-structured-kernel-presentation/179
 *
 * 3. How to figure out the shape/dtype
 * ------------------------------------
 * Unfortunatley there isn't a one-stop-shop for learning the output shape formulae for all
 * operators.  This is partly becuase some operators are not part of our 'public' API, including
 * backward operators which users don't directly invoke.
 *
 * Check our opinfo registry:
 *  https://github.com/pytorch/pytorch/blob/13b859983183ea9938deb5030ac9a0747841f0a8/torch/csrc/jit/runtime/symbolic_shape_registry.cpp
 *
 * Read the manual (for ops that are 1:1 with python frontend):
 *  https://pytorch.org/docs/stable/generated/torch.trace.html
 *
 */

// #include <functional>
#include "torch/csrc/lazy/core/shape.h"
#include <ATen/native/ConvUtils.h>
#include "aten/src/ATen/native/ReduceOpsUtils.h"
//#include "lazy_tensor_core/csrc/ts_backend/LazyShapeInference.h"
#include "torch/csrc/api/include/torch/enum.h"
#include <iostream>
#include <vector>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
using Shape = torch::lazy::Shape;


// Copied from ATen/native/utils/ParamUtils.h, which aparently I can't include from here?
std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

std::vector<Shape> compute_shape_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  if(reduction == at::Reduction::None) {
    return {Shape(self.scalar_type(), self.sizes().vec())};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}


std::vector<Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
  // Based on aten/src/ATen/native/ConstantPadNd.cpp::constant_pad_nd
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ",
            pad.size());

  auto input_sizes = self.sizes();
  auto l_inp = self.dim();

  auto l_pad = pad.size() / 2;
  auto l_diff = l_inp - l_pad;
  TORCH_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
            "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
            l_inp, "dimensions.");

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < (size_t)l_diff; i ++) {
      new_shape.emplace_back(input_sizes[i]);
  }

  for (const auto i : c10::irange((size_t)l_pad)) {
      auto pad_idx = pad.size() - ((i + 1) * 2);
      auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
      TORCH_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                pad[pad_idx], " and ", pad[pad_idx + 1], " resulted in a negative output size, "
                "which is invalid. Check dimension ", l_diff + i, " of your input.");
      new_shape.emplace_back(new_dim);
  }
  return {Shape(self.scalar_type(), new_shape)};
}

std::vector<Shape> compute_shape_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::optional<at::IntArrayRef> bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
  if (bias_sizes.has_value()) {
    return {Shape(input.scalar_type(), input.sizes().vec()),
            Shape(weight.scalar_type(), weight.sizes().vec()),
            Shape(grad_output.scalar_type(), bias_sizes.value().vec())};
  } else {
    // TODO(whc) not sure whether to return 2 shapes here, or a 3rd one that is empty
    return {Shape(input.scalar_type(), input.sizes().vec()),
            Shape(weight.scalar_type(), weight.sizes().vec())};
  }
}

std::vector<Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {

  int64_t dim = weight.ndimension() - 2;
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  // at::convolution performs parameter expansion before running kernels on expanded parameters
  // we must do the same.  Shape formulae access differnent dimensions of e.g. output_padding, but
  // output_padding may be passed in as a scalar.  Sadly, accessing output_padding[1] in this case
  // gives incorrect results rather than indexing error
  auto expanded_stride = expand_param_if_needed(stride, "stride", dim);
  auto expanded_padding = expand_param_if_needed(padding, "padding", dim);
  auto expanded_dilation = expand_param_if_needed(dilation, "dilation", dim);
  if (!transposed) {
    return {Shape(input.scalar_type(), at::native::conv_output_size(input.sizes(), weight.sizes(), expanded_padding, expanded_stride, expanded_dilation))};
  } else {
    auto expanded_output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
    auto out_shape = at::native::conv_input_size(input.sizes(), weight.sizes(), expanded_padding, expanded_output_padding, expanded_stride, expanded_dilation, groups);
    return {Shape(input.scalar_type(), out_shape)};
  }
}

std::vector<Shape> compute_shape_masked_fill(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_masked_fill(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse){
  // Based on aten/src/ATen/native/Embedding.cpp::embedding.
  std::vector<int64_t> out_sizes = indices.sizes().vec();
  out_sizes.emplace_back(weight.size(1));
  return {Shape(weight.scalar_type(), out_sizes)};
}

std::vector<Shape> compute_shape_std(const at::Tensor & self, bool unbiased){
  return compute_shape_std(self, c10::nullopt, c10::nullopt, false);
}
std::vector<Shape> compute_shape_std(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim){
  return compute_shape_std(self, dim, c10::nullopt, keepdim);
}
std::vector<Shape> compute_shape_std(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim){
  if (dim.has_value()) {
    auto shape = at::native::shape_from_dim_mask(self, at::native::make_dim_mask(dim.value(), self.dim()), keepdim);
    return {Shape(self.scalar_type(), std::vector<int64_t>(shape.begin(), shape.end()))};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_embedding_dense_backward(const at::Tensor& grad_output, const at::Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  // Based on aten/src/ATen/native/Embedding.cpp::embedding_dense_backward_cpu.
  return {Shape(grad_output.scalar_type(), {num_weights, grad_output.size(-1)})};
}

std::vector<Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim,
    const at::Tensor & index) {
  // Based on definition of https://pytorch.org/docs/stable/generated/torch.index_select.html.
  // Promote Rank 0 index tensor to a 1 * 1 tensor.
  dim = at::maybe_wrap_dim(dim, self);
  auto index_dim = index.dim() > 0 ? index.dim() : 1;
  auto index_size = index.dim() > 0 ? index.size(0) : 1;
  TORCH_CHECK(index_dim == 1);

  auto self_sizes = self.sizes();
  std::vector<int64_t> output_sizes(self_sizes.begin(), self_sizes.end());
  output_sizes[dim] = index_size;

  return {Shape(self.scalar_type(), output_sizes)};
}

std::vector<Shape> compute_shape_kl_div_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction, bool log_target) {
  // Based on definition of aten/src/ATen/native/Loss.cpp::kl_div_backward_cpu.
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_cat(at::TensorList tensors, int64_t dim) {
  // TODO(whc) support cat in codegen and move this to compute_*_cat functions
  std::vector<int64_t> out_shape(tensors[0].sizes().begin(), tensors[0].sizes().end());

  dim = at::maybe_wrap_dim(dim, tensors);
  size_t extended_dim_shape = 0;
  for (auto& tensor: tensors) {
    extended_dim_shape += tensor.sizes()[dim];
  }
  out_shape[dim] = extended_dim_shape;
  return {Shape(tensors[0].scalar_type(), out_shape)};
}

std::vector<Shape> compute_shape_native_layer_norm(const at::Tensor & input,
    at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias,
    double eps) {
  // Copied from aten/src/ATen/native/layer_norm.cpp::layer_norm_cpu_out.
  auto input_shape = input.sizes().vec();
  const size_t axis = input.dim() - normalized_shape.size();

  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.emplace_back(input_shape[idx]);
  }
  for (const auto idx : c10::irange(axis, input.dim())) {
    (void)idx; // Suppress unused variable warning
    stat_shape.emplace_back(1);
  }

  return {Shape(input.scalar_type(), std::move(input_shape)),
          Shape(input.scalar_type(), stat_shape),
          Shape(input.scalar_type(), std::move(stat_shape))};
}

std::vector<Shape> compute_shape_native_layer_norm_backward(const at::Tensor& grad_out,
    const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& mean, const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, ::std::array<bool,3> output_mask) {
  std::vector<Shape> shapes;
  shapes.push_back(Shape(input.scalar_type(),
                         output_mask[0] ? input.sizes().vec() : std::vector<int64_t>{}));
  shapes.push_back(Shape(weight && weight->defined() ? weight->scalar_type() : input.scalar_type(),
                         output_mask[1] && weight ? weight->sizes().vec() : std::vector<int64_t>{}));
  shapes.push_back(Shape(bias && weight->defined() ? bias->scalar_type() : input.scalar_type(),
                         output_mask[2] && bias ? bias->sizes().vec() : std::vector<int64_t>{}));
  return shapes;
}

std::vector<Shape> compute_shape_mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {Shape(dtype.value(), {})};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_mv(const at::Tensor& self, const at::Tensor& vec) {
  return {Shape(self.scalar_type(), {self.size(0)})};
}

std::vector<Shape> compute_shape_native_dropout(const at::Tensor & input, double p, c10::optional<bool> train) {
  return {Shape(input.scalar_type(), input.sizes().vec()), Shape(c10::ScalarType::Bool, input.sizes().vec())};
}

std::vector<Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale) {
  return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}

std::vector<Shape> compute_shape_relu(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_bitwise_and(const at::Tensor& self, const at::Scalar& other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_sum(
    const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {Shape(dtype.value(), {})};
  }
  // It's undocumented, but torch::sum promotes all integral types to int64_t by
  // default
  if (isIntegralType(self.scalar_type(), /*includeBool*/ true)) {
    return {Shape(c10::ScalarType::Long, {})};
  }
  return {Shape(self.scalar_type(), {})};;
}

std::vector<Shape> compute_shape_zero(at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_trace(const at::Tensor& self) {
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_sort(const at::Tensor & self, int64_t dim, bool descending) {
  return {Shape(self.scalar_type(), self.sizes().vec()),
          Shape(c10::ScalarType::Long, self.sizes().vec())};
}

std::vector<Shape> compute_shape_smooth_l1_loss(
    const at::Tensor& self, const at::Tensor& target, int64_t reduction,
    double beta) {
  // Taken from definition of 'Output' shape here:
  // https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
  switch (reduction) {
    case at::Reduction::None:
      return {Shape(self.scalar_type(), self.sizes().vec())};
    default:
      return {Shape(self.scalar_type(), {})};
  }
}

std::vector<Shape> compute_shape_smooth_l1_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, int64_t reduction, double beta) {
  // The `grad_output` tensor is really the input to this kernel, and while its
  // shape may vary following the logic of the forward output, the output of
  // this kernel should have fixed shapes matching the inputs to the forward
  // kernel.
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_logdet(const at::Tensor & self) {
  // assumes self.shape is {*, n, n} and returns shape *
  TORCH_INTERNAL_ASSERT(self.dim() >= 2);
  std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end() - 2);
  // Doesn't check input dtype, but output dtype either matches it,
  // or the actual logdet operation will throw if it's an unsupported type
  return {Shape(self.scalar_type(), out_sizes)};
}

std::vector<Shape> compute_shape_log_sigmoid_forward(const at::Tensor& self) {
  // Based on definition of aten/src/ATen/native/Activation.cpp::log_sigmoid_forward_out_cpu.
  return {Shape(self.scalar_type(), self.sizes().vec()),
          Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_log_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& buffer) {
  // Based on definition of aten/src/ATen/native/Activation.cpp::log_sigmoid_backward_cpu*.
  return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}

std::vector<Shape> compute_shape_nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  // Based on definition of aten/src/ATen/native/LossNLL2d.cpp:nll_loss2d_forward_cpu
  auto sizes =
      (reduction == at::Reduction::Reduction::None ? target.sizes().vec()
                                                   : std::vector<int64_t>{});
  return {Shape(self.scalar_type(), sizes), Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_nll_loss2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

} // namespace ops
} // namespace ir
} // namespace torch_lazy_tensors