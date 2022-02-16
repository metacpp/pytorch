#include <ATen/native/nestedtensor/nested_tensor_impl.h>
#include <ATen/native/nestedtensor/nested_node.h>

namespace at {
namespace native {

at::Tensor nested_tensor_constructor(
    at::TensorList list,
    at::ScalarType dtype,
    at::Device device,
    // bool requires_grad,
    bool pin_memory,
    bool channels_last) {
  std::vector<torch::nested_tensor::TensorNode> list_nodes;
  for (size_t i = 0; i < list.size(); i++) {
    at::Tensor tmp_tensor = list[i];
    list_nodes.push_back(torch::nested_tensor::TensorNode(std::move(tmp_tensor)));
  }
  Tensor result = wrap_tensor_node(torch::nested_tensor::TensorNode(std::move(list_nodes)));
  Tensor buffer = get_buffer(result);
  buffer = buffer.to(device, dtype);
  if (pin_memory) {
    buffer = buffer.pin_memory();
  }
  return wrap_buffer(std::move(buffer), get_efficient_nested_size(result));
}

}
}
