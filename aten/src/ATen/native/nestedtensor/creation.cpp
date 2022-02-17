#include <ATen/native/nestedtensor/nested_tensor_impl.h>

namespace at {
namespace native {

at::Tensor nested_tensor_constructor(
    at::TensorList list,
    at::ScalarType dtype,
    at::Device device,
    bool pin_memory,
    bool channels_last) {
  std::vector<Tensor> sizes;
  std::vector<Tensor> flat_tensors;
  for (size_t i = 0; i < list.size(); i++) {
    flat_tensors.push_back(list[i].reshape(-1).contiguous());
    sizes.push_back(at::tensor(c10::IntArrayRef(list[i].sizes())));
  }
  Tensor buffer = at::cat(at::TensorList(flat_tensors));
  buffer = buffer.to(device, dtype);
  if (pin_memory) {
    buffer = buffer.pin_memory();
  }
  return wrap_buffer(std::move(buffer),
                     EfficientSizeNode(list.size(),
                                       at::stack(at::TensorList(sizes))));
}

}
}
