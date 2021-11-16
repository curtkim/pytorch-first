#include <torch/script.h>
#include <tuple>

static const std::string file = "../../33_namedtuple.torchscript";

// 아래와 같은 에러가 발생한다.
// terminate called after throwing an instance of 'c10::Error'
// what():  forward() Expected a value of type '__torch__.Point' for argument 'point' but instead found type 'Tuple[Tensor, Tensor]'.
// Position: 1
// Declaration: forward(__torch__.PlaceholderModule self, __torch__.Point point) -> (Tensor)

int main() {
  auto device = torch::Device(torch::kCPU, 0);

  torch::jit::script::Module module = torch::jit::load(file, device);

  std::vector<torch::jit::IValue> inputs = {
      std::tuple<at::Tensor, at::Tensor>{
          at::ones({3, 1}, at::device(device).dtype(at::kInt)),
          at::ones({3, 1}, at::device(device).dtype(at::kInt))
      }
  };

  auto output = module.forward(inputs);
  std::cout << output << std::endl;
}
