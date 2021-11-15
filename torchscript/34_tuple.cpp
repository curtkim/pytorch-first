#include <torch/script.h>
#include <tuple>

static const std::string file = "../../34_tuple.torchscript";

void test(torch::Device device){
  torch::jit::script::Module module = torch::jit::load(file, device);

  at::Tensor a = at::ones({3, 1}, at::device(device).dtype(at::kInt));
  at::Tensor b = at::ones({3, 1}, at::device(device).dtype(at::kInt));

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(a);
  inputs.push_back(b);
  inputs.push_back(std::tuple<int, float, bool>{1, 1.1, true});

  auto output = module.forward(inputs);
  std::cout << output << std::endl;
}

int main() {
  auto cpu = torch::Device(torch::kCPU, 0);
  test(cpu);

  auto gpu = torch::Device(torch::kCUDA, 0);
  test(gpu);
}
