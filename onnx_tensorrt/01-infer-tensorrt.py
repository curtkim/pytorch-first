# error 발생함.
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda   # for cuda.Stream
import torch


model_path = "01_add.onnx"
input_name = "onnx::Add_0"
output_name = "2"


logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
print("flag", flags)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    pass # Error handling code here

config = builder.create_builder_config()
config.max_workspace_size = 1 << 20 # 1 MiB
serialized_engine = builder.build_serialized_network(network, config)
with open("01-add.trt", "wb") as f:
    f.write(serialized_engine)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)


for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    print('binding:', binding, 'size:', size)


context = engine.create_execution_context()

input_idx = engine[input_name]
output_idx = engine[output_name]

input = torch.arange(4, dtype=torch.float32, device='cuda')
output = torch.tensor([0, 0, 0, 0], dtype=torch.float32, device='cuda')

buffers = [None] * 2 # Assuming 1 input and 1 output
buffers[input_idx] = input.data_ptr()
buffers[output_idx] = output.data_ptr()

stream = cuda.Stream()
context.execute_async_v2(buffers, stream.handle)
stream.synchronize()

print('input', input)
print('output', output)
