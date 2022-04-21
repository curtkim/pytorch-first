# error 발생함.
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch

model_path = "01_add.onnx"
input_name = "onnx::Add_0"
output_name = "2"


logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    pass # Error handling code here

config = builder.create_builder_config()
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
serialized_engine = builder.build_serialized_network(network, config)
with open("01-add.trt", "wb") as f:
    f.write(serialized_engine)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
with open("01-add.trt", "rb") as f:
    serialized_engine = f.read()


for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    print('binding:', binding, 'size:', size)

    #host_mem = cuda.pagelocked_empty(size, np.float32)
    #cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    #bindings.append(int(cuda_mem))
    #if engine.binding_is_input(binding):
    #    host_inputs.append(host_mem)
    #    cuda_inputs.append(cuda_mem)
    #else:
    #    host_outputs.append(host_mem)
    #    cuda_outputs.append(cuda_mem)


context = engine.create_execution_context()
# To perform inference, you must pass TensorRT buffers for inputs and outputs, which TensorRT requires you to specify in a list of GPU pointers.
# You can query the engine using the names you provided for input and output tensors to find the right positions in the array
input_idx = engine[input_name]
output_idx = engine[output_name]

# Using these indices, set up GPU buffers for each input and output.
# Several Python packages allow you to allocate memory on the GPU, including,
# but not limited to, PyTorch, the Polygraphy CUDA wrapper, and PyCUDA.

input = torch.arange(4)
output = torch.tensor([0, 0, 0, 0])

buffers = [None] * 2 # Assuming 1 input and 1 output
buffers[input_idx] = input.data_ptr()
buffers[output_idx] = output.data_ptr()

cuda_ctx = cuda.Device(0).make_context()
cuda_ctx.push()

stream = cuda.Stream()
context.execute_async_v2(buffers, stream.handle)
#context.execute_async(
#    batch_size=1,
#    bindings=bindings,
#    stream_handle=stream.handle)
stream.synchronize()
cuda_ctx.pop()

print('input', input)
print('output', output)

del stream
