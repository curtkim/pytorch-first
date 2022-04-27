import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
import common


model_path = "01_add.onnx"
input_name = "onnx::Add_0"
output_name = "2"


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_engine(network, config)


def main():
    engine = build_engine_onnx(model_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    input = np.arange(4)
    pagelocked_buffer = inputs[0].host
    np.copyto(pagelocked_buffer, input)

    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(input)
    print(trt_outputs[0])


if __name__ == '__main__':
    main()