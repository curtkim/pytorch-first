# error 발생
import torch

from effdet.bench import _post_process, _batch_detection

print(type(_batch_detection))
_batch_detection.save('_batch_detection.torchscript')

scripted = torch.jit.script(_post_process)
scripted.save('_post_process.torchscript')

