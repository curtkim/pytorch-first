# logs
#   checkpoints/
#       _metrics.json
#       best.pth, best_full.pth
#       last.pth, last_full.pth
#       train.1.pth, train.1_full.pth
#   logs
#       _epoch_.cvs
#       hparams.yml
#       train.csv
#       valid.csv
#   tensorboard

import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
from catalyst.contrib import MNIST

model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

train_data = MNIST(os.getcwd(), train=True)
valid_data = MNIST(os.getcwd(), train=False)
loaders = {
    "train": DataLoader(train_data, batch_size=32),
    "valid": DataLoader(valid_data, batch_size=32),
}

a, b = train_data[0]
print(a.shape)
print(b)

INPUT_KEY = "features"
OUTPUT_KEY = "logits"
TARGET_KEY = "targets"
LOSS_KEY = "loss"


runner = dl.SupervisedRunner(
    input_key=INPUT_KEY, output_key=OUTPUT_KEY, target_key=TARGET_KEY, loss_key=LOSS_KEY
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=2,
    callbacks=[
        dl.AccuracyCallback(input_key=OUTPUT_KEY, target_key=TARGET_KEY, topk_args=(1, 3, 5)),
        dl.PrecisionRecallF1SupportCallback(
            input_key=OUTPUT_KEY, target_key=TARGET_KEY, num_classes=10
        )
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    load_best_on_end=True,
)

# model evaluation
metrics = runner.evaluate_loader(
    loader=loaders["valid"],
    callbacks=[
        dl.AccuracyCallback(input_key=OUTPUT_KEY, target_key=TARGET_KEY, topk_args=(1, 3, 5)),
    ]
)
print(metrics.keys())
assert "accuracy01" in metrics.keys()


for prediction in runner.predict_loader(loader=loaders["valid"]):
    assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10


features_batch = next(iter(loaders['valid']))[0]
model.load_state_dict(utils.get_averaged_weights_by_path_mask(logdir="./logs", path_mask="*.pth"))

print("trace_model")
utils.trace_model(model=runner.model.cpu(), batch=features_batch)
print("quantize_model")
utils.quantize_model(model=runner.model)
print("prune_model")
utils.prune_model(model=runner.model, pruning_fn="l1_unstructured", amount=0.8)
print("onnx_export")
utils.onnx_export(model=runner.model.cpu(), batch=features_batch, file="./logs/mnist.onnx", verbose=True)
