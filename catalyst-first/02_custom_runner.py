import os
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl, metrics
from catalyst.contrib import MNIST

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
optimizer = optim.Adam(model.parameters(), lr=0.02)

train_data = MNIST(os.getcwd(), train=True)
valid_data = MNIST(os.getcwd(), train=False)
loaders = {
    "train": DataLoader(train_data, batch_size=32),
    "valid": DataLoader(valid_data, batch_size=32),
}


class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device))

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "accuracy01", "accuracy03"]
        }

    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        x, y = batch
        # run model forward pass
        logits = self.model(x)
        # compute the loss
        loss = F.cross_entropy(logits, y)
        # compute the metrics
        accuracy01, accuracy03 = metrics.accuracy(logits, y, topk=(1, 3))
        # log metrics
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
        )
        for key in ["loss", "accuracy01", "accuracy03"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        # run model backward pass
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in ["loss", "accuracy01", "accuracy03"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


runner = CustomRunner()
# model training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logs",
    num_epochs=1,
    verbose=True,
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
)
# model inference
for logits in runner.predict_loader(loader=loaders["valid"]):
    assert logits.detach().cpu().numpy().shape[-1] == 10
