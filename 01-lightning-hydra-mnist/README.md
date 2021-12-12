## console out

    venv/lib/python3.8/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'config.yaml': Defaults list is missing `_self_`. See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order for more information
      warnings.warn(msg, UserWarning)
    [2021-12-09 09:55:51,355][src.utils.utils][INFO] - Disabling python warnings! <config.ignore_warnings=True>
    [2021-12-09 09:55:51,379][src.train][INFO] - Instantiating datamodule <src.datamodules.mnist_datamodule.MNISTDataModule>
    [2021-12-09 09:55:51,383][src.train][INFO] - Instantiating model <src.models.mnist_model.MNISTLitModel>
    [2021-12-09 09:55:51,390][src.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.ModelCheckpoint>
    [2021-12-09 09:55:51,392][src.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
    [2021-12-09 09:55:51,393][src.train][INFO] - Instantiating trainer <pytorch_lightning.Trainer>
    [2021-12-09 09:55:51,436][pytorch_lightning.utilities.distributed][INFO] - GPU available: True, used: True
    [2021-12-09 09:55:51,437][pytorch_lightning.utilities.distributed][INFO] - TPU available: False, using: 0 TPU cores
    [2021-12-09 09:55:51,437][pytorch_lightning.utilities.distributed][INFO] - IPU available: False, using: 0 IPUs
    [2021-12-09 09:55:51,437][src.train][INFO] - Logging hyperparameters!
    [2021-12-09 09:55:51,438][src.train][INFO] - Starting training!
    [2021-12-09 09:55:51,510][pytorch_lightning.accelerators.gpu][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Epoch 0:  92%|█████████▏| 860/939 [00:16<00:01, 50.70it/s, loss=0.0949, v_num=]
    Epoch 0:  94%|█████████▎| 880/939 [00:17<00:01, 51.44it/s, loss=0.0949, v_num=]
    Epoch 0:  96%|█████████▌| 900/939 [00:17<00:00, 51.79it/s, loss=0.0949, v_num=]
    Epoch 0:  98%|█████████▊| 920/939 [00:17<00:00, 52.14it/s, loss=0.0949, v_num=]
    Epoch 0: 100%|██████████| 939/939 [00:18<00:00, 52.13it/s, loss=0.0949, v_num=, val/acc=0.970]


## sinle node multi gpu

    # localhost 2 gpu
    MASTER_ADDR=127.0.0.1 MASTER_PORT=6000 WORLD_SIZE=2 NODE_RANK=0 python run.py trainer=ddp trainer.gpus='[0]'
    MASTER_ADDR=127.0.0.1 MASTER_PORT=6000 WORLD_SIZE=2 NODE_RANK=1 python run.py trainer=ddp trainer.gpus='[1]'

    
## multi node
    
    #ddp.yaml 파일에 num_nodes가 있어야 한다.
    num_nodes: 2

    MASTER_ADDR=server1 MASTER_PORT=5824 WORLD_SIZE=2 NODE_RANK=0 python run.py trainer=ddp trainer.gpus='[0]'
    MASTER_ADDR=server1 MASTER_PORT=5824 WORLD_SIZE=2 NODE_RANK=1 python run.py trainer=ddp trainer.gpus='[0]'

    2 node and each 2 gpus
    MASTER_ADDR=server1 MASTER_PORT=port WORLD_SIZE=4 NODE_RANK=0 python run.py trainer=ddp trainer.gpus='[0,1]'
    MASTER_ADDR=server1 MASTER_PORT=port WORLD_SIZE=4 NODE_RANK=1 python run.py trainer=ddp trainer.gpus='[0,1]'


## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n myenv
conda activate myenv

# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```
