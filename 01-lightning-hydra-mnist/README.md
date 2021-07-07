## Description
What it does

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
