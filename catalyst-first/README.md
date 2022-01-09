## howto

    CUDA_VISIBLE_DEVICES="0" python 03_cifar_ddp.py

    # 아래방법 실패함.
    MASTER_ADDR=localhost MASTER_PORT=23373 WORLD_SIZE=2 NODE_RANK=0 python 05_cifar_custom_all.py
    MASTER_ADDR=localhost MASTER_PORT=23373 WORLD_SIZE=2 NODE_RANK=1 python 05_cifar_custom_all.py


## ICallback

    experiment / stage / epoch / loader / batch


    on_experiment_start
        on_stage_start
            on_epoch_start
                on_loader_start
                    on_batch_start
                    on_batch_end
                on_loader_end
            on_epoch_end
        on_stage_end
    on_experiment_end
    on_exception

## IRunner
- engine
- model
- criterion
- optimizer
- scheduler
- 
- stage_key
- global_epoch_step
- global_batch_step
- global_sample_step
- is_infer_stage
- is_train_loader
- 
- batch_metrics
- loader_metrics
- epoch_metrics

- engine.is_ddp
- engine.is_master_process
