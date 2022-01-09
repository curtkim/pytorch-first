## howto
    

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
