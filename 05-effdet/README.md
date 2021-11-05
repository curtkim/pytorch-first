## 개요
- effdet를 torchlighting으로 training
- _to_torchscript로 변환 (성공)
  torch.jit.script으로 변환 effdet.HeadNet forward가 변환되지 않아서 코드를 수정함.
- torch.jit.trace로 시도 (코드 수정없이)
  어려울듯. warning이 나오기는 했는데, 파일은 생성됨.
- predictor.py torchscript화된 effdet.EfficentNet만을 로드해서
  DetBenchTrain의 일부기능과 EfficientDetModel 일부기능을 빼냄 


## Note
- DetBenchTrain의 결과는
  loss, class_loss, box_loss를 반환하고
  training이 아닐때는 detections를 추가로 반환한다.
  _post_process: top-k를 찾고
  _batch_detection: nms 하는 것 같은데
  

## Reference
- https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f
- https://amaarora.github.io/2021/01/13/efficientdet-pytorch.html

## step

    python train.py 
    # -> trained_effdet 생성
    python _to_torchscript.py
    # -> scripted.torchscript

    trained_effdet : 450M
    scripted.torchscript: 449M


## torchscript
#### torch.jit.script:

#### torch.jit.ignore: 
- function or method should be ignored and left as a Python function
- This allows you to leave code in your model that is not yet TorchScript compatible
- If called from TorchScript, ignored functions will dispatch the call to the Python interpreter
- Models with ignored functions cannot be exported. use @torch.jit.unused instead
#### torch.jit.unused
- This decorator indicates to the compiler that a function or method should be ignored and replaced with the raising of an exception
- This allows you to leave code in your model that is not yet TorchScript compatible and still export your model


## config

        {
        'name': 'tf_efficientnetv2_l', 
        'backbone_name': 'tf_efficientnetv2_l', 
        'backbone_args': {'drop_path_rate': 0.2}, 
        'backbone_indices': None, 
        'image_size': [512, 512], 
        'num_classes': 1, 
        'min_level': 3, 
        'max_level': 7, 
        'num_levels': 5, 
        'num_scales': 3, 
        'aspect_ratios': [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]], 
        'anchor_scale': 4.0, 
        'pad_type': 'same', 
        'act_type': 'swish', 
        'norm_layer': None, 
        'norm_kwargs': {'eps': 0.001, 'momentum': 0.01}, 
        'box_class_repeats': 3, 
        'fpn_cell_repeats': 3, 
        'fpn_channels': 88, 
        'separable_conv': True, 
        'apply_resample_bn': True, 
        'conv_after_downsample': False, 
        'conv_bn_relu_pattern': False, 
        'use_native_resize_op': False, 
        'downsample_type': 'max', 
        'upsample_type': 'nearest', 
        'redundant_bias': True, 
        'head_bn_level_first': False, 
        'head_act_type': None, 
        'fpn_name': None, 
        'fpn_config': None, 
        'fpn_drop_path_rate': 0.0, 
        'alpha': 0.25, 
        'gamma': 1.5, 
        'label_smoothing': 0.0, 
        'legacy_focal': False, 
        'jit_loss': False, 
        'delta': 0.1, 
        'box_loss_weight': 50.0, 
        'soft_nms': False, 
        'max_detection_points': 5000, 
        
        'max_det_per_image': 100,         # 최대 몇개의 bouding box 
        'url': ''
        }