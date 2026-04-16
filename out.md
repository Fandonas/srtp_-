(zccpytorch) D:\pyproject\CLIP-FSAR-master>python runs/run.py --cfg configs/projects/CLIPFSAR/hmdb51/HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot.yaml
C:\Users\ASUS\.conda\envs\zccpytorch\lib\site-packages\torchvision\transforms\_functional_video.py:6: UserWarning: The 'torchvision.transforms._functional_video' module is deprecated since 0.12 and will be removed in the future. Please use the 'torchvision.transforms.functional' module instead.
  warnings.warn(
C:\Users\ASUS\.conda\envs\zccpytorch\lib\site-packages\torchvision\transforms\_transforms_video.py:22: UserWarning: The 'torchvision.transforms._transforms_video' module is deprecated since 0.12 and will be removed in the future. Please use the 'torchvision.transforms' module instead.
  warnings.warn(
D:\pyproject\CLIP-FSAR-master\models\base\few_shot.py:41: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import packaging
Loading config from configs/projects/CLIPFSAR/hmdb51/HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot.yaml.
[10/31 22:54:29][INFO] train_net_few_shot:  473: Train with config:
[10/31 22:54:29][INFO] train_net_few_shot:  474: {
  "TASK_TYPE": "few_shot_action",
  "PRETRAIN": {
    "ENABLE": false
  },
  "LOCALIZATION": {
    "ENABLE": false
  },
  "TRAIN": {
    "ENABLE": true,
    "DATASET": "Ssv2_few_shot",
    "BATCH_SIZE": 1,
    "LOG_FILE": "training_log.log",
    "EVAL_PERIOD": 2,
    "NUM_FOLDS": 1,
    "AUTO_RESUME": true,
    "CHECKPOINT_PERIOD": 10,
    "CHECKPOINT_FILE_PATH": "",
    "CHECKPOINT_TYPE": "pytorch",
    "CHECKPOINT_INFLATE": false,
    "CHECKPOINT_PRE_PROCESS": {
      "ENABLE": false
    },
    "FINE_TUNE": true,
    "ONLY_LINEAR": false,
    "LR_REDUCE": false,
    "TRAIN_VAL_COMBINE": false,
    "LOSS_FUNC": "cross_entropy",
    "USE_CLASSIFICATION": true,
    "USE_CLASSIFICATION_VALUE": 1.2,
    "DATASET_FEW": "HMDB_few_shot",
    "META_BATCH": true,
    "NUM_SAMPLES": 1000000,
    "WAY": 5,
    "SHOT": 1,
    "QUERY_PER_CLASS": 2,
    "QUERY_PER_CLASS_TEST": 1,
    "NUM_TRAIN_TASKS": 5000,
    "NUM_TEST_TASKS": 10000,
    "VAL_FRE_ITER": 300,
    "BATCH_SIZE_PER_TASK": 1,
    "SEMANTIC_LOSS_WEIGHT": 0.5,
    "SEMANTIC_TEMPERATURE": 0.1,
    "CLASS_NAME": [
      "Running fingers or brush through hair repeatedly, then moving from scalp to ends in smooth strokes",
      "Following the trajectory of an airborne object, then hands successfully intercept and secure it", 
      "Placing food in mouth, then jaw moves rhythmically to process and break down the substance",      
      "Bringing palms together with force, then creating a sharp sound that echoes through the space",   
      "Reaching upward to find secure grips, then pushing with feet to lift body weight higher",
      "Stepping onto the first level, then continuing to ascend each subsequent step to reach the top",  
      "Positioning at the platform edge, then launching headfirst into the water below",
      "Grasping the sword handle firmly, then drawing the blade free from its protective sheath",        
      "Dropping the ball to the ground, then it bounces back up to be caught and released again",        
      "Raising the container to mouth level, then allowing liquid to flow smoothly into the opening",    
      "Losing balance and control, then the body falls to the ground surface",
      "Beginning in upright position, then arching backward to complete a full backward somersault",     
      "Planting hands firmly on ground, then raising legs to balance entire body weight upside down",    
      "Opening arms wide, then closing them around another in a warm, protective embrace",
      "Bending knees to prepare, then exploding upward with leg power to reach maximum height",
      "Moving face close to another, then lips meet in a brief, tender connection",
      "Gripping the horizontal bar, then pulling with arms until the body rises above the bar",
      "Drawing arm back to build power, then releasing a forceful punch toward the target",
      "Placing hands against the object, then applying steady pressure to push the object away",
      "Settling into the bicycle seat, then using legs to pedal and hands to guide direction",
      "Mounting the horses back, then using legs and reins to direct the animals movement",
      "Extending hand forward, then grasping anothers hand and moving it in greeting",
      "Drawing the bowstring back with tension, then releasing to send arrow toward target",
      "Lying flat on the surface, then contracting core muscles to raise torso upward",
      "Rising from a lower position, then standing upright until the body is balanced on feet",
      "Wielding the sword with purpose, then moving it through various combat-ready positions",
      "Holding the sword steadily, then executing a series of coordinated training movements",
      "Cocking the arm back behind the body, then whipping forward to launch the object",
      "Rotating around a central axis, then continuing the circular motion in smooth rhythm",
      "Lifting one foot from ground, then placing it forward to advance the bodys position",
      "Raising hand above the head, then moving it back and forth to create a visible signal"
    ]
  },
  "TEST": {
    "ENABLE": false,
    "DATASET": "Ssv2_few_shot",
    "BATCH_SIZE": 4,
    "NUM_SPATIAL_CROPS": 1,
    "SPATIAL_CROPS": "cctltr",
    "NUM_ENSEMBLE_VIEWS": 1,
    "LOG_FILE": "val.log",
    "CHECKPOINT_FILE_PATH": "",
    "CHECKPOINT_TYPE": "pytorch",
    "AUTOMATIC_MULTI_SCALE_TEST": false,
    "TEST_SET": "val",
    "UPLOAD_CLASSIFIER_RESULTS": true,
    "CLASS_NAME": [
      "Holding the sword in combat stance, then executing precise thrusting and parrying movements",     
      "Lifting the leg high, then snapping it forward to strike with precise force",
      "Approaching the ball with foot raised, then making contact to send the ball flying",
      "Reaching toward the object, then fingers close around it to lift the object away",
      "Tilting the container at an angle, then liquid flows smoothly into the receiving vessel",
      "Placing hands on the ground, then lowering and raising the body through arm strength",
      "Moving legs in rapid succession, then covering distance quickly with each forward step",
      "Bending at the knees and waist, then the body settles into a seated position",
      "Holding the smoking device, then inhaling and exhaling smoke in controlled rhythm",
      "Opening the mouth to form words, then vocal cords produce sounds for communication"
    ]
  },
  "VISUALIZATION": {
    "ENABLE": false,
    "NAME": "",
    "FEATURE_MAPS": {
      "ENABLE": false,
      "BASE_OUTPUT_DIR": ""
    }
  },
  "SUBMISSION": {
    "ENABLE": false,
    "SAVE_RESULTS_PATH": "test.json"
  },
  "DATA": {
    "DATA_ROOT_DIR": "D:\\pyproject\\ActionCLIP-master\\data\\hmdb51",
    "ANNO_DIR": "./configs/projects/CLIPFSAR/hmdb51/",
    "NUM_INPUT_FRAMES": 4,
    "NUM_INPUT_CHANNELS": 2304,
    "SAMPLING_MODE": "interval_based",
    "SAMPLING_RATE": 50,
    "TRAIN_JITTER_SCALES": [
      256,
      256
    ],
    "TRAIN_CROP_SIZE": 224,
    "TEST_SCALE": 256,
    "TEST_CROP_SIZE": 224,
    "MEAN": [
      0.48145466,
      0.4578275,
      0.40821073
    ],
    "STD": [
      0.26862954,
      0.26130258,
      0.27577711
    ],
    "MULTI_LABEL": false,
    "ENSEMBLE_METHOD": "sum",
    "TARGET_FPS": 12,
    "MINUS_INTERVAL": false,
    "FPS": 12,
    "NORM_FEATURE": false,
    "USE_AUG_FEATURE": false,
    "AUG": false,
    "LOAD_PROPS": false,
    "TEMPORAL_SCALE": 256,
    "LABELS_TYPE": "cls",
    "LOAD_TYPE": "pickle",
    "DOWNLOAD_FEATURE": true,
    "SAMPLING_UNIFORM": false,
    "CLIP_INTERVAL": 8
  },
  "MODEL": {
    "NAME": "BaseVideoModel",
    "EMA": {
      "ENABLE": false,
      "DECAY": 0.99996
    }
  },
  "VIDEO": {
    "BACKBONE": {
      "DEPTH": null,
      "META_ARCH": "Identity",
      "NUM_FILTERS": null,
      "NUM_INPUT_CHANNELS": 3,
      "NUM_OUT_FEATURES": null,
      "KERNEL_SIZE": null,
      "DOWNSAMPLING": null,
      "DOWNSAMPLING_TEMPORAL": null,
      "NUM_STREAMS": 1,
      "EXPANSION_RATIO": 2,
      "BRANCH": {
        "NAME": null
      },
      "STEM": {
        "NAME": null
      },
      "NONLOCAL": {
        "ENABLE": false,
        "STAGES": [
          5
        ],
        "MASK_ENABLE": false
      },
      "INITIALIZATION": null
    },
    "HEAD": {
      "NAME": "CNN_SEMANTIC_ALIGNMENT_FEW_SHOT",
      "ACTIVATION": "softmax",
      "DROPOUT_RATE": 0.0,
      "NUM_CLASSES": 5,
      "BACKBONE_NAME": "RN50"
    },
    "DIM1D": 256,
    "DIM2D": 128,
    "DIM3D": 512,
    "BACKBONE_LAYER": 2,
    "BACKBONE_GROUPS_NUM": 4
  },
  "OPTIMIZER": {
    "ADJUST_LR": false,
    "BASE_LR": 0.002,
    "LR_POLICY": "cosine",
    "MAX_EPOCH": 300,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": "1e-3",
    "WARMUP_EPOCHS": 10,
    "WARMUP_START_LR": 0.0002,
    "OPTIM_METHOD": "adam",
    "DAMPENING": 0.0,
    "NESTEROV": true
  },
  "BN": {
    "WB_LOCK": false,
    "FREEZE": false,
    "WEIGHT_DECAY": 0.0,
    "MOMENTUM": 0.1,
    "EPS": "1e-3",
    "SYNC": false
  },
  "DATA_LOADER": {
    "NUM_WORKERS": 0,
    "PIN_MEMORY": false,
    "ENABLE_MULTI_THREAD_DECODE": false,
    "COLLATE_FN": null
  },
  "NUM_GPUS": 1,
  "SHARD_ID": 0,
  "NUM_SHARDS": 1,
  "RANDOM_SEED": 42,
  "OUTPUT_DIR": "output/CLIPFSAR_HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot",
  "OUTPUT_CFG_FILE": "configuration.log",
  "LOG_PERIOD": 50,
  "DIST_BACKEND": "nccl",
  "LOG_MODEL_INFO": true,
  "LOG_CONFIG_INFO": true,
  "OSS": {
    "ENABLE": false,
    "KEY": null,
    "SECRET": null,
    "ENDPOINT": null,
    "CHECKPOINT_OUTPUT_PATH": null,
    "SECONDARY_DATA_OSS": {
      "ENABLE": false,
      "KEY": null,
      "SECRET": null,
      "ENDPOINT": null,
      "BUCKETS": [
        ""
      ]
    }
  },
  "AUGMENTATION": {
    "COLOR_AUG": false,
    "BRIGHTNESS": 0.5,
    "CONTRAST": 0.5,
    "SATURATION": 0.5,
    "HUE": 0.25,
    "GRAYSCALE": 0.3,
    "CONSISTENT": true,
    "SHUFFLE": true,
    "GRAY_FIRST": true,
    "RATIO": [
      0.857142857142857,
      1.1666666666666667
    ],
    "USE_GPU": false,
    "MIXUP": {
      "ENABLE": false,
      "ALPHA": 0.0,
      "PROB": 1.0,
      "MODE": "batch",
      "SWITCH_PROB": 0.5
    },
    "CUTMIX": {
      "ENABLE": false,
      "ALPHA": 0.0,
      "MINMAX": null
    },
    "RANDOM_ERASING": {
      "ENABLE": false,
      "PROB": 0.25,
      "MODE": "const",
      "COUNT": [
        1,
        1
      ],
      "NUM_SPLITS": 0,
      "AREA_RANGE": [
        0.02,
        0.33
      ],
      "MIN_ASPECT": 0.3
    },
    "LABEL_SMOOTHING": 0.0,
    "SSV2_FLIP": false,
    "IS_SPLIT": false,
    "NO_RANDOM_ERASE": true
  },
  "PAI": false,
  "USE_MULTISEG_VAL_DIST": false,
  "DETECTION": {
    "ENABLE": false
  },
  "TENSORBOARD": {
    "ENABLE": false
  },
  "SOLVER": {
    "BASE_LR": 1e-05,
    "LR_POLICY": "steps_with_relative_lrs",
    "MAX_EPOCH": 10,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": "5e-5",
    "WARMUP_EPOCHS": 1,
    "WARMUP_START_LR": "1e-06",
    "OPTIM_METHOD": "adam",
    "DAMPENING": 0.0,
    "NESTEROV": true,
    "STEPS_ITER": 700,
    "STEPS": [
      0,
      4,
      6
    ],
    "LRS": [
      1,
      0.1,
      0.01
    ]
  },
  "PRE_DOWNLOAD": {
    "ENABLE": false
  }
}

[10/31 22:54:31][INFO] models.base.few_shot: 2725: ✓ 使用默认 PROMPT: 'a photo of {}'
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\ASUS\.conda\envs\zccpytorch\lib\logging\__init__.py", line 1086, in emit
    stream.write(msg + self.terminator)
UnicodeEncodeError: 'gbk' codec can't encode character '\u2713' in position 51: illegal multibyte sequence
Call stack:
  File "D:\pyproject\CLIP-FSAR-master\runs\run.py", line 104, in <module>
    main()
  File "D:\pyproject\CLIP-FSAR-master\runs\run.py", line 98, in main
    launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])
  File "D:\pyproject\CLIP-FSAR-master\utils\launcher.py", line 36, in launch_task
    func(cfg=cfg)
  File "D:\pyproject\CLIP-FSAR-master\runs\train_net_few_shot.py", line 477, in train_few_shot
    model, model_ema = build_model(cfg)
  File "D:\pyproject\CLIP-FSAR-master\models\base\builder.py", line 32, in build_model
    model = BaseVideoModel(cfg)
  File "D:\pyproject\CLIP-FSAR-master\models\base\models.py", line 40, in __init__
    self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
  File "D:\pyproject\CLIP-FSAR-master\models\base\semantic_alignment_few_shot.py", line 36, in __init__  
    super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).__init__(cfg)
  File "D:\pyproject\CLIP-FSAR-master\models\base\few_shot.py", line 2725, in __init__
    logger.info("✓ 使用默认 PROMPT: 'a photo of {}'")
Message: "✓ 使用默认 PROMPT: 'a photo of {}'"
Arguments: ()
[10/31 22:54:31][INFO] models.base.few_shot: 2759: CNN_OTAM_CLIPFSAR initialized with default fusion weights (no TEXT_COFF parameter)
[10/31 22:54:31][INFO] utils.misc:  155: Model:
BaseVideoModel(
  (backbone): Identity()
  (head): CNN_SEMANTIC_ALIGNMENT_FEW_SHOT(
    (backbone): ModifiedResNet(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU(inplace=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU(inplace=True)
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu3): ReLU(inplace=True)
      (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
          (downsample): Sequential(
            (-1): AvgPool2d(kernel_size=1, stride=1, padding=0)
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
          (downsample): Sequential(
            (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu3): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
          (downsample): Sequential(
            (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
          (downsample): Sequential(
            (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)       
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        
          (relu2): ReLU(inplace=True)
          (avgpool): Identity()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)       
          (relu3): ReLU(inplace=True)
        )
      )
      (attnpool): AttentionPool2d(
        (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
        (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
        (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
        (c_proj): Linear(in_features=2048, out_features=1024, bias=True)
      )
    )
    (mid_layer): Sequential()
    (classification_layer): Sequential()
    (context2): Transformer_v1(
      (layers): ModuleList(
        (0): ModuleList(
          (0): PreNormattention_qkv(
            (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (fn): Attention_qkv(
              (attend): Softmax(dim=-1)
              (to_q): Linear(in_features=1024, out_features=1024, bias=False)
              (to_k): Linear(in_features=1024, out_features=1024, bias=False)
              (to_v): Linear(in_features=1024, out_features=1024, bias=False)
              (to_out): Sequential(
                (0): Linear(in_features=1024, out_features=1024, bias=True)
                (1): Dropout(p=0.2, inplace=False)
              )
            )
          )
          (1): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=1024, out_features=2048, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.05, inplace=False)
              (3): Linear(in_features=2048, out_features=1024, bias=True)
              (4): Dropout(p=0.05, inplace=False)
            )
          )
        )
      )
    )
  )
)
[10/31 22:54:31][INFO] utils.misc:  156: Params: 46,711,649
[10/31 22:54:31][INFO] utils.misc:  157: Mem: 0.46422863006591797 MB
[10/31 22:54:31][INFO] utils.misc:  164: nvidia-smi
Fri Oct 31 22:54:31 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.65                 Driver Version: 576.65         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   46C    P0             20W /  160W |    3047MiB /  16303MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2628      C   ...da\envs\zccpytorch\python.exe      N/A      |
|    0   N/A  N/A            4236    C+G   D:\app\QQ\QQ.exe                      N/A      |
|    0   N/A  N/A            4648    C+G   ....0.3537.99\msedgewebview2.exe      N/A      |
|    0   N/A  N/A            7292    C+G   D:\app\WeMeet\wemeetapp.exe           N/A      |
|    0   N/A  N/A            9148    C+G   ...crosoft OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A            9228    C+G   D:\app\Trae\Trae.exe                  N/A      |
|    0   N/A  N/A            9316    C+G   D:\app\WeMeet\wemeetapp.exe           N/A      |
|    0   N/A  N/A           11588    C+G   ...IA app\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           13372    C+G   ...yb3d8bbwe\Notepad\Notepad.exe      N/A      |
|    0   N/A  N/A           14808    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           15084    C+G   ...indows\System32\ShellHost.exe      N/A      |
|    0   N/A  N/A           15284    C+G   ...IA app\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           18864    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           19208    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           19216    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           19248    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A           19276    C+G   ...cw5n1h2txyewy\WidgetBoard.exe      N/A      |
|    0   N/A  N/A           20316    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           20660    C+G   ...mba6cd70vzyy\ArmouryCrate.exe      N/A      |
|    0   N/A  N/A           22568    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           22956    C+G   ...8wekyb3d8bbwe\M365Copilot.exe      N/A      |
|    0   N/A  N/A           25032    C+G   ...ef.win7x64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A           26032    C+G   ....0.3537.99\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           27256    C+G   ....0.3537.99\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           30988    C+G   ...t\3.37.10.401\XnnExternal.exe      N/A      |
|    0   N/A  N/A           31804    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           31816    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+
[10/31 22:54:31][INFO] models.utils.optimizer:   83: Optimized parameters constructed. Parameters without weight decay: []
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  110: Reading video list from file: train_few_shot.txt
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  145: Loading HMDB_few_shot dataset list for split 'train'...
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:   57: loaded 4280 videos from train dataset: HMDB_few_shot !
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  171: Dataset HMDB_few_shot split train loaded. Length 4280.
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  110: Reading video list from file: test_few_shot.txt
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  145: Loading HMDB_few_shot dataset list for split 'test'...
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:   57: loaded 1292 videos from test dataset: HMDB_few_shot !
[10/31 22:54:31][INFO] datasets.base.ssv2_few_shot:  171: Dataset HMDB_few_shot split test loaded. Length 1292.
[10/31 22:54:31][INFO] train_net_few_shot:  511: Mixup/cutmix disabled.
[10/31 22:54:31][INFO] train_net_few_shot:  523: Start epoch: 1
[10/31 22:54:31][INFO] train_net_few_shot:   55: Norm training: True
D:\pyproject\CLIP-FSAR-master\runs\train_net_few_shot.py:140: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:836.)
  if math.isnan(loss):
Debug - Similarity matrix range (after temperature): [-0.0498, 0.4171]
Debug - Semantic temperature: 0.1
Debug - OTAM output: -0.6784
Debug - Similarity matrix range (after temperature): [0.0678, 0.2590]
Debug - Semantic temperature: 0.1
Debug - OTAM output: -0.7365
Debug - Similarity matrix range (after temperature): [0.2194, 0.4923]
Debug - Semantic temperature: 0.1
Debug - OTAM output: -0.7982