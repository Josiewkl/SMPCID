# model settings
model = dict(
    type='SingleStageInsParsingDetector',
    backbone=dict(
        type='ResNet',
        depth=101,  #网络层数
        num_stages=4,  #resnet的stage数量
        out_indices=(0, 1, 2, 3), #输出的stage的序号
        frozen_stages=1, #冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        style='pytorch', #网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
        dcn=dict(
            type='DCNv2', #可变性卷积
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], #输入的各个stage的通道数
        out_channels=256, # FPN的各个特征图的输出通道数
        start_level=0, ## 从0号输入特征图开始fpn
        num_outs=5), ## 输出的特征层的数量 # 控制输出特征图个数
    mask_head=dict(
        type='CID_Parsing_basic',
        cfg=dict(
            CFG_NAME='cid-hrnet32',
            OUTPUT_DIR='runs',
            WORKERS=0,
            PRINT_FREQ=100,
            DIST_BACKEND='nccl',
            AUTO_RESUME=True,
            PIN_MEMORY=True,
            VERBOSE=False,
            DDP=False,
            CUDNN=dict(
                BENCHMARK=True,
                DETERMINISTIC=False,
                ENABLED=True),
            MODEL=dict(
                NAME='CID',
                DEVICE='cuda',
                INIT_WEIGHTS=True,
                PRETRAINED='model/imagenet/hrnet_w32-36af842e.pth',
                SYNC_BN=False,
                # BACKBONE=dict(
                #     PRETRAINED_LAYERS= ['*'],
                #     STAGES=dict(
                #         NUM_STAGES= 3,
                #         NUM_MODULES=[1, 4, 3],
                #         NUM_BRANCHES=[2, 3, 4],
                #         BLOCK=['BASIC', 'BASIC', 'BASIC'],
                #         NUM_BLOCKS=[[4, 4], [4, 4, 4], [4, 4, 4, 4]],
                #         NUM_CHANNELS=[[32, 64], [32, 64, 128], [32, 64, 128, 256]],
                #         FUSE_METHOD=['SUM', 'SUM', 'SUM'])),
                BIAS_PROB=0.01,
                IIA=dict(
                    IN_CHANNELS=1280,
                    OUT_CHANNELS=59,
                    OUT_RES=40),
                GFD=dict(
                    IN_CHANNELS=1280,
                    CHANNELS=32,
                    OUT_CHANNELS=58)),
            LOSS=dict(
                MULTI_HEATMAP_LOSS_WEIGHT=1.0,
                CONTRASTIVE_LOSS_WEIGHT=1.0,
                SINGLE_HEATMAP_LOSS_WEIGHT=4.0),
            DATASET=dict(
                ROOT= 'data/crowdpose',
                DATASET='crowdpose', 
                NUM_KEYPOINTS=58,
                MAX_INSTANCES=200,
                TRAIN='trainval',
                TEST='test',
                FILTER_IMAGE=False,
                SIGMA=2.0,
                FLIP=0.5,
                FLIP_INDEX=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13],
                MAX_ROTATION=30,
                MIN_SCALE=0.75,
                MAX_SCALE=1.5,
                SCALE_TYPE='short',
                MAX_TRANSLATE=40,
                INPUT_SIZE=512,
                OUTPUT_SIZE=128),
            TRAIN=dict(
                LR_SCHEDULER='step',
                LR_FACTOR=0.1,
                LR_STEP=[200, 260], 
                LR=0.001,
                OPTIMIZER='adam',
                MOMENTUM=0.9,
                WD=0.0001, 
                NESTEROV=False,
                GAMMA1=0.99,
                GAMMA2=0.0,
                BEGIN_EPOCH=0,
                END_EPOCH=300,
                RESUME=False,
                CHECKPOINT='',
                IMAGES_PER_GPU=20,
                SHUFFLE=True),
            TEST=dict(
                FLIP_TEST=True,
                IMAGES_PER_GPU=1,
                MODEL_FILE='',
                OKS_SCORE=0.8,
                OKS_SIGMAS=[0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.79, 0.79],
                MAX_PROPOSALS=30,
                KEYPOINT_THRESHOLD=0.01,
                CENTER_POOL_KERNEL=3,
                POOL_THRESHOLD1=300,
                POOL_THRESHOLD2=200),),
        num_classes=59,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        enable_center=True,
        enable_offset=True,
        enable_heatmaploss=False,
        enable_ori_grid= True, #['cate', 'kernel', False]
        enhance_cate = None, #['large kernel', 'dilation','deep_branch']/None
        enable_cate_decouple = False,
        enable_cate_eval = False, # apply orignal solo category segment inference method
        enable_moi = False,
        enable_keypoints = False,
        mask_feature_head=dict(
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_center=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_offset=dict(
            type='SmoothL1Loss',
            beta=1. / 9, 
            reduction='mean', 
            loss_weight=10.0)),
    train_cfg = dict(),
    test_cfg = dict(
        nms_pre=500,
        ctr_score=0.1,
        score_thr=0.1,
        cate_score_thr=0.3,
        mask_thr=0.5,
        cate_update_thr=0.1,
        update_thr=0.1,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100,
        debug_heatmap=False)

)


#_base_=['../_base_/default_runtime.py']
# dataset settings
dataset_type = 'MHP'
data_root = '/home/ubuntu/wyf/multi-parsing/dataset/LV-MHP-v2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_mhp'),
    dict(type='LoadAnnotations_mhp', with_mask=True, with_seg=True ,with_mhp_parsing=True, with_keypoints=True),
    dict(type='Resize_Parsing',
         img_scale=[(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip_Parsing', flip_ratio=0.5, direction='horizontal', dataset='mhp'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad_Parsing', size_divisor=32),
    dict(type='DefaultFormatBundle_Parsing'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_parsing', 'gt_semantic_seg', 'gt_keypoints'],
                         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile_mhp'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize_Parsing', keep_ratio=True),
            dict(type='RandomFlip_Parsing', flip_ratio=None, direction='horizontal', dataset='mhp'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad_Parsing', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/parsing_annos/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/parsing_annos/',
        img_prefix=data_root + 'val/images/',
        test_mode = True,
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/parsing_annos/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        data_root=data_root
        ))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.02,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])

device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/MHP_release_r101_fpn_8gpu_1x_offset_parsing_v1/ori_gird_DCN_1x/'
load_from = None
resume_from = None
workflow = [('train', 1)]