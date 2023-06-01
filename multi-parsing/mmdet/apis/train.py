# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg) #用于确保配置文件与当前代码版本兼容
    logger = get_root_logger(log_level=cfg.log_level) #设置一个记录有关训练过程信息的日志;get_root_logger函数就是获取一个logger对象，之后打印输出信息都是用它

    # prepare data loaders准备数据加载器
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset] #确保数据集是列表或元组

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type'] #设置要用于训练的运行程序类型

    train_dataloader_default_args = dict(
        samples_per_gpu=2,   #batch_size = samples_per_gpu,sampers_per_gpu是决定每张卡取多少sampers
        workers_per_gpu=2,   #num_workers = workers_per_gpu,workers_per_gpu决定读取数据时每个gpu分配的线程数
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset] #为列表中的每个数据集构建数据加载器

    # put model on gpus
    if distributed:
        # TODO: find_ununsed_parameters can be setup as False to improve trainning speed.
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp( #将模型构建成分布式数据并行模型
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids) #将模型构建成数据并行模型

    # build optimizer构建优化器
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(   #runner就是一个运行器，包含一堆和训练有关的属性和函数
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same 把时间戳赋值给runner的timestamp属性
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks寄存器挂钩,创建了学习率更新hook，优化器hook，检查点hook，日志hook，还有动量hook
    # lr hook：学习率更新hook，它主要在训练过程中执行学习率更新动作，有时候训练过程中学习率会随着epoch的增加而逐渐衰减，他就是完成这个衰减功能的
    # optimizer hook：优化器hook，它主要执行优化器需要干的事情
    # checkpoint hook：检查点hook，它主要是在训练过程中保存检查点文件
    # log hook：日志hook，它负责在训练过程中打印日志；
    # momentum hook：动量hook，它负责训练过程中更新动量等关于动量的操作        
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook()) #DistSamplerSeedHook是一个训练钩子，在分布式训练中用于设置数据加载器（dataloader）的种子（seed）
            #以确保每个进程使用的随机数种子不同，从而避免数据加载器产生重复的数据

    # register eval hooks注册评估钩子，用于记录训练时每个迭代的时间
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False) #判断是否需要进行模型验证（评估）

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation
        # 如果batch_size大于1，就需要对数据处理做一些修改，但是我们往往设置val_samples_per_gpu为1，
        # 也就是下面这个if里面的东西不会执行

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        # 构建验证数据集和dataloader        
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook  #获取验证相关的一些配置信息，如果是分布式就注册DistEvalHook，如果不是就注册EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW') #使用EvalHook来进行模型评估，代码中传入的参数包括val_dataloader（验证集数据加载器）、eval_cfg（评估配置）和priority（hook的优先级）等
            #EvalHook是一个钩子函数，可以在训练过程中定期对模型进行验证评估，priority参数指定了该hook的优先级，'LOW'表示该hook的优先级较低,比如每隔5个epoch就跑一次val数据集的推理，然后评估结果，Eval hook就是完成这些操作的

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    # 判断一下是否从指定从某个checkpoint重新执行或者加载checkpoint继续执行
    if cfg.resume_from:
        runner.resume(cfg.resume_from) # 加载resume的checkpoint，并设置一些基础信息
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from) # 加载checkpoint的参数
    runner.run(data_loaders, cfg.workflow) #开始训练模型 
