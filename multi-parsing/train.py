# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')#argparse:命令行解析模块，description描述这个程度做什么以及怎么做
    parser.add_argument('config', help='train config file path') #调用 add_argument() 方法添加参数给 ArgumentParser 添加程序参数信息
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',#action:命令行遇到参数时的动作，默认值是 store
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        default=True, #default:不指定参数时的默认值
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int, #type:命令行参数应该被转换成的类型。
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+', #nargs:应该读取的命令行参数个数, * 号，表示 0 或多个参数； + 号表示 1 或多个参数
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args() #使用 parse_args() 解析添加的参数,将检查命令行，把每个参数转换为适当的类型然后调用相应的操作
    if 'LOCAL_RANK' not in os.environ: #检查是否存在环境变量"LOCAL_RANK"，如果不存在，则将其设置为命令行参数args.local_rank对应的值
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options: #检查命令行参数是否同时包含--options和--cfg-options，如果是，则抛出ValueError异常，提示--options已经被弃用，应该使用--cfg-options
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options: #如果命令行参数只包含--options，打印警告信息，提示--options已经被弃用，应该使用--cfg-options，并将--options的值赋给--cfg-options
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args() #解析命令行参数

    cfg = Config.fromfile(args.config) #从配置文件中读取配置,Config 类用于操作配置文件，它支持从多种文件格式中加载配置，包括 python, json 和 yaml。本句以字典的形式返回预先写好的一个配置文件中的信息。

    # replace the ${key} with the value of cfg.key 使用replace_cfg_vals()函数替换配置文件中的占位符
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS 根据MMDET_DATASETS更新数据
    update_data_root(cfg)

    if args.cfg_options is not None: #如果存在args.cfg_options，则将args.cfg_options中的配置加载到cfg中
        cfg.merge_from_dict(args.cfg_options)                                               

    if args.auto_scale_lr: #是否存在自动缩放学习率
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True #cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # 设置从哪个checkpoint重新开始训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.初始化分布式环境,如果未设置的话，默认是非分布式训练
    if args.launcher == 'none': #args.launcher不指定就是不使用分布式
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params) # 初始化pytorch的分布式
        # re-set gpu_ids with distributed training mode使用分布式训练模式重新设置gpu_ids
        # 获取分布式信息，返回的是当前进程的id：rank，和总进程数量world_size
        _, world_size = get_dist_info()
        # 总进程数量也就是要使用的gpu数量，如果world_size为4，则使用的gpu_ids为[0,1,2,3]
        cfg.gpu_ids = range(world_size)

    # create work_dir创建工作目录并将配置文件保存到工作目录中
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config))) #将配置文件 cfg 的内容输出到指定路径，路径为 cfg.work_dir 中的文件夹中，文件名为 args.config 中的文件名；
    #osp.join() 函数用于将两个路径拼接成一个完整的路径；osp.basename() 函数用于提取路径中的文件名部分;cfg.dump() 函数用于将 cfg 的内容输出到该路径下的文件中
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) #time strftime()函数用于格式化时间
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log') #log_file 的值就是cfg.work_dir 目录下，文件名为时间戳加上 .log 后缀的路径
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)#初始化日志记录器

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    #初始化meta字典来记录一些重要的信息，比如环境变量之类的，把他们打印到日志中
    meta = dict()
    # log env info
    env_info_dict = collect_env() # 收集系统环境信息
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line) # 打印系统环境信息
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text # 记录配置文件的名称
    # log some basic info
    logger.info(f'Distributed training: {distributed}') #将一个信息级别为INFO的日志记录添加到logger对象中，输出信息为"Distributed training: {distributed}"，其中，{distributed}是一个占位符，将在输出时被实际的变量值所代替
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device() #获取当前可用的设备并将其存储在 cfg.device 中
    # set random seeds
    #初始化随机种子，以确保在训练过程中每次生成的随机数都是一致的，使得模型的训练结果更加稳定
    seed = init_random_seed(args.seed, device=cfg.device)  #init_random_seed根据传入的随机种子args.seed以及设备类型cfg.device生成一个初始的随机种子seed
    seed = seed + dist.get_rank() if args.diff_seed else seed  #如果args.diff_seed为True，则会根据当前进程的rank对seed进行调整，以确保不同进程的随机种子不同
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)#不太懂
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, #获取模型类型
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights() #对模型权重进行初始化

    datasets = [build_dataset(cfg.data.train)] #构建数据集列表（此时只有训练数据集，没有验证数据集）
    if len(cfg.workflow) == 2: #列表长度，判断是否需要添加验证数据集
        assert 'val' in [mode for (mode, _) in cfg.workflow] #判断 'val' 是否在 cfg.workflow 列表中
        val_dataset = copy.deepcopy(cfg.data.val) #构建验证数据集
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))  #调用build_dataset函数，使用val_dataset的配置项构建一个数据集对象，并将其添加到datasets列表中
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data  将 mmdet 版本、配置文件内容和类名保存在 checkpoints 中作为元数据
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],  #mmdet_version'是当前 mmdet 库的版本号加上前7位 git hash，表示当前代码的版本信息
            CLASSES=datasets[0].CLASSES) #CLASSES：值为第一个数据集的类别名称列表，表示训练数据的类别信息
    # add an attribute for visualization convenience 添加属性以方便可视化
    #训练模型
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed, #是否分布式训练
        validate=(not args.no_validate), #是否需要验证
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
