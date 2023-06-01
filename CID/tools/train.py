import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import models
from config import get_cfg, update_config
from core.trainer import Trainer
from dataset import make_train_dataloader
from utils.logging import create_checkpoint, setup_logger
from utils.utils import get_optimizer, save_checkpoint

def parse_args():
    #argparse用于命令项选项与参数解析的模块
    #创建解析器，ArgumentParser对象包含将命令行解析成python数据类型所需的全部信息
    parser = argparse.ArgumentParser(description='Train CID')
    # general
    #调用add_argument（）方法添加参数
    #加载指定的yaml配置文件
    parser.add_argument('--cfg', #选项字符串的名字或者列表
                        help='experiment configure file name', #参数的帮助信息
                        required=True, #可选参数是否可以省略
                        type=str) #命令行参数应该被转换成的类型
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None, #不指定参数时的默认值
                        nargs=argparse.REMAINDER)
    # # distributed training分布式训练
    parser.add_argument('--gpus',
                        help='gpu ids for ddp training',
                        type=str)
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--port',
                        default='23459',
                        type=str,
                        help='port used to set up distributed training')
    #使用parse_args()解析添加的参数
    args = parser.parse_args()

    return args

def main():
    #解析输入参数，并更新cfg
    args = parse_args()
    cfg = get_cfg()
    update_config(cfg, args)

    final_output_dir = create_checkpoint(cfg, 'train')

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    dist_url = args.dist_url + ':{}'.format(args.port)

    ngpus_per_node = torch.cuda.device_count()
    if cfg.DDP:
        world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(world_size, dist_url, final_output_dir, args))
    else:
        main_worker(0, 1, dist_url, final_output_dir, args)

def main_worker(rank, world_size, dist_url, final_output_dir, args):
    cfg = get_cfg()
    update_config(cfg, args)
    # setup logger 用于输出运行日志
    logger, _ = setup_logger(final_output_dir, rank, 'train') #配置日志记录
    if not cfg.DDP or (cfg.DDP and rank == 0):
        logger.info(pprint.pformat(args))
        logger.info(cfg)

    # cudnn related setting GPU设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print("Use GPU: {} for training".format(rank))
    if cfg.DDP:
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.format(dist_url, world_size, rank))
        #初始化进程组
        dist.init_process_group(
            backend=cfg.DIST_BACKEND, #通信所用的后端
            init_method=dist_url, #指定如何初始化进程组的URL
            world_size=world_size, #参与任务的进程数
            rank=rank #当前进程的rank
        )

    model = models.create(cfg.MODEL.NAME, cfg, is_train=True) #该目录下没有models文件夹，但是由于前面import _init_path所以可以返回上一级目录在lib中找到models
    #进入到models以后首先是__init__,里面包含了create函数

    #用于训练信息的图形化显示
    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tblog')),
        'train_global_steps': 0
    }

    if cfg.DDP:
        if cfg.MODEL.SYNC_BN:
            print('use sync bn') #SyncBatchNorm仅支持在DDP模式下使用，且要求每个显卡部署一个进程
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #在DDP模式下进行网络训练前将BatchNorm*D调整为SyncBatchNorm
        torch.cuda.set_device(rank) #可以更方便地将模型和数据加载到对应GPU上(每一个Process都应该运行在它自己对应的GPU上)
        model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]) #一种数据并行化方式：一个Device负责处理数据的一个切片 (即Batch的一部分)
    else:
        model = torch.nn.DataParallel(model).cuda() #一种数据并行化方式

    # Data loading code（数据加载代码）
    train_loader = make_train_dataloader(cfg, distributed=cfg.DDP) 
    logger.info(train_loader.dataset)

    #优化策略相关配置
    best_perf = -1
    last_epoch = -1
    optimizer = get_optimizer(cfg, model.parameters()) #获取优化器配置

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH #获取优化器配置，开始的轮次
    checkpoint_file = os.path.join(
        final_output_dir, 'model', 'checkpoint.pth.tar') #获取优化器配置，类似存档
    #断点的自动恢复
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict']) #加入模型参数
        optimizer.load_state_dict(checkpoint['optimizer']) #加入优化器的参数
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # save config file
    if not cfg.DDP or (cfg.DDP and rank == 0):
        src_folder = os.path.join(final_output_dir, 'src')
        if not os.path.exists(os.path.join(src_folder, 'lib')):
            shutil.copytree('lib', os.path.join(src_folder, 'lib'))
            shutil.copytree('tools', os.path.join(src_folder, 'tools'))
            shutil.copy2(args.cfg, src_folder)
        else:
            logger.info("=> src files are already existed in: {}".format(src_folder))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( #学习率
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    trainer = Trainer(cfg, model, rank, final_output_dir, writer_dict) #声明trainer类
    #开始训练
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if cfg.DDP:
            train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch, train_loader, optimizer)

        lr_scheduler.step()

        perf_indicator = epoch
        #对比结果，保存最优模型
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.DDP or (cfg.DDP and rank == 0):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            #保存断点
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'model', 'final_state{}.pth.tar'.format(rank)
    ) #最后模型输出的位置
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()