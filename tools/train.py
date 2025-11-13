import argparse
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model
from mmcv.runner import get_dist_info, init_dist
from mmdet3d.utils import get_root_logger, collect_env
import os
import time
from mmdet.apis import set_random_seed
from mmcv import mkdir_or_exist

def parse_args():
    parese = argparse.ArgumentParser(description="Train a detector")
    parese.add_argument("config")
    parese.add_argument('--resume-from')

    parese.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none"
    )
    parese.add_argument("--seed", type=int, default=0)
    parese.add_argument("--deterministic", action="store_true")
    parese.add_argument("--autoscale-lr", action='store_true')
    args = parese.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # 创建工作空间
    cfg.work_dir = os.path.join(
        "./work_dirs", os.path.splitext(os.path.basename(args.config))[0]
    )
    mkdir_or_exist(cfg.work_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name="mmdet")

    # meta 用来保存关键的信息
    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['exp_name'] = os.path.basename(args.config)

    # 分布式计算与gpu数量
    if args.launcher == "none":
        distributed = False
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    logger.info(f'Distributed training: {distributed}')

    # 随机种子设置
    if args.seed is not None:
        set_random_seed(args.seed, args.deterministic) 
        logger.info(f'set random seed to {args.seed}, deterministic: {args.deterministic}')
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # 自动学习率设置
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
    
    # 中断恢复点
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    datasets = build_dataset(cfg.data.train)
    model = build_model(cfg.model)
    train_model(model, datasets, cfg)


if __name__ == "__main__":
    main()
